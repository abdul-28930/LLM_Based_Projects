"""Discord bot for message scraping and analytics."""

import os
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from database import Database
from monitoring import MetricsLogger, monitor_performance, log_command

# Set up logging with UTF-8 encoding
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Use stdout instead of default stderr
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DiscordScraper(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix="!", intents=intents)
        
        # Initialize components
        self.db = Database()
        self.metrics = MetricsLogger(log_interval=300)  # Log metrics every 5 minutes
        
        # Dictionary to store different log channel IDs per guild
        self.log_channels = {}
        
    @monitor_performance
    async def setup_hook(self):
        """Initialize the bot's database and sync commands."""
        try:
            await self.db.initialize()
            
            # Load extensions
            extensions = [
                'analytics.commands',
                'dashboard.commands'
            ]
            
            for extension in extensions:
                try:
                    await self.load_extension(extension)
                    logger.info(f"Loaded {extension} extension")
                except Exception as e:
                    logger.error(f"Failed to load {extension}: {e}")
                    
            await self.tree.sync()
            await self.metrics.start()
            logger.info("Bot setup completed")
            
        except Exception as e:
            logger.error(f"Error in bot setup: {e}")
            raise

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info("------")

    async def ensure_log_channels(self, guild: discord.Guild) -> dict:
        """Ensure all log channels exist in the guild."""
        # Check if bot has required permissions
        bot_member = guild.get_member(self.user.id)
        if not bot_member:
            logger.error("Could not find bot member in guild")
            raise discord.errors.Forbidden(None, "Bot not found in guild")

        # Look for existing category or create new one
        category_name = "dyno utility"
        category = discord.utils.get(guild.categories, name=category_name)
        
        # Set up permissions
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(view_channel=False),
            bot_member: discord.PermissionOverwrite(
                view_channel=True,
                send_messages=True,
                embed_links=True,
                attach_files=True,
                read_message_history=True,
                manage_messages=True
            ),
            guild.me: discord.PermissionOverwrite(
                view_channel=True,
                send_messages=True,
                embed_links=True,
                attach_files=True,
                read_message_history=True,
                manage_messages=True
            )
        }

        if not category:
            try:
                category = await guild.create_category(name=category_name, overwrites=overwrites)
            except discord.Forbidden:
                logger.error("Missing permissions to create category")
                raise discord.errors.Forbidden(None, "Missing permissions to create category")
            except Exception as e:
                logger.error(f"Error creating category: {e}")
                raise

        # Channel configurations
        log_channels = {
            'message_logs': ('new-messages', '📝 New message tracking initialized'),
            'channel_logs': ('channel-updates', '🔄 Channel tracking initialized'),
            'deletion_logs': ('deletions', '❌ Deletion tracking initialized'),
            'edit_logs': ('edits', '📝 Edit tracking initialized'),
            'archive_logs': ('archives', '🗄️ Archive tracking initialized')
        }

        channels = {}
        guild_channels = {}

        # Create or get each log channel
        for channel_type, (base_name, init_message) in log_channels.items():
            try:
                # Check for existing channel
                channel = discord.utils.get(category.channels, name=base_name)
                
                if not channel:
                    try:
                        channel = await guild.create_text_channel(
                            name=base_name,
                            category=category,
                            overwrites=overwrites,
                            topic=f"Log channel for {channel_type}"
                        )
                        await channel.send(f"🔄 {init_message}. Events will be logged here.")
                    except discord.Forbidden:
                        logger.error(f"Missing permissions to create channel {base_name}")
                        continue
                    except Exception as e:
                        logger.error(f"Error creating channel {base_name}: {e}")
                        continue

                # Verify bot has permission to send messages
                if not channel.permissions_for(bot_member).send_messages:
                    logger.error(f"Bot doesn't have permission to send messages in {channel.name}")
                    try:
                        await channel.edit(overwrites=overwrites)
                    except:
                        continue

                channels[channel_type] = channel
                guild_channels[channel_type] = channel.id

            except Exception as e:
                logger.error(f"Error handling channel {base_name}: {e}")

        if not channels:
            raise discord.errors.Forbidden(None, "Could not create or access any log channels. Please check bot permissions.")

        self.log_channels[guild.id] = guild_channels
        return channels

    async def log_event(self, guild: discord.Guild, channel_type: str, embed: discord.Embed):
        """Log an event to the appropriate log channel."""
        try:
            channels = await self.ensure_log_channels(guild)
            if channel_type in channels:
                await channels[channel_type].send(embed=embed)
        except Exception as e:
            logger.error(f"Error logging {channel_type} event: {e}")

    @monitor_performance("message_processing")
    async def on_message(self, message: discord.Message):
        """Event handler for new messages."""
        if message.author.bot or not message.guild:
            return

        # Skip messages in log channels
        if message.channel.id in self.log_channels.get(message.guild.id, {}).values():
            return

        try:
            message_data = {
                'id': message.id,
                'channel_id': message.channel.id,
                'guild_id': message.guild.id,
                'author_id': message.author.id,
                'author_name': message.author.name,
                'content': message.content,
                'created_at': message.created_at.isoformat()
            }
            
            if await self.db.add_message(message_data):
                embed = discord.Embed(
                    title="✉️ New Message",
                    description=f"Message sent in {message.channel.mention}",
                    color=discord.Color.green(),
                    timestamp=message.created_at
                )
                embed.add_field(name="Author", value=f"{message.author.name} ({message.author.id})", inline=False)
                embed.add_field(name="Content", value=message.content[:1024] if message.content else "[No content]", inline=False)
                await self.log_event(message.guild, 'message_logs', embed)

            # Process commands after handling the message
            await self.process_commands(message)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics.monitor.track_error("message_processing_error")

    @monitor_performance("message_deletion")
    async def on_message_delete(self, message: discord.Message):
        """Event handler for deleted messages."""
        if message.author.bot or not message.guild:
            return

        if message.channel.id in self.log_channels.get(message.guild.id, {}).values():
            return

        try:
            if await self.db.mark_message_deleted(message.id):
                embed = discord.Embed(
                    title="🗑️ Message Deleted",
                    description=f"Message deleted in {message.channel.mention}",
                    color=discord.Color.red(),
                    timestamp=datetime.utcnow()
                )
                embed.add_field(name="Author", value=f"{message.author.name} ({message.author.id})", inline=False)
                embed.add_field(name="Content", value=message.content[:1024] if message.content else "[No content]", inline=False)
                await self.log_event(message.guild, 'deletion_logs', embed)

        except Exception as e:
            logger.error(f"Error handling message deletion: {e}")
            self.metrics.monitor.track_error("message_deletion_error")

    @monitor_performance("message_edit")
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Event handler for edited messages."""
        if before.author.bot or not before.guild:
            return

        if before.channel.id in self.log_channels.get(before.guild.id, {}).values():
            return

        try:
            if before.content != after.content:
                if await self.db.update_message(after.id, after.content):
                    embed = discord.Embed(
                        title="✏️ Message Edited",
                        description=f"Message edited in {before.channel.mention}",
                        color=discord.Color.blue(),
                        timestamp=datetime.utcnow()
                    )
                    embed.add_field(name="Author", value=f"{before.author.name} ({before.author.id})", inline=False)
                    embed.add_field(name="Before", value=before.content[:1024] if before.content else "[No content]", inline=False)
                    embed.add_field(name="After", value=after.content[:1024] if after.content else "[No content]", inline=False)
                    await self.log_event(before.guild, 'edit_logs', embed)

        except Exception as e:
            logger.error(f"Error handling message edit: {e}")
            self.metrics.monitor.track_error("message_edit_error")

    @monitor_performance("channel_deletion")
    async def on_guild_channel_delete(self, channel: discord.abc.GuildChannel):
        """Event handler for channel deletions."""
        try:
            if await self.db.archive_channel(channel.id, channel.guild.id):
                embed = discord.Embed(
                    title="📂 Channel Archived",
                    description="Channel and its messages have been moved to archive",
                    color=discord.Color.orange(),
                    timestamp=datetime.utcnow()
                )
                embed.add_field(name="Channel", value=f"{channel.name} ({channel.id})", inline=False)
                embed.add_field(name="Type", value=str(channel.type), inline=False)
                await self.log_event(channel.guild, 'archive_logs', embed)

        except Exception as e:
            logger.error(f"Error archiving channel: {e}")
            self.metrics.monitor.track_error("channel_archive_error")

    async def setup_bot(self):
        """Set up the bot and load all extensions."""
        try:
            await self.load_extension('analytics.commands')
            logger.info("Loaded analytics commands extension")
        except Exception as e:
            logger.error(f"Failed to load analytics commands: {e}")
            self.metrics.monitor.track_error("extension_load_error")

    async def close(self):
        """Clean up resources when shutting down."""
        await self.metrics.stop()  # Stop metrics logging
        await super().close()

bot = DiscordScraper()

@bot.tree.command(name="track", description="Track all channels in the server")
@app_commands.checks.has_permissions(manage_guild=True)
async def track(interaction: discord.Interaction):
    """Track all channels in the server and save their information."""
    await interaction.response.defer(thinking=True)
    
    try:
        guild = interaction.guild
        if not guild:
            await interaction.followup.send("❌ This command can only be used in a server!")
            return

        # Ensure log channels exist
        log_channels = await bot.ensure_log_channels(guild)

        # Initialize counters
        new_channels = []
        archived_channels = []
        updated_channels = []
        errors = []
        
        # Create embed for progress
        embed = discord.Embed(
            title="📊 Channel Tracking Progress",
            description="Scanning all channels...",
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        progress_msg = await interaction.followup.send(embed=embed)

        # Get current channel IDs from database
        current_channel_ids = set(await bot.db.get_current_channel_ids(guild.id))
        
        # Get all current Discord channel IDs
        discord_channel_ids = set()

        # Track all channels
        for channel in guild.channels:
            # Skip log channels
            if channel.id in bot.log_channels.get(guild.id, {}).values():
                continue

            try:
                discord_channel_ids.add(channel.id)
                channel_data = {
                    'id': channel.id,
                    'guild_id': guild.id,
                    'name': channel.name,
                    'type': str(channel.type),
                    'parent_id': channel.category.id if hasattr(channel, 'category') and channel.category else None,
                    'position': getattr(channel, 'position', 0),
                    'created_at': channel.created_at.isoformat()
                }
                
                if channel.id in current_channel_ids:
                    if await bot.db.add_channel(channel_data):
                        updated_channels.append(channel.name)
                        # Log channel update
                        update_embed = discord.Embed(
                            title="🔄 Channel Updated",
                            description=f"Channel information updated",
                            color=discord.Color.blue(),
                            timestamp=datetime.utcnow()
                        )
                        update_embed.add_field(name="Channel", value=f"{channel.name} ({channel.id})", inline=False)
                        await bot.log_event(guild, 'channel_logs', update_embed)
                else:
                    if await bot.db.add_channel(channel_data):
                        new_channels.append(channel.name)
                        # Log new channel
                        new_embed = discord.Embed(
                            title="✨ New Channel Added",
                            description=f"New channel tracked",
                            color=discord.Color.green(),
                            timestamp=datetime.utcnow()
                        )
                        new_embed.add_field(name="Channel", value=f"{channel.name} ({channel.id})", inline=False)
                        await bot.log_event(guild, 'channel_logs', new_embed)

                # Update progress
                if len(new_channels) % 5 == 0 or len(updated_channels) % 5 == 0:
                    embed.description = f"Processing... Found {len(new_channels)} new, {len(updated_channels)} updated channels"
                    await progress_msg.edit(embed=embed)
                    
            except Exception as e:
                logger.error(f"Error tracking channel {channel.name}: {e}")
                errors.append(f"Error with {channel.name}: {str(e)}")

        # Archive channels that no longer exist
        channels_to_archive = current_channel_ids - discord_channel_ids
        for channel_id in channels_to_archive:
            if await bot.db.archive_channel(channel_id, guild.id):
                archived_channels.append(str(channel_id))
                # Log channel archival
                archive_embed = discord.Embed(
                    title="📂 Channel Archived",
                    description="Channel and its messages have been moved to archive",
                    color=discord.Color.orange(),
                    timestamp=datetime.utcnow()
                )
                archive_embed.add_field(name="Channel ID", value=str(channel_id), inline=False)
                await bot.log_event(guild, 'archive_logs', archive_embed)

        # Create final embed
        final_embed = discord.Embed(
            title="✅ Channel Tracking Complete",
            description=f"Successfully tracked all channels in {guild.name}!",
            color=discord.Color.green(),
            timestamp=datetime.utcnow()
        )
        
        # Add statistics summary
        total_tracked = len(new_channels) + len(updated_channels)
        final_embed.add_field(
            name="📊 Summary",
            value=f"• Total Channels Tracked: {total_tracked}\n"
                  f"• New Channels: {len(new_channels)}\n"
                  f"• Updated Channels: {len(updated_channels)}\n"
                  f"• Archived Channels: {len(archived_channels)}",
            inline=False
        )
        
        # Add new channels field
        if new_channels:
            new_channels_text = "\n".join(f"• {name}" for name in new_channels[:10])
            if len(new_channels) > 10:
                new_channels_text += f"\n• ...and {len(new_channels) - 10} more"
            final_embed.add_field(
                name=f"✨ New Channels",
                value=new_channels_text or "None",
                inline=False
            )
        
        # Add updated channels field
        if updated_channels:
            updated_channels_text = "\n".join(f"• {name}" for name in updated_channels[:10])
            if len(updated_channels) > 10:
                updated_channels_text += f"\n• ...and {len(updated_channels) - 10} more"
            final_embed.add_field(
                name=f"🔄 Updated Channels",
                value=updated_channels_text or "None",
                inline=False
            )
        
        # Add archived channels field
        if archived_channels:
            archived_channels_text = "\n".join(f"• ID: {id}" for id in archived_channels[:10])
            if len(archived_channels) > 10:
                archived_channels_text += f"\n• ...and {len(archived_channels) - 10} more"
            final_embed.add_field(
                name=f"📁 Archived Channels",
                value=archived_channels_text or "None",
                inline=False
            )
        
        # Add errors field if any
        if errors:
            errors_text = "\n".join(f"• {error}" for error in errors[:5])
            if len(errors) > 5:
                errors_text += f"\n• ...and {len(errors) - 5} more errors"
            final_embed.add_field(
                name=f"⚠️ Errors",
                value=errors_text,
                inline=False
            )
        
        final_embed.set_footer(text=f"Requested by {interaction.user.name}")
        await progress_msg.edit(embed=final_embed)

        # Send success message
        success_message = (
            f"✅ **Channel tracking completed in {guild.name}!**\n"
            f"• Tracked {total_tracked} channels\n"
            f"• Found {len(new_channels)} new channels\n"
            f"• Updated {len(updated_channels)} existing channels\n"
            f"• Archived {len(archived_channels)} deleted channels\n\n"
            f"📝 Events will be logged in the following channels:\n"
            f"• {log_channels['message_logs'].mention} - New messages\n"
            f"• {log_channels['channel_logs'].mention} - Channel updates\n"
            f"• {log_channels['deletion_logs'].mention} - Deletions\n"
            f"• {log_channels['edit_logs'].mention} - Message edits\n"
            f"• {log_channels['archive_logs'].mention} - Archives"
        )
        if errors:
            success_message += f"\n⚠️ Encountered {len(errors)} errors during tracking"
        
        await interaction.followup.send(success_message)

    except Exception as e:
        logger.error(f"Error in track command: {e}")
        await interaction.followup.send(f"❌ An error occurred: {str(e)}")

@track.error
async def track_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    """Error handler for the track command."""
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message(
            "❌ You need the 'Manage Server' permission to use this command!",
            ephemeral=True
        )
    else:
        logger.error(f"Unexpected error in track command: {error}")
        await interaction.response.send_message(
            "❌ An unexpected error occurred!",
            ephemeral=True
        )

@bot.tree.command(name="sync", description="Start tracking all messages in the server")
@app_commands.checks.has_permissions(manage_guild=True)
async def sync_messages(interaction: discord.Interaction):
    """Initialize message tracking and sync message history."""
    await interaction.response.defer(thinking=True)
    
    try:
        guild = interaction.guild
        if not guild:
            await interaction.followup.send("❌ This command can only be used in a server!")
            return

        # Ensure log channel exists
        log_channel = await bot.ensure_log_channels(guild)
        
        # Initialize counters
        messages_synced = 0
        channels_processed = 0
        errors = []
        
        # Create progress embed
        embed = discord.Embed(
            title="📥 Message Sync Progress",
            description="Starting message sync...",
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        progress_msg = await interaction.followup.send(embed=embed)

        # Process each text channel
        for channel in guild.channels:
            if isinstance(channel, discord.TextChannel) and channel.id != log_channel['message_logs'].id:
                try:
                    channel_messages = 0
                    async for message in channel.history(limit=None):
                        if not message.author.bot:  # Skip bot messages
                            message_data = {
                                'id': message.id,
                                'channel_id': channel.id,
                                'guild_id': guild.id,
                                'author_id': message.author.id,
                                'author_name': message.author.name,
                                'content': message.content,
                                'created_at': message.created_at.isoformat(),
                                'edited_at': message.edited_at.isoformat() if message.edited_at else None
                            }
                            
                            if await bot.db.add_message(message_data):
                                messages_synced += 1
                                channel_messages += 1
                                
                                # Update progress every 100 messages
                                if messages_synced % 100 == 0:
                                    embed.description = (
                                        f"Syncing messages...\n\n"
                                        f"📥 Processed: {messages_synced:,} messages\n"
                                        f"📁 Current: {channel.name}\n"
                                        f"✅ Completed: {channels_processed} channels"
                                    )
                                    await progress_msg.edit(embed=embed)
                    
                    channels_processed += 1
                    logger.info(f"Synced {channel_messages} messages from {channel.name}")
                    
                except Exception as e:
                    error_msg = f"Error in channel {channel.name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

        # Create final embed
        final_embed = discord.Embed(
            title="✅ Message Tracking Initialized",
            description=(
                f"Successfully synced message history in {guild.name}!\n\n"
                "📝 **The bot will now automatically track:**\n"
                "• All new messages\n"
                "• Message edits\n"
                "• Message deletions\n\n"
                f"📊 Events will be logged in {log_channel['message_logs'].mention}"
            ),
            color=discord.Color.green(),
            timestamp=datetime.utcnow()
        )
        
        # Add statistics
        final_embed.add_field(
            name="📊 Sync Summary",
            value=(
                f"• Messages Synced: {messages_synced:,}\n"
                f"• Channels Processed: {channels_processed}\n"
                f"• Tracking Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            ),
            inline=False
        )
        
        # Add errors if any
        if errors:
            errors_text = "\n".join(f"• {error}" for error in errors[:5])
            if len(errors) > 5:
                errors_text += f"\n• ...and {len(errors) - 5} more errors"
            final_embed.add_field(
                name="⚠️ Errors",
                value=errors_text,
                inline=False
            )
        
        final_embed.set_footer(text=f"Requested by {interaction.user.name}")
        await progress_msg.edit(embed=final_embed)

        # Send success message
        success_message = (
            f"✅ **Message tracking initialized in {guild.name}!**\n"
            f"• Synced {messages_synced:,} messages\n"
            f"• Processed {channels_processed} channels\n"
            f"• All future message events will be logged in {log_channel['message_logs'].mention}"
        )
        if errors:
            success_message += f"\n⚠️ Encountered {len(errors)} errors during sync"
        
        await interaction.followup.send(success_message)

        # Send initialization message to log channel
        log_embed = discord.Embed(
            title="🚀 Message Tracking Started",
            description=(
                "Message tracking has been initialized for this server.\n\n"
                "**The following events will be logged:**\n"
                "• ✉️ New messages\n"
                "• ✏️ Message edits\n"
                "• 🗑️ Message deletions"
            ),
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        log_embed.set_footer(text=f"Initialized by {interaction.user.name}")
        await log_channel['message_logs'].send(embed=log_embed)

    except Exception as e:
        logger.error(f"Error in sync command: {e}")
        await interaction.followup.send(f"❌ An error occurred: {str(e)}")

@sync_messages.error
async def sync_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    """Error handler for the sync command."""
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message(
            "❌ You need the 'Manage Server' permission to use this command!",
            ephemeral=True
        )
    else:
        logger.error(f"Unexpected error in sync command: {error}")
        await interaction.response.send_message(
            "❌ An unexpected error occurred!",
            ephemeral=True
        )

def run_bot():
    """Run the Discord bot."""
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        raise ValueError("No Discord token found in environment variables")
    
    try:
        bot.run(token)
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        raise

if __name__ == "__main__":
    run_bot() 