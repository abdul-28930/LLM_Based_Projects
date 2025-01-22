import os
import asyncio
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

from services.database import Database
from services.sentiment import SentimentAnalyzer
from services.rag import RAGSystem
from services.llm import DeepSeekLLM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/discord_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Bot configuration
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize services
db = Database()
sentiment_analyzer = SentimentAnalyzer()
rag_system = RAGSystem()
llm = DeepSeekLLM()

@bot.event
async def on_ready():
    """Called when the bot is ready and connected to Discord."""
    logger.info(f'{bot.user} has connected to Discord!')
    await db.initialize()
    backup_data.start()

@tasks.loop(hours=24)
async def backup_data():
    """Backup data daily"""
    try:
        logger.info("Starting daily backup...")
        rag_system.save_index()
        logger.info("Daily backup completed successfully")
    except Exception as e:
        logger.error(f"Error during backup: {e}")

@bot.event
async def on_message(message: discord.Message):
    """Process and store each new message."""
    if message.author == bot.user:
        return

    try:
        # Extract message data
        message_data = {
            'message_id': str(message.id),
            'content': message.content,
            'author_id': str(message.author.id),
            'author_name': message.author.name,
            'channel_id': str(message.channel.id),
            'channel_name': message.channel.name,
            'guild_id': str(message.guild.id),
            'timestamp': message.created_at.isoformat(),
        }

        # Store message in database
        await db.store_message(message_data)

        # Analyze sentiment
        sentiment = await sentiment_analyzer.analyze(message.content)
        await db.store_sentiment(message.id, sentiment)

        # Add to RAG system
        await rag_system.add_message(message_data)

        # Process commands after storing
        await bot.process_commands(message)

    except Exception as e:
        logger.error(f"Error processing message: {e}")

@bot.command(name='analyze')
async def analyze_channel(ctx: commands.Context, days: int = 7, focus: str = "general"):
    """
    Analyze channel messages.
    Usage: !analyze [days=7] [focus=general|sentiment|engagement|topics]
    """
    try:
        # Send initial response
        initial_message = await ctx.send("🔍 Analyzing messages... Please wait.")

        # Get messages
        messages = await db.get_channel_messages(ctx.channel.id, days)
        
        if not messages:
            await initial_message.edit(content="No messages found in the specified time period.")
            return

        # Get analysis based on focus
        if focus == "sentiment":
            analysis = await analyze_sentiment_trends(messages)
        else:
            analysis = await llm.analyze_messages(messages)

        # Create embed
        embed = discord.Embed(
            title=f"Channel Analysis - Last {days} days",
            description=analysis,
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text=f"Requested by {ctx.author.name}")
        
        # Add statistics
        stats = calculate_channel_stats(messages)
        embed.add_field(name="Statistics", value=stats, inline=False)

        await initial_message.edit(content=None, embed=embed)

    except Exception as e:
        logger.error(f"Error in analyze command: {e}")
        await ctx.send(f"❌ An error occurred during analysis: {str(e)}")

@bot.command(name='query')
async def query_messages(ctx: commands.Context, *, query: str):
    """
    Search for similar messages using RAG.
    Usage: !query <your search query>
    """
    try:
        # Send initial response
        initial_message = await ctx.send("🔍 Searching messages... Please wait.")

        # Get similar messages
        response = await rag_system.query(query)
        
        # Create embed
        embed = discord.Embed(
            title="Search Results",
            description=response,
            color=discord.Color.green(),
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text=f"Requested by {ctx.author.name}")

        await initial_message.edit(content=None, embed=embed)

    except Exception as e:
        logger.error(f"Error in query command: {e}")
        await ctx.send(f"❌ An error occurred during search: {str(e)}")

@bot.command(name='stats')
async def channel_stats(ctx: commands.Context, days: int = 7):
    """
    Get channel statistics.
    Usage: !stats [days=7]
    """
    try:
        messages = await db.get_channel_messages(ctx.channel.id, days)
        stats = calculate_channel_stats(messages)
        
        embed = discord.Embed(
            title=f"Channel Statistics - Last {days} days",
            description=stats,
            color=discord.Color.gold(),
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text=f"Requested by {ctx.author.name}")
        
        await ctx.send(embed=embed)

    except Exception as e:
        logger.error(f"Error in stats command: {e}")
        await ctx.send(f"❌ An error occurred: {str(e)}")

def calculate_channel_stats(messages: List[Dict[str, Any]]) -> str:
    """Calculate channel statistics."""
    if not messages:
        return "No messages found in the specified time period."

    total_messages = len(messages)
    unique_authors = len(set(msg['author_id'] for msg in messages))
    avg_length = sum(len(msg['content']) for msg in messages) / total_messages
    
    sentiment_counts = {
        'POSITIVE': sum(1 for msg in messages if msg.get('sentiment_label') == 'POSITIVE'),
        'NEGATIVE': sum(1 for msg in messages if msg.get('sentiment_label') == 'NEGATIVE'),
        'NEUTRAL': sum(1 for msg in messages if msg.get('sentiment_label') == 'NEUTRAL')
    }

    return f"""
• Total Messages: {total_messages}
• Unique Authors: {unique_authors}
• Average Message Length: {avg_length:.1f} characters
• Sentiment Distribution:
  - Positive: {sentiment_counts['POSITIVE']} ({(sentiment_counts['POSITIVE']/total_messages)*100:.1f}%)
  - Negative: {sentiment_counts['NEGATIVE']} ({(sentiment_counts['NEGATIVE']/total_messages)*100:.1f}%)
  - Neutral: {sentiment_counts['NEUTRAL']} ({(sentiment_counts['NEUTRAL']/total_messages)*100:.1f}%)
"""

async def analyze_sentiment_trends(messages: List[Dict[str, Any]]) -> str:
    """Analyze sentiment trends in messages."""
    if not messages:
        return "No messages found for sentiment analysis."

    # Group by day
    messages_by_day = defaultdict(list)
    for msg in messages:
        date = datetime.fromisoformat(msg['timestamp']).date()
        messages_by_day[date].append(msg)

    # Calculate daily sentiments
    daily_sentiments = []
    for date, day_messages in sorted(messages_by_day.items()):
        positive = sum(1 for msg in day_messages if msg.get('sentiment_label') == 'POSITIVE')
        negative = sum(1 for msg in day_messages if msg.get('sentiment_label') == 'NEGATIVE')
        total = len(day_messages)
        
        daily_sentiments.append({
            'date': date,
            'positive_ratio': positive/total if total > 0 else 0,
            'negative_ratio': negative/total if total > 0 else 0,
            'message_count': total
        })

    # Format analysis
    analysis = "📊 **Sentiment Analysis Report**\n\n"
    for day in daily_sentiments:
        analysis += f"**{day['date']}**:\n"
        analysis += f"• Messages: {day['message_count']}\n"
        analysis += f"• Positive: {day['positive_ratio']*100:.1f}%\n"
        analysis += f"• Negative: {day['negative_ratio']*100:.1f}%\n\n"

    return analysis

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