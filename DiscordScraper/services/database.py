import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
import sqlite3
import json
import aiosqlite
from pathlib import Path

class Database:
    def __init__(self, db_path: str = "data/discord_data.db"):
        self.db_path = db_path
        Path(os.path.dirname(db_path)).mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialize the database with required tables."""
        async with aiosqlite.connect(self.db_path) as db:
            # Create messages table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    content TEXT,
                    author_id TEXT,
                    author_name TEXT,
                    channel_id TEXT,
                    channel_name TEXT,
                    guild_id TEXT,
                    timestamp TEXT,
                    embedding BLOB
                )
            """)

            # Create sentiments table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sentiments (
                    message_id TEXT PRIMARY KEY,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    FOREIGN KEY (message_id) REFERENCES messages(message_id)
                )
            """)

            await db.commit()

    async def store_message(self, message_data: Dict[str, Any]):
        """Store a message in the database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO messages 
                (message_id, content, author_id, author_name, channel_id, 
                channel_name, guild_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_data['message_id'],
                message_data['content'],
                message_data['author_id'],
                message_data['author_name'],
                message_data['channel_id'],
                message_data['channel_name'],
                message_data['guild_id'],
                message_data['timestamp']
            ))
            await db.commit()

    async def store_sentiment(self, message_id: str, sentiment_data: Dict[str, Any]):
        """Store sentiment analysis results."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO sentiments 
                (message_id, sentiment_score, sentiment_label)
                VALUES (?, ?, ?)
            """, (
                message_id,
                sentiment_data['score'],
                sentiment_data['label']
            ))
            await db.commit()

    async def get_channel_messages(self, channel_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Retrieve messages from a channel within the specified time period."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT m.*, s.sentiment_score, s.sentiment_label
                FROM messages m
                LEFT JOIN sentiments s ON m.message_id = s.message_id
                WHERE m.channel_id = ? AND m.timestamp > ?
                ORDER BY m.timestamp DESC
            """, (channel_id, cutoff_date)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def search_messages(self, query: str) -> List[Dict[str, Any]]:
        """Search messages using full-text search."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT m.*, s.sentiment_score, s.sentiment_label
                FROM messages m
                LEFT JOIN sentiments s ON m.message_id = s.message_id
                WHERE m.content LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT 100
            """, (f"%{query}%",)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows] 