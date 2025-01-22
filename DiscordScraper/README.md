# Discord AI Analysis Bot

A powerful Discord bot that collects messages and performs advanced AI analysis using sentiment analysis, RAG (Retrieval-Augmented Generation), and LLM-based insights.

## Features

- Message collection and storage
- Real-time sentiment analysis
- RAG-based message retrieval and analysis
- LLM-powered insights using DeepSeek
- Channel analysis and user engagement metrics

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- Discord Bot Token
- 8GB+ RAM

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd DiscordScraper
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Discord bot token and other configurations
```

## Usage

1. Start the bot:
```bash
python bot.py
```

2. Available commands:
- `!analyze [days=7]`: Analyze channel messages for the specified number of days
- `!query [text]`: Search for similar messages using RAG
- More commands coming soon...

## Project Structure

```
DiscordScraper/
├── bot.py                 # Main bot file
├── requirements.txt       # Project dependencies
├── .env.example          # Example environment variables
├── README.md             # This file
├── data/                 # Data storage
│   ├── discord_data.db   # SQLite database
│   └── faiss_index       # FAISS vector index
└── services/             # Core services
    ├── database.py       # Database operations
    ├── sentiment.py      # Sentiment analysis
    ├── rag.py           # RAG implementation
    └── llm.py           # LLM service
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 