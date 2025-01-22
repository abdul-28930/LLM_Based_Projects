from typing import Dict, Any
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize the sentiment analyzer with a lightweight model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze the sentiment of a given text."""
        try:
            # Truncate text if it's too long
            max_length = self.tokenizer.model_max_length
            truncated_text = self.tokenizer.decode(
                self.tokenizer.encode(text, truncation=True, max_length=max_length)
            )

            # Get sentiment prediction
            result = self.sentiment_pipeline(truncated_text)[0]
            
            return {
                'score': float(result['score']),
                'label': result['label'],
                'original_length': len(text),
                'truncated_length': len(truncated_text)
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {
                'score': 0.0,
                'label': 'NEUTRAL',
                'error': str(e)
            }

    async def batch_analyze(self, texts: list[str]) -> list[Dict[str, Any]]:
        """Analyze sentiment for multiple texts in batch."""
        try:
            results = self.sentiment_pipeline(texts, batch_size=32)
            return [
                {
                    'score': float(result['score']),
                    'label': result['label']
                }
                for result in results
            ]
        except Exception as e:
            print(f"Error in batch sentiment analysis: {e}")
            return [
                {
                    'score': 0.0,
                    'label': 'NEUTRAL',
                    'error': str(e)
                }
            ] * len(texts) 