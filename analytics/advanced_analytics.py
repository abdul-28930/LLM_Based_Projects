from textblob import TextBlob
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    def __init__(self):
        """Initialize the advanced analytics module."""
        logger.info("Initializing advanced analytics")
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text using TextBlob.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Convert polarity to label
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
                
            return {
                "label": label,
                "score": polarity,
                "confidence": abs(polarity)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "label": "neutral",
                "score": 0.0,
                "confidence": 0.0
            }
            
    def batch_analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries containing sentiment analysis results
        """
        return [self.analyze_sentiment(text) for text in texts] 