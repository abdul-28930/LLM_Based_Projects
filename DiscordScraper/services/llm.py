from typing import List, Dict, Any
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime, timedelta
from collections import defaultdict

class DeepSeekLLM:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-6.7b-base"):
        """Initialize the DeepSeek LLM for analysis."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

    def _format_messages_for_analysis(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for analysis input."""
        # Group messages by day
        messages_by_day = defaultdict(list)
        for msg in messages:
            date = datetime.fromisoformat(msg['timestamp']).date()
            messages_by_day[date].append(msg)

        # Format the input
        formatted_input = "Analyze the following Discord chat messages:\n\n"
        for date, day_messages in sorted(messages_by_day.items()):
            formatted_input += f"=== {date} ===\n"
            for msg in day_messages:
                formatted_input += f"{msg['author_name']}: {msg['content']}\n"
            formatted_input += "\n"

        return formatted_input

    def _generate_analysis_prompt(self, formatted_messages: str) -> str:
        """Generate the analysis prompt."""
        return f"""
{formatted_messages}

Please provide a comprehensive analysis of these Discord messages, including:
1. Key discussion topics and themes
2. Overall sentiment and mood
3. User engagement patterns
4. Notable interactions or important information
5. Recommendations for community engagement

Format the analysis in a clear, structured way.
"""

    async def analyze_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Analyze a collection of messages using the LLM."""
        try:
            # Format messages and create prompt
            formatted_messages = self._format_messages_for_analysis(messages)
            prompt = self._generate_analysis_prompt(formatted_messages)

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=1000,
                    temperature=0.7,
                    top_p=0.95,
                    num_return_sequences=1
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the analysis part (after the prompt)
            analysis = response.split("Format the analysis in a clear, structured way.")[-1].strip()
            
            return analysis

        except Exception as e:
            return f"Error analyzing messages: {str(e)}"

    async def generate_insights(self, messages: List[Dict[str, Any]], 
                              focus_area: str = "general") -> str:
        """Generate specific insights based on a focus area."""
        focus_prompts = {
            "general": "Provide general insights about the conversation.",
            "engagement": "Analyze user engagement patterns and suggest improvements.",
            "sentiment": "Analyze the emotional tone and sentiment trends.",
            "topics": "Identify and analyze key discussion topics and themes."
        }

        prompt = focus_prompts.get(focus_area, focus_prompts["general"])
        formatted_messages = self._format_messages_for_analysis(messages)
        
        try:
            full_prompt = f"{formatted_messages}\n\n{prompt}"
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=500,
                    temperature=0.7,
                    top_p=0.95,
                    num_return_sequences=1
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            insights = response.split(prompt)[-1].strip()
            
            return insights

        except Exception as e:
            return f"Error generating insights: {str(e)}" 