import os
from typing import List, Dict, Any
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from datetime import datetime

class RAGSystem:
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "data/faiss_index"):
        """Initialize the RAG system with FAISS index and embedding model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model).to(self.device)
        self.index_path = index_path
        self.dimension = 384  # Dimension of the embedding model
        
        # Initialize or load FAISS index
        if os.path.exists(f"{index_path}.index"):
            self.index = faiss.read_index(f"{index_path}.index")
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using the model."""
        inputs = self.tokenizer(text, return_tensors="pt", 
                              truncation=True, max_length=512,
                              padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings[0]

    async def add_message(self, message_data: Dict[str, Any]):
        """Add a message to the RAG system."""
        embedding = self._get_embedding(message_data['content'])
        self.index.add(np.array([embedding]))
        
        # Save index periodically
        if self.index.ntotal % 100 == 0:  # Save every 100 messages
            faiss.write_index(self.index, f"{self.index_path}.index")

    async def query(self, query: str, k: int = 5) -> str:
        """Query the RAG system with a question."""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Search similar vectors
        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # Invalid index
                continue
                
            result = {
                'distance': float(dist),
                'index': int(idx),
                'similarity': 1 / (1 + float(dist))  # Convert distance to similarity
            }
            results.append(result)
        
        # Format response
        response = f"Found {len(results)} relevant messages:\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. Similarity: {result['similarity']:.2f}\n"
            # Add message content here when integrated with database
        
        return response

    def save_index(self):
        """Save the FAISS index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.index") 