"""
Embedding Generator Module
Handles creation of text embeddings using OpenAI-compatible APIs.
"""

from typing import List, Union, Dict, Any
from openai import OpenAI
from src.utils.logger import app_logger as logger
import time

from ..utils.exceptions import EmbeddingError
from config.settings import settings


class EmbeddingGenerator:
    """Generate text embeddings for vector storage and retrieval."""
    
    def __init__(self):
        """Initialize the embedding generator with Requesty API."""
        try:
            self.client = OpenAI(
                api_key=settings.requesty_api_key,
                base_url=settings.requesty_base_url
            )
            self.model = settings.embedding_model
            self.dimension = settings.embedding_dimension
            self.batch_size = 100  # Process embeddings in batches
            self.max_retries = 3
            self.retry_delay = 1  # seconds
            
            logger.info(f"EmbeddingGenerator initialized with model: {self.model}, dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingGenerator: {str(e)}")
            raise EmbeddingError(f"EmbeddingGenerator initialization failed: {str(e)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty text")
        
        # Truncate text if too long (most models have token limits)
        text = self._truncate_text(text, max_tokens=8000)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    dimensions=self.dimension
                )
                
                embedding = response.data[0].embedding
                
                # Validate embedding
                if not embedding or len(embedding) != self.dimension:
                    raise EmbeddingError(
                        f"Invalid embedding dimension: expected {self.dimension}, got {len(embedding)}"
                    )
                
                logger.debug(f"Generated embedding for text of length {len(text)}")
                return embedding
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to generate embedding after {self.max_retries} attempts")
                    raise EmbeddingError(f"Embedding generation failed: {str(e)}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {self.batch_size}")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._generate_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Processed batch {i // self.batch_size + 1}/{(len(texts) - 1) // self.batch_size + 1}")
        
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def _generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a single batch.
        
        Args:
            texts: Batch of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Truncate texts if needed
        truncated_texts = [self._truncate_text(text, max_tokens=8000) for text in texts]
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=truncated_texts,
                    dimensions=self.dimension
                )
                
                embeddings = [item.embedding for item in response.data]
                
                # Validate all embeddings
                for idx, embedding in enumerate(embeddings):
                    if not embedding or len(embedding) != self.dimension:
                        raise EmbeddingError(
                            f"Invalid embedding dimension at index {idx}: "
                            f"expected {self.dimension}, got {len(embedding)}"
                        )
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"Batch attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to generate batch embeddings after {self.max_retries} attempts")
                    raise EmbeddingError(f"Batch embedding generation failed: {str(e)}")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding specifically for a search query.
        
        This method can be extended to apply query-specific preprocessing
        or use different embedding strategies for queries vs documents.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        # For now, use the same method as document embedding
        # Can be extended with query-specific preprocessing
        logger.debug(f"Generating query embedding for: {query[:100]}...")
        return self.generate_embedding(query)
    
    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """
        Truncate text to fit within token limits.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens (approximate)
            
        Returns:
            Truncated text
        """
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        logger.warning(f"Truncating text from {len(text)} to {max_chars} characters")
        return text[:max_chars]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding configuration.
        
        Returns:
            Dictionary with embedding stats
        """
        return {
            "model": self.model,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries
        }
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate an embedding vector.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not embedding:
            return False
        
        if len(embedding) != self.dimension:
            logger.warning(f"Invalid embedding dimension: expected {self.dimension}, got {len(embedding)}")
            return False
        
        # Check if all values are numbers
        try:
            for val in embedding:
                if not isinstance(val, (int, float)):
                    return False
        except Exception:
            return False
        
        return True
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if len(embedding1) != len(embedding2):
            raise EmbeddingError("Embeddings must have the same dimension")
        
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Compute magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        # Compute cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        
        # Normalize to 0-1 range
        return (similarity + 1) / 2
