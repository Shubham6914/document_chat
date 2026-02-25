"""
Retrieval Service Module
Handles document retrieval with hybrid search (semantic + keyword).
"""

from typing import List, Dict, Any, Optional
from src.utils.logger import app_logger as logger
import re
from collections import Counter

from ..embeddings.embedding_service import EmbeddingGenerator
from ..utils.exceptions import RetrievalError


class RetrievalService:
    """
    Advanced retrieval service with hybrid search capabilities.
    Combines semantic search (embeddings) with keyword-based search.
    """
    
    def __init__(self, vector_store):
        """
        Initialize retrieval service.
        
        Args:
            vector_store: VectorStore instance for semantic search
        """
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.semantic_weight = 0.7  # Weight for semantic search
        self.keyword_weight = 0.3   # Weight for keyword search
        
        logger.info("RetrievalService initialized with hybrid search")
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            rerank: Whether to rerank results
            
        Returns:
            List of retrieved chunks with hybrid scores
        """
        try:
            logger.info(f"Performing hybrid search for query: {query[:100]}...")
            
            # Get more candidates for reranking
            candidate_k = top_k * 3 if rerank else top_k
            
            # 1. Semantic search
            semantic_results = self.vector_store.search(
                query=query,
                top_k=candidate_k,
                filter_dict=filter_dict
            )
            
            # 2. Keyword scoring
            keyword_scores = self._compute_keyword_scores(query, semantic_results)
            
            # 3. Combine scores
            hybrid_results = self._combine_scores(
                semantic_results,
                keyword_scores,
                self.semantic_weight,
                self.keyword_weight
            )
            
            # 4. Rerank if requested
            if rerank:
                hybrid_results = self._rerank_results(query, hybrid_results)
            
            # 5. Return top_k results
            final_results = hybrid_results[:top_k]
            
            logger.info(f"Hybrid search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise RetrievalError(f"Hybrid search failed: {str(e)}")
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform pure semantic search using embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of retrieved chunks
        """
        try:
            results = self.vector_store.search(
                query=query,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            logger.info(f"Semantic search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            raise RetrievalError(f"Semantic search failed: {str(e)}")
    
    def _compute_keyword_scores(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[int, float]:
        """
        Compute keyword-based relevance scores for chunks.
        
        Args:
            query: Search query
            chunks: List of document chunks
            
        Returns:
            Dictionary mapping chunk index to keyword score
        """
        # Extract keywords from query
        query_keywords = self._extract_keywords(query)
        
        if not query_keywords:
            return {i: 0.0 for i in range(len(chunks))}
        
        keyword_scores = {}
        
        for idx, chunk in enumerate(chunks):
            text = chunk.get('text', '').lower()
            
            # Count keyword matches
            matches = 0
            total_keywords = len(query_keywords)
            
            for keyword in query_keywords:
                # Use word boundaries for exact matching
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches += len(re.findall(pattern, text, re.IGNORECASE))
            
            # Normalize score
            score = min(matches / total_keywords, 1.0) if total_keywords > 0 else 0.0
            keyword_scores[idx] = score
        
        return keyword_scores
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text)
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
            'which', 'who', 'when', 'where', 'why', 'how'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _combine_scores(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_scores: Dict[int, float],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword scores.
        
        Args:
            semantic_results: Results from semantic search
            keyword_scores: Keyword-based scores
            semantic_weight: Weight for semantic score
            keyword_weight: Weight for keyword score
            
        Returns:
            Combined results sorted by hybrid score
        """
        combined_results = []
        
        for idx, result in enumerate(semantic_results):
            semantic_score = result.get('score', 0.0)
            keyword_score = keyword_scores.get(idx, 0.0)
            
            # Compute hybrid score
            hybrid_score = (
                semantic_weight * semantic_score +
                keyword_weight * keyword_score
            )
            
            # Add hybrid score to result
            result['hybrid_score'] = hybrid_score
            result['semantic_score'] = semantic_score
            result['keyword_score'] = keyword_score
            
            combined_results.append(result)
        
        # Sort by hybrid score
        combined_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return combined_results
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using additional relevance signals.
        
        Args:
            query: Original query
            results: Initial results
            
        Returns:
            Reranked results
        """
        # For now, use a simple reranking based on text length and position
        # Can be extended with cross-encoder models
        
        for idx, result in enumerate(results):
            text = result.get('text', '')
            
            # Penalize very short or very long chunks
            length_score = 1.0
            if len(text) < 100:
                length_score = 0.8
            elif len(text) > 2000:
                length_score = 0.9
            
            # Slight boost for earlier results (trust initial ranking)
            position_score = 1.0 - (idx * 0.02)
            
            # Adjust hybrid score
            result['hybrid_score'] *= length_score * position_score
        
        # Re-sort
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks and format as context.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            filter_dict: Optional metadata filter
            include_metadata: Whether to include full metadata
            
        Returns:
            Dictionary with formatted context and metadata
        """
        try:
            # Perform hybrid search
            chunks = self.hybrid_search(
                query=query,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            if not chunks:
                return {
                    'context': 'No relevant context found in the document.',
                    'chunks': [],
                    'sources': []
                }
            
            # Format context
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(
                    f"[Section {i}] (Source: {chunk['file_name']}, "
                    f"Relevance: {chunk['hybrid_score']:.2f})\n"
                    f"{chunk['text']}"
                )
            
            context = "\n\n".join(context_parts)
            
            # Extract sources
            sources = self._extract_sources(chunks)
            
            result = {
                'context': context,
                'chunks': chunks if include_metadata else [],
                'sources': sources,
                'num_chunks': len(chunks)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieve with context failed: {str(e)}")
            raise RetrievalError(f"Failed to retrieve context: {str(e)}")
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract unique source information from chunks.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            List of unique sources
        """
        sources = []
        seen_files = set()
        
        for chunk in chunks:
            file_name = chunk.get('file_name', '')
            if file_name and file_name not in seen_files:
                sources.append({
                    'file_name': file_name,
                    'file_path': chunk.get('file_path', ''),
                    'relevance_score': chunk.get('hybrid_score', chunk.get('score', 0))
                })
                seen_files.add(file_name)
        
        return sources
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval configuration statistics.
        
        Returns:
            Dictionary with retrieval stats
        """
        return {
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight,
            'embedding_model': self.embedding_generator.model,
            'embedding_dimension': self.embedding_generator.dimension
        }
    
    def set_weights(self, semantic_weight: float, keyword_weight: float):
        """
        Adjust weights for hybrid search.
        
        Args:
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        """
        total = semantic_weight + keyword_weight
        if total == 0:
            raise ValueError("Weights cannot both be zero")
        
        # Normalize weights
        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total
        
        logger.info(f"Updated weights: semantic={self.semantic_weight:.2f}, keyword={self.keyword_weight:.2f}")
