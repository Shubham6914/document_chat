"""
Retrieval Module
Handles vector storage and retrieval using Pinecone.
"""

from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from src.utils.logger import app_logger as logger

from ..utils.exceptions import RetrievalError
from config.settings import settings


class VectorStore:
    """Manage vector storage and retrieval using Pinecone."""
    
    def __init__(self):
        """Initialize Pinecone vector store."""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index_name = settings.pinecone_index_name
            self.dimension = settings.embedding_dimension  # Embedding dimension (matching Pinecone index)
            
            # Initialize OpenAI client for embeddings
            self.client = OpenAI(
                api_key=settings.requesty_api_key,
                base_url=settings.requesty_base_url
            )
            self.embedding_model = settings.embedding_model
            
            # Create or connect to index
            self._initialize_index()
            
            logger.info(f"VectorStore initialized with Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {str(e)}")
            raise RetrievalError(f"VectorStore initialization failed: {str(e)}")
    
    def _initialize_index(self):
        """Create Pinecone index if it doesn't exist."""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=settings.pinecone_environment
                    )
                )
                logger.info(f"Index {self.index_name} created successfully")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {str(e)}")
            raise RetrievalError(f"Index initialization failed: {str(e)}")
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                dimensions=self.dimension  # Specify output dimension
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to create embedding: {str(e)}")
            raise RetrievalError(f"Embedding creation failed: {str(e)}")
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks to vector store with enhanced metadata including section info.
        
        Args:
            chunks: List of document chunks with text and metadata
        """
        try:
            logger.info(f"Adding {len(chunks)} chunks to vector store")
            
            vectors = []
            for chunk in chunks:
                # Create embedding
                embedding = self.create_embedding(chunk['text'])
                
                # Prepare enhanced metadata with section information
                metadata = {
                    'text': chunk['text'],
                    'file_name': chunk.get('file_name', ''),
                    'file_path': chunk.get('file_path', ''),
                    'file_type': chunk.get('file_type', ''),
                    'chunk_id': chunk.get('chunk_id', 0),
                    'start_char': chunk.get('start_char', 0),
                    'end_char': chunk.get('end_char', 0),
                    'page': chunk.get('page', 0),
                    'article': chunk.get('article', ''),
                    'section': chunk.get('section', ''),
                    'paragraph': chunk.get('paragraph', ''),
                    'section_id': chunk.get('section_id', '')
                }
                
                # Create vector ID
                vector_id = f"{chunk.get('file_name', 'doc')}_{chunk.get('chunk_id', 0)}"
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise RetrievalError(f"Failed to add documents: {str(e)}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks with enhanced section metadata.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of relevant chunks with scores and section information
        """
        try:
            # Create query embedding
            query_embedding = self.create_embedding(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results with enhanced metadata
            chunks = []
            for match in results.matches:
                chunks.append({
                    'id': match.id,
                    'text': match.metadata.get('text', ''),
                    'score': match.score,
                    'file_name': match.metadata.get('file_name', ''),
                    'file_path': match.metadata.get('file_path', ''),
                    'file_type': match.metadata.get('file_type', ''),
                    'chunk_id': match.metadata.get('chunk_id', 0),
                    'start_char': match.metadata.get('start_char', 0),
                    'end_char': match.metadata.get('end_char', 0),
                    'page': match.metadata.get('page', 0),
                    'article': match.metadata.get('article', ''),
                    'section': match.metadata.get('section', ''),
                    'paragraph': match.metadata.get('paragraph', ''),
                    'section_id': match.metadata.get('section_id', ''),
                    'metadata': match.metadata
                })
            
            logger.info(f"Found {len(chunks)} relevant chunks for query")
            return chunks
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise RetrievalError(f"Search failed: {str(e)}")
    
    def delete_document(self, file_name: str) -> None:
        """
        Delete all chunks for a specific document.
        
        Args:
            file_name: Name of the file to delete
        """
        try:
            # Delete by metadata filter
            self.index.delete(filter={'file_name': file_name})
            logger.info(f"Deleted all chunks for document: {file_name}")
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            raise RetrievalError(f"Failed to delete document: {str(e)}")
    
    def clear_index(self) -> None:
        """Clear all vectors from the index."""
        try:
            self.index.delete(delete_all=True)
            logger.info("Cleared all vectors from index")
        except Exception as e:
            logger.error(f"Failed to clear index: {str(e)}")
            raise RetrievalError(f"Failed to clear index: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index stats
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}
