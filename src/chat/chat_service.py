"""
Chat Service Module
Enhanced chat engine with structured responses and hallucination guards.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.utils.logger import app_logger as logger
import json

from ..retrieval.retrieval_service import RetrievalService
from ..utils.exceptions import ChatError
from config.settings import settings
from config.prompts import prompts


class ChatService:
    """
    Enhanced conversational interface with structured responses and hallucination prevention.
    """
    
    def __init__(self, retrieval_service: RetrievalService):
        """
        Initialize chat service.
        
        Args:
            retrieval_service: RetrievalService instance for context retrieval
        """
        self.retrieval_service = retrieval_service
        self.client = OpenAI(
            api_key=settings.requesty_api_key,
            base_url=settings.requesty_base_url
        )
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.chat_history: List[Dict[str, str]] = []
        self.max_history = settings.chat_history_length
        
        logger.info("ChatService initialized with structured responses and hallucination guards")
    
    def chat(
        self,
        user_message: str,
        document_filter: Optional[Dict[str, Any]] = None,
        include_context: bool = True,
        use_hybrid_search: bool = True
    ) -> Dict[str, Any]:
        """
        Process user message and generate structured response.
        
        Args:
            user_message: User's question or message
            document_filter: Optional filter for document retrieval
            include_context: Whether to include retrieved context
            use_hybrid_search: Whether to use hybrid search (vs pure semantic)
            
        Returns:
            Dictionary with structured response and metadata
        """
        try:
            logger.info(f"Processing chat message: {user_message[:100]}...")
            
            # Retrieve relevant context if needed
            context = ""
            retrieved_chunks = []
            
            if include_context:
                if use_hybrid_search:
                    retrieved_chunks = self.retrieval_service.hybrid_search(
                        query=user_message,
                        top_k=settings.max_context_chunks,
                        filter_dict=document_filter
                    )
                else:
                    retrieved_chunks = self.retrieval_service.semantic_search(
                        query=user_message,
                        top_k=settings.max_context_chunks,
                        filter_dict=document_filter
                    )
                
                context = self._format_context(retrieved_chunks)
            
            # Build messages for LLM
            messages = self._build_messages(user_message, context)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            assistant_message = response.choices[0].message.content
            
            # Parse structured response
            structured_response = self._parse_structured_response(assistant_message)
            
            # Validate response against context (hallucination check)
            if include_context and context:
                validation = self._validate_response(structured_response, context)
                structured_response['validation'] = validation
            
            # Update chat history
            self._update_history(user_message, assistant_message)
            
            # Prepare final response with only top source
            all_sources = self._extract_sources(retrieved_chunks)
            top_source = all_sources[:1] if all_sources else []  # Keep only the most relevant source
            
            result = {
                'response': structured_response.get('answer', assistant_message),
                'structured_response': structured_response,
                'context_used': context if include_context else None,
                'sources': top_source,  # Only the top source
                'confidence': structured_response.get('confidence', 'medium'),
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                'search_method': 'hybrid' if use_hybrid_search else 'semantic'
            }
            
            logger.info("Chat response generated successfully with structured format")
            return result
            
        except Exception as e:
            logger.error(f"Chat processing failed: {str(e)}")
            raise ChatError(f"Failed to process chat message: {str(e)}")
    
    def _build_messages(
        self,
        user_message: str,
        context: str
    ) -> List[Dict[str, str]]:
        """
        Build message list for LLM with structured response instructions.
        
        Args:
            user_message: User's message
            context: Retrieved context
            
        Returns:
            List of message dictionaries
        """
        messages = [
            {
                "role": "system",
                "content": prompts.CHAT_SYSTEM_PROMPT
            }
        ]
        
        # Add chat history (limited)
        history_to_include = self.chat_history[-(self.max_history * 2):]
        messages.extend(history_to_include)
        
        # Add current message with context
        user_content = prompts.format_chat_prompt(user_message, context)
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string with precise section citations.
        
        Args:
            chunks: List of retrieved chunks with metadata
            
        Returns:
            Formatted context string with precise citations
        """
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            score = chunk.get('hybrid_score', chunk.get('score', 0))
            page = chunk.get('page', 0)
            article = chunk.get('article', '')
            section = chunk.get('section', '')
            
            # Build citation
            citation_parts = []
            if page > 0:
                citation_parts.append(f"Page {page}")
            if article:
                citation_parts.append(article)
            if section:
                citation_parts.append(section)
            
            citation = ", ".join(citation_parts) if citation_parts else chunk['file_name']
            
            context_parts.append(
                f"[Section {i}] (Source: {citation}, Relevance: {score:.2f})\n"
                f"{chunk['text']}"
            )
        
        return "\n\n".join(context_parts)
    
    def _parse_structured_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse response into structured format.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Structured response dictionary
        """
        # Try to extract structured components
        structured = {
            'answer': '',
            'source': '',
            'supporting_text': '',
            'confidence': 'medium'
        }
        
        lines = response_text.split('\n')
        current_section = 'answer'
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if line_lower.startswith('**answer:**') or line_lower.startswith('answer:'):
                current_section = 'answer'
                continue
            elif line_lower.startswith('**source:**') or line_lower.startswith('source:'):
                current_section = 'source'
                continue
            elif line_lower.startswith('**supporting text:**') or line_lower.startswith('supporting text:'):
                current_section = 'supporting_text'
                continue
            elif line_lower.startswith('**confidence:**') or line_lower.startswith('confidence:'):
                current_section = 'confidence'
                # Extract confidence value
                confidence_text = line.split(':', 1)[1].strip().lower()
                if 'high' in confidence_text:
                    structured['confidence'] = 'high'
                elif 'low' in confidence_text:
                    structured['confidence'] = 'low'
                else:
                    structured['confidence'] = 'medium'
                continue
            
            # Add content to current section
            if line.strip():
                if structured[current_section]:
                    structured[current_section] += '\n' + line
                else:
                    structured[current_section] = line
        
        # If no structured format detected, use full response as answer
        if not structured['answer']:
            structured['answer'] = response_text
        
        return structured
    
    def _validate_response(
        self,
        structured_response: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """
        Validate response against context to detect hallucinations.
        
        Args:
            structured_response: Structured response
            context: Retrieved context
            
        Returns:
            Validation result
        """
        answer = structured_response.get('answer', '')
        
        # Simple validation: check if key terms in answer appear in context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        answer_words -= stop_words
        context_words -= stop_words
        
        # Calculate overlap
        if answer_words:
            overlap = len(answer_words & context_words) / len(answer_words)
        else:
            overlap = 0.0
        
        validation = {
            'overlap_score': overlap,
            'is_supported': overlap > 0.3,  # At least 30% overlap
            'confidence_adjustment': 0.0
        }
        
        # Adjust confidence based on validation
        if overlap < 0.3:
            validation['confidence_adjustment'] = -0.2
            validation['warning'] = "Low overlap with source context - response may contain unsupported information"
        elif overlap > 0.7:
            validation['confidence_adjustment'] = 0.1
        
        return validation
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract detailed source information from chunks with precise citations.
        Now uses pre-extracted metadata for accurate citations.
        
        Args:
            chunks: Retrieved chunks with metadata
            
        Returns:
            List of source dictionaries with precise citations
        """
        sources = []
        
        for chunk in chunks:
            file_name = chunk.get('file_name', '')
            page = chunk.get('page', 0)
            score = chunk.get('hybrid_score', chunk.get('score', 0))
            text = chunk.get('text', '')
            
            # Use pre-extracted metadata from chunking process
            article = chunk.get('article', '')
            section = chunk.get('section', '')
            paragraph = chunk.get('paragraph', '')
            
            # Build precise citation
            citation_parts = []
            
            if page > 0:
                citation_parts.append(f"Page {page}")
            
            # Add section information in order of specificity
            if article:
                citation_parts.append(article)
            if section:
                citation_parts.append(section)
            if paragraph:
                citation_parts.append(paragraph)
            
            # Format final citation
            if citation_parts:
                source_citation = ", ".join(citation_parts)
            else:
                source_citation = file_name
            
            # Build section display string
            section_display = []
            if article:
                section_display.append(article)
            if section:
                section_display.append(section)
            if paragraph:
                section_display.append(paragraph)
            
            section_str = ", ".join(section_display) if section_display else 'N/A'
            
            sources.append({
                'file_name': file_name,
                'file_path': chunk.get('file_path', ''),
                'page': page if page > 0 else 'N/A',
                'section': section_str,
                'citation': source_citation,
                'relevance_score': round(score * 100, 1),
                'text_preview': text[:150] + '...' if len(text) > 150 else text
            })
        
        return sources
    
    def _update_history(self, user_message: str, assistant_message: str):
        """
        Update chat history.
        
        Args:
            user_message: User's message
            assistant_message: Assistant's response
        """
        self.chat_history.append({
            "role": "user",
            "content": user_message
        })
        self.chat_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Trim history if too long
        max_messages = self.max_history * 2
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        logger.info("Chat history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get chat history.
        
        Returns:
            List of message dictionaries
        """
        return self.chat_history.copy()
    
    def ask_with_extraction(
        self,
        question: str,
        extracted_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Answer question using both extracted fields and document context.
        
        Args:
            question: User's question
            extracted_fields: Previously extracted structured fields
            
        Returns:
            Response dictionary
        """
        try:
            # Format extracted fields
            fields_context = self._format_extracted_fields(extracted_fields)
            
            # Get document context using hybrid search
            retrieved_chunks = self.retrieval_service.hybrid_search(question)
            doc_context = ""
            if retrieved_chunks:
                doc_context = self._format_context(retrieved_chunks)
            
            # Build combined prompt
            combined_prompt = prompts.format_extraction_with_fields_prompt(
                question=question,
                fields_context=fields_context,
                doc_context=doc_context
            )
            
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": prompts.CHAT_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": combined_prompt
                }
            ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            assistant_message = response.choices[0].message.content
            structured_response = self._parse_structured_response(assistant_message)
            
            # Only return top source
            all_sources = self._extract_sources(retrieved_chunks)
            top_source = all_sources[:1] if all_sources else []
            
            return {
                'response': structured_response.get('answer', assistant_message),
                'structured_response': structured_response,
                'sources': top_source,  # Only the top source
                'used_extracted_fields': True
            }
            
        except Exception as e:
            logger.error(f"Ask with extraction failed: {str(e)}")
            raise ChatError(f"Failed to process question: {str(e)}")
    
    def _format_extracted_fields(self, extracted_fields: Dict[str, Any]) -> str:
        """
        Format extracted fields for context.
        
        Args:
            extracted_fields: Extracted fields dictionary
            
        Returns:
            Formatted string
        """
        if not extracted_fields or 'fields' not in extracted_fields:
            return "No extracted fields available."
        
        lines = []
        for field_name, field_data in extracted_fields['fields'].items():
            value = field_data.get('value', 'N/A')
            confidence = field_data.get('confidence', 0)
            source = field_data.get('source', '')
            
            lines.append(f"- **{field_name}**: {value}")
            lines.append(f"  Confidence: {confidence:.2f}")
            if source:
                lines.append(f"  Source: {source[:100]}...")
        
        return "\n".join(lines)
    
    def get_chat_stats(self) -> Dict[str, Any]:
        """
        Get chat statistics.
        
        Returns:
            Dictionary with chat stats
        """
        return {
            'history_length': len(self.chat_history),
            'max_history': self.max_history,
            'model': self.model,
            'temperature': self.temperature,
            'retrieval_stats': self.retrieval_service.get_retrieval_stats()
        }
