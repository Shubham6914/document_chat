"""
Document Processor Module
Handles loading and processing of legal documents (PDF, DOCX).
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
import re
import PyPDF2
import pdfplumber
from docx import Document
from src.utils.logger import app_logger as logger

from ..utils.exceptions import DocumentProcessingError
from ..utils.helpers import clean_text, extract_metadata


class DocumentProcessor:
    """Process and extract text from legal documents."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.supported_formats = ['.pdf', '.docx', '.txt']
        logger.info("DocumentProcessor initialized")
    
    def load_document(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Load a document and extract its content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document text, metadata, and page information
            
        Raises:
            DocumentProcessingError: If document cannot be processed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise DocumentProcessingError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        logger.info(f"Loading document: {file_path.name}")
        
        try:
            if file_path.suffix.lower() == '.pdf':
                return self._load_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                return self._load_docx(file_path)
            elif file_path.suffix.lower() == '.txt':
                return self._load_txt(file_path)
        except Exception as e:
            logger.error(f"Error loading document {file_path.name}: {str(e)}")
            raise DocumentProcessingError(f"Failed to load document: {str(e)}")
    
    def _load_pdf(self, file_path: Path) -> Dict[str, any]:
        """
        Load and extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with document content and metadata
        """
        try:
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(file_path) as pdf:
                pages = []
                full_text = []
                
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    pages.append({
                        'page_number': i + 1,
                        'text': clean_text(page_text),
                        'raw_text': page_text
                    })
                    full_text.append(page_text)
                
                metadata = self._extract_pdf_metadata(file_path)
                
                return {
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'file_type': 'pdf',
                    'full_text': clean_text('\n\n'.join(full_text)),
                    'pages': pages,
                    'page_count': len(pages),
                    'metadata': metadata
                }
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {str(e)}")
            return self._load_pdf_pypdf2(file_path)
    
    def _load_pdf_pypdf2(self, file_path: Path) -> Dict[str, any]:
        """
        Fallback PDF loader using PyPDF2.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with document content and metadata
        """
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pages = []
            full_text = []
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                pages.append({
                    'page_number': i + 1,
                    'text': clean_text(page_text),
                    'raw_text': page_text
                })
                full_text.append(page_text)
            
            metadata = self._extract_pdf_metadata(file_path)
            
            return {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_type': 'pdf',
                'full_text': clean_text('\n\n'.join(full_text)),
                'pages': pages,
                'page_count': len(pages),
                'metadata': metadata
            }
    
    def _load_docx(self, file_path: Path) -> Dict[str, any]:
        """
        Load and extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary with document content and metadata
        """
        doc = Document(file_path)
        
        paragraphs = []
        full_text = []
        
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                paragraphs.append({
                    'paragraph_number': i + 1,
                    'text': clean_text(para.text),
                    'raw_text': para.text
                })
                full_text.append(para.text)
        
        metadata = {
            'author': doc.core_properties.author,
            'created': doc.core_properties.created,
            'modified': doc.core_properties.modified,
            'title': doc.core_properties.title
        }
        
        return {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': 'docx',
            'full_text': clean_text('\n\n'.join(full_text)),
            'paragraphs': paragraphs,
            'paragraph_count': len(paragraphs),
            'metadata': metadata
        }
    
    def _load_txt(self, file_path: Path) -> Dict[str, any]:
        """
        Load and extract text from a TXT file.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            Dictionary with document content and metadata
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        return {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': 'txt',
            'full_text': clean_text(text),
            'raw_text': text,
            'metadata': extract_metadata(file_path)
        }
    
    def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata or {}
                
                return {
                    'title': metadata.get('/Title', ''),
                    'author': metadata.get('/Author', ''),
                    'subject': metadata.get('/Subject', ''),
                    'creator': metadata.get('/Creator', ''),
                    'producer': metadata.get('/Producer', ''),
                    'creation_date': metadata.get('/CreationDate', ''),
                    'modification_date': metadata.get('/ModDate', '')
                }
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {str(e)}")
            return {}
    
    def _extract_section_from_text(self, text: str) -> str:
        """
        Simple extraction of section/article info from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Section identifier string or empty string
        """
        # Check first 300 characters for section markers
        search_text = text[:300]
        
        # Simple patterns for common formats
        patterns = [
            (r'ARTICLE\s+([IVXLCDM]+|\d+)', 'Article'),
            (r'Article\s+([IVXLCDM]+|\d+)', 'Article'),
            (r'SECTION\s+(\d+\.?\d*)', 'Section'),
            (r'Section\s+(\d+\.?\d*)', 'Section'),
        ]
        
        for pattern, prefix in patterns:
            match = re.search(pattern, search_text)
            if match:
                return f"{prefix} {match.group(1)}"
        
        return ''
    
    def chunk_document(
        self,
        document: Dict[str, any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, any]]:
        """
        Split document into chunks with simple section metadata extraction.
        
        Args:
            document: Document dictionary from load_document
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks with metadata including page and section info
        """
        chunks = []
        chunk_id = 0
        
        # For PDFs, chunk by page first to preserve page numbers
        if document['file_type'] == 'pdf' and 'pages' in document:
            for page_info in document['pages']:
                page_num = page_info['page_number']
                page_text = page_info['text']
                
                # If page is small enough, keep it as one chunk
                if len(page_text) <= chunk_size:
                    section_info = self._extract_section_from_text(page_text)
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': page_text.strip(),
                        'page': page_num,
                        'article': section_info if 'Article' in section_info else '',
                        'section': section_info if 'Section' in section_info else '',
                        'paragraph': '',
                        'section_id': section_info,
                        'file_name': document['file_name'],
                        'file_path': document['file_path'],
                        'file_type': document['file_type']
                    })
                    chunk_id += 1
                else:
                    # Split large pages into smaller chunks
                    start = 0
                    while start < len(page_text):
                        end = start + chunk_size
                        chunk_text = page_text[start:end]
                        
                        # Try to break at sentence boundary
                        if end < len(page_text):
                            last_period = chunk_text.rfind('.')
                            last_newline = chunk_text.rfind('\n')
                            break_point = max(last_period, last_newline)
                            
                            if break_point > chunk_size * 0.5:
                                end = start + break_point + 1
                                chunk_text = page_text[start:end]
                        
                        section_info = self._extract_section_from_text(chunk_text)
                        chunks.append({
                            'chunk_id': chunk_id,
                            'text': chunk_text.strip(),
                            'page': page_num,
                            'article': section_info if 'Article' in section_info else '',
                            'section': section_info if 'Section' in section_info else '',
                            'paragraph': '',
                            'section_id': section_info,
                            'file_name': document['file_name'],
                            'file_path': document['file_path'],
                            'file_type': document['file_type']
                        })
                        
                        chunk_id += 1
                        start = end - chunk_overlap
        else:
            # For non-PDF documents, use simple chunking
            text = document['full_text']
            start = 0
            
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                
                # Try to break at sentence boundary
                if end < len(text):
                    last_period = chunk_text.rfind('.')
                    last_newline = chunk_text.rfind('\n')
                    break_point = max(last_period, last_newline)
                    
                    if break_point > chunk_size * 0.5:
                        end = start + break_point + 1
                        chunk_text = text[start:end]
                
                section_info = self._extract_section_from_text(chunk_text)
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text.strip(),
                    'start_char': start,
                    'end_char': end,
                    'article': section_info if 'Article' in section_info else '',
                    'section': section_info if 'Section' in section_info else '',
                    'paragraph': '',
                    'section_id': section_info,
                    'file_name': document['file_name'],
                    'file_path': document['file_path'],
                    'file_type': document['file_type']
                })
                
                chunk_id += 1
                start = end - chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from {document['file_name']}")
        return chunks
