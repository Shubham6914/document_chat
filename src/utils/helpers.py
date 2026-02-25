"""Utility helper functions."""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\$\%\&]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_dates(text: str) -> List[str]:
    """
    Extract potential dates from text.
    
    Args:
        text: Text to search for dates
        
    Returns:
        List of potential date strings
    """
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
        r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
        r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return dates


def extract_currency(text: str) -> List[str]:
    """
    Extract currency amounts from text.
    
    Args:
        text: Text to search for currency
        
    Returns:
        List of currency amounts
    """
    currency_patterns = [
        r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # $1,000.00
        r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars?)',  # 1000 USD
    ]
    
    amounts = []
    for pattern in currency_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        amounts.extend(matches)
    
    return amounts


def normalize_currency(amount_str: str) -> float:
    """
    Normalize currency string to float.
    
    Args:
        amount_str: Currency string (e.g., "$1,000.00")
        
    Returns:
        Float value
    """
    # Remove currency symbols and text
    cleaned = re.sub(r'[^\d\.]', '', amount_str)
    
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def calculate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Calculate estimated reading time in minutes.
    
    Args:
        text: Text to calculate reading time for
        words_per_minute: Average reading speed
        
    Returns:
        Reading time in minutes
    """
    word_count = len(text.split())
    minutes = max(1, round(word_count / words_per_minute))
    return minutes


def generate_document_id(content: str) -> str:
    """
    Generate unique document ID based on content hash.
    
    Args:
        content: Document content
        
    Returns:
        Unique document ID
    """
    return hashlib.md5(content.encode()).hexdigest()


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix


def format_confidence_score(score: float) -> str:
    """
    Format confidence score as percentage with label.
    
    Args:
        score: Confidence score (0.0 to 1.0)
        
    Returns:
        Formatted string (e.g., "High (85%)")
    """
    percentage = int(score * 100)
    
    if score >= 0.8:
        label = "High"
    elif score >= 0.6:
        label = "Medium"
    else:
        label = "Low"
    
    return f"{label} ({percentage}%)"


def chunk_text_by_sentences(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Chunk text by sentences with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap size in characters
        
    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_text = ' '.join(current_chunk)
            if len(overlap_text) > overlap:
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_length
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def extract_section_headers(text: str) -> List[Dict[str, Any]]:
    """
    Extract section headers from text.
    
    Args:
        text: Text to search for headers
        
    Returns:
        List of headers with positions
    """
    # Common patterns for section headers
    patterns = [
        r'^(?:SECTION|Section|ARTICLE|Article)\s+(\d+[\.\d]*)\s*[:\-]?\s*(.+)$',
        r'^(\d+[\.\d]*)\s+(.+)$',
        r'^([A-Z][A-Z\s]+)$',  # ALL CAPS headers
    ]
    
    headers = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                headers.append({
                    'line_number': i,
                    'text': line,
                    'level': len(match.groups()) if match.groups() else 1
                })
                break
    
    return headers


def extract_metadata(file_path) -> Dict[str, Any]:
    """
    Extract basic metadata from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file metadata
    """
    from pathlib import Path
    import os
    
    path = Path(file_path)
    
    if not path.exists():
        return {}
    
    stat = os.stat(path)
    
    return {
        'file_name': path.name,
        'file_size': stat.st_size,
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'extension': path.suffix
    }
