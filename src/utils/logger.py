"""
Simple logging configuration for the Legal Document Chat application.
Logs are displayed only on the terminal.
"""

import logging
import sys
from config.settings import settings


def setup_logger():
    """Configure the application logger for terminal output only."""
    
    # Create logger
    logger = logging.getLogger("legal_document_chat")
    logger.setLevel(logging.DEBUG if settings.debug_mode else logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler - terminal output only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.debug_mode else logging.INFO)
    
    # Simple format for terminal
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.info("Logger initialized (terminal output only)")
    
    return logger


# Initialize logger
app_logger = setup_logger()
