"""Custom exceptions for the Legal Document Chat application."""


class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass


class ExtractionError(Exception):
    """Raised when field extraction fails."""
    pass


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class RetrievalError(Exception):
    """Raised when document retrieval fails."""
    pass


class ChatError(Exception):
    """Raised when chat processing fails."""
    pass


class APIError(Exception):
    """Raised when API calls fail."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when file type is not supported."""
    pass


class EmptyDocumentError(DocumentProcessingError):
    """Raised when document has no extractable content."""
    pass


class LowConfidenceError(ExtractionError):
    """Raised when extraction confidence is below threshold."""
    pass


class ConflictDetectedError(ValidationError):
    """Raised when conflicting data is detected."""
    pass


class UploadError(Exception):
    """Raised when file upload fails."""
    pass


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass
