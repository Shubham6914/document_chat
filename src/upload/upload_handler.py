"""
Upload Handler Module
Handles document upload, validation, and processing orchestration.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
from src.utils.logger import app_logger as logger
import hashlib
import shutil

from ..utils.exceptions import UploadError, DocumentProcessingError
from config.settings import settings, PROCESSED_DIR


class UploadHandler:
    """Handle document uploads with validation and processing."""
    
    def __init__(self):
        """Initialize upload handler."""
        self.supported_formats = ['.pdf', '.docx', '.txt']
        self.max_file_size = 50 * 1024 * 1024  # 50 MB
        self.upload_dir = PROCESSED_DIR
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"UploadHandler initialized with upload dir: {self.upload_dir}")
    
    def handle_upload(
        self,
        file_data: bytes,
        file_name: str,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Handle file upload with validation.
        
        Args:
            file_data: File content as bytes
            file_name: Original file name
            validate: Whether to validate the file
            
        Returns:
            Dictionary with upload information
            
        Raises:
            UploadError: If upload fails validation or processing
        """
        try:
            logger.info(f"Handling upload for file: {file_name}")
            
            # Validate file if requested
            if validate:
                self._validate_file(file_data, file_name)
            
            # Generate unique file path
            file_path = self._generate_file_path(file_name)
            
            # Save file
            self._save_file(file_data, file_path)
            
            # Compute file hash for deduplication
            file_hash = self._compute_hash(file_data)
            
            result = {
                'file_name': file_name,
                'file_path': str(file_path),
                'file_size': len(file_data),
                'file_hash': file_hash,
                'file_type': file_path.suffix.lower().lstrip('.'),
                'status': 'uploaded'
            }
            
            logger.info(f"File uploaded successfully: {file_name} ({len(file_data)} bytes)")
            return result
            
        except Exception as e:
            logger.error(f"Upload failed for {file_name}: {str(e)}")
            raise UploadError(f"Failed to upload file: {str(e)}")
    
    def handle_streamlit_upload(
        self,
        uploaded_file,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Handle Streamlit UploadedFile object.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            validate: Whether to validate the file
            
        Returns:
            Dictionary with upload information
        """
        try:
            file_data = uploaded_file.getbuffer().tobytes()
            file_name = uploaded_file.name
            
            return self.handle_upload(file_data, file_name, validate)
            
        except Exception as e:
            logger.error(f"Streamlit upload failed: {str(e)}")
            raise UploadError(f"Failed to handle Streamlit upload: {str(e)}")
    
    def _validate_file(self, file_data: bytes, file_name: str):
        """
        Validate uploaded file.
        
        Args:
            file_data: File content
            file_name: File name
            
        Raises:
            UploadError: If validation fails
        """
        # Check file size
        if len(file_data) == 0:
            raise UploadError("File is empty")
        
        if len(file_data) > self.max_file_size:
            raise UploadError(
                f"File size ({len(file_data)} bytes) exceeds maximum "
                f"allowed size ({self.max_file_size} bytes)"
            )
        
        # Check file extension
        file_path = Path(file_name)
        if file_path.suffix.lower() not in self.supported_formats:
            raise UploadError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        # Basic content validation
        self._validate_content(file_data, file_path.suffix.lower())
        
        logger.debug(f"File validation passed for: {file_name}")
    
    def _validate_content(self, file_data: bytes, file_extension: str):
        """
        Validate file content based on type.
        
        Args:
            file_data: File content
            file_extension: File extension
            
        Raises:
            UploadError: If content validation fails
        """
        # Check PDF magic number
        if file_extension == '.pdf':
            if not file_data.startswith(b'%PDF'):
                raise UploadError("Invalid PDF file: missing PDF header")
        
        # Check DOCX magic number (ZIP format)
        elif file_extension == '.docx':
            if not file_data.startswith(b'PK'):
                raise UploadError("Invalid DOCX file: missing ZIP header")
        
        # Check TXT is valid UTF-8
        elif file_extension == '.txt':
            try:
                file_data.decode('utf-8')
            except UnicodeDecodeError:
                raise UploadError("Invalid TXT file: not valid UTF-8 text")
    
    def _generate_file_path(self, file_name: str) -> Path:
        """
        Generate unique file path for upload.
        
        Args:
            file_name: Original file name
            
        Returns:
            Path object for the file
        """
        file_path = self.upload_dir / file_name
        
        # Handle duplicate file names
        if file_path.exists():
            base_name = file_path.stem
            extension = file_path.suffix
            counter = 1
            
            while file_path.exists():
                new_name = f"{base_name}_{counter}{extension}"
                file_path = self.upload_dir / new_name
                counter += 1
        
        return file_path
    
    def _save_file(self, file_data: bytes, file_path: Path):
        """
        Save file to disk.
        
        Args:
            file_data: File content
            file_path: Path to save file
            
        Raises:
            UploadError: If save fails
        """
        try:
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            logger.debug(f"File saved to: {file_path}")
            
        except Exception as e:
            raise UploadError(f"Failed to save file: {str(e)}")
    
    def _compute_hash(self, file_data: bytes) -> str:
        """
        Compute SHA-256 hash of file content.
        
        Args:
            file_data: File content
            
        Returns:
            Hex string of file hash
        """
        return hashlib.sha256(file_data).hexdigest()
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """
        Delete uploaded file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if deleted successfully
        """
        try:
            file_path = Path(file_path)
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False
    
    def list_uploaded_files(self) -> list[Dict[str, Any]]:
        """
        List all uploaded files.
        
        Returns:
            List of file information dictionaries
        """
        try:
            files = []
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    files.append({
                        'file_name': file_path.name,
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size,
                        'file_type': file_path.suffix.lower().lstrip('.'),
                        'modified_time': file_path.stat().st_mtime
                    })
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list uploaded files: {str(e)}")
            return []
    
    def clear_uploads(self) -> int:
        """
        Clear all uploaded files.
        
        Returns:
            Number of files deleted
        """
        try:
            count = 0
            
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    count += 1
            
            logger.info(f"Cleared {count} uploaded files")
            return count
            
        except Exception as e:
            logger.error(f"Failed to clear uploads: {str(e)}")
            return 0
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """
        Get upload statistics.
        
        Returns:
            Dictionary with upload stats
        """
        files = self.list_uploaded_files()
        
        total_size = sum(f['file_size'] for f in files)
        
        return {
            'total_files': len(files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'supported_formats': self.supported_formats,
            'max_file_size_mb': self.max_file_size / (1024 * 1024),
            'upload_directory': str(self.upload_dir)
        }
