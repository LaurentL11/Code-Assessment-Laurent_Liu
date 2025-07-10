import os
import tempfile
import logging
import mimetypes
from typing import Optional
from pathlib import Path
import aiofiles
from fastapi import UploadFile

from config import settings

logger = logging.getLogger(__name__)

def validate_file(file: UploadFile) -> bool:
    """
    Validate uploaded file type and size
    
    Args:
        file: Uploaded file object
        
    Returns:
        Whether validation passed
    """
    try:
        # Check file extension
        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in settings.ALLOWED_EXTENSIONS:
                logger.warning(f"File extension {file_extension} not allowed")
                return False
        
        # Check file size
        if hasattr(file, 'size') and file.size:
            if file.size > settings.MAX_FILE_SIZE:
                logger.warning(f"File size {file.size} exceeds maximum {settings.MAX_FILE_SIZE}")
                return False
        
        # Check MIME type
        if hasattr(file, 'content_type') and file.content_type:
            allowed_mime_types = {
                'application/pdf',
                'image/jpeg',
                'image/jpg', 
                'image/png',
                'image/gif'
            }
            if file.content_type not in allowed_mime_types:
                logger.warning(f"MIME type {file.content_type} not allowed")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating file: {str(e)}")
        return False

async def save_temp_file(file: UploadFile) -> str:
    """
    Save uploaded file to temporary directory
    
    Args:
        file: Uploaded file object
        
    Returns:
        Temporary file path
    """
    try:
        # Ensure temporary directory exists
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        # Generate temporary filename
        file_extension = ""
        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            dir=settings.TEMP_DIR,
            suffix=file_extension,
            delete=False
        ) as temp_file:
            temp_file_path = temp_file.name
        
        # Asynchronously write file content
        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"Saved temporary file: {temp_file_path}")
        return temp_file_path
        
    except Exception as e:
        logger.error(f"Error saving temporary file: {str(e)}")
        raise

def cleanup_temp_files() -> None:
    """Clean up temporary file directory"""
    try:
        temp_dir = Path(settings.TEMP_DIR)
        if temp_dir.exists():
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        logger.debug(f"Deleted temporary file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file {file_path}: {str(e)}")
        
        logger.info("Temporary files cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during temporary files cleanup: {str(e)}")

def get_file_mime_type(file_path: str) -> Optional[str]:
    """
    Get file MIME type
    
    Args:
        file_path: File path
        
    Returns:
        MIME type string
    """
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type
    except Exception as e:
        logger.warning(f"Failed to detect MIME type for {file_path}: {str(e)}")
        return None

def get_file_size(file_path: str) -> int:
    """
    Get file size
    
    Args:
        file_path: File path
        
    Returns:
        File size in bytes
    """
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.warning(f"Failed to get file size for {file_path}: {str(e)}")
        return 0

def is_file_readable(file_path: str) -> bool:
    """
    Check if file is readable
    
    Args:
        file_path: File path
        
    Returns:
        Whether file is readable
    """
    try:
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)
    except Exception:
        return False

def create_safe_filename(filename: str) -> str:
    """
    Create safe filename
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove dangerous characters
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
    safe_filename = ''.join(c for c in filename if c in safe_chars)
    
    # Ensure filename is not empty
    if not safe_filename:
        safe_filename = "uploaded_file"
    
    # Limit filename length
    if len(safe_filename) > 100:
        name_part = safe_filename[:90]
        ext_part = safe_filename[-10:]
        safe_filename = name_part + ext_part
    
    return safe_filename 