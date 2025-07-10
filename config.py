import os
from typing import List


try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="D:/Code/.env")
except ImportError:
    # Skip if python-dotenv is not installed
    pass

class Settings:
    """Application configuration"""
    
    # Basic configuration
    APP_NAME: str = "OCR Document Extraction API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # File configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg"]
    TEMP_DIR: str = "temp"
    
    # OCR configuration
    OCR_USE_ANGLE_CLS: bool = True
    OCR_USE_GPU: bool = False
    OCR_LANG: str = "en"  # English
    
    # OpenAI API configuration (from environment variables)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    
    # Document classification threshold
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create settings instance
settings = Settings()

# Ensure temporary directory exists
if not os.path.exists(settings.TEMP_DIR):
    os.makedirs(settings.TEMP_DIR) 