"""
Model package - Contains document classification and OCR processing models
"""

from .document_classifier import DocumentClassifier
from .ocr_processor import OCRProcessor

__all__ = ["DocumentClassifier", "OCRProcessor"] 