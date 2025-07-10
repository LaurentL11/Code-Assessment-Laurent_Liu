import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
from paddleocr import PaddleOCR

from config import settings

logger = logging.getLogger(__name__)

class OCRProcessor:
    """OCR processor - Use PaddleOCR to extract text"""
    
    def __init__(self):
        """Initialize OCR processor"""
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=settings.OCR_USE_ANGLE_CLS,
                lang=settings.OCR_LANG,
                use_gpu=settings.OCR_USE_GPU,
                show_log=False
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            self.ocr = None
    
    async def extract_text(self, file_path: str) -> str:
        """
        Extract text from file
        
        Args:
            file_path: File path
            
        Returns:
            Extracted text content
        """
        if self.ocr is None:
            raise RuntimeError("OCR not initialized")
        
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.pdf':
                return await self._extract_pdf_text(file_path)
            else:
                return await self._extract_image_text(file_path)
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file using PyMuPDF rendering (no Poppler required)"""
        try:
            # First try to extract PDF text directly
            doc = fitz.open(str(file_path))
            direct_text = ""
            for page in doc:
                direct_text += page.get_text()
            doc.close()
            
            # If directly extracted text is sufficient, return directly
            if len(direct_text.strip()) > 50:
                logger.info(f"Extracted text directly from PDF: {len(direct_text)} characters")
                return direct_text
            
            # Otherwise convert to images using PyMuPDF and perform OCR
            logger.info("PDF text extraction insufficient, converting to images for OCR using PyMuPDF")
            doc = fitz.open(str(file_path))
            all_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Create a transformation matrix for higher resolution
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                # Render page to image
                pix = page.get_pixmap(matrix=mat)
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                
                # Use OCR on the image
                page_text = await self._ocr_image(image)
                all_text += f"\n--- Page {page_num+1} ---\n" + page_text
            
            doc.close()
            return all_text
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    async def _extract_image_text(self, file_path: Path) -> str:
        """Extract text from image file"""
        try:
            image = Image.open(str(file_path))
            text = await self._ocr_image(image)
            logger.info(f"Extracted text from image: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {str(e)}")
            raise
    
    async def _ocr_image(self, image: Image.Image) -> str:
        """Perform OCR on single image"""
        try:
            # Convert PIL image to numpy array
            image_array = np.array(image)
            
            # Run OCR in thread pool (since PaddleOCR is synchronous)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.ocr.ocr, image_array)
            
            # Extract text
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) > 1:
                        text_lines.append(line[1][0])
            
            return "\n".join(text_lines)
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return ""
    
    async def extract_text_with_coordinates(self, file_path: str) -> List[Dict]:
        """
        Extract text with coordinate information
        
        Args:
            file_path: File path
            
        Returns:
            List containing text and coordinate information
        """
        if self.ocr is None:
            raise RuntimeError("OCR not initialized")
        
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.pdf':
                return await self._extract_pdf_text_with_coordinates(file_path)
            else:
                return await self._extract_image_text_with_coordinates(file_path)
                
        except Exception as e:
            logger.error(f"Error extracting text with coordinates: {str(e)}")
            raise
    
    async def _extract_pdf_text_with_coordinates(self, file_path: Path) -> List[Dict]:
        """Extract text and coordinates from PDF using PyMuPDF rendering (no Poppler required)"""
        try:
            doc = fitz.open(str(file_path))
            all_results = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Create a transformation matrix for higher resolution
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                # Render page to image
                pix = page.get_pixmap(matrix=mat)
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                
                # Use OCR on the image with coordinates
                page_results = await self._ocr_image_with_coordinates(image)
                for result in page_results:
                    result["page"] = page_num + 1
                all_results.extend(page_results)
            
            doc.close()
            return all_results
            
        except Exception as e:
            logger.error(f"Error processing PDF coordinates: {str(e)}")
            raise
    
    async def _extract_image_text_with_coordinates(self, file_path: Path) -> List[Dict]:
        """Extract text and coordinates from image"""
        try:
            image = Image.open(str(file_path))
            results = await self._ocr_image_with_coordinates(image)
            return results
            
        except Exception as e:
            logger.error(f"Error processing image coordinates: {str(e)}")
            raise
    
    async def _ocr_image_with_coordinates(self, image: Image.Image) -> List[Dict]:
        """Perform OCR on image and return coordinate information"""
        try:
            image_array = np.array(image)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.ocr.ocr, image_array)
            
            results = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) > 1:
                        coordinates = line[0]
                        text_info = line[1]
                        
                        results.append({
                            "text": text_info[0],
                            "confidence": text_info[1],
                            "coordinates": coordinates,
                            "bbox": {
                                "x1": min(coord[0] for coord in coordinates),
                                "y1": min(coord[1] for coord in coordinates),
                                "x2": max(coord[0] for coord in coordinates),
                                "y2": max(coord[1] for coord in coordinates)
                            }
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in OCR coordinate processing: {str(e)}")
            return []
    
    def is_available(self) -> bool:
        """Check if OCR is available"""
        return self.ocr is not None 