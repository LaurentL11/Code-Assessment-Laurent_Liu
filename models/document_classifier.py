import asyncio
import logging
from typing import Dict, List, Optional
from pathlib import Path
import re
from PIL import Image
import fitz  # PyMuPDF
import io
import numpy as np

logger = logging.getLogger(__name__)

class DocumentClassifier:
    """Document classifier - Identify document type"""
    
    def __init__(self, ocr_processor=None):
        """Initialize classifier, optionally with an OCR processor"""
        # Weighted keyword dictionaries for each document type
        self.referral_keywords = {
            'thank you for seeing': 2.0,
            'please see': 1.8,
            'kindly assist': 1.8,
            'assist to': 1.5,
            'please assist': 1.5,
            'referral': 2.0,
            'referred': 2.0,
            'referred to': 2.0,
            'referral for': 2.0,
            'referred for': 2.0,
            'above patient': 1.5,
            'consultation': 1.0,
            'specialist': 1.0,
            'appointment': 1.0,
            'opinion': 1.0,
            'assessment': 0.8,
            'evaluation': 0.8,
            'treatment plan': 0.8,
            'dear dr': 1.8,
            'dear': 1.0,
            'kind regards': 1.8,
            'regards': 1.0,
            'further management': 1.5,
            'request for opinion': 1.5,
            'mole check': 1.5,
            'pt.': 1.2,
            'recommend': 1.0
        }
        self.medical_keywords = {
            'medical certificate': 2.5,
            'certify': 2.0,
            'fit for work': 2.0,
            'unfit for duty': 2.0,
            'clinic': 1.5,
            'doctor': 1.0,
            'certificate': 1.5,
            'medical': 1.0,
            'hospital': 1.5,
            'physician': 1.0,
            'outpatient': 1.5,
            'hospitalisation leave': 2.0,
            'maternity leave': 2.0,
            'admitted': 1.5,
            'discharged': 1.5,
            'time chit': 1.5,
            'sick leave': 2.0,
            'valid for': 1.5,
            'light duty': 1.5
        }
        self.receipt_keywords = {
            'receipt': 2.5,
            'tax invoice': 2.2,
            'invoice': 2.0,
            'bill': 2.0,
            'bill date': 1.5,
            'visit date': 1.5,
            'invoice no.': 1.5,
            'total': 2.2,
            'sub-total': 1.5,
            'subtotal': 1.5,
            'amount': 1.8,
            'amount paid': 2.0,
            'balance due': 2.0,
            'gst': 2.0,
            'vat': 2.0,
            'cashier': 1.5,
            'payment': 1.8,
            'pay': 1.0,
            'price': 1.0,
            'cost': 1.0,
            'charge': 1.0,
            'purchase': 1.0,
            'transaction': 1.0,
            'statement': 1.0,
            'account': 1.0,
            'consultation': 1.2,
            'pharmaceutical': 1.2,
            'practice cost': 1.2,
            'qty': 1.0,
            'description': 1.0,
            'adjustment': 1.0
        }
        self.document_types = [
            "referral_letter",
            "medical_certificate",
            "receipt"
        ]
        self.ocr_processor = ocr_processor
    
    async def classify(self, file_path: str) -> str:
        """
        Classify document type using normalized weighted keyword scoring
        """
        try:
            # Extract text for classification
            text = await self._extract_text_for_classification(file_path)
            # Keyword-based classification with weighted scoring
            classification_scores = self._classify_by_keywords(text)
            text_length = max(len(text), 1)  # Avoid division by zero
            normalized_scores = {k: v / (text_length / 100) for k, v in classification_scores.items()}
            max_score = max(normalized_scores.values())
            if max_score < 0.5:  # Threshold for normalized score per 100 chars
                logger.warning(f"Document type not supported. Normalized max score: {max_score:.3f}")
                raise ValueError("unsupported_document_type")
            # Handle cases where multiple types have the same score
            best_type = self._resolve_tie_scores(normalized_scores, file_path, text)
            confidence = normalized_scores[best_type]
            logger.info(f"Document classified as: {best_type} (confidence: {confidence:.3f})")
            logger.debug(f"All normalized scores: {normalized_scores}")
            return best_type
        except ValueError as e:
            # Re-raise ValueError for unsupported document type
            raise e
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            # Check if we can extract any meaningful text
            try:
                text = await self._extract_text_for_classification(file_path)
                if len(text.strip()) < 10:  # Very little text extracted
                    raise ValueError("unsupported_document_type")
            except:
                pass
            # If we still can't determine, raise unsupported document type
            raise ValueError("unsupported_document_type")
    
    async def _extract_text_for_classification(self, file_path: str) -> str:
        """Extract text for classification (PDF or image, with OCR fallback)"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return await self._extract_pdf_text(file_path)
        else:
            return await self._extract_image_text(file_path)
    
    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF, fallback to OCR if needed"""
        try:
            doc = fitz.open(str(file_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if len(text.strip()) > 50:
                return text
        except Exception as e: 
            logger.warning(f"Failed to extract PDF text: {str(e)}")
        # Fallback to OCR if available
        if self.ocr_processor:
            try:
                # Use PyMuPDF to convert PDF pages to images
                doc = fitz.open(str(file_path))
                all_text = ""
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Get page dimensions
                    rect = page.rect
                    # Create a transformation matrix for higher resolution
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                    # Render page to image
                    pix = page.get_pixmap(matrix=mat)
                    # Convert to PIL Image
                    img_data = pix.tobytes("ppm")
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Use OCR on the image
                    page_text = await self.ocr_processor._ocr_image(image)
                    all_text += page_text + "\n"
                
                doc.close()
                return all_text
            except Exception as e:
                logger.warning(f"Failed to extract PDF text with OCR: {str(e)}")
        return ""
    
    async def _extract_image_text(self, file_path: Path) -> str:
        """Extract text from image using OCR"""
        if self.ocr_processor:
            try:
                image = Image.open(str(file_path))
                return await self.ocr_processor._ocr_image(image)
            except Exception as e:
                logger.warning(f"Failed to extract image text with OCR: {str(e)}")
                return ""
        return ""
    
    def _classify_by_keywords(self, text: str) -> Dict[str, float]:
        """Weighted keyword-based classification with detailed matching information"""
        text_lower = text.lower()
        
        # Calculate weighted scores and track matched keywords
        scores = {}
        matched_keywords = {}
        
        # Medical certificate scoring
        medical_score = 0
        medical_matches = []
        for keyword, weight in self.medical_keywords.items():
            if keyword in text_lower:
                medical_score += weight
                medical_matches.append((keyword, weight))
        scores['medical_certificate'] = medical_score
        matched_keywords['medical_certificate'] = medical_matches
        
        # Receipt scoring
        receipt_score = 0
        receipt_matches = []
        for keyword, weight in self.receipt_keywords.items():
            if keyword in text_lower:
                receipt_score += weight
                receipt_matches.append((keyword, weight))
        scores['receipt'] = receipt_score
        matched_keywords['receipt'] = receipt_matches
        
        # Referral letter scoring
        referral_score = 0
        referral_matches = []
        for keyword, weight in self.referral_keywords.items():
            if keyword in text_lower:
                referral_score += weight
                referral_matches.append((keyword, weight))
        scores['referral_letter'] = referral_score
        matched_keywords['referral_letter'] = referral_matches
        
        # Store matched keywords for detailed analysis
        self.last_matched_keywords = matched_keywords
        
        # If all scores are 0, try to classify by filename
        if all(score == 0 for score in scores.values()):
            scores = self._classify_by_filename(text)
        
        return scores
    
    def _classify_by_filename(self, filename: str) -> Dict[str, float]:
        """Classify by filename"""
        filename_lower = filename.lower()
        scores = {doc_type: 0.0 for doc_type in self.document_types}
        
        if "referral" in filename_lower:
            scores["referral_letter"] = 0.8
        elif "certificate" in filename_lower or "medical" in filename_lower:
            scores["medical_certificate"] = 0.8
        elif "receipt" in filename_lower or "invoice" in filename_lower:
            scores["receipt"] = 0.8
        else:
            # Return all zeros for unsupported document types
            pass
        
        return scores
    
    def get_document_types(self) -> List[str]:
        """Get supported document types"""
        return self.document_types.copy()
    
    def get_last_matched_keywords(self) -> Dict[str, List[tuple]]:
        """Get the matched keywords and their weights from the last classification"""
        return getattr(self, 'last_matched_keywords', {})
    
    def get_classification_breakdown(self, text: str) -> Dict[str, Dict]:
        """Get detailed classification breakdown with matched keywords and scores"""
        text_lower = text.lower()
        breakdown = {}
        
        # Medical certificate breakdown
        medical_matches = []
        medical_score = 0
        for keyword, weight in self.medical_keywords.items():
            if keyword in text_lower:
                medical_score += weight
                medical_matches.append((keyword, weight))
        breakdown['medical_certificate'] = {
            'score': medical_score,
            'matched_keywords': medical_matches,
            'total_keywords': len(self.medical_keywords)
        }
        
        # Receipt breakdown
        receipt_matches = []
        receipt_score = 0
        for keyword, weight in self.receipt_keywords.items():
            if keyword in text_lower:
                receipt_score += weight
                receipt_matches.append((keyword, weight))
        breakdown['receipt'] = {
            'score': receipt_score,
            'matched_keywords': receipt_matches,
            'total_keywords': len(self.receipt_keywords)
        }
        
        # Referral letter breakdown
        referral_matches = []
        referral_score = 0
        for keyword, weight in self.referral_keywords.items():
            if keyword in text_lower:
                referral_score += weight
                referral_matches.append((keyword, weight))
        breakdown['referral_letter'] = {
            'score': referral_score,
            'matched_keywords': referral_matches,
            'total_keywords': len(self.referral_keywords)
        }
        
        return breakdown
    
    def _resolve_tie_scores(self, scores: Dict[str, float], file_path: str, text: str) -> str:
        """Resolve tie scores using additional heuristics"""
        # Find the maximum score
        max_score = max(scores.values())
        
        # Get all types with the maximum score
        tied_types = [doc_type for doc_type, score in scores.items() if score == max_score]
        
        if len(tied_types) == 1:
            # No tie, return the single winner
            return tied_types[0]
        
        logger.info(f"Tie detected between: {tied_types} (score: {max_score:.3f})")
        
        # Apply tie-breaking heuristics
        return self._apply_tie_breakers(tied_types, file_path, text, scores)
    
    def _apply_tie_breakers(self, tied_types: List[str], file_path: str, text: str, scores: Dict[str, float]) -> str:
        """Apply tie-breaking heuristics in order of priority"""
        
        # 1. Check if any type has significantly more weighted keyword matches
        keyword_scores = {}
        text_lower = text.lower()
        
        for doc_type in tied_types:
            if doc_type == 'medical_certificate':
                keyword_scores[doc_type] = sum(weight for keyword, weight in self.medical_keywords.items() if keyword in text_lower)
            elif doc_type == 'receipt':
                keyword_scores[doc_type] = sum(weight for keyword, weight in self.receipt_keywords.items() if keyword in text_lower)
            elif doc_type == 'referral_letter':
                keyword_scores[doc_type] = sum(weight for keyword, weight in self.referral_keywords.items() if keyword in text_lower)
        
        max_keywords = max(keyword_scores.values())
        best_keyword_types = [t for t in tied_types if keyword_scores[t] == max_keywords]
        
        if len(best_keyword_types) == 1:
            logger.info(f"Tie broken by keyword count: {best_keyword_types[0]}")
            return best_keyword_types[0]
        
        # 2. Use filename analysis for remaining ties
        filename_scores = self._classify_by_filename(file_path)
        filename_best = max(filename_scores.items(), key=lambda x: x[1])
        
        if filename_best[1] > 0.5:  # Only use filename if confidence is reasonable
            for doc_type in best_keyword_types:
                if doc_type == filename_best[0]:
                    logger.info(f"Tie broken by filename analysis: {doc_type}")
                    return doc_type
        
        # 3. Use document-specific heuristics
        if "referral_letter" in best_keyword_types and "medical_certificate" in best_keyword_types:
            # Check for referral-specific terms
            referral_terms = ["referral", "refer", "recommend", "patient", "consultation"]
            if any(term in text.lower() for term in referral_terms):
                logger.info("Tie broken by referral-specific terms")
                return "referral_letter"
            else:
                logger.info("Tie broken by defaulting to medical_certificate")
                return "medical_certificate"
        
        if "receipt" in best_keyword_types:
            # Check for receipt-specific terms
            receipt_terms = ["invoice", "tax", "gst", "payment", "total", "amount", "receipt"]
            if any(term in text.lower() for term in receipt_terms):
                logger.info("Tie broken by receipt-specific terms")
                return "receipt"
        
        # 4. Default priority order: referral_letter > medical_certificate > receipt
        priority_order = ["referral_letter", "medical_certificate", "receipt"]
        for doc_type in priority_order:
            if doc_type in best_keyword_types:
                logger.info(f"Tie broken by priority order: {doc_type}")
                return doc_type
        
        # 5. Last resort: return the first one
        logger.warning(f"Unable to break tie, returning first type: {best_keyword_types[0]}")
        return best_keyword_types[0] 