import asyncio
import logging
from typing import Dict, List, Optional, Any
import re
from datetime import datetime

from .llm_service import LLMService

logger = logging.getLogger(__name__)

class ExtractionService:
    """Extraction service - Integrates LLM and OCR results for field extraction"""
    
    def __init__(self, llm_service: LLMService):
        """
        Initialize extraction service
        
        Args:
            llm_service: LLM service instance
        """
        self.llm_service = llm_service
    
    async def extract_fields(self, document_type: str, text: str) -> Dict[str, Any]:
        """
        Extract structured fields from text
        
        Args:
            document_type: Document type
            text: OCR extracted text
            
        Returns:
            Extracted structured fields
        """
        try:
            # First use LLM to extract fields
            if self.llm_service.is_available():
                llm_result = await self.llm_service.extract_fields(document_type, text)
                
                # Enhance LLM result with rules
                enhanced_result = self._enhance_with_rules(document_type, text, llm_result)
                
                # Validate and clean result
                validated_result = self._validate_and_clean(document_type, enhanced_result)
                
                return validated_result
            else:
                # If LLM is not available, use rule-based extraction
                logger.warning("LLM service not available, using rule-based extraction")
                return self._rule_based_extraction(document_type, text)
                
        except Exception as e:
            logger.error(f"Error in field extraction: {str(e)}")
            # Return basic error result
            return {
                "error": str(e),
                "confidence": 0.0,
                "document_type": document_type,
                "extraction_method": "error"
            }
    
    def _enhance_with_rules(self, document_type: str, text: str, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance LLM result with rules"""
        enhanced_result = llm_result.copy()
        
        # General rule enhancement
        enhanced_result = self._enhance_dates(text, enhanced_result)
        enhanced_result = self._enhance_amounts(text, enhanced_result)
        enhanced_result = self._enhance_names(text, enhanced_result)
        
        # Document-type specific rule enhancement
        if document_type == "referral_letter":
            enhanced_result = self._enhance_referral_letter(text, enhanced_result)
        elif document_type == "medical_certificate":
            enhanced_result = self._enhance_medical_certificate(text, enhanced_result)
        elif document_type == "receipt":
            enhanced_result = self._enhance_receipt(text, enhanced_result)
        
        return enhanced_result
    
    def _enhance_dates(self, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance date extraction"""
        # Date regex patterns
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # DD-MM-YYYY or DD/MM/YYYY
            # Removed Chinese date patterns
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates_found.extend(matches)
        
        # If dates found but not extracted by LLM, try to fill
        if dates_found:
            for key, value in result.items():
                if "date" in key.lower() and (value is None or value == ""):
                    if dates_found:
                        result[key] = dates_found[0]
                        dates_found.pop(0)
        
        return result
    
    def _enhance_amounts(self, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance amount extraction"""
        # Amount regex patterns
        amount_patterns = [
            r'\$\s*(\d+(?:\.\d{2})?)',      # $123.45
            r'(\d+(?:\.\d{2})?)\s*dollars?', # 123.45 dollars
            r'Total\s*[:：]?\s*(\d+(?:\.\d{2})?)',  # Total: 123.45
            r'Amount\s*[:：]?\s*(\d+(?:\.\d{2})?)', # Amount: 123.45
        ]
        
        amounts_found = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts_found.extend(matches)
        
        # Fill amount fields
        if amounts_found:
            for key, value in result.items():
                if "amount" in key.lower() and (value is None or value == ""):
                    if amounts_found:
                        try:
                            result[key] = float(amounts_found[0])
                            amounts_found.pop(0)
                        except ValueError:
                            pass
        
        return result
    
    def _enhance_names(self, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance name extraction"""
        # Simple name patterns (can be extended as needed)
        name_patterns = [
            r'Patient\s*[:：]\s*([^\s\n]+)',      # Patient: Name
            r'Doctor\s*[:：]\s*([^\s\n]+)',       # Doctor: Name
            r'Name\s*[:：]\s*([^\s\n]+)',         # Name: Name
        ]
        
        names_found = []
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            names_found.extend(matches)
        
        # Fill name fields
        if names_found:
            for key, value in result.items():
                if "name" in key.lower() and (value is None or value == ""):
                    if names_found:
                        result[key] = names_found[0].strip()
                        names_found.pop(0)
        
        return result
    
    def _enhance_referral_letter(self, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance referral letter specific fields"""
        # Referral letter specific patterns
        patterns = {
            "urgency_level": [
                r'urgent|emergency',
                r'routine|normal',
            ]
        }
        
        for field, pattern_list in patterns.items():
            if result.get(field) is None:
                for pattern in pattern_list:
                    if re.search(pattern, text, re.IGNORECASE):
                        if "urgent" in pattern or "emergency" in pattern:
                            result[field] = "urgent"
                        else:
                            result[field] = "routine"
                        break
        
        return result
    
    def _enhance_medical_certificate(self, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance medical certificate specific fields"""
        # Medical certificate specific patterns
        if result.get("follow_up_required") is None:
            if re.search(r'follow.*up|revisit', text, re.IGNORECASE):
                result["follow_up_required"] = True
            else:
                result["follow_up_required"] = False
        
        return result
    
    def _enhance_receipt(self, text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance receipt specific fields"""
        # Receipt specific patterns
        if result.get("payment_method") is None:
            payment_methods = {
                "Cash": r'cash',
                "Bank Card": r'card',
                "Alipay": r'alipay',
                "WeChat": r'wechat',
                "Other": r'other'
            }
            
            for method, pattern in payment_methods.items():
                if re.search(pattern, text, re.IGNORECASE):
                    result["payment_method"] = method
                    break
        
        return result
    
    def _validate_and_clean(self, document_type: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean results"""
        cleaned_result = {}
        
        for key, value in result.items():
            # Clean empty strings
            if value == "":
                cleaned_result[key] = None
            # Clean overly long strings
            elif isinstance(value, str) and len(value) > 500:
                cleaned_result[key] = value[:500] + "..."
            # Validate numeric values
            elif key.endswith("_amount") and isinstance(value, str):
                try:
                    cleaned_result[key] = float(value)
                except ValueError:
                    cleaned_result[key] = None
            else:
                cleaned_result[key] = value
        
        # Ensure confidence field exists
        if "confidence" not in cleaned_result:
            cleaned_result["confidence"] = 0.7
        
        # Add extraction metadata
        cleaned_result["extraction_timestamp"] = datetime.now().isoformat()
        cleaned_result["extraction_method"] = "llm_enhanced"
        
        return cleaned_result
    
    def _rule_based_extraction(self, document_type: str, text: str) -> Dict[str, Any]:
        """Rule-based extraction (fallback when LLM is not available)"""
        result = {
            "document_type": document_type,
            "confidence": 0.3,
            "extraction_method": "rule_based",
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        # Basic rule extraction
        result = self._enhance_dates(text, result)
        result = self._enhance_amounts(text, result)
        result = self._enhance_names(text, result)
        
        # Add specific fields based on document type
        if document_type == "referral_letter":
            result.update({
                "patient_name": None,
                "referring_doctor": None,
                "referral_date": None,
                "referral_reason": None
            })
        elif document_type == "medical_certificate":
            result.update({
                "patient_name": None,
                "issuing_doctor": None,
                "issue_date": None,
                "medical_condition": None
            })
        elif document_type == "receipt":
            result.update({
                "receipt_number": None,
                "transaction_date": None,
                "total_amount": None,
                "merchant_name": None
            })
        
        return result 