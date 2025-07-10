import asyncio
import logging
from typing import Dict, List, Optional, Any
import json
import re
import openai
from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service - Use OpenAI GPT API for text processing"""
    
    def __init__(self):
        """Initialize LLM service"""
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not set, LLM service will not be available")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")

    LLM_CLASSIFY_PROMPT = (
    "I will give you a medical content below, and you are required to tell what kind of document it is from the content based on these three document types: [Referral Letters, Medical Certificates, Receipts]. Your response should be just telling the answer from these three types, no other words. If the context based on your decision is not in any of these three types, just return False.\n\n{text}"
    )
    DOCUMENT_TYPES = [
        'Referral Letters',
        'Medical Certificates',
        'Receipts',
    ]
    
    def _clean_json_response(self, text: str) -> str:
        """Clean LLM response that might contain markdown formatting"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = re.sub(r'^```\s*', '', text)
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _create_extraction_prompt(self, document_type: str, text: str) -> str:
        """Create extraction prompt based on document type"""

        if document_type == "referral_letter":
            return f"""
You are an expert in information extraction for insurance and healthcare claims.
Given a referral letter, extract ONLY the following fields and output them in strict JSON format:

- claimant_name: Patient Name (the main person referred in the letter)
- provider_name: Provider / Lab name (must not contain the literal string \"Fullerton Health\")
- signature_presence: true if a handwritten signature is detected, otherwise false
- total_amount_paid: Number, keep decimal point if present. If not available, return null.
- total_approved_amount: Number, keep decimal point if present. If not available, return null.
- total_requested_amount: Number, keep decimal point if present. If not available, return null.

The output must be a valid JSON object, with only the above fields, and no extra fields, comments, or explanations. If a field is not found, set its value to null.

Referral letter content:
{text}
"""

        elif document_type == "medical_certificate":
            return f"""
You are a medical document extraction expert.
Given a medical certificate, extract ONLY the following fields and output them in strict JSON format:

- claimant_name: Claimant Name (person for whom the certificate is issued)
- claimant_address: Address (if available, otherwise null)
- claimant_date_of_birth: Date of birth (format: DD/MM/YYYY; if not present, null)
- diagnosis_name: Diagnosis or reason for leave (if stated; null if not present)
- discharge_date_time: Discharge date, if relevant (format: DD/MM/YYYY; null if not present)
- icd_code: ICD code if given, otherwise null
- provider_name: Provider / Lab name (must not contain \"Fullerton Health\")
- submission_date_time: Admission date/time or certificate issue date (format: DD/MM/YYYY)
- date_of_mc: Date of Medical Certificate (DD/MM/YYYY; if not present, null)
- mc_days: Number of MC days as an integer, if available; otherwise null

The output must be a valid JSON object, with only the above fields, and no extra fields, comments, or explanations. If a field is not found, set its value to null.

Medical certificate content:
{text}
"""

        elif document_type == "receipt":
            return f"""
You are an information extraction specialist.
Given a medical receipt, extract ONLY the following fields and output them in strict JSON format:

- claimant_name: Claimant Name (person who received the service)
- claimant_address: Address, if available, else null
- claimant_date_of_birth: Date of birth (format: DD/MM/YYYY, if present, else null)
- provider_name: Provider / Lab name (must not contain \"Fullerton Health\")
- tax_amount: Number, keep decimal point if present (e.g. 3.65). If not present, null.
- total_amount: Number, keep decimal point if present (e.g. 49.45). If not present, null.

The output must be a valid JSON object, with only the above fields, and no extra fields, comments, or explanations. If a field is not found, set its value to null.

Receipt content:
{text}
"""

        else:
            return f"""
Please extract key information from the following document and return it in JSON format:

Document content:
{text}

Please automatically identify and extract relevant fields based on the document type.
"""

    def _filter_fields(self, data: dict, allowed_fields: list) -> dict:
        return {k: v for k, v in data.items() if k in allowed_fields}

    async def extract_fields(self, document_type: str, text: str) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("LLM service not available - API key not set")
        try:
            prompt = self._create_extraction_prompt(document_type, text)
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional document information extraction assistant. Please extract the specified fields from the provided document content and return them in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=settings.OPENAI_TEMPERATURE
            )
            result_text = response.choices[0].message.content
            cleaned_text = self._clean_json_response(result_text)
            try:
                result = json.loads(cleaned_text)
                # 字段过滤
                if document_type == "referral_letter":
                    allowed = [
                        "claimant_name", "provider_name", "signature_presence",
                        "total_amount_paid", "total_approved_amount", "total_requested_amount"
                    ]
                elif document_type == "medical_certificate":
                    allowed = [
                        "claimant_name", "claimant_address", "claimant_date_of_birth",
                        "diagnosis_name", "discharge_date_time", "icd_code", "provider_name",
                        "submission_date_time", "date_of_mc", "mc_days"
                    ]
                elif document_type == "receipt":
                    allowed = [
                        "claimant_name", "claimant_address", "claimant_date_of_birth",
                        "provider_name", "tax_amount", "total_amount"
                    ]
                else:
                    allowed = list(result.keys())
                return self._filter_fields(result, allowed)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning raw text")
                return {"raw_response": result_text, "error": "JSON parsing failed"}
        except Exception as e:
            logger.error(f"Error in LLM extraction: {str(e)}")
            raise
    
    def classify_document_with_llm(self, text):
        prompt = self.LLM_CLASSIFY_PROMPT.format(text=text)
        response = openai.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.OPENAI_MAX_TOKENS,
            temperature=settings.OPENAI_TEMPERATURE
        )
        answer = response.choices[0].message.content.strip()
        # If LLM returns False, it means unsupported document type
        if answer.lower() == 'false':
            raise ValueError('unsupported_document_type')
        # Keyword matching to prevent abnormal answers
        for doc_type in self.DOCUMENT_TYPES:
            if doc_type.lower() in answer.lower():
                # Map to compatible format for field extraction
                if doc_type == "Referral Letters":
                    return "referral_letter"
                elif doc_type == "Medical Certificates":
                    return "medical_certificate"
                elif doc_type == "Receipts":
                    return "receipt"
        raise ValueError('unsupported_document_type')
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.client is not None 