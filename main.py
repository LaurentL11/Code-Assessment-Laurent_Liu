from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any
import logging
from pathlib import Path
import os
from datetime import datetime
import time

from models.document_classifier import DocumentClassifier
from models.ocr_processor import OCRProcessor
from services.llm_service import LLMService
from services.extraction_service import ExtractionService
from utils.file_utils import validate_file, save_temp_file
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="OCR Document Extraction API",
    description="OCR API for extracting key information from referral letters, medical certificates, and receipts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    debug=True  # Enable debug mode
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ocr_processor = OCRProcessor()
document_classifier = DocumentClassifier(ocr_processor=ocr_processor)
llm_service = LLMService()
extraction_service = ExtractionService(llm_service)

ALLOWED_FIELDS = {
    "referral_letter": [
        "claimant_name", "provider_name", "signature_presence",
        "total_amount_paid", "total_approved_amount", "total_requested_amount"
    ],
    "medical_certificate": [
        "claimant_name", "claimant_address", "claimant_date_of_birth",
        "diagnosis_name", "discharge_date_time", "icd_code", "provider_name",
        "submission_date_time", "date_of_mc", "mc_days"
    ],
    "receipt": [
        "claimant_name", "claimant_address", "claimant_date_of_birth",
        "provider_name", "tax_amount", "total_amount"
    ]
}

@app.get("/")
async def root():
    """Root path, returns API information"""
    return {
        "message": "OCR Document Extraction API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "extract": "/OCR",
            "docs": "/docs"
        }
    }

@app.post("/OCR", summary="Extract document information", operation_id="extract_document")
async def extract_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    start_time = time.time()
    temp_file_path = None
    try:
        # 1. File validation
        if not validate_file(file):
            return JSONResponse(status_code=400, content={"error": "file_missing"})
        # 2. Save temp file
        temp_file_path = await save_temp_file(file)
        logger.info(f"Processing file: {file.filename}")
        # 3. Try direct PDF text extraction (fitz) if PDF
        extracted_text = None
        if file.filename.lower().endswith('.pdf'):
            import fitz
            try:
                doc = fitz.open(temp_file_path)
                direct_text = ""
                for page in doc:
                    direct_text += page.get_text()
                doc.close()
                if len(direct_text.strip()) > 50:
                    extracted_text = direct_text
                    logger.info(f"Directly extracted {len(direct_text)} chars from PDF using fitz.")
            except Exception as e:
                logger.warning(f"Direct PDF text extraction failed: {e}")
        # If not enough text, use OCR (for all file types)
        if extracted_text is None:
            extracted_text = await ocr_processor.extract_text(temp_file_path)
            logger.info(f"OCR extracted {len(extracted_text)} characters")
        # 5. Document type classification
        try:
            document_type = llm_service.classify_document_with_llm(extracted_text)
        except ValueError as e:
            if str(e) == "unsupported_document_type":
                if temp_file_path:
                    try: os.unlink(temp_file_path)
                    except: pass
                return Response(
                    content='{"error": "unsupported_document_type"}',
                    status_code=442,
                    media_type="application/json",
                    headers={"X-Reason": "Unsupported Document Type"}
                )
            try:
                document_type = await document_classifier.classify(temp_file_path)
            except ValueError as e:
                if temp_file_path:
                    try: os.unlink(temp_file_path)
                    except: pass
                return JSONResponse(status_code=422, content={"error": "unsupported_document_type"})
        # 6. LLM field extraction
        extracted_fields = await extraction_service.extract_fields(document_type, extracted_text)
        # 7. Clean up temp file
        if temp_file_path:
            try: os.unlink(temp_file_path)
            except: pass
        # 8. Return result in required format
        total_time = round(time.time() - start_time, 2)
        allowed = ALLOWED_FIELDS.get(document_type, list(extracted_fields.keys()))
        final_json = {k: v for k, v in extracted_fields.items() if k in allowed}
        return {
            "message": "Processing completed.",
            "result": {
                "document_type": document_type,
                "total_time": total_time,
                "finalJson": final_json
            }
        }
    except Exception as e:
        logger.error(f"Error processing file {getattr(file, 'filename', 'unknown')}: {str(e)}")
        if temp_file_path:
            try: os.unlink(temp_file_path)
            except: pass
        return JSONResponse(status_code=500, content={"error": "internal_server_error"})

@app.get("/supported-types")
async def get_supported_types():
    """Return supported document types"""
    return {
        "supported_types": [
            {
                "type": "referral_letter",
                "name": "Referral Letter",
                "description": "Doctor referral recommendation letter"
            },
            {
                "type": "medical_certificate", 
                "name": "Medical Certificate",
                "description": "Medical diagnosis certificate"
            },
            {
                "type": "receipt",
                "name": "Receipt",
                "description": "Medical expense receipt"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 