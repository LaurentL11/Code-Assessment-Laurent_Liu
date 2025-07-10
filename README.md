# OCR Document Extraction API

An intelligent document information extraction system based on FastAPI, PaddleOCR, and OpenAI GPT.

## Features

- **Automatic Document Type Recognition**: Supports referral letters, medical certificates, and receipts
- **OCR Text Extraction**: Integrates PaddleOCR, supports PDF, PNG, JPG, JPEG formats
- **Intelligent Field Extraction**: Uses GPT API for structured information extraction
- **High-Performance Asynchronous Processing**: Powered by FastAPI
- **Comprehensive Error Handling and Logging**

## Project Structure

```
Code/
├── main.py                  # FastAPI main application
├── config.py                # Configuration file
├── requirements.txt         # Dependency list
├── models/                  # Classification & OCR models
├── services/                # Business logic
├── utils/                   # Utility functions
├── temp/                    # Temporary files
└── Assessment_Documents/    # Example documents
```

## Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**

   Create a `.env` file in the project root with your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_MAX_TOKENS=1000
   OPENAI_TEMPERATURE=0.3
   ```

   > **Note**: Other configuration options like server settings, OCR parameters, and file limits are set in `config.py` and can be modified there if needed.

3. **Run the Application**

   ```bash
   python main.py
   ```
   Or with uvicorn (hot reload):
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the API**

   - Swagger Docs: http://localhost:8000/docs
   - Main Endpoint: http://localhost:8000/OCR

## API Endpoints

### POST `/OCR`

Upload a document and extract structured information.

- **Parameter**: `file` (PDF/PNG/JPG/JPEG)
- **Response Example**:

  ```json
  {
    "success": true,
    "document_type": "referral_letter",
    "extracted_fields": {
      "patient_name": "John Doe",
      "patient_age": "35",
      "referring_doctor": "Dr. Smith"
    },
    "processing_time": "2024-01-15T10:30:00"
  }
  ```

### GET `/supported-types`

Get the list of supported document types.

## Supported Document Types

- **Referral Letter (`referral_letter`)**: Patient, referring doctor, diagnosis, etc.
- **Medical Certificate (`medical_certificate`)**: Patient, medical condition, validity period, etc.
- **Receipt (`receipt`)**: Transaction, amount, payment method, etc.

## System Requirements

- Python 3.8+
- Sufficient memory for PaddleOCR
- OpenAI API key
- Optional: GPU for better OCR performance

## Development Guide

- **Add New Document Types**: Edit `models/document_classifier.py`, `services/llm_service.py`, and `services/extraction_service.py`
- **Customize OCR Settings**: Modify `config.py`
- **Error Handling**: See logging and exception handling in each module

## Performance & Security Tips

- Enable GPU: `OCR_USE_GPU=true`
- Set appropriate file size limits
- Add authentication and access control for production
- Regularly clean up temporary files

---
