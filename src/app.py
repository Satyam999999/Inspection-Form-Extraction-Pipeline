"""
FastAPI application for Industrial Inspection Form Extractor.

Accepts multipart/form-data (file upload) OR JSON with base64-encoded file.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.pipeline import run_pipeline
from src.schema import ExtractionResponse

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Industrial Inspection Form Extractor",
    description=(
        "Extract structured JSON from scanned or photographed industrial inspection "
        "forms using Gemini 1.5 Flash, LangGraph, and Pydantic."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Base64ExtractRequest(BaseModel):
    """Schema for base64-encoded JSON extraction request."""
    file: str = Field(..., description="Base64-encoded file bytes")
    filename: str = Field(default="upload.jpg", description="Original file name with extension")


@app.get("/", tags=["Health"])
async def root() -> Dict[str, Any]:
    """Root endpoint with service metadata and documentation links."""
    return {
        "service": "Industrial Inspection Form Extractor",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint for container orchestrators and load balancers."""
    return {"status": "healthy"}


@app.post(
    "/extract",
    response_model=ExtractionResponse,
    tags=["Extraction"],
    summary="Extract inspection form data",
    description=(
        "Upload an image (PNG, JPG, TIFF, etc.) or PDF document. "
        "Supports both multipart/form-data file uploads and application/json with base64 encoded file."
    ),
)
async def extract_form(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Uploaded inspection form (Image or PDF)"),
) -> JSONResponse:
    content_type = request.headers.get("content-type", "").lower()
    file_bytes: bytes = b""
    filename: str = "upload.jpg"

    try:
        if "multipart/form-data" in content_type:
            if file is None:
                form = await request.form()
                upload_item = form.get("file")
                if upload_item is None or not hasattr(upload_item, "read"):
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"success": False, "error": "Multipart body must contain a 'file' field."},
                    )
                file_bytes = await upload_item.read()
                filename = getattr(upload_item, "filename", "upload.jpg") or "upload.jpg"
            else:
                file_bytes = await file.read()
                filename = file.filename or "upload.jpg"
        elif "application/json" in content_type:
            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"success": False, "error": f"Invalid JSON payload: {e}"},
                )
            if not isinstance(body, dict) or "file" not in body:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"success": False, "error": "JSON body must contain a 'file' field (base64 string)."},
                )
            filename = body.get("filename", "upload.jpg")
            try:
                file_bytes = base64.b64decode(body["file"])
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"success": False, "error": f"Invalid base64 in 'file': {e}"},
                )
        else:
            raw_body = await request.body()
            if not raw_body:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"success": False, "error": "Empty request body. Send multipart/form-data or application/json."},
                )
            try:
                body = json.loads(raw_body)
                if isinstance(body, dict) and "file" in body:
                    filename = body.get("filename", "upload.jpg")
                    file_bytes = base64.b64decode(body["file"])
                else:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"success": False, "error": "Unsupported Content-Type. Use multipart/form-data or application/json."},
                    )
            except Exception:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"success": False, "error": "Unsupported Content-Type. Use multipart/form-data or application/json."},
                )

    except Exception as exc:
        logger.exception("Failed to parse request")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": f"Bad request: {exc}"},
        )

    if not file_bytes:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Empty file body."},
        )

    logger.info("Processing file: %s (%d bytes)", filename, len(file_bytes))

    try:
        result = run_pipeline(file_bytes, filename)
    except Exception as exc:
        logger.exception("Pipeline execution error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(exc)},
        )

    status_code = status.HTTP_200_OK if result.success else 422
    return JSONResponse(
        status_code=status_code,
        content=result.model_dump(exclude_none=False),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
