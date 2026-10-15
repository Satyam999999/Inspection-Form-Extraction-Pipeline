"""
Image pre-processing utilities.

Converts PDF pages to PIL images, auto-rotates, sharpens blurry scans,
and resizes to Gemini's recommended maximum (2 048 px on the longest side).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

MAX_SIDE_PX = 2048          # Gemini 1.5 flash optimal
BLUR_VARIANCE_THRESHOLD = 80  # Below this → apply unsharp mask
MIN_BRIGHTNESS = 30           # Below this → enhance brightness
MAX_BRIGHTNESS = 225          # Above this → reduce brightness


# ──────────────────────────────────────────────
# PDF → images
# ──────────────────────────────────────────────

def pdf_to_pil_images(pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """Convert every PDF page to a PIL Image at *dpi* dots-per-inch."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pymupdf: pip install pymupdf") from exc

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[Image.Image] = []
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    for page in doc:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        images.append(img)
    doc.close()
    return images


# ──────────────────────────────────────────────
# Quality assessment
# ──────────────────────────────────────────────

def _laplacian_variance(img: Image.Image) -> float:
    """Measure sharpness; lower = blurrier."""
    gray = np.array(img.convert("L"))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_image_quality(img: Image.Image) -> dict:
    """Return a dict of quality metrics for diagnostic purposes."""
    gray = np.array(img.convert("L"))
    blur_score = _laplacian_variance(img)
    brightness = float(gray.mean())
    return {
        "width": img.width,
        "height": img.height,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "is_blurry": blur_score < BLUR_VARIANCE_THRESHOLD,
        "is_too_dark": brightness < MIN_BRIGHTNESS,
        "is_too_bright": brightness > MAX_BRIGHTNESS,
    }


def quality_notes(metrics: dict) -> str:
    """Human-readable quality warning string, or empty string if OK."""
    issues: List[str] = []
    if metrics["is_blurry"]:
        issues.append(f"blurry (Laplacian var={metrics['blur_score']})")
    if metrics["is_too_dark"]:
        issues.append(f"dark scan (brightness={metrics['brightness']})")
    if metrics["is_too_bright"]:
        issues.append(f"overexposed (brightness={metrics['brightness']})")
    if metrics["width"] < 400 or metrics["height"] < 400:
        issues.append(f"low resolution ({metrics['width']}×{metrics['height']}px)")
    return "; ".join(issues)


# ──────────────────────────────────────────────
# Enhancement
# ──────────────────────────────────────────────

def _auto_deskew(img: Image.Image) -> Image.Image:
    """Deskew using Hough-line rotation detection (best-effort)."""
    gray = np.array(img.convert("L"))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return img
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        np.array(img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return Image.fromarray(rotated)


def enhance_image(img: Image.Image) -> Image.Image:
    """Apply sharpening, contrast & brightness corrections."""
    metrics = assess_image_quality(img)

    # Deskew
    img = _auto_deskew(img)

    # Sharpen if blurry
    if metrics["is_blurry"]:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    # Adjust brightness
    if metrics["is_too_dark"]:
        img = ImageEnhance.Brightness(img).enhance(1.4)
    elif metrics["is_too_bright"]:
        img = ImageEnhance.Brightness(img).enhance(0.8)

    return img


def resize_for_model(img: Image.Image, max_side: int = MAX_SIDE_PX) -> Image.Image:
    """Down-scale so the longest side ≤ max_side; preserve aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ──────────────────────────────────────────────
# Main entry-point
# ──────────────────────────────────────────────

def prepare_images(
    file_bytes: bytes,
    filename: str,
    enhance: bool = True,
) -> Tuple[List[Image.Image], str]:
    """
    Load, optionally enhance, and resize images ready for the VLM.

    Returns
    -------
    images : list of PIL Images
    quality_note : human-readable quality warnings (empty = clean)
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        raw_images = pdf_to_pil_images(file_bytes)
    elif ext in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}:
        raw_images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
    else:
        raise ValueError(f"Unsupported file extension: {ext!r}")

    all_notes: List[str] = []
    processed: List[Image.Image] = []

    for i, img in enumerate(raw_images):
        metrics = assess_image_quality(img)
        note = quality_notes(metrics)
        if note:
            prefix = f"page {i+1}: " if len(raw_images) > 1 else ""
            all_notes.append(prefix + note)
            logger.warning("Image quality issue — %s%s", prefix, note)

        if enhance:
            img = enhance_image(img)
        img = resize_for_model(img)
        processed.append(img)

    return processed, "; ".join(all_notes)
