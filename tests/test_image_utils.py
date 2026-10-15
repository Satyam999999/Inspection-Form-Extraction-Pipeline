"""
Unit tests for image and PDF preprocessing routines.
"""

from __future__ import annotations

import io
from PIL import Image

import pytest
from src.image_utils import (
    assess_image_quality,
    enhance_image,
    prepare_images,
    quality_notes,
    resize_for_model,
)


class TestImageUtils:
    def test_assess_image_quality_clean(self):
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        metrics = assess_image_quality(img)
        assert metrics["width"] == 200
        assert metrics["height"] == 200
        assert "blur_score" in metrics
        assert "brightness" in metrics
        assert not metrics["is_too_dark"]
        assert not metrics["is_too_bright"]

    def test_assess_image_quality_dark(self):
        img = Image.new("RGB", (100, 100), color=(10, 10, 10))
        metrics = assess_image_quality(img)
        assert metrics["is_too_dark"] is True
        notes = quality_notes(metrics)
        assert "dark scan" in notes

    def test_resize_for_model(self):
        large_img = Image.new("RGB", (3000, 1500), color=(255, 255, 255))
        resized = resize_for_model(large_img, max_side=2048)
        assert resized.width == 2048
        assert resized.height == 1024

    def test_resize_for_model_already_small(self):
        small_img = Image.new("RGB", (500, 500), color=(255, 255, 255))
        resized = resize_for_model(small_img, max_side=2048)
        assert resized.size == (500, 500)

    def test_enhance_image_runs(self):
        img = Image.new("RGB", (150, 150), color=(120, 120, 120))
        enhanced = enhance_image(img)
        assert enhanced.size == (150, 150)

    def test_prepare_images_png(self):
        img = Image.new("RGB", (250, 250), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        file_bytes = buf.getvalue()

        images, notes = prepare_images(file_bytes, "inspection_test.png")
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)

    def test_prepare_images_unsupported_ext(self):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            prepare_images(b"random-bytes", "document.docx")
