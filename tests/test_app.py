"""
Unit tests for the FastAPI application and endpoints.
"""

from __future__ import annotations

import base64
import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.schema import ExtractionResponse, InspectionForm


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_form_dict():
    return {
        "form_type": "SAFETY_WALKTHROUGH",
        "form_id": "TEST-100",
        "inspection_date": "05/05/2024",
        "inspector": {
            "name": "Alex Vance",
            "id_number": "AV-100",
            "company": "Safety Corp",
            "signature_present": True,
        },
        "site": {
            "site_name": "Sector 7",
            "location": "North Wing",
            "department": "Security",
            "permit_number": "P-99",
        },
        "overall_result": "PASS",
        "checklist_items": [
            {
                "item_number": "1",
                "description": "Emergency lights functional",
                "result": "PASS",
                "remarks": None,
                "risk_level": None,
            }
        ],
        "equipment_items": [],
        "non_conformances": [],
    }


class TestFastAPIApp:
    def test_root_endpoint(self, client):
        res = client.get("/")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "online"
        assert body["docs"] == "/docs"

    def test_health_check(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "healthy"}

    def test_cors_headers(self, client):
        res = client.options(
            "/extract",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") in ["*", "http://localhost:3000"]

    def test_empty_body_bad_request(self, client):
        res = client.post(
            "/extract",
            content=b"",
            headers={"Content-Type": "application/json"},
        )
        assert res.status_code == 400
        body = res.json()
        assert not body["success"]

    @patch("src.app.run_pipeline")
    def test_json_payload_success(self, mock_run_pipeline, client, sample_form_dict):
        mock_form = InspectionForm.model_validate(sample_form_dict)
        mock_run_pipeline.return_value = ExtractionResponse(
            success=True,
            form=mock_form,
            retries_used=0,
            model_used="gemini-1.5-flash",
            processing_time_ms=120.0,
        )

        b64_img = base64.b64encode(b"simulated-image-bytes").decode("utf-8")
        res = client.post(
            "/extract",
            json={"filename": "test.png", "file": b64_img},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["form"]["form_id"] == "TEST-100"

    @patch("src.app.run_pipeline")
    def test_multipart_payload_success(self, mock_run_pipeline, client, sample_form_dict):
        mock_form = InspectionForm.model_validate(sample_form_dict)
        mock_run_pipeline.return_value = ExtractionResponse(
            success=True,
            form=mock_form,
            retries_used=0,
            model_used="gemini-1.5-flash",
            processing_time_ms=150.0,
        )

        file_content = b"fake-png-bytes"
        res = client.post(
            "/extract",
            files={"file": ("inspection.png", io.BytesIO(file_content), "image/png")},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["form"]["form_id"] == "TEST-100"

    @patch("src.app.run_pipeline")
    def test_pipeline_failure_returns_422(self, mock_run_pipeline, client):
        mock_run_pipeline.return_value = ExtractionResponse(
            success=False,
            error="Validation failed after retries",
            retries_used=3,
            model_used="gemini-1.5-flash",
            processing_time_ms=350.0,
        )

        b64_img = base64.b64encode(b"corrupt-data").decode("utf-8")
        res = client.post(
            "/extract",
            json={"filename": "corrupt.png", "file": b64_img},
        )
        assert res.status_code == 422
        body = res.json()
        assert body["success"] is False
        assert "Validation failed" in body["error"]

    @patch("src.app.run_pipeline")
    def test_pipeline_exception_returns_500(self, mock_run_pipeline, client):
        mock_run_pipeline.side_effect = RuntimeError("Fatal pipeline crash")

        b64_img = base64.b64encode(b"valid-data").decode("utf-8")
        res = client.post(
            "/extract",
            json={"filename": "sample.png", "file": b64_img},
        )
        assert res.status_code == 500
        body = res.json()
        assert body["success"] is False
        assert "Fatal pipeline crash" in body["error"]
