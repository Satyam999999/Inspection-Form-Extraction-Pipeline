"""
Unit and integration tests for the inspection-form extraction pipeline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import (
    ChecklistItem,
    ExtractionResponse,
    FormType,
    InspectionForm,
    InspectionResult,
    Inspector,
    RiskLevel,
    SiteInfo,
)
from eval.evaluate import evaluate_form, _normalize, _str_match


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def minimal_form_dict():
    return {
        "form_type": "SAFETY_WALKTHROUGH",
        "form_id": "TEST-001",
        "inspection_date": "01/01/2024",
        "inspector": {
            "name": "Alice Smith",
            "id_number": "AS-001",
            "company": None,
            "signature_present": False,
        },
        "site": {
            "site_name": "Test Site",
            "location": "Block A",
            "department": None,
            "permit_number": None,
        },
        "overall_result": "PASS",
        "checklist_items": [
            {
                "item_number": "1",
                "description": "PPE worn by all workers",
                "result": "PASS",
                "remarks": None,
                "risk_level": None,
            }
        ],
        "equipment_items": [],
        "non_conformances": [],
    }


# ──────────────────────────────────────────────
# Schema tests
# ──────────────────────────────────────────────

class TestInspectionFormSchema:
    def test_valid_minimal_form(self, minimal_form_dict):
        form = InspectionForm.model_validate(minimal_form_dict)
        assert form.form_type == FormType.SAFETY_WALKTHROUGH
        assert form.inspector.name == "Alice Smith"
        assert len(form.checklist_items) == 1

    def test_invalid_form_type_raises(self, minimal_form_dict):
        minimal_form_dict["form_type"] = "INVALID_TYPE"
        with pytest.raises(Exception):
            InspectionForm.model_validate(minimal_form_dict)

    def test_empty_checklist_and_equipment_raises(self, minimal_form_dict):
        minimal_form_dict["checklist_items"] = []
        minimal_form_dict["equipment_items"] = []
        with pytest.raises(Exception):
            InspectionForm.model_validate(minimal_form_dict)

    def test_none_checklist_coerced_to_empty_list(self, minimal_form_dict):
        minimal_form_dict["non_conformances"] = None
        form = InspectionForm.model_validate(minimal_form_dict)
        assert form.non_conformances == []

    def test_result_enum_values(self):
        assert InspectionResult.PASS == "PASS"
        assert InspectionResult.FAIL == "FAIL"
        assert InspectionResult.NA == "N/A"

    def test_serialisation_roundtrip(self, minimal_form_dict):
        form = InspectionForm.model_validate(minimal_form_dict)
        dumped = form.model_dump()
        form2 = InspectionForm.model_validate(dumped)
        assert form2.form_id == form.form_id

    def test_extraction_response_success(self, minimal_form_dict):
        form = InspectionForm.model_validate(minimal_form_dict)
        resp = ExtractionResponse(
            success=True,
            form=form,
            retries_used=0,
            model_used="gemini-1.5-flash",
            processing_time_ms=500.0,
        )
        assert resp.success
        assert resp.form is not None

    def test_extraction_response_failure(self):
        resp = ExtractionResponse(
            success=False,
            error="Validation error",
            retries_used=3,
            model_used="gemini-1.5-flash",
            processing_time_ms=1200.0,
        )
        assert not resp.success
        assert resp.error == "Validation error"


# ──────────────────────────────────────────────
# Evaluation metric tests
# ──────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_normalize_uppercases_and_strips(self):
        assert _normalize("  Hello World ") == "HELLOWORLD"

    def test_normalize_removes_dashes_slashes(self):
        assert _normalize("01/01/2024") == "01012024"
        assert _normalize("TEST-001") == "TEST001"

    def test_str_match_case_insensitive(self):
        assert _str_match("PASS", "pass")
        assert _str_match("Alpha Site", "ALPHA SITE")

    def test_str_match_mismatch(self):
        assert not _str_match("PASS", "FAIL")

    def test_perfect_eval(self):
        gt = {
            "form_type": "SAFETY_WALKTHROUGH",
            "form_id": "TEST-001",
            "inspection_date": "01/01/2024",
            "overall_result": "PASS",
            "inspector": {"name": "Alice Smith", "id_number": "AS-001"},
            "site": {"site_name": "Test Site", "location": "Block A"},
            "checklist_items": [
                {"item_number": "1", "description": "PPE worn", "result": "PASS", "remarks": None},
            ],
            "non_conformances": [],
            "quality": "clean",
        }
        metrics = evaluate_form(gt, gt)
        assert metrics["scalar_f1"] == 1.0
        assert metrics["checklist_f1"] == 1.0
        assert metrics["overall_f1"] == 1.0

    def test_missing_checklist_item_lowers_recall(self):
        gt = {
            "form_type": "SAFETY_WALKTHROUGH",
            "form_id": "T-001",
            "inspection_date": "01/01/2024",
            "overall_result": "PASS",
            "inspector": {"name": "A", "id_number": "1"},
            "site": {"site_name": "S", "location": "L"},
            "checklist_items": [
                {"item_number": "1", "description": "PPE worn", "result": "PASS"},
                {"item_number": "2", "description": "Exits clear", "result": "PASS"},
            ],
            "non_conformances": [],
            "quality": "clean",
        }
        pred = dict(gt)
        pred["checklist_items"] = [
            {"item_number": "1", "description": "PPE worn", "result": "PASS"},
        ]
        metrics = evaluate_form(pred, gt)
        assert metrics["checklist_recall"] < 1.0


# ──────────────────────────────────────────────
# Pipeline smoke test (mocked LLM)
# ──────────────────────────────────────────────

class TestPipelineSmoke:
    @patch("src.pipeline.get_llm")
    @patch("src.pipeline.prepare_images")
    def test_pipeline_success(self, mock_prepare, mock_get_llm, minimal_form_dict):
        """Pipeline should succeed when LLM returns valid JSON."""
        # Mock image preparation
        mock_img = MagicMock()
        mock_prepare.return_value = ([mock_img], "")

        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(minimal_form_dict))
        mock_get_llm.return_value = mock_llm

        from src.pipeline import run_pipeline

        # Create minimal fake image bytes
        result = run_pipeline(b"fake-png-bytes", "test_form.png")
        assert result.success
        assert result.form is not None
        assert result.retries_used == 0

    @patch("src.pipeline.get_llm")
    @patch("src.pipeline.prepare_images")
    def test_pipeline_retry_on_invalid_json(self, mock_prepare, mock_get_llm, minimal_form_dict):
        """Pipeline should retry after receiving invalid JSON."""
        mock_img = MagicMock()
        mock_prepare.return_value = ([mock_img], "")

        mock_llm = MagicMock()
        # First call returns invalid JSON; second returns valid
        mock_llm.invoke.side_effect = [
            MagicMock(content="not valid json {{"),
            MagicMock(content=json.dumps(minimal_form_dict)),
        ]
        mock_get_llm.return_value = mock_llm

        from src.pipeline import run_pipeline

        result = run_pipeline(b"fake-png-bytes", "test_form.png")
        assert result.success
        assert result.retries_used == 1

    @patch("src.pipeline.get_llm")
    @patch("src.pipeline.prepare_images")
    def test_pipeline_exhausts_retries(self, mock_prepare, mock_get_llm):
        """Pipeline returns failure after MAX_RETRIES exhausted."""
        mock_img = MagicMock()
        mock_prepare.return_value = ([mock_img], "")

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="always invalid {{{")
        mock_get_llm.return_value = mock_llm

        import src.pipeline as pipeline_module
        original_max = pipeline_module.MAX_RETRIES
        pipeline_module.MAX_RETRIES = 2
        pipeline_module._GRAPH = None   # reset compiled graph

        try:
            from src.pipeline import run_pipeline
            result = run_pipeline(b"fake-png-bytes", "test_form.png")
        finally:
            pipeline_module.MAX_RETRIES = original_max
            pipeline_module._GRAPH = None

        assert not result.success
        assert result.retries_used >= 1
