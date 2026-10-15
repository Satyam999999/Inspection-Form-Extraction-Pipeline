"""
Pydantic v2 schemas for inspection-form extraction.

Covers: General Industrial / Construction Safety Inspection forms.
All fields are typed and annotated so the LLM is prompted correctly and
Pydantic validates the returned JSON.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

class InspectionResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NA = "N/A"
    PENDING = "PENDING"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FormType(str, Enum):
    SAFETY_WALKTHROUGH = "SAFETY_WALKTHROUGH"
    EQUIPMENT_INSPECTION = "EQUIPMENT_INSPECTION"
    SITE_HAZARD_ASSESSMENT = "SITE_HAZARD_ASSESSMENT"
    FIRE_SAFETY = "FIRE_SAFETY"
    ELECTRICAL_INSPECTION = "ELECTRICAL_INSPECTION"
    SCAFFOLD_INSPECTION = "SCAFFOLD_INSPECTION"


# ──────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────

class Inspector(BaseModel):
    name: str = Field(..., description="Full name of the inspector")
    id_number: Optional[str] = Field(None, description="Badge / employee ID")
    company: Optional[str] = Field(None, description="Inspector's company or contractor")
    signature_present: bool = Field(False, description="True if a signature is visible")


class SiteInfo(BaseModel):
    site_name: str = Field(..., description="Name or identifier of the site / facility")
    location: Optional[str] = Field(None, description="Address or GPS coordinates")
    department: Optional[str] = Field(None, description="Department or work area")
    permit_number: Optional[str] = Field(None, description="Work permit or project number")


class ChecklistItem(BaseModel):
    item_number: Optional[str] = Field(None, description="Row / item number in the checklist")
    description: str = Field(..., description="What is being checked")
    result: InspectionResult = Field(..., description="PASS / FAIL / N/A / PENDING")
    remarks: Optional[str] = Field(None, description="Free-text remarks or observations")
    risk_level: Optional[RiskLevel] = Field(None, description="Risk severity if FAIL")


class NonConformance(BaseModel):
    nc_id: Optional[str] = Field(None, description="NC reference number")
    description: str = Field(..., description="Description of the non-conformance")
    corrective_action: Optional[str] = Field(None, description="Required corrective action")
    responsible_party: Optional[str] = Field(None, description="Person / team accountable")
    due_date: Optional[str] = Field(None, description="Target completion date (any format found)")
    closed: bool = Field(False, description="True if the NC has been resolved")


class EquipmentItem(BaseModel):
    equipment_id: Optional[str] = Field(None, description="Asset / tag number")
    equipment_name: str = Field(..., description="Name or description of the equipment")
    last_maintenance_date: Optional[str] = Field(None, description="Date of last maintenance")
    condition: Optional[str] = Field(None, description="Overall condition description")
    result: InspectionResult = Field(..., description="Pass/fail/N/A for this equipment")


# ──────────────────────────────────────────────
# Root extraction model
# ──────────────────────────────────────────────

class InspectionForm(BaseModel):
    """Top-level extraction schema for an industrial / construction inspection form."""

    form_type: FormType = Field(..., description="Type of inspection form detected")
    form_id: Optional[str] = Field(None, description="Unique form / report ID printed on the form")
    form_version: Optional[str] = Field(None, description="Version or revision of the form template")

    inspection_date: Optional[str] = Field(
        None,
        description="Date the inspection was conducted (preserve original format)",
    )
    report_date: Optional[str] = Field(
        None,
        description="Date the report was filed (if different from inspection_date)",
    )

    inspector: Inspector = Field(..., description="Details of the primary inspector")
    second_inspector: Optional[Inspector] = Field(None, description="Second or co-inspector if present")
    approved_by: Optional[str] = Field(None, description="Name of the approving authority")

    site: SiteInfo = Field(..., description="Site / facility details")

    overall_result: Optional[InspectionResult] = Field(
        None, description="Overall pass/fail result for the entire inspection"
    )
    overall_risk_level: Optional[RiskLevel] = Field(
        None, description="Overall risk level assigned to the inspection"
    )

    checklist_items: List[ChecklistItem] = Field(
        default_factory=list,
        description="All checklist rows extracted from the form",
    )
    equipment_items: List[EquipmentItem] = Field(
        default_factory=list,
        description="Equipment-specific inspection records",
    )
    non_conformances: List[NonConformance] = Field(
        default_factory=list,
        description="Non-conformances or deficiencies recorded",
    )

    general_observations: Optional[str] = Field(
        None, description="Free-text general observations or summary comments"
    )
    recommendations: Optional[str] = Field(
        None, description="Recommendations for improvement"
    )
    next_inspection_date: Optional[str] = Field(
        None, description="Scheduled date of next inspection"
    )

    # Image-quality diagnostics (filled by the pipeline, not the LLM)
    image_quality_notes: Optional[str] = Field(
        None, description="Pipeline notes on scan quality (blurry, low-res, handwritten, etc.)"
    )

    model_config = {"use_enum_values": True}

    @field_validator("checklist_items", "equipment_items", "non_conformances", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> List:
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    @model_validator(mode="after")
    def at_least_one_checklist_or_equipment(self) -> "InspectionForm":
        if not self.checklist_items and not self.equipment_items:
            raise ValueError(
                "Extracted form must contain at least one checklist item or equipment record."
            )
        return self


# ──────────────────────────────────────────────
# Lightweight response wrapper
# ──────────────────────────────────────────────

class ExtractionResponse(BaseModel):
    """API response envelope returned by the extraction API."""

    success: bool
    form: Optional[InspectionForm] = None
    error: Optional[str] = None
    retries_used: int = 0
    model_used: str = ""
    processing_time_ms: float = 0.0
    raw_llm_output: Optional[str] = Field(None, description="Raw LLM JSON (debug only)")
