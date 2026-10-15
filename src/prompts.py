"""
Prompt templates for inspection-form extraction via Gemini vision.
"""

from __future__ import annotations

import json

from src.schema import InspectionForm

# ─────────────────────────────────────────────
# Build the JSON schema snippet once at import time
# ─────────────────────────────────────────────
_SCHEMA_JSON = json.dumps(InspectionForm.model_json_schema(), indent=2)

SYSTEM_PROMPT = """\
You are an expert document digitisation assistant specialising in industrial
and construction safety inspection forms.

Your task is to extract EVERY field visible in the supplied image(s) into a
single, well-structured JSON object that conforms exactly to the JSON Schema
shown below.

Rules
-----
1. Output ONLY the JSON object — no markdown fences, no preamble, no trailing
   commentary.
2. Use null for fields that are not present or illegible; never omit a key.
3. Dates: preserve the original format found in the document (do NOT convert).
4. Enumerations must be one of the allowed values (case-sensitive).
5. Checklist rows that have no result checkbox ticked should be "N/A".
6. If a checklist section is present but empty, return an empty list [].
7. Signatures are indicated by "signature_present": true / false.
8. For handwritten text that is partially legible, transcribe your best
   interpretation and append " [illegible]" at the end of the field value.
9. image_quality_notes: leave as null — the pipeline fills this field.

JSON Schema to follow
---------------------
{schema}
"""

USER_PROMPT_SINGLE = """\
Extract ALL fields from this inspection form image and return the JSON.
"""

USER_PROMPT_MULTI = """\
Extract ALL fields from these {n} inspection form pages and return a single
merged JSON object representing the complete form.
"""

RETRY_SYSTEM_PROMPT = """\
Your previous JSON output failed Pydantic validation with this error:

{error}

Please correct the JSON so it validates successfully against the schema.
Output ONLY the corrected JSON — nothing else.

Original (invalid) JSON
-----------------------
{bad_json}
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(schema=_SCHEMA_JSON)


def build_retry_prompt(error: str, bad_json: str) -> str:
    return RETRY_SYSTEM_PROMPT.format(error=error, bad_json=bad_json)
