"""
LangGraph extraction pipeline.

Graph topology
──────────────

  START
    │
    ▼
  preprocess_node   ← load & enhance images
    │
    ▼
  extract_node      ← call Gemini 1.5 Flash with vision
    │
    ▼
  validate_node     ← Pydantic validation
    │
   ┌┴──────────────────┐
   ▼                   ▼
  return_node        retry_node   (up to MAX_RETRIES times)
   │                   │
   │                   ▼
   │               extract_node  (re-infer with error context)
   │                   │
   └───────────────────┘
         │
         ▼
       END
"""

from __future__ import annotations

import json
import logging
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.image_utils import prepare_images
from src.prompts import build_system_prompt, build_retry_prompt, USER_PROMPT_SINGLE, USER_PROMPT_MULTI
from src.schema import ExtractionResponse, InspectionForm

logger = logging.getLogger(__name__)

MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


# ──────────────────────────────────────────────
# LangGraph state
# ──────────────────────────────────────────────

class PipelineState(TypedDict):
    # Inputs
    file_bytes: bytes
    filename: str

    # Intermediate
    images: List[Any]               # PIL Images
    quality_note: str

    # LLM communication
    raw_llm_output: str
    last_error: Optional[str]
    retries: int

    # Outputs
    form: Optional[Dict[str, Any]]
    success: bool
    error_message: Optional[str]
    model_used: str
    start_time: float


# ──────────────────────────────────────────────
# Helper: PIL image → base64 for Gemini
# ──────────────────────────────────────────────

def _pil_to_base64(img: Any) -> str:
    import base64
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────
# Build the Gemini client (singleton per process)
# ──────────────────────────────────────────────

def _build_llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0,
        max_retries=2,          # LangChain-level network retries
    )


_LLM: Optional[ChatGoogleGenerativeAI] = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _LLM
    if _LLM is None:
        _LLM = _build_llm()
    return _LLM


# ──────────────────────────────────────────────
# Graph nodes
# ──────────────────────────────────────────────

def preprocess_node(state: PipelineState) -> PipelineState:
    """Load bytes → PIL images with quality enhancement."""
    logger.info("preprocess_node: loading %s", state["filename"])
    images, quality_note = prepare_images(state["file_bytes"], state["filename"])
    return {**state, "images": images, "quality_note": quality_note, "retries": 0}


def _build_messages_first(images: List[Any]) -> List:
    """Build Gemini multi-modal messages for the first extraction attempt."""
    system = SystemMessage(content=build_system_prompt())

    content_parts: List[Dict] = []
    if len(images) == 1:
        text = USER_PROMPT_SINGLE
    else:
        text = USER_PROMPT_MULTI.format(n=len(images))

    content_parts.append({"type": "text", "text": text})
    for img in images:
        b64 = _pil_to_base64(img)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    human = HumanMessage(content=content_parts)
    return [system, human]


def _build_messages_retry(bad_json: str, error: str) -> List:
    """Build messages for a retry attempt, providing the error context."""
    system = SystemMessage(content=build_system_prompt())
    human = HumanMessage(content=build_retry_prompt(error=error, bad_json=bad_json))
    return [system, human]


def extract_node(state: PipelineState) -> PipelineState:
    """Call Gemini vision and capture raw JSON output."""
    logger.info(
        "extract_node: attempt %d/%d", state.get("retries", 0) + 1, MAX_RETRIES + 1
    )
    llm = get_llm()

    if state.get("last_error") and state.get("raw_llm_output"):
        messages = _build_messages_retry(
            bad_json=state["raw_llm_output"],
            error=state["last_error"],
        )
    else:
        messages = _build_messages_first(state["images"])

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown fences if the model wrapped the JSON anyway
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

    return {**state, "raw_llm_output": raw, "model_used": GEMINI_MODEL}


def validate_node(state: PipelineState) -> PipelineState:
    """Validate raw_llm_output against InspectionForm Pydantic schema."""
    raw = state.get("raw_llm_output", "")
    try:
        data = json.loads(raw)
        # Inject quality note from preprocessing
        if state.get("quality_note"):
            data["image_quality_notes"] = state["quality_note"]
        form = InspectionForm.model_validate(data)
        return {
            **state,
            "form": form.model_dump(),
            "success": True,
            "error_message": None,
            "last_error": None,
        }
    except (json.JSONDecodeError, ValidationError) as exc:
        error_msg = str(exc)
        logger.warning("validate_node: validation failed — %s", error_msg[:300])
        return {
            **state,
            "form": None,
            "success": False,
            "error_message": error_msg,
            "last_error": error_msg,
        }


def retry_node(state: PipelineState) -> PipelineState:
    """Increment retry counter."""
    return {**state, "retries": state.get("retries", 0) + 1}


def return_node(state: PipelineState) -> PipelineState:
    """Terminal node — nothing to do; result is already in state."""
    return state


# ──────────────────────────────────────────────
# Routing logic
# ──────────────────────────────────────────────

def should_retry(state: PipelineState) -> str:
    """After validate_node: retry if failed and budget not exhausted."""
    if state["success"]:
        return "return"
    if state.get("retries", 0) < MAX_RETRIES:
        return "retry"
    return "return"         # Give up, return partial / error response


# ──────────────────────────────────────────────
# Build the graph
# ──────────────────────────────────────────────

def build_graph() -> Any:
    sg = StateGraph(PipelineState)

    sg.add_node("preprocess", preprocess_node)
    sg.add_node("extract", extract_node)
    sg.add_node("validate", validate_node)
    sg.add_node("retry", retry_node)
    sg.add_node("return", return_node)

    sg.add_edge(START, "preprocess")
    sg.add_edge("preprocess", "extract")
    sg.add_edge("extract", "validate")
    sg.add_conditional_edges(
        "validate",
        should_retry,
        {"return": "return", "retry": "retry"},
    )
    sg.add_edge("retry", "extract")
    sg.add_edge("return", END)

    return sg.compile()


# Compile once at module level for warm server invocations
_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def run_pipeline(file_bytes: bytes, filename: str) -> ExtractionResponse:
    """
    Run the full extraction pipeline for a single file.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the PDF or image file.
    filename : str
        Original filename (used for extension detection).

    Returns
    -------
    ExtractionResponse
    """
    t0 = time.perf_counter()
    graph = get_graph()

    initial_state: PipelineState = {
        "file_bytes": file_bytes,
        "filename": filename,
        "images": [],
        "quality_note": "",
        "raw_llm_output": "",
        "last_error": None,
        "retries": 0,
        "form": None,
        "success": False,
        "error_message": None,
        "model_used": GEMINI_MODEL,
        "start_time": t0,
    }

    final_state = graph.invoke(initial_state)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if final_state["success"]:
        form = InspectionForm.model_validate(final_state["form"])
        return ExtractionResponse(
            success=True,
            form=form,
            retries_used=final_state.get("retries", 0),
            model_used=final_state["model_used"],
            processing_time_ms=round(elapsed_ms, 1),
            raw_llm_output=final_state.get("raw_llm_output"),
        )
    else:
        return ExtractionResponse(
            success=False,
            error=final_state.get("error_message", "Unknown error"),
            retries_used=final_state.get("retries", 0),
            model_used=final_state.get("model_used", GEMINI_MODEL),
            processing_time_ms=round(elapsed_ms, 1),
            raw_llm_output=final_state.get("raw_llm_output"),
        )
