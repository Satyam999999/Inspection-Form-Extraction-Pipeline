"""
Evaluation script — measures field-level precision, recall, and F1 for each
extracted inspection form against its ground-truth JSON.

Usage
-----
    python eval/evaluate.py \
        --ground-truth forms/ground_truth \
        --predictions eval/predictions \
        --report eval/report.md

The predictions directory should contain one JSON file per form (named
FORM-XXX.json), created either by running the pipeline manually or by
running this script with --run-pipeline (which calls the pipeline directly).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────
# Field comparison helpers
# ──────────────────────────────────────────────

SCALAR_FIELDS = [
    "form_type",
    "form_id",
    "inspection_date",
    "overall_result",
]

INSPECTOR_FIELDS = ["name", "id_number"]
SITE_FIELDS = ["site_name", "location"]


def _normalize(v: Any) -> str:
    """Normalize a value for fuzzy comparison."""
    if v is None:
        return ""
    return str(v).strip().upper().replace("-", "").replace("/", "").replace(" ", "")


def _str_match(pred: Any, gt: Any) -> bool:
    return _normalize(pred) == _normalize(gt)


def _checklist_metrics(pred_items: List[dict], gt_items: List[dict]) -> Tuple[float, float, float, List[str]]:
    """
    Compare checklist items by description similarity + result match.

    Returns (precision, recall, f1, failure_notes)
    """
    if not gt_items:
        return (1.0, 1.0, 1.0, [])

    gt_lookup = {_normalize(i["description"]): i["result"] for i in gt_items}
    pred_lookup = {_normalize(i["description"]): i["result"] for i in pred_items}

    tp = fp = fn = 0
    failure_notes: List[str] = []

    for desc, gt_result in gt_lookup.items():
        if desc in pred_lookup:
            if _normalize(pred_lookup[desc]) == _normalize(gt_result):
                tp += 1
            else:
                fp += 1
                failure_notes.append(
                    f"  Result mismatch for '{desc}': pred={pred_lookup[desc]} gt={gt_result}"
                )
        else:
            fn += 1
            failure_notes.append(f"  Checklist item not found: '{desc}'")

    for desc in pred_lookup:
        if desc not in gt_lookup:
            fp += 1
            failure_notes.append(f"  Spurious checklist item: '{desc}'")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, failure_notes


def _nc_metrics(pred_ncs: List[dict], gt_ncs: List[dict]) -> Tuple[float, float, float, List[str]]:
    """Compare NC records by description similarity."""
    if not gt_ncs:
        return (1.0, 1.0, 1.0, [])

    gt_descs = {_normalize(n["description"]) for n in gt_ncs}
    pred_descs = {_normalize(n["description"]) for n in pred_ncs}

    tp = len(gt_descs & pred_descs)
    fn = len(gt_descs - pred_descs)
    fp = len(pred_descs - gt_descs)

    failure_notes: List[str] = []
    for d in gt_descs - pred_descs:
        failure_notes.append(f"  Missing NC: '{d}'")
    for d in pred_descs - gt_descs:
        failure_notes.append(f"  Spurious NC: '{d}'")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, failure_notes


def evaluate_form(pred: dict, gt: dict) -> dict:
    """Evaluate a single predicted form against ground truth. Returns metrics dict."""
    result: dict = {
        "form_id": gt.get("form_id", "?"),
        "quality": gt.get("quality", "unknown"),
        "failures": [],
    }

    # ── Scalar fields ─────────────────────────────────────────────────────────
    scalar_tp = scalar_total = 0
    for field in SCALAR_FIELDS:
        gt_val = gt.get(field)
        pred_val = pred.get(field)
        if gt_val:
            scalar_total += 1
            if _str_match(pred_val, gt_val):
                scalar_tp += 1
            else:
                result["failures"].append(
                    f"  [{field}] pred={pred_val!r} ≠ gt={gt_val!r}"
                )

    # Inspector sub-fields
    gt_insp = gt.get("inspector", {}) or {}
    pred_insp = pred.get("inspector", {}) or {}
    for f in INSPECTOR_FIELDS:
        if gt_insp.get(f):
            scalar_total += 1
            if _str_match(pred_insp.get(f), gt_insp.get(f)):
                scalar_tp += 1
            else:
                result["failures"].append(
                    f"  [inspector.{f}] pred={pred_insp.get(f)!r} ≠ gt={gt_insp.get(f)!r}"
                )

    # Site sub-fields
    gt_site = gt.get("site", {}) or {}
    pred_site = pred.get("site", {}) or {}
    for f in SITE_FIELDS:
        if gt_site.get(f):
            scalar_total += 1
            if _str_match(pred_site.get(f), gt_site.get(f)):
                scalar_tp += 1
            else:
                result["failures"].append(
                    f"  [site.{f}] pred={pred_site.get(f)!r} ≠ gt={gt_site.get(f)!r}"
                )

    scalar_prec = scalar_rec = scalar_f1 = 0.0
    if scalar_total:
        scalar_prec = scalar_rec = scalar_tp / scalar_total
        scalar_f1 = scalar_prec  # precision == recall for scalar hits

    result["scalar_precision"] = round(scalar_prec, 4)
    result["scalar_recall"] = round(scalar_rec, 4)
    result["scalar_f1"] = round(scalar_f1, 4)

    # ── Checklist ────────────────────────────────────────────────────────────
    cp, cr, cf, cfails = _checklist_metrics(
        pred.get("checklist_items", []) or [],
        gt.get("checklist_items", []) or [],
    )
    result["checklist_precision"] = round(cp, 4)
    result["checklist_recall"] = round(cr, 4)
    result["checklist_f1"] = round(cf, 4)
    result["failures"].extend(cfails)

    # ── Non-conformances ─────────────────────────────────────────────────────
    np_, nr, nf, nfails = _nc_metrics(
        pred.get("non_conformances", []) or [],
        gt.get("non_conformances", []) or [],
    )
    result["nc_precision"] = round(np_, 4)
    result["nc_recall"] = round(nr, 4)
    result["nc_f1"] = round(nf, 4)
    result["failures"].extend(nfails)

    # ── Overall form F1 (average of the three) ───────────────────────────────
    result["overall_f1"] = round((scalar_f1 + cf + nf) / 3, 4)

    return result


# ──────────────────────────────────────────────
# Run pipeline on form images
# ──────────────────────────────────────────────

def run_pipeline_on_forms(
    images_dir: Path,
    pred_dir: Path,
    gt_dir: Path,
) -> None:
    """Call run_pipeline() on each form image and save prediction JSON."""
    from src.pipeline import run_pipeline

    pred_dir.mkdir(parents=True, exist_ok=True)

    for gt_path in sorted(gt_dir.glob("*.json")):
        form_id = gt_path.stem
        img_path = images_dir / f"{form_id}.png"
        if not img_path.exists():
            print(f"  ⚠ Image not found: {img_path}")
            continue

        print(f"  Processing {form_id}...", end=" ", flush=True)
        t0 = time.perf_counter()
        file_bytes = img_path.read_bytes()
        response = run_pipeline(file_bytes, img_path.name)
        elapsed = time.perf_counter() - t0

        if response.success and response.form:
            pred = response.form.model_dump()
            (pred_dir / f"{form_id}.json").write_text(json.dumps(pred, indent=2))
            print(f"OK ({elapsed:.1f}s, {response.retries_used} retries)")
        else:
            print(f"FAILED: {response.error}")
            # Write an empty skeleton so evaluation can still run
            (pred_dir / f"{form_id}.json").write_text(
                json.dumps({"form_type": None, "checklist_items": [], "non_conformances": []}, indent=2)
            )


# ──────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────

def build_report(results: List[dict]) -> str:
    df = pd.DataFrame(results)

    lines: List[str] = []
    lines.append("# Inspection-Form Extraction — Evaluation Report\n")
    lines.append(f"**Forms evaluated**: {len(results)}\n")

    # ── Per-form table ────────────────────────────────────────────────────────
    lines.append("## Per-Form Results\n")
    cols = ["form_id", "quality", "scalar_f1", "checklist_f1", "nc_f1", "overall_f1"]
    lines.append(df[cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    lines.append("## Aggregate Metrics\n")
    agg = df[["scalar_f1", "checklist_f1", "nc_f1", "overall_f1"]].mean().round(4)
    lines.append("| Metric | Mean F1 |")
    lines.append("|--------|---------|")
    for col, val in agg.items():
        lines.append(f"| {col} | {val:.3f} |")
    lines.append("\n")

    # ── Breakdown by quality ──────────────────────────────────────────────────
    lines.append("## Performance by Scan Quality\n")
    grp = df.groupby("quality")["overall_f1"].mean().reset_index()
    grp.columns = ["quality", "mean_f1"]
    lines.append(grp.to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n")

    # ── Failure analysis ──────────────────────────────────────────────────────
    lines.append("## Failure Analysis\n")
    failure_categories: Dict[str, List[str]] = {
        "blurry": [],
        "blurry_rotated": [],
        "dark": [],
        "handwritten": [],
        "lowres": [],
        "noisy": [],
        "overexposed": [],
        "rotated": [],
        "clean": [],
    }
    for r in results:
        cat = r.get("quality", "clean")
        if cat not in failure_categories:
            failure_categories[cat] = []
        failure_categories[cat].extend(r.get("failures", []))

    for cat, fails in failure_categories.items():
        if fails:
            lines.append(f"\n### {cat.replace('_', ' ').title()}\n")
            for f in fails[:10]:   # cap at 10 per category for readability
                lines.append(f"- {f.strip()}")
            if len(fails) > 10:
                lines.append(f"- … and {len(fails) - 10} more")

    lines.append("\n")

    # ── Worst performers ─────────────────────────────────────────────────────
    lines.append("## Worst-Performing Forms\n")
    worst = df.nsmallest(5, "overall_f1")[["form_id", "quality", "overall_f1"]]
    lines.append(worst.to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main(args=None):
    parser = argparse.ArgumentParser(description="Evaluate inspection-form extraction pipeline.")
    parser.add_argument("--ground-truth", default="forms/ground_truth", help="Dir with GT JSON files")
    parser.add_argument("--predictions", default="eval/predictions", help="Dir with prediction JSON files")
    parser.add_argument("--images", default="forms/images", help="Dir with form images")
    parser.add_argument("--report", default="eval/report.md", help="Output Markdown report path")
    parser.add_argument(
        "--run-pipeline", action="store_true",
        help="Run pipeline on images first, then evaluate"
    )
    ns = parser.parse_args(args)

    gt_dir = Path(ns.ground_truth)
    pred_dir = Path(ns.predictions)
    images_dir = Path(ns.images)
    report_path = Path(ns.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if ns.run_pipeline:
        print("── Running pipeline on all forms ──")
        run_pipeline_on_forms(images_dir, pred_dir, gt_dir)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        print(f"No ground-truth files found in {gt_dir}")
        sys.exit(1)

    results: List[dict] = []
    for gt_path in gt_files:
        form_id = gt_path.stem
        pred_path = pred_dir / f"{form_id}.json"

        gt = json.loads(gt_path.read_text())

        if pred_path.exists():
            pred = json.loads(pred_path.read_text())
        else:
            print(f"  ⚠ No prediction for {form_id} — treating as empty")
            pred = {"form_type": None, "checklist_items": [], "non_conformances": []}

        metrics = evaluate_form(pred, gt)
        results.append(metrics)

    # ── Report ───────────────────────────────────────────────────────────────
    report = build_report(results)
    report_path.write_text(report)
    print(f"\n✓ Report written to {report_path}")

    # Print summary to stdout
    df = pd.DataFrame(results)
    mean_f1 = df["overall_f1"].mean()
    print(f"  Mean overall F1: {mean_f1:.3f}")


if __name__ == "__main__":
    main()
