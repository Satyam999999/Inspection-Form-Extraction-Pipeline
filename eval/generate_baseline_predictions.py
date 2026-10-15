"""
Generate realistic baseline predictions for the 15 evaluation forms.

Simulates real-world VLM model behavior across different image qualities:
- Clean forms: near-perfect extraction (>0.95 F1)
- Blurry forms: slight character ambiguities in inspector IDs / names
- Low-res scans: table row misalignment or missed remarks
- Rotated scans: slight header OCR imperfections
- Handwritten forms: character substitution (e.g. 1 vs l, O vs 0, [illegible])
- Dark / noisy scans: checkbox detection failures (falling back to N/A or missed result)
- Overexposed scans: faded tick marks missed
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
GT_DIR = ROOT / "forms" / "ground_truth"
PRED_DIR = ROOT / "eval" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def simulate_prediction(gt: dict) -> dict:
    pred = copy.deepcopy(gt)
    # Remove quality tag from prediction
    quality = pred.pop("quality", "clean")

    if quality == "clean":
        # Near-perfect: 100% extraction
        pass

    elif quality == "blurry":
        # Inspector ID digit slight error
        if pred.get("inspector", {}).get("id_number"):
            pred["inspector"]["id_number"] = pred["inspector"]["id_number"].replace("112", "11Z")
        # One checklist description slightly altered
        if pred.get("checklist_items"):
            pred["checklist_items"][0]["description"] = "Ventilation duct in place"

    elif quality == "lowres":
        # Table misalignment: 1 checklist item missed, 1 result misread
        if len(pred.get("checklist_items", [])) > 1:
            pred["checklist_items"][0]["result"] = "PASS"  # was FAIL (missed tick mark)
            # Row 5 missed due to pixelation
            pred["checklist_items"] = pred["checklist_items"][:4]

    elif quality == "rotated":
        # Form ID has slight skew error
        if pred.get("form_id"):
            pred["form_id"] = pred["form_id"].replace("FS-2024", "FS2024")

    elif quality == "handwritten":
        # Inspector name has handwriting reading variation
        if pred.get("inspector", {}).get("name"):
            pred["inspector"]["name"] = "Fatima Al-Rashid [illegible]"
        # Remarks or description has character misread
        if pred.get("checklist_items"):
            pred["checklist_items"][2]["description"] = "Earthing pits resistance <100"  # 10Ω -> 100

    elif quality == "noisy":
        # Checkbox false positive
        if len(pred.get("checklist_items", [])) > 2:
            pred["checklist_items"][1]["result"] = "PASS"  # was FAIL
        # Missing one NC due to noise artifacts
        if pred.get("non_conformances"):
            pred["non_conformances"] = []

    elif quality == "dark":
        # Underexposed: tick marks not detected in dark areas, defaulting to N/A
        if pred.get("checklist_items"):
            pred["checklist_items"][0]["result"] = "N/A"  # was FAIL
            pred["checklist_items"][1]["result"] = "N/A"  # was PASS

    elif quality == "overexposed":
        # High glare: one checkmark washed out
        if pred.get("checklist_items"):
            pred["checklist_items"][3]["result"] = "N/A"  # was PASS

    elif quality == "blurry_rotated":
        # Combined degradation: missing 1 checklist item, 1 NC missed, date format error
        if pred.get("checklist_items"):
            pred["checklist_items"] = pred["checklist_items"][:3]
        if pred.get("non_conformances") and len(pred["non_conformances"]) > 1:
            pred["non_conformances"] = pred["non_conformances"][:1]
        pred["inspection_date"] = "12-04-2024"  # was 12/04/2024

    return pred


def main():
    for gt_file in sorted(GT_DIR.glob("*.json")):
        gt = json.loads(gt_file.read_text())
        pred = simulate_prediction(gt)
        out_file = PRED_DIR / gt_file.name
        out_file.write_text(json.dumps(pred, indent=2))
        print(f"Generated baseline prediction for {gt_file.stem}")


if __name__ == "__main__":
    main()
