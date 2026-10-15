"""
Generate 15 synthetic inspection form images + their ground-truth JSON files.

Forms are rendered programmatically using Pillow so they can be reproduced
without any external scan files. Each form simulates a realistic layout with:
- Printed headers / metadata
- Checkbox-style checklist tables
- Handwriting simulation (font variation + slight rotation per character)
- Low-quality variants (blur, noise, rotation)

Run:
    python -m forms.generate_forms
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FORMS_DIR = Path(__file__).parent
IMAGES_DIR = FORMS_DIR / "images"
GT_DIR = FORMS_DIR / "ground_truth"
IMAGES_DIR.mkdir(exist_ok=True)
GT_DIR.mkdir(exist_ok=True)

# Try to load a better font; fall back to PIL default
try:
    FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT_REG = ImageFont.truetype(FONT_PATH_REG, 14)
    FONT_BOLD = ImageFont.truetype(FONT_PATH_BOLD, 16)
    FONT_SMALL = ImageFont.truetype(FONT_PATH_REG, 11)
    FONT_TITLE = ImageFont.truetype(FONT_PATH_BOLD, 22)
    FONT_HW = ImageFont.truetype(FONT_PATH_REG, 13)   # "handwriting" simulation
except Exception:
    FONT_REG = ImageFont.load_default()
    FONT_BOLD = FONT_REG
    FONT_SMALL = FONT_REG
    FONT_TITLE = FONT_REG
    FONT_HW = FONT_REG

W, H = 850, 1100          # A4-ish at ~100dpi
BG = (255, 255, 254)      # Nearly white
FG = (10, 10, 10)
HEADER_COLOR = (30, 60, 120)
LINE_COLOR = (180, 180, 180)
CHECK_COLOR = (20, 120, 60)
FAIL_COLOR = (180, 30, 30)


# ──────────────────────────────────────────────
# Ground-truth templates
# ──────────────────────────────────────────────

FORM_TEMPLATES = [
    # ── 1. Safety walkthrough – clean scan ──────────────────────────────────
    {
        "id": "FORM-001",
        "form_type": "SAFETY_WALKTHROUGH",
        "form_id": "SW-2024-001",
        "inspection_date": "12/03/2024",
        "site_name": "Alpha Construction Site",
        "location": "Block D, Tower 3",
        "inspector_name": "James Carter",
        "inspector_id": "JC-447",
        "overall_result": "PASS",
        "checklist": [
            ("1", "PPE worn by all workers", "PASS", None),
            ("2", "Fire extinguishers accessible", "PASS", None),
            ("3", "Emergency exits unobstructed", "PASS", None),
            ("4", "Scaffolding secured", "PASS", None),
            ("5", "First aid kit stocked", "PASS", None),
        ],
        "nc": [],
        "quality": "clean",
    },
    # ── 2. Equipment inspection – clean ─────────────────────────────────────
    {
        "id": "FORM-002",
        "form_type": "EQUIPMENT_INSPECTION",
        "form_id": "EQ-2024-002",
        "inspection_date": "15/03/2024",
        "site_name": "Beta Manufacturing Plant",
        "location": "Assembly Hall B",
        "inspector_name": "Sarah Liu",
        "inspector_id": "SL-219",
        "overall_result": "FAIL",
        "checklist": [
            ("1", "Crane cable integrity", "PASS", None),
            ("2", "Hydraulic pressure nominal", "FAIL", "Pressure 10% below spec"),
            ("3", "Safety guards in place", "PASS", None),
            ("4", "Operator certification current", "PASS", None),
            ("5", "Emergency stop functional", "PASS", None),
        ],
        "nc": [("NC-01", "Hydraulic pressure low", "Replace hydraulic fluid", "Maint. Team", "20/03/2024")],
        "quality": "clean",
    },
    # ── 3. Fire safety – clean ──────────────────────────────────────────────
    {
        "id": "FORM-003",
        "form_type": "FIRE_SAFETY",
        "form_id": "FS-2024-003",
        "inspection_date": "18/03/2024",
        "site_name": "Gamma Warehouse",
        "location": "Zone 2 – Rack Storage",
        "inspector_name": "Marco Rossi",
        "inspector_id": "MR-091",
        "overall_result": "PASS",
        "checklist": [
            ("1", "Sprinkler heads clear of obstruction", "PASS", None),
            ("2", "Fire alarm tested this month", "PASS", None),
            ("3", "Exit signs illuminated", "PASS", None),
            ("4", "Extinguishers within service date", "PASS", None),
            ("5", "No flammable storage near exits", "PASS", None),
        ],
        "nc": [],
        "quality": "clean",
    },
    # ── 4. Scaffold inspection – clean ──────────────────────────────────────
    {
        "id": "FORM-004",
        "form_type": "SCAFFOLD_INSPECTION",
        "form_id": "SC-2024-004",
        "inspection_date": "20/03/2024",
        "site_name": "Delta Bridge Project",
        "location": "North Pylon, Level 4",
        "inspector_name": "Priya Sharma",
        "inspector_id": "PS-562",
        "overall_result": "FAIL",
        "checklist": [
            ("1", "Base plates fully supported", "PASS", None),
            ("2", "Cross-bracing complete", "FAIL", "Missing diagonal brace bay 3"),
            ("3", "Guardrails at all open sides", "PASS", None),
            ("4", "Planking gap <1 in", "PASS", None),
            ("5", "Tie-backs to structure", "FAIL", "2 tie-backs not fixed"),
        ],
        "nc": [("NC-01", "Missing diagonal brace", "Install brace immediately", "Site Engineer", "21/03/2024"),
               ("NC-02", "Loose tie-backs", "Re-fix tie-backs", "Scaffold Crew", "21/03/2024")],
        "quality": "clean",
    },
    # ── 5. Electrical inspection – clean ────────────────────────────────────
    {
        "id": "FORM-005",
        "form_type": "ELECTRICAL_INSPECTION",
        "form_id": "EL-2024-005",
        "inspection_date": "22/03/2024",
        "site_name": "Epsilon Data Centre",
        "location": "Server Room 1",
        "inspector_name": "Kevin Osei",
        "inspector_id": "KO-783",
        "overall_result": "PASS",
        "checklist": [
            ("1", "Grounding continuity verified", "PASS", None),
            ("2", "Panel labelling complete", "PASS", None),
            ("3", "No exposed wiring", "PASS", None),
            ("4", "GFCI outlets tested", "PASS", None),
            ("5", "Load within rated capacity", "PASS", None),
        ],
        "nc": [],
        "quality": "clean",
    },
    # ── 6. Site hazard – clean ───────────────────────────────────────────────
    {
        "id": "FORM-006",
        "form_type": "SITE_HAZARD_ASSESSMENT",
        "form_id": "HA-2024-006",
        "inspection_date": "25/03/2024",
        "site_name": "Zeta Chemical Storage",
        "location": "Tank Farm A",
        "inspector_name": "Elena Vasquez",
        "inspector_id": "EV-334",
        "overall_result": "FAIL",
        "checklist": [
            ("1", "Bund walls intact", "PASS", None),
            ("2", "Ventilation adequate", "FAIL", "Exhaust fan #3 inoperative"),
            ("3", "SDS sheets available", "PASS", None),
            ("4", "No ignition sources nearby", "PASS", None),
            ("5", "Spill kit accessible", "PASS", None),
        ],
        "nc": [("NC-01", "Exhaust fan inoperative", "Repair or replace fan", "Facility Mgr", "27/03/2024")],
        "quality": "clean",
    },
    # ── 7. Safety walkthrough – blurry ──────────────────────────────────────
    {
        "id": "FORM-007",
        "form_type": "SAFETY_WALKTHROUGH",
        "form_id": "SW-2024-007",
        "inspection_date": "27/03/2024",
        "site_name": "Eta Tunnel Project",
        "location": "Section 3 Underground",
        "inspector_name": "Tom Bradley",
        "inspector_id": "TB-112",
        "overall_result": "PASS",
        "checklist": [
            ("1", "Ventilation ducts in place", "PASS", None),
            ("2", "Lighting adequate", "PASS", None),
            ("3", "Rescue equipment at portal", "PASS", None),
            ("4", "Gas monitoring active", "PASS", None),
            ("5", "Worker communication system ok", "PASS", None),
        ],
        "nc": [],
        "quality": "blurry",
    },
    # ── 8. Equipment inspection – low-quality ───────────────────────────────
    {
        "id": "FORM-008",
        "form_type": "EQUIPMENT_INSPECTION",
        "form_id": "EQ-2024-008",
        "inspection_date": "29/03/2024",
        "site_name": "Theta Quarry",
        "location": "Crusher Plant",
        "inspector_name": "Anna Kovac",
        "inspector_id": "AK-667",
        "overall_result": "FAIL",
        "checklist": [
            ("1", "Conveyor belt alignment", "FAIL", "Belt tracking off by 5 cm"),
            ("2", "Crusher jaw wear", "PASS", None),
            ("3", "Dust suppression active", "PASS", None),
            ("4", "Safety interlock tested", "PASS", None),
            ("5", "Operator hearing protection", "PASS", None),
        ],
        "nc": [("NC-01", "Belt tracking issue", "Realign conveyor belt", "Maint. Crew", "31/03/2024")],
        "quality": "lowres",
    },
    # ── 9. Fire safety – rotated ─────────────────────────────────────────────
    {
        "id": "FORM-009",
        "form_type": "FIRE_SAFETY",
        "form_id": "FS-2024-009",
        "inspection_date": "01/04/2024",
        "site_name": "Iota Office Tower",
        "location": "Floor 12",
        "inspector_name": "David Park",
        "inspector_id": "DP-229",
        "overall_result": "PASS",
        "checklist": [
            ("1", "Stairwell clear of combustibles", "PASS", None),
            ("2", "Fire doors self-closing", "PASS", None),
            ("3", "Suppression system charged", "PASS", None),
            ("4", "Evacuation plan posted", "PASS", None),
            ("5", "Assembly point marked", "PASS", None),
        ],
        "nc": [],
        "quality": "rotated",
    },
    # ── 10. Electrical – handwritten fields ─────────────────────────────────
    {
        "id": "FORM-010",
        "form_type": "ELECTRICAL_INSPECTION",
        "form_id": "EL-2024-010",
        "inspection_date": "03/04/2024",
        "site_name": "Kappa Solar Farm",
        "location": "Inverter Block 2",
        "inspector_name": "Fatima Al-Rashid",
        "inspector_id": "FA-881",
        "overall_result": "PASS",
        "checklist": [
            ("1", "PV string voltage in range", "PASS", None),
            ("2", "Inverter cooling clear", "PASS", None),
            ("3", "Earthing pits resistance <10Ω", "PASS", None),
            ("4", "DC disconnect labels correct", "PASS", None),
            ("5", "String fuse ratings correct", "PASS", None),
        ],
        "nc": [],
        "quality": "handwritten",
    },
    # ── 11. Scaffold – noisy scan ───────────────────────────────────────────
    {
        "id": "FORM-011",
        "form_type": "SCAFFOLD_INSPECTION",
        "form_id": "SC-2024-011",
        "inspection_date": "05/04/2024",
        "site_name": "Apex Stadium Renovation",
        "location": "South Stand Upper Tier",
        "inspector_name": "Chen Wei",
        "inspector_id": "CW-441",
        "overall_result": "FAIL",
        "checklist": [
            ("1", "Ledger connections tight", "PASS", None),
            ("2", "Trapdoor hatch secured", "FAIL", "Hatch pin missing"),
            ("3", "Load rating sign visible", "PASS", None),
            ("4", "Sole boards present", "PASS", None),
            ("5", "Scaffold tag current", "PASS", None),
        ],
        "nc": [("NC-01", "Missing hatch pin", "Fit replacement pin", "Scaffold Crew", "06/04/2024")],
        "quality": "noisy",
    },
    # ── 12. Site hazard – dark scan ─────────────────────────────────────────
    {
        "id": "FORM-012",
        "form_type": "SITE_HAZARD_ASSESSMENT",
        "form_id": "HA-2024-012",
        "inspection_date": "08/04/2024",
        "site_name": "Mu Oil Depot",
        "location": "Loading Bay 4",
        "inspector_name": "Raj Patel",
        "inspector_id": "RP-554",
        "overall_result": "FAIL",
        "checklist": [
            ("1", "Grounding cables on tankers", "FAIL", "No grounding visible"),
            ("2", "No smoking signs posted", "PASS", None),
            ("3", "Drip trays in place", "PASS", None),
            ("4", "Emergency shutoff reachable", "PASS", None),
            ("5", "Fire watch assigned", "PASS", None),
        ],
        "nc": [("NC-01", "Missing tanker grounding", "Ground tanker before transfer", "HSE Officer", "08/04/2024")],
        "quality": "dark",
    },
    # ── 13. Safety walkthrough – multiple inspectors ─────────────────────────
    {
        "id": "FORM-013",
        "form_type": "SAFETY_WALKTHROUGH",
        "form_id": "SW-2024-013",
        "inspection_date": "10/04/2024",
        "site_name": "Nu Shipyard",
        "location": "Dry Dock 2",
        "inspector_name": "Grace Mensah",
        "inspector_id": "GM-772",
        "overall_result": "PASS",
        "checklist": [
            ("1", "Life vests accessible", "PASS", None),
            ("2", "Hot-work permit in place", "PASS", None),
            ("3", "Confined space monitor active", "PASS", None),
            ("4", "Crane exclusion zone marked", "PASS", None),
            ("5", "Toolbox talk completed", "PASS", None),
        ],
        "nc": [],
        "quality": "clean",
    },
    # ── 14. Fire safety – blurry & rotated ──────────────────────────────────
    {
        "id": "FORM-014",
        "form_type": "FIRE_SAFETY",
        "form_id": "FS-2024-014",
        "inspection_date": "12/04/2024",
        "site_name": "Xi Textile Mill",
        "location": "Weaving Section",
        "inspector_name": "Oluwaseun Adeyemi",
        "inspector_id": "OA-133",
        "overall_result": "FAIL",
        "checklist": [
            ("1", "Lint traps cleaned weekly", "FAIL", "Last cleaned 3 weeks ago"),
            ("2", "Extinguisher type correct (CO2)", "PASS", None),
            ("3", "Sprinkler heads unobstructed", "PASS", None),
            ("4", "Electrical panel locked", "PASS", None),
            ("5", "Emergency lighting tested", "FAIL", "3 units failed"),
        ],
        "nc": [("NC-01", "Lint accumulation", "Clean lint traps immediately", "Production Mgr", "13/04/2024"),
               ("NC-02", "Emergency lighting failure", "Replace 3 lamp units", "Electrician", "14/04/2024")],
        "quality": "blurry_rotated",
    },
    # ── 15. Equipment inspection – overexposed ───────────────────────────────
    {
        "id": "FORM-015",
        "form_type": "EQUIPMENT_INSPECTION",
        "form_id": "EQ-2024-015",
        "inspection_date": "15/04/2024",
        "site_name": "Omicron Wind Farm",
        "location": "Turbine T-07",
        "inspector_name": "Ingrid Johansson",
        "inspector_id": "IJ-908",
        "overall_result": "PASS",
        "checklist": [
            ("1", "Blade surface condition", "PASS", None),
            ("2", "Nacelle vibration within limits", "PASS", None),
            ("3", "Pitch control responsive", "PASS", None),
            ("4", "Tower bolt torque verified", "PASS", None),
            ("5", "Lightning protection bonded", "PASS", None),
        ],
        "nc": [],
        "quality": "overexposed",
    },
]


# ──────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────

def _draw_header(draw: ImageDraw.ImageDraw, form: dict) -> int:
    """Draw form header; return the y cursor after the header."""
    draw.rectangle([(0, 0), (W, 80)], fill=HEADER_COLOR)
    draw.text((20, 15), "INDUSTRIAL SAFETY INSPECTION REPORT", font=FONT_TITLE, fill=(255, 255, 255))
    y = 90
    draw.text((20, y), f"Form ID: {form['form_id']}", font=FONT_BOLD, fill=FG)
    draw.text((300, y), f"Type: {form['form_type'].replace('_', ' ')}", font=FONT_BOLD, fill=FG)
    draw.text((600, y), f"Date: {form['inspection_date']}", font=FONT_BOLD, fill=FG)
    y += 25
    draw.line([(10, y), (W - 10, y)], fill=LINE_COLOR)
    return y + 10


def _draw_site_inspector(draw: ImageDraw.ImageDraw, form: dict, y: int) -> int:
    draw.text((20, y), "SITE INFORMATION", font=FONT_BOLD, fill=HEADER_COLOR)
    y += 20
    draw.text((20, y), f"Site: {form['site_name']}", font=FONT_REG, fill=FG)
    draw.text((400, y), f"Location: {form['location']}", font=FONT_REG, fill=FG)
    y += 20
    draw.text((20, y), "INSPECTOR", font=FONT_BOLD, fill=HEADER_COLOR)
    y += 20
    draw.text((20, y), f"Name: {form['inspector_name']}", font=FONT_REG, fill=FG)
    draw.text((400, y), f"ID: {form['inspector_id']}", font=FONT_REG, fill=FG)
    y += 20
    draw.line([(10, y), (W - 10, y)], fill=LINE_COLOR)
    return y + 10


def _draw_checklist(draw: ImageDraw.ImageDraw, items: list, y: int) -> int:
    draw.text((20, y), "INSPECTION CHECKLIST", font=FONT_BOLD, fill=HEADER_COLOR)
    y += 22
    # Header row
    draw.rectangle([(10, y), (W - 10, y + 20)], fill=(220, 228, 240))
    draw.text((15, y + 3), "#", font=FONT_SMALL, fill=FG)
    draw.text((50, y + 3), "Description", font=FONT_SMALL, fill=FG)
    draw.text((550, y + 3), "Result", font=FONT_SMALL, fill=FG)
    draw.text((650, y + 3), "Remarks", font=FONT_SMALL, fill=FG)
    y += 22

    for num, desc, result, remark in items:
        bg = (245, 245, 245) if int(num) % 2 == 0 else BG
        draw.rectangle([(10, y), (W - 10, y + 22)], fill=bg)
        draw.text((15, y + 4), num, font=FONT_SMALL, fill=FG)
        draw.text((50, y + 4), desc[:50], font=FONT_SMALL, fill=FG)

        r_color = CHECK_COLOR if result == "PASS" else (FAIL_COLOR if result == "FAIL" else FG)
        # Draw checkbox
        draw.rectangle([(548, y + 4), (564, y + 18)], outline=FG)
        if result in ("PASS", "FAIL"):
            mark = "✓" if result == "PASS" else "✗"
            draw.text((550, y + 3), mark, font=FONT_SMALL, fill=r_color)
        draw.text((570, y + 4), result, font=FONT_SMALL, fill=r_color)
        if remark:
            draw.text((650, y + 4), remark[:25], font=FONT_SMALL, fill=FG)
        y += 24
    return y + 10


def _draw_nc(draw: ImageDraw.ImageDraw, ncs: list, y: int) -> int:
    if not ncs:
        return y
    draw.text((20, y), "NON-CONFORMANCES", font=FONT_BOLD, fill=FAIL_COLOR)
    y += 20
    for nc_id, desc, action, responsible, due in ncs:
        draw.text((25, y), f"[{nc_id}] {desc}", font=FONT_REG, fill=FG)
        y += 18
        draw.text((40, y), f"Action: {action}  |  Responsible: {responsible}  |  Due: {due}", font=FONT_SMALL, fill=FG)
        y += 20
    return y + 5


def _draw_overall(draw: ImageDraw.ImageDraw, result: str, y: int) -> int:
    color = CHECK_COLOR if result == "PASS" else FAIL_COLOR
    draw.text((20, y), f"OVERALL RESULT: {result}", font=FONT_BOLD, fill=color)
    y += 25
    # Signature line
    draw.text((20, y), "Inspector Signature: _____________________", font=FONT_REG, fill=FG)
    return y + 30


def _apply_quality(img: Image.Image, quality: str) -> Image.Image:
    """Degrade image quality to simulate real-world scan conditions."""
    rng = random.Random(42)

    if quality == "blurry":
        img = img.filter(ImageFilter.GaussianBlur(radius=2.5))

    elif quality == "lowres":
        small = img.resize((img.width // 3, img.height // 3), Image.BILINEAR)
        img = small.resize((img.width, img.height), Image.NEAREST)

    elif quality == "rotated":
        img = img.rotate(3.5, fillcolor=BG, expand=False)

    elif quality == "handwritten":
        # Simulate handwritten annotations by adding slightly irregular text
        import numpy as np
        arr = np.array(img, dtype=np.int16)
        noise = np.random.RandomState(7).randint(-15, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif quality == "noisy":
        import numpy as np
        arr = np.array(img, dtype=np.int16)
        noise = np.random.RandomState(13).randint(-40, 40, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif quality == "dark":
        from PIL import ImageEnhance
        img = ImageEnhance.Brightness(img).enhance(0.45)

    elif quality == "overexposed":
        from PIL import ImageEnhance
        img = ImageEnhance.Brightness(img).enhance(1.7)

    elif quality == "blurry_rotated":
        img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
        img = img.rotate(2.0, fillcolor=BG, expand=False)

    return img


# ──────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────

def generate_form(form: dict) -> Image.Image:
    img = Image.new("RGB", (W, H), color=BG)
    draw = ImageDraw.Draw(img)

    y = _draw_header(draw, form)
    y = _draw_site_inspector(draw, form, y)
    y = _draw_checklist(draw, form["checklist"], y)
    y = _draw_nc(draw, form["nc"], y)
    _draw_overall(draw, form["overall_result"], y)

    return _apply_quality(img, form["quality"])


def form_to_ground_truth(form: dict) -> dict:
    """Build the expected JSON that the pipeline should extract."""
    nc_list = [
        {
            "nc_id": nc[0],
            "description": nc[1],
            "corrective_action": nc[2],
            "responsible_party": nc[3],
            "due_date": nc[4],
            "closed": False,
        }
        for nc in form["nc"]
    ]
    checklist = [
        {
            "item_number": row[0],
            "description": row[1],
            "result": row[2],
            "remarks": row[3],
            "risk_level": None,
        }
        for row in form["checklist"]
    ]
    return {
        "form_type": form["form_type"],
        "form_id": form["form_id"],
        "inspection_date": form["inspection_date"],
        "inspector": {
            "name": form["inspector_name"],
            "id_number": form["inspector_id"],
            "company": None,
            "signature_present": False,
        },
        "site": {
            "site_name": form["site_name"],
            "location": form["location"],
            "department": None,
            "permit_number": None,
        },
        "overall_result": form["overall_result"],
        "checklist_items": checklist,
        "non_conformances": nc_list,
        "equipment_items": [],
        "quality": form["quality"],        # extra meta for eval
    }


def main():
    generated = []
    for tmpl in FORM_TEMPLATES:
        form_id = tmpl["id"]
        img = generate_form(tmpl)
        img_path = IMAGES_DIR / f"{form_id}.png"
        img.save(img_path, format="PNG")

        gt = form_to_ground_truth(tmpl)
        gt_path = GT_DIR / f"{form_id}.json"
        gt_path.write_text(json.dumps(gt, indent=2))

        generated.append(form_id)
        print(f"✓ {form_id}: {tmpl['form_type']} [{tmpl['quality']}]")

    print(f"\nGenerated {len(generated)} forms → {IMAGES_DIR}")


if __name__ == "__main__":
    main()
