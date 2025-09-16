#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ~1000 relevant prospects (companies) for IIC Ingredients & Packaging,
with a focus on buyers/users of pea protein (food & pet food manufacturers,
functional foods, dairy alternatives, snacks/bakery) and packaging-relevant
leads that could also buy ingredients from IIC AG / The Ingredients Experts.

- Uses OpenAI Responses API (Python SDK) to expand seed segments into batches.
- De-duplicates, validates basic relevance, prioritizes EU/DACH but allows global.
- Saves to CSV with columns:
  Company, Segment, Country/Region, WhyRelevant, Website, Priority

Before running, set your API key:
  export OPENAI_API_KEY="sk-..."
"""

import os
import csv
import time
import json
import re
from typing import List, Dict, Set

# ---- OpenAI client (Responses API) ----
# Docs: https://platform.openai.com/docs/quickstart?api-mode=responses&language=python
try:
    from openai import OpenAI  # new SDK (>=1.x)
except Exception:
    # Fallback for older installs:
    from openai import OpenAI  # type: ignore

client = OpenAI()

# -------------- CONFIG -----------------
OUTPUT_CSV = "prospects_iic_pea_protein.csv"
TARGET_COUNT = 1000
BATCH_SIZE = 40           # companies requested per API call
MAX_CALLS = 40            # safety cap (40*40=1600 max potential)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SYSTEM_ROLE = (
    "Du bist ein B2B-Research- und Lead-Generation-Analyst für IIC International AG "
    "(IIC Packaging, The Ingredients Experts). Deine Aufgabe: Finde reale, existierende "
    "Unternehmen (B2B) in Europa (Priorität DACH/EU), Großbritannien und global, die "
    "mit hoher Wahrscheinlichkeit Erbsenprotein (Pea Protein) als Zutat einsetzen "
    "oder relevante Kunden für IICs Zutaten- und Verpackungsportfolio sind. "
    "Gib NUR echte Firmennamen aus (keine Fantasie-/Shop-Namen), vermeide Dubletten."
)

# Seed segments from the user's example + relevant additions
SEED_SEGMENTS = [
    # 1) Meat alternatives
    ("Hersteller pflanzlicher Fleischalternativen",
     "Burger, Nuggets, Würstchen, Hack, Schnitzel; etablierte Marken & Private Label"),
    # 2) Protein powders / functional foods
    ("Protein- & Functional-Food-Hersteller",
     "Sportnahrung, Shakes, Riegel, Complete Meals, RTD-Proteindrinks"),
    # 3) Dairy alternatives
    ("Hersteller von Molkereialternativen",
     "Milchalternativen, Joghurts, Käsealternativen, Eis"),
    # 4) Snacks & Bakery
    ("Hersteller von Snacks und Backwaren",
     "Chips, Kekse, Riegel, Müsli/Cereals, Backwaren mit Protein"),
    # 5) Pet food
    ("Tiernahrungshersteller (Pet Food)",
     "Hunde-/Katzenfutter, Premium & Spezialfutter"),
    # 6) Contract manufacturers / Co-Packers
    ("Lohnhersteller & Co-Packer Food/Drinks",
     "Produzieren im Auftrag – gut für schnelle Skalierung"),
    # 7) Ready Meals / Convenience
    ("Hersteller von Fertiggerichten & Convenience",
     "Bowls, Suppen, Saucen, Tiefkühlkost"),
    # 8) B2B Ingredient Distributors
    ("B2B-Zutatenhändler / Distributoren",
     "Multiplikatoren für Markteintritt; EU-weit und global"),
    # 9) Health/Medical Nutrition
    ("Hersteller medizinischer / seniorengerechter Ernährung",
     "Klinik-/Seniorenernährung, proteinangereichert"),
    # 10) Packaging-relevant food brands (cross-sell)
    ("Marken mit Verpackungsbedarf (Cross-Sell Packaging)",
     "Lebensmittel-/Getränkemarken, die auch Zutaten einsetzen könnten"),
]

# Basic validation/normalization helpers
def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())

def looks_like_company(name: str) -> bool:
    # Filter out clearly non-company items
    bad_patterns = [
        r"^siehe", r"^n/a$", r"^keine", r"^unbekannt", r"^sample", r"http", r"\.com\b", r"\.de\b"
    ]
    for bp in bad_patterns:
        if re.search(bp, name, flags=re.I):
            return False
    # avoid entirely generic non-entities
    if len(name.split()) == 1 and name.isalpha() and len(name) <= 3:
        return False
    return True

def dedupe_and_filter(rows: List[Dict[str, str]], seen: Set[str]) -> List[Dict[str, str]]:
    clean = []
    for r in rows:
        key = normalize_name(r.get("Company",""))
        if not key or key in seen:
            continue
        if looks_like_company(r.get("Company","")):
            seen.add(key)
            clean.append(r)
    return clean

def build_prompt(segment_title: str, segment_desc: str, avoid: List[str]) -> str:
    avoid_text = "; ".join(sorted(set(avoid)))[:6000]  # cap token bloat
    return f"""
Kontext:
- Auftraggeber: IIC International AG (IIC Packaging, The Ingredients Experts)
- Fokus: potenzielle B2B-Kunden für Erbsenprotein (und weitere pflanzliche Zutaten),
  zusätzlich Verpackungs-Cross-Sell möglich.
- Priorität: Europa/DACH, dann UK/Global. Nur reale Unternehmen, keine Dubletten.

Segment: {segment_title}
Beschreibung: {segment_desc}

AUFGABE:
Nenne mir {BATCH_SIZE} neue, reale Unternehmen in diesem Segment, die als Käufer oder Anwender
von Erbsenprotein in Frage kommen. Liefere **CSV ohne Kopfzeile**, mit genau diesen Spalten:
Company,Segment,Country/Region,WhyRelevant,Website,Priority

Definitionen:
- Priority: 'A' (sehr passend, hohe Wahrscheinlichkeit), 'B' (passend), 'C' (möglich).
- WhyRelevant: 1 kurze Begründung (Produktkategorie, Proteinbezug, EU-Präsenz etc.).
- Country/Region: möglichst EU/DACH/UK; sonst Land angeben.
- Website: wenn bekannt, sonst leer lassen.
- KEINE Dubletten. KEINE Erklärtexte. Nur CSV-Zeilen.

Vermeide diese Unternehmen (bereits gesammelt):
{avoid_text}
""".strip()

def parse_csv_block(text: str, expected_cols=6) -> List[Dict[str, str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        # naive CSV split (commas). If commas in fields are likely, a smarter parser could be used.
        parts = [p.strip() for p in re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", line)]
        if len(parts) < expected_cols:
            # try semicolon fallback
            parts = [p.strip() for p in re.split(r";(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", line)]
        if len(parts) >= expected_cols:
            rows.append({
                "Company": parts[0].strip('"'),
                "Segment": parts[1].strip('"') if len(parts) > 1 else "",
                "Country/Region": parts[2].strip('"') if len(parts) > 2 else "",
                "WhyRelevant": parts[3].strip('"') if len(parts) > 3 else "",
                "Website": parts[4].strip('"') if len(parts) > 4 else "",
                "Priority": parts[5].strip('"') if len(parts) > 5 else "",
            })
    return rows

def generate_batch(segment_title: str, segment_desc: str, avoid_list: List[str]) -> List[Dict[str, str]]:
    prompt = build_prompt(segment_title, segment_desc, avoid_list)
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_ROLE},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    # Extract text (Responses API returns output_text convenience or choose first item)
    try:
        text = resp.output_text
    except Exception:
        # fallback
        text = ""
        try:
            if resp.output and len(resp.output) and hasattr(resp.output[0], "content"):
                text = resp.output[0].content[0].text
        except Exception:
            pass
    if not text:
        return []
    rows = parse_csv_block(text)
    return rows

def main():
    print("== Generating prospects for IIC Ingredients & Packaging ==")
    seen: Set[str] = set()
    results: List[Dict[str, str]] = []

    # a rolling "avoid" list (recent N) to control token growth
    AVOID_ROLLING_MAX = 300

    call_count = 0
    seg_index = 0
    while len(results) < TARGET_COUNT and call_count < MAX_CALLS:
        seg_title, seg_desc = SEED_SEGMENTS[seg_index % len(SEED_SEGMENTS)]
        seg_index += 1

        avoid_list = [r["Company"] for r in results[-AVOID_ROLLING_MAX:]]
        batch = generate_batch(seg_title, seg_desc, avoid_list)
        call_count += 1

        if not batch:
            print(f"[WARN] Empty batch on call {call_count}, segment {seg_title}. Retrying after short pause.")
            time.sleep(2.0)
            continue

        batch_clean = dedupe_and_filter(batch, seen)
        if not batch_clean:
            print(f"[INFO] No new unique companies found in batch {call_count}.")
            continue

        results.extend(batch_clean)
        print(f"[OK] {len(batch_clean)} added (total {len(results)}). Segment: {seg_title}")
        time.sleep(0.8)  # gentle pacing

    # Trim to exact target if we overshoot
    results = results[:TARGET_COUNT]

    # Save CSV
    fieldnames = ["Company", "Segment", "Country/Region", "WhyRelevant", "Website", "Priority"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} prospects to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
