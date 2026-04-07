# Quantum Policy NLP — Quadruple Helix Analysis

> Master's Thesis · Technical University of Munich (TUM)
> Python 3.10+ · Flair NER · Adobe PDF Extract API

---

## Abstract

This repository implements a four-step NLP pipeline for analysing national quantum technology strategy documents from **17 countries** (2018–2025) through the lens of the **Quadruple Helix (QH) innovation framework** (Ranga & Etzkowitz 2013). The pipeline extracts named entities from policy PDFs, applies a five-level manual classification schema to assign actors to helix categories (government, industry, academia, civil society, intermediary), and computes paragraph-level co-occurrence networks. Three research questions examine actor prominence, helix interaction patterns and Triple Helix spaces, and country-level QH system type classification.

---

## Theoretical Framework

The **Quadruple Helix** model extends the Triple Helix (Etzkowitz & Leydesdorff 1995) by adding civil society as a fourth institutional sphere:

| Helix | Description |
|-------|-------------|
| `government` | State and public-sector bodies |
| `industry` | Commercial and private-sector actors |
| `academia` | Universities and research institutes |
| `civil_society` | NGOs, citizen groups, media |
| `intermediary` | Hybrid bridge organisations (multi-sphere) |

### Five-Level Classification Schema

```
Level 1  actor_type        institutional | individual
Level 2  sphere_boundary   single-sphere | multi-sphere (hybrid)
Level 3  exact_category    e.g. "national government institutions", "universities"
Level 4  innovation_type   R&D | Non R&D | Both
Level 5  helix             government | industry | academia | civil_society | intermediary
```

---

## Pipeline Overview

```
data/raw/ (PDFs)
     │
     ▼
┌─────────────────────────────────────────────────┐
│ Step 0  step0_preprocess.py                     │
│         Adobe PDF Extract API → structured text │
└─────────────────┬───────────────────────────────┘
                  │  data/processed/step0/text/*.txt
                  ▼
┌─────────────────────────────────────────────────┐
│ Step 1  step1_ner.py                            │
│         Flair NER (ORG + PER) → JSONL mentions  │
└─────────────────┬───────────────────────────────┘
                  │  data/processed/step1/entities_ner.jsonl
                  ▼
┌─────────────────────────────────────────────────┐
│ Step 2  step2_classify.py                       │
│         Manual QH labels → classified entities  │
└─────────────────┬───────────────────────────────┘
                  │  data/processed/step2/entities_classified.jsonl
                  ▼
┌─────────────────────────────────────────────────┐
│ Step 3  step3_cooccurrence.py                   │
│         Paragraph-level helix co-occurrence     │
└─────────────────┬───────────────────────────────┘
                  │  data/processed/step3/cooccurrence.jsonl
                  ▼
┌─────────────────────────────────────────────────┐
│ Analysis  src/analysis/                         │
│   Descriptives · RQ1 · RQ2 · RQ3               │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
           outputs/  (JSON · PNG · CSV)
```

---

## Repository Structure

```
Master-Thesis-NLP/
│
├── run.py                          ← Unified CLI entry point
│
├── src/
│   ├── config.py                   ← All paths and settings (single source of truth)
│   ├── utils.py                    ← Shared I/O and helper functions
│   │
│   ├── pipeline/
│   │   ├── run_pipeline.py         ← Pipeline step runner
│   │   ├── step0_preprocess.py     ← PDF extraction via Adobe API
│   │   ├── step1_ner.py            ← Flair NER (ORG + PER)
│   │   ├── step2_classify.py       ← QH manual classification
│   │   └── step3_cooccurrence.py   ← Paragraph co-occurrence
│   │
│   └── analysis/
│       ├── run_deliverables.py     ← Analysis orchestrator
│       ├── descriptives.py         ← Corpus overview statistics
│       ├── methodology.py          ← Pipeline validation metrics
│       ├── rq1.py                  ← Actor prominence & helix balance (HBI)
│       ├── rq2.py                  ← Helix pair interactions & TH space emphasis
│       ├── rq3.py                  ← QH system type classification per country
│       ├── rq_extended.py          ← Supplementary figures
│       └── _helpers.py             ← Shared output utilities
│
├── data/
│   ├── raw/                        ← Input PDFs (place documents here)
│   └── processed/                  ← Intermediate pipeline outputs
│       ├── step0/                  ← Extracted text + structured JSON
│       ├── step1/                  ← NER mention JSONL
│       ├── step2/                  ← Classified entity JSONL + manual labels JSON
│       └── step3/                  ← Co-occurrence JSONL + paragraph alignment
│
├── outputs/                        ← Thesis-ready figures, tables, JSON
│   ├── descriptives/
│   ├── rq1/  rq2/  rq3/
│
├── configs/                        ← Page-exclusion configuration templates
├── requirements.txt
└── .env.example
```

---

## Setup

### Requirements

- **Python >= 3.10** (tested on Python 3.11)
- Adobe PDF Services credentials (Step 0 only — free tier available)
- ~8 GB RAM for Flair NER model loading (Step 1)

### Installation

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd Master-Thesis-NLP

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your Adobe credentials (required for Step 0 only)
```

---

## Running the Pipeline

Steps must be run in order. Each step produces output consumed by the next.

```bash
# Step 0: Extract text from PDFs (requires Adobe credentials in .env)
python run.py pipeline --step 0

# Step 1: Named Entity Recognition — extracts ORG and PER mentions
python run.py pipeline --step 1

# Step 2: Quadruple Helix classification — applies manual labels
# On first run, creates data/processed/step2/manual_classification_labels.json
# Fill in the labels file, then re-run to apply them.
python run.py pipeline --step 2

# Step 3: Paragraph co-occurrence — builds helix interaction network
python run.py pipeline --step 3

# Or run all steps in sequence:
python run.py pipeline --step all
```

You can also use the module runner directly:
```bash
python -m src.pipeline.run_pipeline --step [0|1|2|3|all]
```

---

## Running the Analysis

```bash
# Generate all thesis outputs (JSON + PNG figures + CSV tables)
python run.py analysis

# Or via module runner:
python -m src.analysis.run_deliverables
```

### Output per Research Question

| RQ | Output directory | Contents |
|----|-----------------|----------|
| Descriptives | `outputs/descriptives/` | Corpus overview CSV + helix distribution figure |
| Methodology | `outputs/methodology_summary.json` | Pipeline validation statistics |
| RQ1 | `outputs/rq1/` | Actor prominence & Helix Balance Index (HBI) by country |
| RQ2 | `outputs/rq2/` | Helix pair co-occurrence + TH space emphasis (Knowledge / Innovation / Consensus) |
| RQ3 | `outputs/rq3/` | QH system type profiles: Balanced / Statist / Laissez-faire / Mixed |

---

## Data Formats

### Step 1 output — `entities_ner.jsonl`

```json
{
  "doc_id": 1, "doc_name": "GER_2023_Quantum Technologies",
  "entity_id": 42, "mention_id": 137,
  "mention": "Federal Ministry of Education",
  "canonical_mention": "federal ministry of education",
  "label": "ORG",
  "start_char": 4821, "end_char": 4850
}
```

### Step 2 manual labels — `manual_classification_labels.json`

```json
[
  {
    "doc_id": 1, "entity_id": 42,
    "status": "entity",
    "level_1_actor_type": "institutional",
    "level_2_sphere_boundary": "single-sphere",
    "level_3_exact_category": "national government institutions",
    "level_4_innovation_type": "Non R&D",
    "level_5_helix": "government"
  }
]
```

### Step 3 output — `cooccurrence.jsonl`

```json
{
  "doc_id": "GER_2023_Quantum Technologies",
  "paragraph_id": 17,
  "country": "GER", "year": "2023",
  "entity_id_1": 42, "h1": "government",
  "entity_id_2": 88, "h2": "academia",
  "pair": "academia\u2013government"
}
```

---

## Configuration Reference

All settings are loaded from `.env` at startup. See `.env.example` for a full template.

| Variable | Description | Default |
|----------|-------------|---------|
| `ADOBE_CLIENT_ID` | Adobe PDF Services client ID | — |
| `ADOBE_CLIENT_SECRET` | Adobe PDF Services client secret | — |
| `PREPROCESS_EXCLUDE_PATH_TAGS` | Structural tags to skip (comma-separated) | `TOC,TOCI,FOOTNOTE` |
| `PREPROCESS_EXCLUDE_PAGES` | Global page exclusions (1-based, comma-separated) | — |
| `PREPROCESS_EXCLUDE_PAGES_BY_COUNTRY` | Country-level exclusions e.g. `AUS:1,2;FIN:3` | — |
| `PREPROCESS_EXCLUDE_PAGES_BY_DOC` | Document-level exclusions | — |
| `PREPROCESS_EXCLUDE_PAGES_BY_DOC_FILE` | Path to a line-by-line exclusion file | — |

Page numbers are always **1-based** (as printed in the document).

---

## Citation

```bibtex
@mastersthesis{vering2025quantum,
  author    = {Luis Vering},
  title     = {Decoding National Quantum Strategies:
               A Design Science Approach to Emerging Technology Governance},
  school    = {Technical University of Munich},
  year      = {2025},
  type      = {Master's Thesis},
}
```

---

## References

- Ranga, M., & Etzkowitz, H. (2013). Triple Helix systems: An analytical framework for innovation policy and practice in the Knowledge Society. *Industry and Higher Education, 27*(4), 237–262.
- Etzkowitz, H., & Leydesdorff, L. (1995). The Triple Helix — University-Industry-Government relations: A laboratory for knowledge-based economic development. *EASST Review, 14*(1), 14–19.
