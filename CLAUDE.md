# CLAUDE.md

## Project Overview

Master's thesis NLP pipeline for Quadruple Helix analysis of national quantum strategy documents. The pipeline extracts named entities from 23 policy PDFs across 17 countries, manually classifies actors into helix categories, and produces thesis-ready analysis outputs (JSON + figures + CSVs).

## Research Questions

**Overarching RQ:** To what extent do national quantum strategies reflect a Quadruple Helix model of innovation, as demonstrated by the positioning of key innovation actors (academia, industry, government, and civil society)?

| | Question | Code |
|---|---|---|
| **RQ1** | How prominently are key innovation actors represented in national quantum strategies? | `src/analysis/rq1.py` |
| **RQ2** | How are interactions (and Triple Helix spaces) between innovation actors emphasized in national quantum strategies? | `src/analysis/rq2.py` |
| **RQ3** | What Quadruple Helix system types characterize national quantum strategies? | `src/analysis/rq3.py` |

## Theoretical Framework — Triple/Quadruple Helix (Ranga & Etzkowitz 2013)

**Source:** Ranga, M. & Etzkowitz, H. (2013). "Triple Helix systems: an analytical framework for innovation policy and practice in the Knowledge Society." *Industry & Higher Education*, 27(3), 237–262.

The thesis extends the **Triple Helix** (university–industry–government) to a **Quadruple Helix** by adding **civil society** as the fourth sphere, applying it to national quantum strategy documents.

### Triple Helix Systems: Three Core Elements

**1. Components** — the institutional spheres, with three key actor distinctions:

| Distinction | Types | Mapping to `level_*` fields |
|---|---|---|
| Actor type | Individual / Institutional | `level_1_actor_type` |
| Innovation activity | R&D / Non-R&D | `level_4_innovation_type` |
| Sphere boundary | Single-sphere / Multi-sphere (hybrid) | `level_2_sphere_boundary` |

- **Individual innovators**: scientists, entrepreneurs, policy makers, students, venture capitalists — tracked via `PER` NER label
- **Institutional innovators**: organizations, agencies, firms, universities — tracked via `ORG` NER label
- **Single-sphere institutions**: rigid boundaries, operate within one sphere only
- **Multi-sphere (hybrid) institutions**: operate at intersections (e.g. tech transfer offices, science parks, incubators, public–private partnerships)
- **R&D innovators**: academic research groups, company R&D divisions, public research labs
- **Non-R&D innovators**: design, production, marketing, training, incubation, financing units

**2. Relationships between components** (five types):

| Relationship | Description |
|---|---|
| Technology transfer | Market/non-market knowledge exchange; core activity of the system |
| Collaboration & conflict moderation | Triadic entities moderate tension better than dyads; task conflict (constructive) vs. relationship conflict (destructive) |
| Collaborative leadership | "Innovation organizers" bridge spheres, build consensus, coordinate top-down and bottom-up |
| Substitution | A stronger sphere fills gaps when another is weak (e.g. government providing venture capital; universities doing firm formation) |
| Networking | Formal/informal networks at national, regional, international level |

**3. Functions — the Triple Helix Spaces:**

| Space | Purpose |
|---|---|
| Knowledge space | Aggregation of R&D and non-R&D knowledge resources; avoids duplication; strengthens knowledge base |
| Innovation space | Hybrid organizations + entrepreneurial individuals; develops local firms, attracts talent, creates IP |
| Consensus space | "Blue-sky" thinking, stakeholder dialogue, governance; brings all spheres together to build shared agendas |

Spaces interact non-linearly and diachronically. A missing space can be catalysed by the others. Regional development moves through four stages: **Genesis → Implementation → Consolidation & adjustment → Self-sustaining growth and renewal**.

### Three Helix Configurations (Etzkowitz & Leydesdorff 2000)

| Configuration | Description | Innovation implication |
|---|---|---|
| Statist | Government dominates, drives academia and industry | Limits innovative capacity |
| Laissez-faire | Industry leads; university and government are ancillary | Limited cross-sphere synergy |
| Balanced | University, industry, and government partner as equals; hybrid organizations emerge | Most favorable for innovation; targeted by this thesis |

The **balanced configuration** is the ideal and is used as the normative benchmark when interpreting actor distributions and co-occurrence patterns across strategies.

### How the Framework Maps to the Pipeline

| Helix concept | Pipeline step | Field / output |
|---|---|---|
| Actor identification (ORG/PER) | Step 1 NER | `entities_ner.jsonl` |
| Actor type & sphere classification | Step 2 manual labels | `level_1` – `level_5` fields |
| Actor co-occurrence (interactions) | Step 3 | `cooccurrence.jsonl` |
| Actor prominence per country | RQ1 | `rq1_actor_prominence_and_balance.json` |
| Helix pair interaction + TH space emphasis | RQ2 | `rq2_relation_by_helix_pair.png` etc. |
| QH system type per country (configuration classification) | RQ3 | `rq3_profiles.json` |

**RQ2 — TH Space mapping (static helix-pair → space):**

| Space | Helix pairs |
|---|---|
| Knowledge space | academia–academia, academia–government |
| Innovation space | academia–industry, academia–intermediary, industry–intermediary |
| Consensus space | government–industry, government–intermediary |
| Civil society | any pair involving civil_society |

**RQ3 — Configuration classification rules (HBI-based):**

| Configuration | Rule |
|---|---|
| Balanced | HBI ≥ 0.80 |
| Statist | HBI < 0.80 AND government share > 50% |
| Laissez-faire | HBI < 0.80 AND government ≤ 50% AND industry leads non-gov helices |
| Mixed | HBI < 0.80 AND government ≤ 50% AND academia or civil_society leads |

### Civil Society as the Fourth Helix

The thesis extends Triple Helix → **Quadruple Helix** by treating `civil_society` as a full helix alongside `government`, `industry`, and `academia`. The `intermediary` category captures hybrid actors that span multiple spheres and cannot be assigned to a single helix.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in Adobe credentials

# Run pipeline steps
python run.py pipeline --step 0   # PDF extraction (requires Adobe credentials)
python run.py pipeline --step 1   # NER (Flair ner-english-large)
python run.py pipeline --step 2   # Quadruple Helix classification
python run.py pipeline --step 3   # Paragraph co-occurrence extraction
python run.py pipeline --step all # All steps in sequence

# Generate analysis outputs (RQ1–RQ4 + descriptives + methodology)
python run.py analysis

# Run everything end-to-end
python run.py all
```

## Architecture

```
run.py                          # Unified entry point
src/
  config.py                     # All paths and env-var config (single source of truth)
  utils.py                      # Shared I/O helpers
  pipeline/
    step0_preprocess.py         # Adobe PDF Services → structured text
    step1_ner.py                # Flair NER → entities_ner.jsonl
    step2_classify.py           # Manual labels → entities_classified.jsonl
    step3_cooccurrence.py       # Paragraph co-occurrence → relations + NLI scoring
    nli_relation_extraction.py  # NLI scorer (roberta-large-mnli, fp16, batch inference)
  analysis/
    _helpers.py                 # Shared utilities: thesis_style(), write_json(), utc_now_iso()
    descriptives.py             # Corpus overview stats and figures
    methodology.py              # Methodology summary JSON
    rq1.py                      # RQ1: actor prominence, helix shares, HBI
    rq2.py                      # RQ2: helix pair relations + TH space emphasis (static mapping)
    rq3.py                      # RQ3: QH system type classification per country
    rq_extended.py              # Supplementary figures (country trajectories)
    spaces.py                   # Supplementary: NLI-predicted TH space distribution
    run_deliverables.py         # Runs all analyses in sequence
data/
  raw/                          # Input PDFs (gitignored)
  processed/
    step0/{text,json,structured}/  # Extracted text and structured JSON
    step1/entities_ner.jsonl
    step2/entities_classified.jsonl
    step2/manual_classification_labels.json  # The human annotation file
    step3/                      # Relations, co-occurrence, paragraphs
outputs/
  rq1/                          # RQ1 figures, CSV, JSON
  rq2/                          # RQ2 figures, CSV
  rq3/                          # RQ3 figures, CSV, JSON
  spaces/                       # NLI space figures (supplementary)
  descriptives/                 # Corpus overview figures
configs/                        # Page exclusion templates
```

## Configuration (.env)

- `ADOBE_CLIENT_ID` / `ADOBE_CLIENT_SECRET` — required for Step 0
- `PREPROCESS_EXCLUDE_PATH_TAGS` — PDF path tags to strip (default: `TOC,TOCI,FOOTNOTE`)
- `PREPROCESS_EXCLUDE_PAGES` — global page exclusions (1-based, comma-separated)
- `PREPROCESS_EXCLUDE_PAGES_BY_COUNTRY` — country-level exclusions, format: `AUS:1,2;FIN:3`
- `PREPROCESS_EXCLUDE_PAGES_BY_DOC` — document-level exclusions (exact doc stem)
- `PREPROCESS_EXCLUDE_PAGES_BY_DOC_FILE` — path to line-wise doc exclusion file
- `NLI_MODEL_NAME` — HuggingFace model for NLI (default: `roberta-large-mnli`)
- `NLI_THRESHOLD` — minimum entailment confidence to assign a relation (default: `0.5`)
- `NLI_BATCH_SIZE` — NLI batch size (default: `128` on GPU, reduce if OOM)

All page numbers are **1-based** in config; the code converts to 0-based internally.

## Quadruple Helix Classification Schema (Step 2)

Step 2 requires a human-filled `data/processed/step2/manual_classification_labels.json`. On first run it creates a seed template — fill it in and re-run.

| Level | Field | Values |
|-------|-------|--------|
| 1 | `level_1_actor_type` | `institutional` \| `individual` |
| 2 | `level_2_sphere_boundary` | `single-sphere` \| `multi-sphere` |
| 3 | `level_3_exact_category` | fine-grained category (e.g. "national government institutions") |
| 4 | `level_4_innovation_type` | `R&D` \| `Non R&D` \| `Both` |
| 5 | `level_5_helix` | `government` \| `industry` \| `academia` \| `civil_society` \| `intermediary` |

Non-entity statuses (filtered from analysis): `historical_individual`, `ner_error`, `not_actor`, `not_specific`, `non_entity_phrase`, `unclear`.

## Document Corpus

File naming: `<COUNTRY_CODE>_<YEAR>_<Title>.pdf`

23 documents across 16 countries. Country codes: AUS, CAN, CZE, DNK, ESP, FIN, FRA, GBR, GER, IRL, ITA, JPN, KOR, NLD, SVN, USA

## NLI Relation & Space Extraction (Step 3)

Step 3 uses zero-shot NLI (`roberta-large-mnli`, fp16 on CUDA) to classify:
- **Relation type** between each co-occurring helix pair (5 types × 4 templates × 2 directions = 40 calls/pair)
- **TH space** for each pair (4 spaces × 4 templates × 2 directions = 32 calls/pair)

NLI premise uses a **±1 sentence window** (sentence containing the pair + 1 before + 1 after) for richer context.

**Important:** `th_space` (NLI-predicted) in `cooccurrence.jsonl` is **supplementary only**. RQ2 and RQ3 use the **static helix-pair → space mapping** (`TH_SPACE_MAP` in `rq2.py`), which is theoretically grounded. NLI space predictions show strong `knowledge_space` bias (model artifact) and should be interpreted cautiously. A `th_space_confidence ≥ 0.5` filter is applied in `spaces.py`.

For GPU acceleration on RunPod: `NLI_BATCH_SIZE=128 python run.py pipeline --step 3`

## NLI Evaluation (evaluation/)

Manual F1 evaluation of NLI relation predictions against gold-standard annotations.

**Files:**
- `evaluation/annotation.json` — 100 cross-helix pairs to annotate (`true_relation` field)
- `evaluation/codebook.md` — annotation codebook grounded in R&E 2013, mirrors NLI templates
- `evaluation/sample.py` — generates annotation.json (100 entries, cross-helix only)
- `evaluation/evaluate.py` — computes precision/recall/F1 per class + confusion matrix

**Sampling strategy:** For each of the 6 relation classes, selects pairs ranked by highest per-class NLI score from `all_scores` (not threshold-based), ensuring all classes are represented. Only cross-helix pairs (h1 ≠ h2) are included — same-helix pairs are not relevant to the RQs.

**Run:**
```bash
python evaluation/sample.py     # generate annotation.json
# fill in true_relation for each entry using evaluation/codebook.md
python evaluation/evaluate.py   # compute F1
```

## SetFit Space Classification (src/pipeline/spaces_setfit.py)

Few-shot fine-tuned classifier for TH space (Tunstall et al. 2022). Replaces NLI space predictions with a trained sentence-transformer model.

**Sampling strategy for annotation (`--sample`):**
- Only cross-helix sentences (h1 ≠ h2)
- Stratified by **helix pair** (not NLI predictions) using the theoretically grounded TH_SPACE_MAP:
  - `knowledge_space` → academia–government pairs
  - `innovation_space` → academia–industry, academia–intermediary, industry–intermediary pairs
  - `consensus_space` → government–industry, government–intermediary pairs
  - `public_space` → any civil_society pair + keyword-matched sentences (`public`, `ethics`, `society`, `citizen`, `trust`, `equity`, etc.)
- Entities shown are only those that appear literally in the sentence text (from cross-helix pairs only)
- Target: 25 per space (100 total); `public_space` is rare in corpus (~88 civil_society entities total) so keyword augmentation is needed to reach ~15 genuine examples

**Run:**
```bash
python -m src.pipeline.spaces_setfit --sample   # generate spaces_annotation.json
# fill in space field using evaluation/codebook_spaces.md
python -m src.pipeline.spaces_setfit --train    # train SetFit model
python -m src.pipeline.spaces_setfit --predict  # write cooccurrence_setfit.jsonl
```

**Output:** `data/processed/step3/cooccurrence_setfit.jsonl` — adds `th_space_setfit` field. `spaces.py` automatically prefers this file over NLI predictions when it exists.

## Key Notes

- `data/raw/` is gitignored (PDFs); `data/processed/` is tracked in git (~67 MB total)
- Python 3.10+ required (tested on 3.11)
- Step 1 uses `flair/ner-english-large` — GPU recommended; uses `FLAIR_NER_BATCH_SIZE=32`
- Step 2 will raise `FileNotFoundError` on first run intentionally — this is expected; fill the seed file
- `src/config.py` auto-creates all output directories on import
- `MAX_DOC_CHARS = 300_000` caps document text length for NER
- All figures use `thesis_style()` from `src/analysis/_helpers.py` — Times New Roman, 11pt, 300 DPI
- GitHub repo: `https://github.com/ge54xib/emerging-tech-policy-nlp`
