"""Global configuration for the policy-document NLP pipeline."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - fallback when dotenv is not installed
    def load_dotenv(*_args, **_kwargs):  # type: ignore[no-redef]
        return False


def _csv_to_upper_set(raw: str) -> set[str]:
    return {item.strip().upper() for item in (raw or "").split(",") if item.strip()}


def _csv_page_numbers_to_zero_based_set(raw: str) -> set[int]:
    """Parse comma-separated human page numbers (1-based) to 0-based indices."""
    out: set[int] = set()
    for item in (raw or "").split(","):
        token = item.strip()
        if not token:
            continue
        try:
            page_number = int(token)
            if page_number < 1:
                continue
            out.add(page_number - 1)
        except ValueError:
            continue
    return out


def _csv_to_country_int_map(raw: str) -> dict[str, set[int]]:
    """Parse mapping format (1-based): AUS:1,2;FIN:3,4"""
    out: dict[str, set[int]] = {}
    for item in (raw or "").split(";"):
        token = item.strip()
        if not token or ":" not in token:
            continue
        key, value = token.split(":", 1)
        country = key.strip().upper()
        if not country:
            continue
        out[country] = _csv_page_numbers_to_zero_based_set(value)
    return out


def _csv_to_doc_int_map(raw: str) -> dict[str, set[int]]:
    """Parse mapping format (1-based): <doc_stem>:1,2;<doc_stem_2>:4"""
    out: dict[str, set[int]] = {}
    for item in (raw or "").split(";"):
        token = item.strip()
        if not token or ":" not in token:
            continue
        key, value = token.split(":", 1)
        doc_stem = key.strip().lower()
        if not doc_stem:
            continue
        out[doc_stem] = _csv_page_numbers_to_zero_based_set(value)
    return out


def _linewise_doc_int_map(raw: str) -> dict[str, set[int]]:
    """Parse line-wise mapping format (1-based): one '<doc_stem>:1,2,3' per line."""
    out: dict[str, set[int]] = {}
    for line in (raw or "").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        doc_stem = key.strip().lower()
        if not doc_stem:
            continue
        out[doc_stem] = _csv_page_numbers_to_zero_based_set(value)
    return out


def _load_linewise_doc_int_map(path_raw: str, base_dir: Path) -> dict[str, set[int]]:
    """Load line-wise doc exclusions from file if present."""
    path_token = (path_raw or "").strip()
    if not path_token:
        return {}
    path = Path(path_token).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    return _linewise_doc_int_map(content)


# -----------------------------
# Base Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_INPUT_DIR = DATA_DIR / "raw"
DATA_OUTPUT_DIR = DATA_DIR / "processed"
INPUT_PDF_DIR = DATA_INPUT_DIR

load_dotenv(BASE_DIR / ".env")


# -----------------------------
# Credentials
# -----------------------------
ADOBE_CLIENT_ID = os.getenv("ADOBE_CLIENT_ID", "").strip()
ADOBE_CLIENT_SECRET = os.getenv("ADOBE_CLIENT_SECRET", "").strip()


# -----------------------------
# Step 0: Preprocessing
# -----------------------------
STEP0_DIR = DATA_OUTPUT_DIR / "step0"
STEP0_TEXT_DIR = STEP0_DIR / "text"
STEP0_JSON_DIR = STEP0_DIR / "json"
STEP0_STRUCTURED_DIR = STEP0_DIR / "structured"

# Main exclusion by path tags (default: remove TOC/TOCI/Footnote only)
PREPROCESS_EXCLUDE_PATH_TAGS = _csv_to_upper_set(
    os.getenv("PREPROCESS_EXCLUDE_PATH_TAGS", "TOC,TOCI,FOOTNOTE")
)

# Optional page-level exclusions for Step0 rendering (human page numbers, 1-based).
# Example: PREPROCESS_EXCLUDE_PAGES=1,2,3
PREPROCESS_EXCLUDE_PAGES = _csv_page_numbers_to_zero_based_set(
    os.getenv("PREPROCESS_EXCLUDE_PAGES", "")
)

# Optional country-level page exclusions for Step0 rendering (human page numbers, 1-based).
# Applies by file stem country prefix (<COUNTRY>_...), e.g.:
# PREPROCESS_EXCLUDE_PAGES_BY_COUNTRY=AUS:1,2;FIN:3,4
PREPROCESS_EXCLUDE_PAGES_BY_COUNTRY = _csv_to_country_int_map(
    os.getenv("PREPROCESS_EXCLUDE_PAGES_BY_COUNTRY", "")
)

# Optional document-level page exclusions for Step0 rendering (human page numbers, 1-based).
# Applies to exact Step0 file stem, e.g.:
# PREPROCESS_EXCLUDE_PAGES_BY_DOC=FIN_2025_Finland's Quantum Technology Strategy 2025-2035:1,2
PREPROCESS_EXCLUDE_PAGES_BY_DOC = _csv_to_doc_int_map(os.getenv("PREPROCESS_EXCLUDE_PAGES_BY_DOC", ""))

# Optional file path for line-wise doc exclusions (1 row per doc).
# Format per line: <doc_stem>:1,2,3
# Example:
# PREPROCESS_EXCLUDE_PAGES_BY_DOC_FILE=configs/step0_exclude_pages_by_doc.txt
PREPROCESS_EXCLUDE_PAGES_BY_DOC_FILE = os.getenv("PREPROCESS_EXCLUDE_PAGES_BY_DOC_FILE", "").strip()
for _doc_key, _doc_pages in _load_linewise_doc_int_map(PREPROCESS_EXCLUDE_PAGES_BY_DOC_FILE, BASE_DIR).items():
    PREPROCESS_EXCLUDE_PAGES_BY_DOC[_doc_key] = PREPROCESS_EXCLUDE_PAGES_BY_DOC.get(_doc_key, set()) | _doc_pages


# -----------------------------
# Step 1: NER
# -----------------------------
STEP1_DIR = DATA_OUTPUT_DIR / "step1"
STEP1_NER_PATH = STEP1_DIR / "entities_ner.jsonl"
FLAIR_NER_MODEL = "flair/ner-english-large"
FLAIR_NER_BATCH_SIZE = 32


# -----------------------------
# Step 2: Quadruple-Helix Classification
# -----------------------------
STEP2_DIR = DATA_OUTPUT_DIR / "step2"
STEP2_MANUAL_LABELS_PATH = STEP2_DIR / "manual_classification_labels.json"
STEP2_CLASSIFIED_PATH = STEP2_DIR / "entities_classified.jsonl"


# -----------------------------
# Step 3: Co-occurrence extraction (sentence splitting + entity pair building)
# -----------------------------
STEP3_DIR = DATA_OUTPUT_DIR / "step3"
STEP3_RELATIONS_PATH = STEP3_DIR / "relations_mapped.jsonl"

FILE_PARAGRAPHS = STEP3_DIR / "paragraphs.jsonl"
FILE_ALIGNMENT = STEP3_DIR / "paragraph_actor_alignment.jsonl"
FILE_COOCCURRENCE = STEP3_DIR / "cooccurrence.jsonl"

# -----------------------------
# Step 4: NLI Relation & Space Scoring
# -----------------------------
# cooccurrence_nli.jsonl = cooccurrence.jsonl + NLI fields
# (relation_type, confidence, all_scores, th_space, th_space_confidence, th_space_scores)
FILE_COOCCURRENCE_NLI = STEP3_DIR / "cooccurrence_nli.jsonl"

NLI_MODEL_NAME = os.getenv("NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-large")
NLI_THRESHOLD  = float(os.getenv("NLI_THRESHOLD", "0.5"))
NLI_BATCH_SIZE = int(os.getenv("NLI_BATCH_SIZE", "16"))


# -----------------------------
# Analysis Outputs (thesis-ready JSON + figures + CSVs)
# -----------------------------
ANALYSIS_DIR = BASE_DIR / "outputs"
ANALYSIS_METHODOLOGY_PATH = ANALYSIS_DIR / "methodology_summary.json"

ANALYSIS_DESCRIPTIVES_DIR = ANALYSIS_DIR / "descriptives"
ANALYSIS_RQ1_DIR = ANALYSIS_DIR / "rq1"
ANALYSIS_RQ2_DIR = ANALYSIS_DIR / "rq2"
ANALYSIS_RQ3_DIR = ANALYSIS_DIR / "rq3"
ANALYSIS_SPACES_DIR = ANALYSIS_DIR / "spaces"

ANALYSIS_RQ1_PATH = ANALYSIS_RQ1_DIR / "rq1_actor_prominence_and_balance.json"
ANALYSIS_RQ1_FIGURE_PATH = ANALYSIS_RQ1_DIR / "rq1_actor_prominence_and_balance.png"
ANALYSIS_RQ1_COMPONENTS_FIGURE_PATH = ANALYSIS_RQ1_DIR / "rq1_components_breakdown.png"
ANALYSIS_RQ1_CSV_PATH = ANALYSIS_RQ1_DIR / "rq1_table.csv"
ANALYSIS_RQ1_TEMPORAL_FIGURE_PATH = ANALYSIS_RQ1_DIR / "rq1_helix_share_over_time.png"

ANALYSIS_RQ2_RELATION_HELIX_FIGURE_PATH = ANALYSIS_RQ2_DIR / "rq2_relation_by_helix_pair.png"
ANALYSIS_RQ2_RELATION_HELIX_SIMPLE_FIGURE_PATH = ANALYSIS_RQ2_DIR / "rq2_relation_by_helix_pair_simple.png"
ANALYSIS_RQ2_RELATION_HELIX_BUBBLE_ONLY_PATH   = ANALYSIS_RQ2_DIR / "rq2_relation_by_helix_pair_bubble_only.png"
ANALYSIS_RQ2_RELATION_COUNTRY_DETAIL_PATH = ANALYSIS_RQ2_DIR / "rq2_relation_country_detail.png"
ANALYSIS_RQ2_RELATION_TEMPORAL_PATH = ANALYSIS_RQ2_DIR / "rq2_relation_over_time.png"

ANALYSIS_RQ3_PATH = ANALYSIS_RQ3_DIR / "rq3_profiles.json"
ANALYSIS_RQ3_FIGURE_PATH = ANALYSIS_RQ3_DIR / "rq3_country_profiles.png"
ANALYSIS_RQ3_SUMMARY_FIGURE_PATH = ANALYSIS_RQ3_DIR / "rq3_profiles_summary.png"
ANALYSIS_RQ3_CSV_PATH = ANALYSIS_RQ3_DIR / "rq3_table.csv"
ANALYSIS_RQ3_REASONING_FIGURE_PATH = ANALYSIS_RQ3_DIR / "rq3_classification_reasoning.png"
ANALYSIS_RQ3_COUNTRY_TRAJECTORIES_PATH = ANALYSIS_RQ3_DIR / "rq3_country_trajectories.png"

ANALYSIS_SPACES_COUNTRY_HEATMAP_PNG      = ANALYSIS_SPACES_DIR / "spaces_by_country_heatmap.png"
ANALYSIS_SPACES_GLOBAL_PNG               = ANALYSIS_SPACES_DIR / "spaces_global_counts.png"
ANALYSIS_SPACES_STACKED_PNG              = ANALYSIS_SPACES_DIR / "spaces_by_country_stacked.png"
ANALYSIS_SPACES_RELATION_MATRIX_PNG      = ANALYSIS_SPACES_DIR / "spaces_relation_matrix.png"
ANALYSIS_SPACES_CONFIDENCE_PNG           = ANALYSIS_SPACES_DIR / "spaces_confidence.png"
ANALYSIS_SPACES_HELIX_BUBBLE_PATH        = ANALYSIS_SPACES_DIR / "spaces_space_by_helix_pair.png"
ANALYSIS_SPACES_HELIX_BUBBLE_ONLY_PATH   = ANALYSIS_SPACES_DIR / "spaces_space_by_helix_pair_bubble_only.png"
ANALYSIS_SPACES_CSV                      = ANALYSIS_SPACES_DIR / "spaces_table.csv"
ANALYSIS_SPACES_JSON                     = ANALYSIS_SPACES_DIR / "spaces_summary.json"

ANALYSIS_DESCRIPTIVES_CORPUS_CSV = ANALYSIS_DESCRIPTIVES_DIR / "descriptives_corpus_overview.csv"
ANALYSIS_DESCRIPTIVES_COUNTRY_CSV = ANALYSIS_DESCRIPTIVES_DIR / "descriptives_corpus_by_country.csv"
ANALYSIS_DESCRIPTIVES_ACTORS_CSV = ANALYSIS_DESCRIPTIVES_DIR / "descriptives_actor_summary.csv"
ANALYSIS_DESCRIPTIVES_TABLE_ACTORS_PNG = ANALYSIS_DESCRIPTIVES_DIR / "descriptives_table_actors.png"


# -----------------------------
# Shared Runtime
# -----------------------------
MAX_DOC_CHARS = 300_000


# -----------------------------
# Directory Bootstrap
# -----------------------------
ALL_OUTPUT_DIRS = [
    DATA_DIR,
    DATA_INPUT_DIR,
    DATA_OUTPUT_DIR,
    STEP0_DIR,
    STEP1_DIR,
    STEP2_DIR,
    STEP3_DIR,
    ANALYSIS_DIR,
    ANALYSIS_DESCRIPTIVES_DIR,
    ANALYSIS_RQ1_DIR,
    ANALYSIS_RQ2_DIR,
    ANALYSIS_RQ3_DIR,
    ANALYSIS_SPACES_DIR,
    STEP0_TEXT_DIR,
    STEP0_JSON_DIR,
    STEP0_STRUCTURED_DIR,
]


def ensure_directories() -> None:
    """Create all pipeline output directories."""
    for directory in ALL_OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()


# -----------------------------
# Compatibility Aliases
# -----------------------------
FILE_NER_OUTPUT = STEP1_NER_PATH
INPUT_DIR = STEP0_TEXT_DIR
MAX_CHARS_PER_DOC = MAX_DOC_CHARS
