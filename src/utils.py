"""Shared helper utilities for pipeline steps and analysis modules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator


def load_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


# ---------------------------------------------------------------------------
# Shared helpers used across pipeline steps and analysis modules
# ---------------------------------------------------------------------------

def to_str(value) -> str:
    """Safely convert any value to str; returns empty string for None."""
    if value is None:
        return ""
    return str(value)


def to_int(value, default: int = 0) -> int:
    """Safely convert value to int; returns default on failure."""
    try:
        return int(value)
    except Exception:
        return default


def normalize_helix(value: str) -> str:
    """Normalize helix label to lowercase snake_case; maps 'university' → 'academia'."""
    helix = to_str(value).strip().lower().replace(" ", "_")
    if helix == "university":
        return "academia"
    return helix


def country_from_doc(doc_name: str) -> str:
    """Extract the uppercase country code from a document stem (e.g. 'GER_2023_...' → 'GER')."""
    raw = to_str(doc_name).strip()
    if not raw:
        return "UNK"
    return raw.split("_", 1)[0].upper()


def year_from_doc(doc_name: str) -> str:
    """Extract the 4-digit year string from a document stem (e.g. 'GER_2023_...' → '2023')."""
    match = re.search(r"(20\d{2})", to_str(doc_name))
    return match.group(1) if match else ""
