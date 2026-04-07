"""Sample co-occurrence pairs for manual NLI evaluation.

Commands
--------
python evaluation/sample.py          # generate new annotation.json (100 entries)
python evaluation/sample.py --update # refresh NLI predictions in existing annotation.json
                                     # preserving all true_relation annotations
python evaluation/sample.py --extend [--n N]
                                     # append N targeted examples per rare class
                                     # (substitution, collaboration_conflict_moderation)
                                     # using keyword matching; skips training data

Fill in the "true_relation" field for each entry.
Valid labels:
  technology_transfer
  collaboration_conflict_moderation
  collaborative_leadership
  substitution
  networking
  no_explicit_relation

Sampling strategy:
  For each relation class, pick the N_PER_CLASS pairs where that class has
  the highest per-class NLI score (from `all_scores`), regardless of the
  global threshold.  This ensures all six classes are represented even when
  the threshold suppresses rare labels.
  `nli_relation` still reflects the threshold-based prediction so the
  evaluation measures real pipeline output.
"""

import argparse
import json
import re
import random
from collections import defaultdict
from pathlib import Path

_STEP3_DIR = Path(__file__).parent.parent / "data/processed/step3"
# Prefer cooccurrence_nli.jsonl (has all_scores from step 4) over cooccurrence.jsonl
_NLI_FILE = _STEP3_DIR / "cooccurrence_nli.jsonl"
_BASE_FILE = _STEP3_DIR / "cooccurrence.jsonl"
COOCCURRENCE_FILE = _NLI_FILE if _NLI_FILE.exists() else _BASE_FILE
ANNOTATION_V1_FILE = Path(__file__).parent / "annotation_v1.json"
OUTPUT_FILE = Path(__file__).parent / "annotation.json"

RELATION_TYPES = [
    "technology_transfer",
    "collaboration_conflict_moderation",
    "collaborative_leadership",
    "substitution",
    "networking",
    "no_explicit_relation",
]

N_TOTAL = 100
RANDOM_SEED = 42


def _load_cooccurrence():
    """Load cross-helix rows, keyed by (doc_id, entity_1, entity_2, sent_text[:40])."""
    rows = []
    lookup = {}  # key → row (for fast prediction refresh)
    with COOCCURRENCE_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not (row.get("sent_text") and row.get("entity_1") and row.get("entity_2")):
                continue
            if row.get("h1") == row.get("h2"):
                continue
            rows.append(row)
            key = (row.get("doc_id"), row.get("entity_1"), row.get("entity_2"), row.get("sent_text", "")[:40])
            lookup[key] = row
    return rows, lookup


def _get_central_sent_text(row: dict) -> str:
    """Return central_sent_text from a cooccurrence row, falling back to sent_text."""
    return row.get("central_sent_text") or row.get("sent_text", "")


def sample():
    random.seed(RANDOM_SEED)
    rows, _ = _load_cooccurrence()

    print(f"Loaded {len(rows)} cross-helix co-occurrence pairs")
    print("\nNLI prediction counts (threshold-based):")
    pred_counts = defaultdict(int)
    for row in rows:
        rel = row.get("relation_type") or "no_explicit_relation"
        pred_counts[rel] += 1
    for rel in RELATION_TYPES:
        print(f"  {rel}: {pred_counts[rel]}")

    n_classes = len(RELATION_TYPES)
    base = N_TOTAL // n_classes
    remainder = N_TOTAL % n_classes
    per_class = {rel: base + (1 if i < remainder else 0) for i, rel in enumerate(RELATION_TYPES)}

    used_keys = set()
    samples = []

    for rel in RELATION_TYPES:
        if rel == "no_explicit_relation":
            pool = sorted(rows, key=lambda r: max(r.get("all_scores", {}).values(), default=0))
        else:
            pool = sorted(rows, key=lambda r: r.get("all_scores", {}).get(rel, 0), reverse=True)

        taken = 0
        for row in pool:
            if taken >= per_class[rel]:
                break
            key = (row.get("doc_id"), row.get("entity_1"), row.get("entity_2"), row.get("sent_text", "")[:40])
            if key in used_keys:
                continue
            used_keys.add(key)
            nli_rel = row.get("relation_type") or "no_explicit_relation"
            samples.append({
                "doc_id":             row.get("doc_id"),
                "entity_1":           row.get("entity_1"),
                "h1":                 row.get("h1"),
                "entity_2":           row.get("entity_2"),
                "h2":                 row.get("h2"),
                "central_sent_text":  _get_central_sent_text(row),
                "sent_text":          row.get("sent_text"),
                "sampled_for":        rel,
                "nli_relation":       nli_rel,
                "nli_confidence":     row.get("confidence"),
                "nli_scores":         row.get("all_scores"),
                "true_relation":      "",
            })
            taken += 1
        print(f"  {rel}: sampled {taken}")

    random.shuffle(samples)
    OUTPUT_FILE.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(samples)} entries to {OUTPUT_FILE}")
    print('Fill in the "true_relation" field for each entry.')
    print(f"Valid labels: {', '.join(RELATION_TYPES)}")


def update():
    """Refresh nli_relation/nli_confidence/nli_scores from new cooccurrence.jsonl.
    Preserves true_relation and all other annotation fields."""
    if not OUTPUT_FILE.exists():
        print("No annotation.json found — run without --update first.")
        return

    _, lookup = _load_cooccurrence()
    entries = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))

    updated = 0
    not_found = 0
    for e in entries:
        key = (e.get("doc_id"), e.get("entity_1"), e.get("entity_2"), e.get("sent_text", "")[:40])
        row = lookup.get(key)
        if row is None:
            not_found += 1
            continue
        e["nli_relation"]        = row.get("relation_type") or "no_explicit_relation"
        e["nli_confidence"]      = row.get("confidence")
        e["nli_scores"]          = row.get("all_scores")
        # Backfill central_sent_text if missing (added in later pipeline version)
        if not e.get("central_sent_text"):
            e["central_sent_text"] = _get_central_sent_text(row)
        updated += 1

    OUTPUT_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {updated} entries with new NLI predictions.")
    if not_found:
        print(f"  {not_found} entries not found in new cooccurrence.jsonl (sent_text may have changed).")


_TABLE_ROW_PATTERN = re.compile(
    r"^\s*[\d\-\•\*\|]|"           # starts with bullet/number/pipe
    r"\t|"                          # contains tab
    r"\n.*\n|"                      # multi-line
    r"^\s*[A-Z][A-Z ]{10,}$",      # ALL CAPS header
    re.MULTILINE,
)

def _is_clean_sentence(text: str) -> bool:
    """Return True if text looks like a real sentence (not a table row or header)."""
    if len(text) < 40:
        return False
    if _TABLE_ROW_PATTERN.search(text):
        return False
    # Must contain at least one verb-like word (basic sentence check)
    if not re.search(r"\b(is|are|was|were|will|would|has|have|had|"
                     r"provide|support|develop|fund|establish|create|"
                     r"promote|ensure|enable|build|strengthen|work)\b", text, re.IGNORECASE):
        return False
    return True


# Keyword patterns for rare relation classes (cross-helix keyword targeting)
_EXTEND_PATTERNS = {
    "substitution": re.compile(
        r"\b(fill|gap|weak|absent|replac|substitut|step.?in|take.?on|lack|void|"
        r"absence|supplement|compens|bridge.?gap|no.?industry|government.?fund|"
        r"public.?venture|state.?provid|fill.?role|play.?role)\b",
        re.IGNORECASE,
    ),
    "collaboration_conflict_moderation": re.compile(
        r"\b(tension|conflict|compet|rival|interest|align|coordinat|balanc|"
        r"reconcil|harmoniz|moderat|mediat|resolv|bridge|conver|dialogue|"
        r"negotiat|consult|diverge|consens|tripart|triadic|pluralist)\b",
        re.IGNORECASE,
    ),
}


def _load_training_keys() -> set:
    """Load (doc_id, entity_1, entity_2) keys from all training data sources."""
    training_keys: set = set()
    if ANNOTATION_V1_FILE.exists():
        v1 = json.loads(ANNOTATION_V1_FILE.read_text(encoding="utf-8"))
        for e in v1:
            training_keys.add((e.get("doc_id"), e.get("entity_1"), e.get("entity_2")))
    return training_keys


def extend(n_per_class: int = 10) -> None:
    """Append N keyword-matched candidates per rare class to annotation.json.

    Skips pairs already in annotation.json or in annotation_v1.json (training data).
    Leaves true_relation="" for manual labeling.
    """
    if not OUTPUT_FILE.exists():
        print("No annotation.json found — run without flags first.")
        return

    existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    existing_keys = {
        (e.get("doc_id"), e.get("entity_1"), e.get("entity_2"))
        for e in existing
    }
    training_keys = _load_training_keys()
    skip_keys = existing_keys | training_keys

    rows, _ = _load_cooccurrence()

    new_entries = []
    seen_sents: set[str] = {e.get("central_sent_text", "") for e in existing}

    for rel_class, pattern in _EXTEND_PATTERNS.items():
        taken = 0
        candidates = []
        for row in rows:
            key = (row.get("doc_id"), row.get("entity_1"), row.get("entity_2"))
            if key in skip_keys:
                continue
            central = _get_central_sent_text(row)
            if central in seen_sents:
                continue
            if pattern.search(central) and _is_clean_sentence(central):
                candidates.append(row)

        # Sort by NLI score for this class (descending) for best keyword+model agreement
        candidates.sort(
            key=lambda r: r.get("all_scores", {}).get(rel_class, 0),
            reverse=True,
        )

        for row in candidates:
            if taken >= n_per_class:
                break
            key = (row.get("doc_id"), row.get("entity_1"), row.get("entity_2"))
            central = _get_central_sent_text(row)
            if central in seen_sents:
                skip_keys.add(key)
                continue
            skip_keys.add(key)
            seen_sents.add(central)
            new_entries.append({
                "doc_id":            row.get("doc_id"),
                "entity_1":          row.get("entity_1"),
                "h1":                row.get("h1"),
                "entity_2":          row.get("entity_2"),
                "h2":                row.get("h2"),
                "central_sent_text": _get_central_sent_text(row),
                "sent_text":         row.get("sent_text"),
                "sampled_for":       f"extend_{rel_class}",
                "nli_relation":      row.get("relation_type") or "no_explicit_relation",
                "nli_confidence":    row.get("confidence"),
                "nli_scores":        row.get("all_scores"),
                "true_relation":     "",
            })
            taken += 1

        print(f"  {rel_class}: added {taken} new candidates (keyword-matched, not in training)")

    combined = existing + new_entries
    OUTPUT_FILE.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nannotation.json now has {len(combined)} entries ({len(new_entries)} new).")
    print('Fill in true_relation="" entries using evaluation/codebook.md.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="Refresh NLI predictions, preserving true_relation annotations")
    parser.add_argument("--extend", action="store_true",
                        help="Append targeted keyword-matched candidates for rare classes")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of candidates to add per rare class (default: 10)")
    args = parser.parse_args()

    if args.update:
        update()
    elif args.extend:
        extend(n_per_class=args.n)
    else:
        sample()
