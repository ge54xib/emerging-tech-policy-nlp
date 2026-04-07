"""Sample co-occurrence pairs for manual evaluation.

Commands
--------
python evaluation/sample.py          # generate new annotation.json (100 entries)
python evaluation/sample.py --extend [--n N]
                                     # append N targeted examples per rare class
                                     # (substitution, collaboration_conflict_moderation)
                                     # using keyword matching

Sampling strategy:
  Stratified by helix pair (same logic as sample_spaces.py uses PAIR_TO_SPACE).
  Each of the 6 cross-helix pair groups contributes equally (N_TOTAL / n_pairs).
  civil_society pairs are grouped together as one bucket.
  Deduplicated by (doc_id, paragraph_id, sentence_id).

Fill in BOTH fields for each entry:
  true_relation — relation between entity_1 and entity_2
  true_space    — TH space expressed by the sentence

Valid true_relation:
  technology_transfer, collaboration_conflict_moderation,
  collaborative_leadership, substitution, networking, no_explicit_relation

Valid true_space:
  knowledge_space, innovation_space, consensus_space, public_space, no_explicit_space
"""

import argparse
import json
import re
import random
from collections import defaultdict
from pathlib import Path

_STEP3_DIR = Path(__file__).parent.parent / "data/processed/step3"
COOCCURRENCE_FILE = _STEP3_DIR / "cooccurrence.jsonl"
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

# Helix-pair buckets (mirrors PAIR_TO_SPACE in sample_spaces.py)
# civil_society pairs are grouped into one bucket regardless of the other helix
HELIX_PAIR_BUCKETS = [
    frozenset({"academia", "government"}),
    frozenset({"academia", "industry"}),
    frozenset({"academia", "intermediary"}),
    frozenset({"government", "industry"}),
    frozenset({"government", "intermediary"}),
    frozenset({"industry", "intermediary"}),
    "civil_society",  # sentinel: any pair involving civil_society
]

N_TOTAL = 100
RANDOM_SEED = 42


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
    if not re.search(r"\b(is|are|was|were|will|would|has|have|had|"
                     r"provide|support|develop|fund|establish|create|"
                     r"promote|ensure|enable|build|strengthen|work)\b", text, re.IGNORECASE):
        return False
    return True


def _bucket(h1: str, h2: str):
    """Return the helix-pair bucket key for a given (h1, h2) pair."""
    if "civil_society" in (h1, h2):
        return "civil_society"
    return frozenset({h1, h2})


def _get_sentence(row: dict) -> str:
    return row.get("central_sent_text") or row.get("sent_text", "")


def _make_entry(row: dict) -> dict:
    return {
        "doc_id":       row.get("doc_id"),
        "country":      row.get("country", ""),
        "paragraph_id": row.get("paragraph_id"),
        "sentence_id":  row.get("sentence_id"),
        "sentence":     _get_sentence(row),
        "entity_1":     row.get("entity_1"),
        "h1":           row.get("h1"),
        "entity_2":     row.get("entity_2"),
        "h2":           row.get("h2"),
        "entities": [
            {"entity": row.get("entity_1"), "helix": row.get("h1")},
            {"entity": row.get("entity_2"), "helix": row.get("h2")},
        ],
        "true_relation": "",
        "true_space":    "",
    }


def _load_cooccurrence():
    rows = []
    with COOCCURRENCE_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not (row.get("entity_1") and row.get("entity_2")):
                continue
            if not _get_sentence(row):
                continue
            if row.get("h1") == row.get("h2"):
                continue
            rows.append(row)
    return rows


def sample():
    random.seed(RANDOM_SEED)
    rows = _load_cooccurrence()
    print(f"Loaded {len(rows)} cross-helix co-occurrence pairs")

    # Group clean sentences by helix-pair bucket, deduplicated by sentence_id
    by_bucket: dict = defaultdict(list)
    seen_sids: set = set()

    for row in rows:
        if not _is_clean_sentence(_get_sentence(row)):
            continue
        sid = (row.get("doc_id"), row.get("paragraph_id"), row.get("sentence_id"))
        if sid in seen_sids:
            continue
        seen_sids.add(sid)
        key = _bucket(row.get("h1", ""), row.get("h2", ""))
        by_bucket[key].append(row)

    # Shuffle each bucket independently
    for bucket in by_bucket.values():
        random.shuffle(bucket)

    n_buckets = len(HELIX_PAIR_BUCKETS)
    base = N_TOTAL // n_buckets
    remainder = N_TOTAL % n_buckets
    per_bucket = {b: base + (1 if i < remainder else 0) for i, b in enumerate(HELIX_PAIR_BUCKETS)}

    samples = []
    used_sids: set = set()
    leftover: list = []

    for bucket_key in HELIX_PAIR_BUCKETS:
        pool = by_bucket.get(bucket_key, [])
        taken = 0
        for row in pool:
            if taken >= per_bucket[bucket_key]:
                leftover.extend(pool[taken:])
                break
            sid = (row.get("doc_id"), row.get("paragraph_id"), row.get("sentence_id"))
            if sid in used_sids:
                continue
            used_sids.add(sid)
            samples.append(_make_entry(row))
            taken += 1
        bucket_label = " & ".join(sorted(bucket_key)) if isinstance(bucket_key, frozenset) else bucket_key
        print(f"  {bucket_label}: {len(pool)} candidates → sampled {taken}")

    # Fill remaining quota from leftover rows (any bucket)
    if len(samples) < N_TOTAL:
        random.shuffle(leftover)
        for row in leftover:
            if len(samples) >= N_TOTAL:
                break
            sid = (row.get("doc_id"), row.get("paragraph_id"), row.get("sentence_id"))
            if sid in used_sids:
                continue
            used_sids.add(sid)
            samples.append(_make_entry(row))

    random.shuffle(samples)
    OUTPUT_FILE.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(samples)} entries to {OUTPUT_FILE}")
    print('Fill in "true_relation" and "true_space" for each entry.')


# Keyword patterns for rare classes (used by --extend only)
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
    "public_space": re.compile(
        r"\b(public|society|citizen|ethic|equity|inclusion|trust|democratic|"
        r"awareness|engagement|transparency|accountability|civil.?society|"
        r"quadruple|community|social|equality|fairness|oversight|participation|"
        r"fourth.?helix|responsible|diversity|benefit)\b",
        re.IGNORECASE,
    ),
}


def _load_training_keys() -> set:
    training_keys: set = set()
    if ANNOTATION_V1_FILE.exists():
        v1 = json.loads(ANNOTATION_V1_FILE.read_text(encoding="utf-8"))
        for e in v1:
            training_keys.add((e.get("doc_id"), e.get("entity_1"), e.get("entity_2")))
    return training_keys


def extend(n_per_class: int = 10) -> None:
    """Append N keyword-matched candidates per rare class to annotation.json."""
    if not OUTPUT_FILE.exists():
        print("No annotation.json found — run without flags first.")
        return

    existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    existing_keys = {
        (e.get("doc_id"), e.get("entity_1"), e.get("entity_2"))
        for e in existing
    }
    skip_keys = existing_keys | _load_training_keys()
    used_sids: set = {
        (e.get("doc_id"), e.get("paragraph_id"), e.get("sentence_id"))
        for e in existing
    }

    rows = _load_cooccurrence()
    new_entries = []

    for rel_class, pattern in _EXTEND_PATTERNS.items():
        candidates = [
            r for r in rows
            if (r.get("doc_id"), r.get("entity_1"), r.get("entity_2")) not in skip_keys
            and (r.get("doc_id"), r.get("paragraph_id"), r.get("sentence_id")) not in used_sids
            and pattern.search(_get_sentence(r))
            and _is_clean_sentence(_get_sentence(r))
        ]
        taken = 0
        for row in candidates:
            if taken >= n_per_class:
                break
            sid = (row.get("doc_id"), row.get("paragraph_id"), row.get("sentence_id"))
            if sid in used_sids:
                continue
            skip_keys.add((row.get("doc_id"), row.get("entity_1"), row.get("entity_2")))
            used_sids.add(sid)
            new_entries.append(_make_entry(row))
            taken += 1
        print(f"  {rel_class}: added {taken} new candidates (keyword-matched)")

    combined = existing + new_entries
    OUTPUT_FILE.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nannotation.json now has {len(combined)} entries ({len(new_entries)} new).")
    print('Fill in "true_relation" and "true_space" for new entries.')




DEMO_FILE = Path(__file__).parent.parent / "Experiments" / "Relation" / "relation_labels.json"
N_DEMO = 50


def demo() -> None:
    """Sample a separate demo pool for Experiments/Relation/gpt_re/run.py only.

    Writes relation_labels.json (50 entries), excluding sentences already in
    annotation.json (the evaluation set). Annotate true_relation before running gpt_re.
    """
    random.seed(RANDOM_SEED + 1)
    rows = _load_cooccurrence()

    eval_sids: set = set()
    if OUTPUT_FILE.exists():
        for e in json.loads(OUTPUT_FILE.read_text(encoding="utf-8")):
            eval_sids.add((e.get("doc_id"), e.get("paragraph_id"), e.get("sentence_id")))

    by_bucket: dict = defaultdict(list)
    seen_sids: set = set()
    for row in rows:
        if not _is_clean_sentence(_get_sentence(row)):
            continue
        sid = (row.get("doc_id"), row.get("paragraph_id"), row.get("sentence_id"))
        if sid in seen_sids or sid in eval_sids:
            continue
        seen_sids.add(sid)
        key = _bucket(row.get("h1", ""), row.get("h2", ""))
        by_bucket[key].append(row)

    for bucket in by_bucket.values():
        random.shuffle(bucket)

    n_buckets = len(HELIX_PAIR_BUCKETS)
    base = N_DEMO // n_buckets
    remainder = N_DEMO % n_buckets
    per_bucket = {b: base + (1 if i < remainder else 0) for i, b in enumerate(HELIX_PAIR_BUCKETS)}

    samples = []
    used_sids: set = set()
    leftover: list = []
    for bucket_key in HELIX_PAIR_BUCKETS:
        pool = by_bucket.get(bucket_key, [])
        taken = 0
        for row in pool:
            if taken >= per_bucket[bucket_key]:
                leftover.extend(pool[taken:])
                break
            sid = (row.get("doc_id"), row.get("paragraph_id"), row.get("sentence_id"))
            if sid in used_sids:
                continue
            used_sids.add(sid)
            samples.append(_make_entry(row))
            taken += 1
        bucket_label = " & ".join(sorted(bucket_key)) if isinstance(bucket_key, frozenset) else bucket_key
        print(f"  {bucket_label}: {len(pool)} candidates → sampled {taken}")

    if len(samples) < N_DEMO:
        random.shuffle(leftover)
        for row in leftover:
            if len(samples) >= N_DEMO:
                break
            sid = (row.get("doc_id"), row.get("paragraph_id"), row.get("sentence_id"))
            if sid in used_sids:
                continue
            used_sids.add(sid)
            samples.append(_make_entry(row))

    random.shuffle(samples)
    DEMO_FILE.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(samples)} entries to {DEMO_FILE}")
    print('Fill in "true_relation" for each entry (demo pool for gpt_re only).')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extend", action="store_true",
                        help="Append targeted keyword-matched candidates for rare classes")
    parser.add_argument("--demo", action="store_true",
                        help="Generate relation_labels.json — demo pool for gpt_re only")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of candidates to add per rare class (default: 10)")
    args = parser.parse_args()

    if args.extend:
        extend(n_per_class=args.n)
    elif args.demo:
        demo()
    else:
        sample()
