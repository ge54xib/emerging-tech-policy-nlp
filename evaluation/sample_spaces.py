"""Sample sentences for manual SetFit TH-space evaluation.

Commands
--------
python evaluation/sample_spaces.py          # generate annotation_spaces.json
python evaluation/sample_spaces.py --update # refresh setfit_space predictions
                                            # preserving all true_space annotations
python evaluation/sample_spaces.py --extend [--n N]
                                            # append N keyword-matched public_space
                                            # candidates (not in training data)

Fill in the "true_space" field for each entry.
Valid labels: knowledge_space, innovation_space, consensus_space, public_space

Sampling strategy:
  Deduplicated to unique sentences (SetFit classifies at sentence level).
  Stratified by helix-pair → TH_SPACE_MAP (same as training stratification).
  Training sentences (from spaces_review.json) are excluded to avoid leakage.
  25 per space (100 total).
"""

import json
import re
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
_NLI_FILE = ROOT / "data/processed/step3/cooccurrence_nli.jsonl"
_BASE_FILE = ROOT / "data/processed/step3/cooccurrence.jsonl"
COOCCURRENCE_FILE = _NLI_FILE if _NLI_FILE.exists() else _BASE_FILE  # NLI file optional
SETFIT_FILE = ROOT / "data/processed/step3/cooccurrence_setfit.jsonl"
REVIEW_FILE = ROOT / "data/processed/step3/spaces_review.json"
LLM_REVIEW_FILE = ROOT / "data/processed/step3/spaces_llm_review.json"
OUTPUT_FILE = Path(__file__).parent / "annotation_spaces.json"

# Keyword pattern for public_space targeting
_PUBLIC_SPACE_PATTERN = re.compile(
    r"\b(public|society|citizen|ethic|equity|inclusion|trust|democratic|governance|"
    r"awareness|engagement|transparency|accountability|civil.?society|fourth.?helix|"
    r"quadruple|stakeholder|community|social|equality|fairness|oversight|participation)\b",
    re.IGNORECASE,
)

SPACE_LABELS = ["knowledge_space", "innovation_space", "consensus_space", "public_space", "no_explicit_space"]
N_PER_SPACE = 25
RANDOM_SEED = 42

_TABLE_ROW_PATTERN = re.compile(
    r"^\s*[\d\-\•\*\|]|"
    r"\t|"
    r"\n.*\n|"
    r"^\s*[A-Z][A-Z ]{10,}$",
    re.MULTILINE,
)

def _is_clean_sentence(text: str) -> bool:
    if len(text) < 40:
        return False
    if _TABLE_ROW_PATTERN.search(text):
        return False
    if not re.search(r"\b(is|are|was|were|will|would|has|have|had|"
                     r"provide|support|develop|fund|establish|create|"
                     r"promote|ensure|enable|build|strengthen|work)\b", text, re.IGNORECASE):
        return False
    return True

PAIR_TO_SPACE = {
    frozenset({"academia", "government"}):     "knowledge_space",
    frozenset({"academia", "industry"}):       "innovation_space",
    frozenset({"academia", "intermediary"}):   "innovation_space",
    frozenset({"industry", "intermediary"}):   "innovation_space",
    frozenset({"government", "industry"}):     "consensus_space",
    frozenset({"government", "intermediary"}): "consensus_space",
}


def _pair_space(h1: str, h2: str) -> str:
    if "civil_society" in (h1, h2):
        return "public_space"
    return PAIR_TO_SPACE.get(frozenset({h1, h2}), "")


def _load_setfit_predictions() -> dict[str, str]:
    """Return central_sent_text → th_space_setfit from cooccurrence_setfit.jsonl."""
    if not SETFIT_FILE.exists():
        return {}
    preds: dict[str, str] = {}
    with SETFIT_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("central_sent_text") or row.get("sent_text", "")
            if text and "th_space_setfit" in row:
                preds[text] = row["th_space_setfit"]
    return preds


def _training_sentences() -> set[str]:
    """Return set of sentence texts already used in training (to exclude)."""
    sents: set[str] = set()
    for path in [REVIEW_FILE, LLM_REVIEW_FILE]:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for e in data:
            s = str(e.get("sentence") or e.get("central_sentence", "")).strip()
            if s:
                sents.add(s)
    return sents


def sample():
    random.seed(RANDOM_SEED)

    training_sents = _training_sentences()
    setfit_preds = _load_setfit_predictions()

    # Collect unique sentences from cooccurrence, cross-helix only
    seen_keys: set[tuple] = set()
    by_space: dict[str, list[dict]] = defaultdict(list)
    entities_by_key: dict[tuple, list[dict]] = defaultdict(list)

    rows_all = []
    with COOCCURRENCE_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_all.append(row)

    # Build candidate pool per space — entities filtered to those in sentence text
    for row in rows_all:
        if row.get("h1") == row.get("h2"):
            continue
        key = (str(row["doc_id"]), int(row["paragraph_id"]), int(row["sentence_id"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        central = row.get("central_sent_text") or row.get("sent_text", "")
        if len(central) < 40:
            continue
        if central.strip() in training_sents:
            continue
        if not _is_clean_sentence(central):
            continue

        space = _pair_space(row.get("h1", ""), row.get("h2", ""))
        if not space:
            continue

        entities = [
            {"entity": row.get("entity_1"), "helix": row.get("h1")},
            {"entity": row.get("entity_2"), "helix": row.get("h2")},
        ]

        by_space[space].append({
            "doc_id":     str(row["doc_id"]),
            "country":    row.get("country", ""),
            "sentence":   central,
            "entities":   entities,
            "true_space": "",
        })

    samples = []
    for space in SPACE_LABELS:
        pool = by_space[space]
        random.shuffle(pool)
        taken = pool[:N_PER_SPACE]
        samples.extend(taken)
        print(f"  {space}: {len(pool)} candidates → sampled {len(taken)}")

    random.shuffle(samples)
    OUTPUT_FILE.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(samples)} entries to {OUTPUT_FILE}")
    if not setfit_preds:
        print("Note: setfit_space is empty — run --predict first, then --update.")
    print('Fill in the "true_space" field for each entry.')
    print(f"Valid labels: {', '.join(SPACE_LABELS)}")


def update():
    """Refresh setfit_space predictions, preserving true_space annotations."""
    if not OUTPUT_FILE.exists():
        print("No annotation_spaces.json found — run without --update first.")
        return

    setfit_preds = _load_setfit_predictions()
    if not setfit_preds:
        print(f"No SetFit predictions found. Run: python -m src.pipeline.spaces_setfit --predict")
        return

    entries = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    updated = 0
    for e in entries:
        pred = setfit_preds.get(e.get("sentence", ""), "")
        if pred:
            e["setfit_space"] = pred
            updated += 1

    OUTPUT_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated setfit_space for {updated}/{len(entries)} entries.")


def extend(n_per_class: int = 10) -> None:
    """Append N keyword-matched public_space candidates to annotation_spaces.json.

    Excludes sentences already in annotation_spaces.json or in training data
    (spaces_review.json, spaces_llm_review.json). Leaves true_space="" for labeling.
    """
    if not OUTPUT_FILE.exists():
        print("No annotation_spaces.json found — run without flags first.")
        return

    existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    existing_sents = {e.get("sentence", "").strip() for e in existing}
    training_sents = _training_sentences()
    skip_sents = existing_sents | training_sents

    setfit_preds = _load_setfit_predictions()

    seen_keys: set[tuple] = set()
    entities_by_key: dict[tuple, list[dict]] = defaultdict(list)
    candidates = []

    rows_all = []
    with COOCCURRENCE_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_all.append(row)

    # Collect entities per sentence key
    for row in rows_all:
        key = (str(row["doc_id"]), int(row["paragraph_id"]), int(row["sentence_id"]))
        if row.get("h1") != row.get("h2"):
            for ef, hf in [("entity_1", "h1"), ("entity_2", "h2")]:
                ent = row.get(ef, "")
                helix = row.get(hf, "")
                if ent:
                    entry = {"entity": ent, "helix": helix}
                    if entry not in entities_by_key[key]:
                        entities_by_key[key].append(entry)

    for row in rows_all:
        if row.get("h1") == row.get("h2"):
            continue
        key = (str(row["doc_id"]), int(row["paragraph_id"]), int(row["sentence_id"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        central = row.get("central_sent_text") or row.get("sent_text", "")
        if len(central) < 20:
            continue
        if central.strip() in skip_sents:
            continue

        # Target public_space: either civil_society pair OR keyword match
        is_civil = "civil_society" in (row.get("h1", ""), row.get("h2", ""))
        has_keyword = bool(_PUBLIC_SPACE_PATTERN.search(central))
        if not (is_civil or has_keyword):
            continue

        candidates.append({
            "doc_id":       str(row["doc_id"]),
            "country":      row.get("country", ""),
            "sentence":     central,
            "entities":     entities_by_key[key],
            "pair_space":   "public_space" if is_civil else "keyword_match",
            "setfit_space": setfit_preds.get(central, ""),
            "true_space":   "",
        })

    # Prefer civil_society pairs, then keyword-only matches
    candidates.sort(key=lambda c: (0 if c["pair_space"] == "public_space" else 1))
    taken = candidates[:n_per_class]

    combined = existing + taken
    OUTPUT_FILE.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  public_space: added {len(taken)} new candidates "
          f"({sum(1 for c in taken if c['pair_space'] == 'public_space')} civil_society pairs, "
          f"{sum(1 for c in taken if c['pair_space'] == 'keyword_match')} keyword-only)")
    print(f"\nannotation_spaces.json now has {len(combined)} entries ({len(taken)} new).")
    print('Fill in true_space="" entries using evaluation/codebook_spaces.md.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="Refresh SetFit predictions, preserving true_space annotations")
    parser.add_argument("--extend", action="store_true",
                        help="Append keyword-matched public_space candidates")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of candidates to add (default: 10)")
    args = parser.parse_args()

    if args.update:
        update()
    elif args.extend:
        extend(n_per_class=args.n)
    else:
        sample()
