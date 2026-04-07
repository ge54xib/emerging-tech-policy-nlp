"""GLiREL relation label quality validation.

Two modes:
  python -m src.analysis.validate_glirel sample   → export stratified sample CSV for manual annotation
  python -m src.analysis.validate_glirel score    → compute precision from your annotations

Workflow:
  1. Run `sample` → outputs/validation/glirel_sample.csv
  2. Open the CSV, fill the `correct` column: 1 = label fits the paragraph, 0 = wrong
  3. Run `score`  → prints per-label and overall precision
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src import config
from src.utils import to_str

SAMPLE_PER_LABEL = 20   # how many rows to draw per relation type
CONFIDENCE_MIN   = 0.0  # lower bound — set e.g. 0.5 to only sample high-conf rows
RANDOM_SEED      = 42

EXPLICIT_RELATION_TYPES = [
    "technology_transfer",
    "collaboration_conflict_moderation",
    "collaborative_leadership",
    "substitution",
    "networking",
]

OUTPUT_DIR   = Path(config.ANALYSIS_DIR) / "validation"
SAMPLE_PATH  = OUTPUT_DIR / "glirel_sample.csv"
FIELDNAMES = [
    "id", "country", "year", "doc_name",
    "e1_mention", "e1_helix",
    "e2_mention", "e2_helix",
    "relation_type", "confidence",
    "paragraph_text",
    "correct",   # ← you fill this: 1 correct / 0 wrong / leave blank = unsure
    "notes",     # ← optional free-text comment
]


def _load_cooccurrence() -> list[dict]:
    # Build sentence text lookup from alignment file: (doc_id, paragraph_id, sentence_id) -> sentence text
    # Fall back to paragraph text if sentence_id is not present
    from src.pipeline.step3_cooccurrence import _sentence_spans, _build_sentence_splitter

    para_text_map: dict[tuple, str] = {}
    para_path = config.FILE_PARAGRAPHS
    if para_path.exists():
        with para_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                    key = (p.get("doc_id"), p.get("paragraph_id"))
                    para_text_map[key] = p.get("text", "")
                except json.JSONDecodeError:
                    pass

    # Build sentence-level lookup
    nlp = _build_sentence_splitter()
    sent_text_map: dict[tuple, str] = {}
    for (doc_id, para_id), para_text in para_text_map.items():
        sents = _sentence_spans(para_text, 0, nlp)
        for s in sents:
            sent_text_map[(doc_id, para_id, s["sentence_id"])] = s["text"]

    path = config.FILE_COOCCURRENCE
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                sent_key = (r.get("doc_id"), r.get("paragraph_id"), r.get("sentence_id"))
                para_key = (r.get("doc_id"), r.get("paragraph_id"))
                r["paragraph_text"] = (
                    sent_text_map.get(sent_key)
                    or para_text_map.get(para_key, "")
                )
                rows.append(r)
            except json.JSONDecodeError:
                pass
    return rows


def cmd_sample() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_cooccurrence()

    # Only keep rows with an explicit relation label above confidence threshold
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rel = to_str(r.get("relation_type", "")).strip()
        if rel not in EXPLICIT_RELATION_TYPES:
            continue
        try:
            conf = float(r.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < CONFIDENCE_MIN:
            continue
        by_label[rel].append(r)

    rng = random.Random(RANDOM_SEED)
    sample_rows = []
    for rel in EXPLICIT_RELATION_TYPES:
        pool = by_label[rel]
        n = min(SAMPLE_PER_LABEL, len(pool))
        drawn = rng.sample(pool, n)
        for i, r in enumerate(drawn):
            sample_rows.append({
                "id":             f"{rel}_{i+1:02d}",
                "country":        to_str(r.get("country", "")),
                "year":           to_str(r.get("year", "")),
                "doc_name":       to_str(r.get("doc_id", "")),
                "e1_mention":     to_str(r.get("e1_mention", r.get("entity1", ""))),
                "e1_helix":       to_str(r.get("h1", r.get("helix1", ""))),
                "e2_mention":     to_str(r.get("e2_mention", r.get("entity2", ""))),
                "e2_helix":       to_str(r.get("h2", r.get("helix2", ""))),
                "relation_type":  rel,
                "confidence":     f"{float(r.get('confidence', 0) or 0):.4f}",
                "paragraph_text": to_str(r.get("paragraph_text", r.get("paragraph", ""))).replace("\n", " ").strip(),
                "correct":        "",
                "notes":          "",
            })

    with SAMPLE_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sample_rows)

    print(f"[OK] Wrote {len(sample_rows)} rows → {SAMPLE_PATH}")
    print("\nDrawn per label:")
    counts = Counter(r["relation_type"] for r in sample_rows)
    for rel in EXPLICIT_RELATION_TYPES:
        pool_size = len(by_label[rel])
        drawn = counts.get(rel, 0)
        print(f"  {rel:<40} drawn={drawn:>3}  pool={pool_size}")
    print(f"\nNext: open {SAMPLE_PATH}, fill the 'correct' column (1/0), then run:\n"
          f"  python -m src.analysis.validate_glirel score")


def cmd_score() -> None:
    if not SAMPLE_PATH.exists():
        print(f"[ERROR] Sample file not found: {SAMPLE_PATH}")
        print("Run `python -m src.analysis.validate_glirel sample` first.")
        sys.exit(1)

    rows = []
    with SAMPLE_PATH.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    annotated = [r for r in rows if r.get("correct", "").strip() in ("0", "1")]
    if not annotated:
        print("[ERROR] No annotated rows found. Fill the 'correct' column (1 or 0) first.")
        sys.exit(1)

    by_label: dict[str, list[int]] = defaultdict(list)
    for r in annotated:
        by_label[r["relation_type"]].append(int(r["correct"]))

    print(f"\nGLiREL validation results  (N={len(annotated)} annotated rows)\n")
    print(f"{'Relation type':<40} {'N':>4}  {'Correct':>7}  {'Precision':>9}")
    print("-" * 65)
    total_n, total_correct = 0, 0
    for rel in EXPLICIT_RELATION_TYPES:
        vals = by_label.get(rel, [])
        if not vals:
            print(f"  {rel:<38}  {'—':>4}  {'—':>7}  {'—':>9}")
            continue
        n = len(vals)
        correct = sum(vals)
        precision = correct / n
        total_n += n
        total_correct += correct
        bar = "█" * int(precision * 20)
        print(f"  {rel:<38}  {n:>4}  {correct:>7}  {precision:>8.1%}  {bar}")

    print("-" * 65)
    overall = total_correct / total_n if total_n else 0
    print(f"  {'OVERALL':<38}  {total_n:>4}  {total_correct:>7}  {overall:>8.1%}")

    # Confidence correlation: are high-conf predictions more accurate?
    conf_buckets: dict[str, list[int]] = defaultdict(list)
    for r in annotated:
        try:
            conf = float(r.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        bucket = f"{int(conf * 10) / 10:.1f}–{int(conf * 10) / 10 + 0.1:.1f}"
        conf_buckets[bucket].append(int(r["correct"]))

    print("\nPrecision by confidence bucket:")
    for bucket in sorted(conf_buckets):
        vals = conf_buckets[bucket]
        prec = sum(vals) / len(vals)
        print(f"  conf {bucket}  N={len(vals):>3}  precision={prec:.1%}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if cmd == "sample":
        cmd_sample()
    elif cmd == "score":
        cmd_score()
    else:
        print(f"Unknown command: {cmd}. Use 'sample' or 'score'.")
        sys.exit(1)
