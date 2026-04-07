"""Step 4: NLI zero-shot Relation and Space scoring.

Reads cooccurrence.jsonl (built by Step 3) and scores each entity pair using
NLI entailment (Sainz et al. 2021) against 5 Ranga & Etzkowitz (2013)
relation types and 4 Triple Helix activity spaces.

Uses central_sent_text (single sentence) as the NLI premise — not the ±1
sentence window used in the old combined Step 3. This avoids multi-sentence
and bullet-list inputs that degrade classifier quality.

Output: cooccurrence_nli.jsonl — same rows as cooccurrence.jsonl with NLI
fields appended:
  relation_type, confidence, all_scores,
  th_space, th_space_confidence, th_space_scores

Run:
    python run.py pipeline --step 4
    # or with GPU:
    NLI_BATCH_SIZE=128 python run.py pipeline --step 4
"""
from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from src import config


def run() -> None:
    print(">>> STEP 4: NLI relation and space scoring")

    if not config.FILE_COOCCURRENCE.exists():
        raise FileNotFoundError(
            f"cooccurrence.jsonl not found: {config.FILE_COOCCURRENCE}\n"
            "Run step 3 first: python run.py pipeline --step 3"
        )

    # Load all rows
    rows: list[dict] = []
    with config.FILE_COOCCURRENCE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"[INFO] Loaded {len(rows)} co-occurrence pairs from cooccurrence.jsonl")

    # Detect already-processed rows
    existing_keys: set[tuple] = set()
    if config.FILE_COOCCURRENCE_NLI.exists():
        with config.FILE_COOCCURRENCE_NLI.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    existing_keys.add((
                        r.get("doc_id"), r.get("paragraph_id"),
                        r.get("sentence_id"), r.get("entity_1"), r.get("entity_2"),
                    ))
                except Exception:
                    pass

    new_rows = [
        r for r in rows
        if (r.get("doc_id"), r.get("paragraph_id"),
            r.get("sentence_id"), r.get("entity_1"), r.get("entity_2"))
        not in existing_keys
    ]

    if not new_rows:
        print(f"[OK] Step 4: all {len(rows)} rows already scored, nothing to do.")
        return

    print(f"[INFO] Skipping {len(existing_keys)} already-scored rows, scoring {len(new_rows)} new rows.")

    from src.pipeline.nli_relation_extraction import NLIRelationScorer
    scorer = NLIRelationScorer(
        model_name=config.NLI_MODEL_NAME,
        threshold=config.NLI_THRESHOLD,
        batch_size=config.NLI_BATCH_SIZE,
    )

    # Cache space results by sentence text (sentence-level property)
    space_cache: dict[str, dict] = {}

    total = 0
    with config.FILE_COOCCURRENCE_NLI.open("a", encoding="utf-8") as out:
        for row in tqdm(new_rows, desc="NLI scoring"):
            premise = row.get("central_sent_text") or row.get("sent_text", "")
            e1 = row.get("entity_1", "")
            e2 = row.get("entity_2", "")

            # Relation scoring
            nli_results = scorer.score_pairs_batch([(premise, e1, e2)])
            nli = nli_results[0]

            # Space scoring — cached per sentence
            if premise not in space_cache:
                space_cache[premise] = scorer.classify_space(premise)
            space = space_cache[premise]

            out_row = dict(row)
            out_row["relation_type"]       = nli["relation_type"]
            out_row["confidence"]          = nli["confidence"]
            out_row["all_scores"]          = nli["all_scores"]
            out_row["th_space"]            = space["th_space"]
            out_row["th_space_confidence"] = space["th_space_confidence"]
            out_row["th_space_scores"]     = space["th_space_scores"]

            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            total += 1

    print(f"[OK] Wrote {total} NLI-scored rows → {config.FILE_COOCCURRENCE_NLI}")


if __name__ == "__main__":
    run()
