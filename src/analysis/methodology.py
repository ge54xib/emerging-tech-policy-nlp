"""Methodology diagnostics written as thesis deliverable JSON."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from statistics import mean

from src import config, utils
from src.analysis._helpers import utc_now_iso, write_json
from src.utils import to_int, to_str


def run() -> None:
    print(">>> Analysis: Methodology Statistics")

    summary: dict = {
        "title": "Methodology Summary",
        "generated_utc": utc_now_iso(),
        "inputs": {
            "step1_ner": str(config.STEP1_NER_PATH),
            "step2_labels": str(config.STEP2_MANUAL_LABELS_PATH),
            "step3_cooccurrence": str(config.FILE_COOCCURRENCE),
        },
    }

    # Step 1
    if not config.STEP1_NER_PATH.exists():
        raise FileNotFoundError(f"Missing Step1 file: {config.STEP1_NER_PATH}")
    ner_rows = list(utils.load_jsonl(config.STEP1_NER_PATH))
    label_counts = Counter(to_str(row.get("label", "")).upper() for row in ner_rows)
    mentions_per_doc: dict[str, int] = defaultdict(int)
    entities_per_doc: dict[str, set[int]] = defaultdict(set)
    for row in ner_rows:
        doc_name = to_str(row.get("doc_name", "")).strip()
        if not doc_name:
            continue
        mentions_per_doc[doc_name] += 1
        entity_id = to_int(row.get("entity_id", 0), 0)
        if entity_id:
            entities_per_doc[doc_name].add(entity_id)

    summary["step1_ner"] = {
        "rows": len(ner_rows),
        "label_counts": dict(sorted(label_counts.items(), key=lambda kv: kv[0])),
        "documents": len(mentions_per_doc),
        "avg_mentions_per_doc": mean(mentions_per_doc.values()) if mentions_per_doc else 0.0,
        "avg_entities_per_doc": (
            mean(len(v) for v in entities_per_doc.values()) if entities_per_doc else 0.0
        ),
    }

    # Step 2
    if config.STEP2_MANUAL_LABELS_PATH.exists():
        payload = json.loads(config.STEP2_MANUAL_LABELS_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            status_counts = Counter(to_str(row.get("status", "")).strip().lower() or "<empty>" for row in payload)
            helix_counts = Counter(
                to_str(row.get("level_5_helix", "")).strip().lower().replace(" ", "_") or "<empty>"
                for row in payload
            )
            summary["step2_labels"] = {
                "rows": len(payload),
                "status_counts": dict(
                    sorted(status_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                ),
                "level_5_helix_counts": dict(
                    sorted(helix_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                ),
            }

    # Step 4
    if config.FILE_COOCCURRENCE.exists():
        pair_counts = Counter()
        country_counts = Counter()
        total_rows = 0
        for row in utils.load_jsonl(config.FILE_COOCCURRENCE):
            total_rows += 1
            pair = to_str(row.get("pair", "")).strip()
            if not pair:
                h1 = to_str(row.get("h1", "")).strip()
                h2 = to_str(row.get("h2", "")).strip()
                pair = "–".join(sorted([h1, h2]))
            pair_counts[pair] += 1
            country = to_str(row.get("country", "")).strip().upper() or "UNK"
            country_counts[country] += 1

        summary["step3_cooccurrence"] = {
            "rows": total_rows,
            "country_counts": dict(
                sorted(country_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "top_helix_pairs": [
                {"pair": pair, "count": count}
                for pair, count in pair_counts.most_common(20)
            ],
        }

    write_json(config.ANALYSIS_METHODOLOGY_PATH, summary)
    print(f"[OK] Wrote: {config.ANALYSIS_METHODOLOGY_PATH}")


if __name__ == "__main__":
    run()
