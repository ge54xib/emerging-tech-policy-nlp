"""GLiREL test on all co-occurrence pairs from the pipeline.

Install: pip install glirel
Run:     python experiments/glirel/test_glirel.py
"""

import json
import re
from collections import Counter
from pathlib import Path

COOCCURRENCE_FILE = Path(__file__).parent.parent.parent / "data/processed/step3/cooccurrence.jsonl"
OUTPUT_FILE = Path(__file__).parent / "glirel_results.jsonl"

RELATION_LABELS = [
    "technology transfer or knowledge exchange",
    "collaboration or conflict moderation",
    "collaborative leadership or coordination",
    "substitution where one actor fills the role of another",
    "networking or partnership formation",
]

LABEL_MAP = {
    "technology transfer or knowledge exchange": "technology_transfer",
    "collaboration or conflict moderation": "collaboration_conflict_moderation",
    "collaborative leadership or coordination": "collaborative_leadership",
    "substitution where one actor fills the role of another": "substitution",
    "networking or partnership formation": "networking",
}

THRESHOLD = 0.05


def _tokenize(text: str):
    """Tokenize with the same regex GLiREL uses internally."""
    tokens = []
    for m in re.finditer(r'\w+(?:[-_]\w+)*|\S', text):
        tokens.append(m.group())
    return tokens


def _find_span(tokens: list, entity_text: str):
    """Return (start_tok, end_tok) for entity in tokens (case-insensitive)."""
    e_words = re.findall(r'\w+(?:[-_]\w+)*|\S', entity_text.lower())
    n = len(e_words)
    if n == 0:
        return None, None
    tok_lower = [t.lower() for t in tokens]
    for i in range(len(tok_lower) - n + 1):
        if tok_lower[i:i + n] == e_words:
            return i, i + n - 1
    return None, None


MAX_TOKENS = 512

def _trim_to_window(tokens: list, s1: int, e1t: int, s2: int, e2t: int):
    """If tokens exceed MAX_TOKENS, trim the window to be centered around
    both entity spans so neither gets cut off."""
    if len(tokens) <= MAX_TOKENS:
        return tokens, s1, e1t, s2, e2t

    # Span covering both entities
    span_start = min(s1, s2)
    span_end = max(e1t, e2t)
    span_len = span_end - span_start + 1

    # Budget of context to distribute around the span
    budget = MAX_TOKENS - span_len
    left_ctx = budget // 2
    right_ctx = budget - left_ctx

    new_start = max(0, span_start - left_ctx)
    new_end = min(len(tokens) - 1, span_end + right_ctx)

    # Shift entity indices by the offset
    offset = new_start
    return (
        tokens[new_start:new_end + 1],
        s1 - offset, e1t - offset,
        s2 - offset, e2t - offset,
    )


def main():
    from glirel import GLiREL

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GLiREL model on {device}...")
    model = GLiREL.from_pretrained(
        "jackboyla/glirel-large-v0", force_download=False, use_safetensors=True
    )
    model = model.to(device)

    rows = []
    with COOCCURRENCE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} co-occurrence pairs")

    relation_counter = Counter()
    written = 0
    skipped_no_span = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            if i % 500 == 0:
                print(f"  [{i}/{len(rows)}] written={written} skipped={skipped_no_span}...")

            text = row.get("sent_text", "")
            e1 = row.get("entity_1", "")
            e2 = row.get("entity_2", "")

            if not text or not e1 or not e2:
                continue

            tokens = _tokenize(text)
            s1, e1t = _find_span(tokens, e1)
            s2, e2t = _find_span(tokens, e2)

            if s1 is None or s2 is None:
                # At least one entity not found literally in the text — skip
                skipped_no_span += 1
                relation_type = "entity_not_in_text"
                confidence = 0.0
            else:
                tokens, s1, e1t, s2, e2t = _trim_to_window(tokens, s1, e1t, s2, e2t)
                ner = [[s1, e1t, "ORG"], [s2, e2t, "ORG"]]
                try:
                    relations = model.predict_relations(
                        tokens,  # pass token list directly (GLiREL accepts pre-tokenized)
                        RELATION_LABELS,
                        threshold=THRESHOLD,
                        ner=ner,
                    )
                except Exception:
                    relations = []

                if relations:
                    best = max(relations, key=lambda r: r.get("score", 0))
                    relation_type = LABEL_MAP.get(best["label"], best["label"])
                    confidence = round(best.get("score", 0), 4)
                else:
                    relation_type = "no_explicit_relation"
                    confidence = 0.0

            relation_counter[relation_type] += 1

            result = {
                "doc_id": row.get("doc_id"),
                "paragraph_id": row.get("paragraph_id"),
                "sentence_id": row.get("sentence_id"),
                "entity_1": e1,
                "h1": row.get("h1"),
                "entity_2": e2,
                "h2": row.get("h2"),
                "sent_text": text,
                "glirel_relation": relation_type,
                "glirel_confidence": confidence,
                "nli_relation": row.get("relation_type"),
                "nli_confidence": row.get("confidence"),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone. {written} results written to {OUTPUT_FILE}")
    print(f"Skipped (entity not in text): {skipped_no_span}")
    print("\nGLiREL relation distribution:")
    for k, v in relation_counter.most_common():
        pct = 100 * v / written if written else 0
        print(f"  {k}: {v} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
