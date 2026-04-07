"""R8 — GLiREL: zero-shot relation extraction with natural language label descriptions.

Paper: Zaratiana, U., et al. (2024). GLiREL: Generalist and Lightweight Model for
       Zero-Shot Relation Extraction. arXiv 2024.

Method (from paper):
  - Architecture: GLiNER-based span encoder + label encoder
  - Entity spans are encoded with a token-level span representation
  - Relation labels provided as free-text descriptions → encoded by the same model
  - Score = dot product between (span1, span2) representation and label embedding
  - Zero-shot: no fine-tuning on this dataset — label descriptions drive inference
  - Trained on a mixture of RE datasets with diverse label sets for generalisation

Key difference from NLI-Sainz: dedicated RE architecture vs. generic NLI model.
Key difference from GPT-RE: local inference, no API, no in-context demos.

Model: jackboyla/glirel-large-v0  (GLiNER-large fine-tuned on RE datasets)
Threshold: 0.05  (low to capture rare relation types)

Label descriptions (natural language, matched to R&E 2013 relation types):
  technology_transfer             → "technology transfer or knowledge exchange"
  collaboration_conflict_moderation → "collaboration or conflict moderation"
  collaborative_leadership        → "collaborative leadership or coordination"
  substitution                    → "substitution where one actor fills the role of another"
  networking                      → "networking or partnership formation"
  no_explicit_relation            → fallback when no label exceeds threshold

Run:
    pip install glirel
    python Experiments/Relation/glirel/run.py

Requires:
    pip install glirel torch scikit-learn
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    RELATION_LABELS,
    load_relation_eval,
    save_outputs,
)

MODEL_NAME = "jackboyla/glirel-large-v0"
THRESHOLD  = 0.05

GLIREL_LABELS = [
    "technology transfer or knowledge exchange",
    "collaboration or conflict moderation",
    "collaborative leadership or coordination",
    "substitution where one actor fills the role of another",
    "networking or partnership formation",
]

LABEL_MAP = {
    "technology transfer or knowledge exchange":            "technology_transfer",
    "collaboration or conflict moderation":                  "collaboration_conflict_moderation",
    "collaborative leadership or coordination":              "collaborative_leadership",
    "substitution where one actor fills the role of another": "substitution",
    "networking or partnership formation":                   "networking",
}

MAX_TOKENS = 512


def _tokenize(text: str) -> list[str]:
    return [m.group() for m in re.finditer(r'\w+(?:[-_]\w+)*|\S', text)]


def _find_span(tokens: list[str], entity: str) -> tuple[int | None, int | None]:
    e_words = re.findall(r'\w+(?:[-_]\w+)*|\S', entity.lower())
    n = len(e_words)
    if not n:
        return None, None
    tok_lower = [t.lower() for t in tokens]
    for i in range(len(tok_lower) - n + 1):
        if tok_lower[i:i + n] == e_words:
            return i, i + n - 1
    return None, None


def _trim_window(tokens, s1, e1, s2, e2):
    if len(tokens) <= MAX_TOKENS:
        return tokens, s1, e1, s2, e2
    span_start = min(s1, s2)
    span_end   = max(e1, e2)
    span_len   = span_end - span_start + 1
    budget     = MAX_TOKENS - span_len
    left_ctx   = budget // 2
    new_start  = max(0, span_start - left_ctx)
    new_end    = min(len(tokens) - 1, span_end + (budget - left_ctx))
    offset     = new_start
    return tokens[new_start:new_end + 1], s1 - offset, e1 - offset, s2 - offset, e2 - offset


def predict(entries: list[dict]) -> tuple[list[str], list[str]]:
    import torch
    from glirel import GLiREL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GLiREL ({MODEL_NAME}) on {device}...")
    model = GLiREL.from_pretrained(MODEL_NAME, force_download=False, use_safetensors=True)
    model = model.to(device)

    true_labels, pred_labels = [], []
    skipped = 0

    for entry in entries:
        text = entry.get("sentence") or entry.get("central_sent_text", "")
        e1   = entry.get("entity_1", "")
        e2   = entry.get("entity_2", "")

        tokens = _tokenize(text)
        s1, et1 = _find_span(tokens, e1)
        s2, et2 = _find_span(tokens, e2)

        if s1 is None or s2 is None:
            # Entity not found literally in text — fall back to no_explicit_relation
            skipped += 1
            pred = "no_explicit_relation"
        else:
            tokens, s1, et1, s2, et2 = _trim_window(tokens, s1, et1, s2, et2)
            ner = [[s1, et1, "ORG"], [s2, et2, "ORG"]]
            try:
                relations = model.predict_relations(
                    tokens,
                    GLIREL_LABELS,
                    threshold=THRESHOLD,
                    ner=ner,
                )
            except Exception:
                relations = []

            if relations:
                best = max(relations, key=lambda r: r.get("score", 0))
                pred = LABEL_MAP.get(best["label"], "no_explicit_relation")
            else:
                pred = "no_explicit_relation"

        true_labels.append(entry["true_relation"])
        pred_labels.append(pred)

    if skipped:
        print(f"[WARN] {skipped} entries skipped (entity not found in text) → no_explicit_relation")
    return true_labels, pred_labels


def main() -> None:
    entries = load_relation_eval()
    print(f"Loaded {len(entries)} labeled relation examples")
    print(f"Model: {MODEL_NAME}  threshold: {THRESHOLD}")

    true_labels, pred_labels = predict(entries)

    predictions = [
        {
            "id":   i,
            "true": t,
            "pred": p,
            "text": entries[i].get("sentence") or entries[i].get("central_sent_text", ""),
        }
        for i, (t, p) in enumerate(zip(true_labels, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, true_labels, pred_labels, RELATION_LABELS)


if __name__ == "__main__":
    main()
