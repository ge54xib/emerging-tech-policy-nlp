"""R3 — Self-consistency majority voting for relation extraction.

Paper: Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought
       Reasoning in Language Models. ICLR 2023.

Method (from paper):
  - Sample k diverse reasoning paths from the same CoT prompt at temperature > 0
  - Take majority vote over final answers from all k paths
  - The correct answer is more likely to be consistent across diverse reasoning paths
  - Wang et al. use k=40; we use k=10 for cost (same method, reduced k)
  - Same prompt as R2 (claude_cot) — enables direct comparison

Key implementation details:
  - temperature=0.7 (same as paper ablation)
  - k=10 sampled paths per example
  - Majority vote; ties broken by first response
  - Stores agreement rate per example (fraction of k paths that agree on final label)

Run:
    export ANTHROPIC_API_KEY=...
    python Experiments/Relation/self_consistency/run.py

Requires:
    pip install anthropic scikit-learn
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    RELATION_LABELS,
    load_relation_eval,
    mark_entities_typed,
    save_outputs,
)

# Reuse the exact same prompt as claude_cot (Wang et al. 2023 requires same prompt)
MODEL = "claude-3-5-haiku-20241022"   # cheaper for k=10 samples per example
K = 10
TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are an expert in Triple Helix innovation theory (Ranga & Etzkowitz 2013). "
    "Classify the relation between two institutional actors in a national quantum "
    "technology policy document. Respond with step-by-step reasoning followed by "
    "exactly one label on the final line."
)

USER_TEMPLATE = """\
Sentence: {sentence}
Actors: <{h1}>{e1}</{h1}> and <{h2}>{e2}</{h2}>

Think step by step:
1. What interaction is described between these two actors?
2. Is there any market or non-market transfer of technology, knowledge, or IP?
3. Does one actor fill a gap because the other sphere is weak — taking the role of the other?
4. Does one actor exercise convening power, bringing together the other and coordinating processes?
5. Is there tension, conflict, or an effort to turn diverging interests into convergence?
6. Is there a formal or informal network formed as a manifestation of collective science and innovation?
7. Given the above reasoning, which ONE label fits best?

Available labels (apply in priority order — higher labels override lower when both apply):
- substitution: one sphere fills gaps that emerge when another is weak; one actor takes the role of the other
- technology_transfer: market or non-market transfer of technology or knowledge; core activity of innovation systems
- collaborative_leadership: an innovation organizer bridges spheres, coordinates top-down and bottom-up processes, exercises convening power
- networking: formal or informal network; collective nature of science, technology and innovation
- collaboration_conflict_moderation: turning tension and conflict of interest into convergence of interest; triadic moderation
- no_explicit_relation: no Triple Helix relationship is described in this sentence

Reasoning:
Final answer (label only):"""


def _parse_label(text: str) -> str:
    lines = text.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.lower().startswith("final answer"):
            candidate = re.sub(r"^final answer[:\s]*", "", line, flags=re.IGNORECASE).strip().strip("*")
            if candidate in RELATION_LABELS:
                return candidate
            for lbl in RELATION_LABELS:
                if lbl in candidate:
                    return lbl
    for line in reversed(lines):
        line = line.strip().strip("*")
        if line in RELATION_LABELS:
            return line
    return "no_explicit_relation"


def _majority_vote(labels: list[str]) -> tuple[str, float]:
    """Return (most_common_label, agreement_rate)."""
    counts = Counter(labels)
    winner, count = counts.most_common(1)[0]
    return winner, count / len(labels)


def predict(entries: list[dict]) -> tuple[list[str], list[str]]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    true_labels, pred_labels = [], []
    all_predictions = []

    for i, entry in enumerate(entries):
        e1   = entry["entity_1"]
        h1   = entry["h1"]
        e2   = entry["entity_2"]
        h2   = entry["h2"]
        text = entry.get("sentence") or entry.get("central_sent_text", "")
        sentence = mark_entities_typed(text, e1, h1, e2, h2)
        prompt = USER_TEMPLATE.format(sentence=sentence, h1=h1, e1=e1, h2=h2, e2=e2)

        # Sample k=10 reasoning paths
        sampled_labels = []
        reasoning_paths = []
        for _ in range(K):
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            resp_text = response.content[0].text
            lbl = _parse_label(resp_text)
            sampled_labels.append(lbl)
            reasoning_paths.append(resp_text)

        pred, agreement = _majority_vote(sampled_labels)
        true_labels.append(entry["true_relation"])
        pred_labels.append(pred)
        all_predictions.append({
            "id":              i,
            "true":            entry["true_relation"],
            "pred":            pred,
            "text":            text,
            "agreement":       round(agreement, 3),
            "sampled_labels":  sampled_labels,
            "reasoning_paths": reasoning_paths,
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(entries)} done")

    return true_labels, pred_labels, all_predictions


def main() -> None:
    entries = load_relation_eval()
    print(f"Loaded {len(entries)} labeled relation examples")
    print(f"Model: {MODEL}  k={K}  temperature={TEMPERATURE}")

    true_labels, pred_labels, all_preds = predict(entries)

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, all_preds, true_labels, pred_labels, RELATION_LABELS)


if __name__ == "__main__":
    main()
