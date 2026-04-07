"""R2 — Chain-of-Thought prompting for relation extraction.

Paper: Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in
       Large Language Models. NeurIPS 2022.

Method (from paper):
  - Zero-shot CoT: elicit a step-by-step reasoning chain BEFORE the final answer
  - Each reasoning step maps to one decision criterion from the codebook
  - The step order mirrors the priority rule: substitution > technology_transfer >
    collaborative_leadership > networking > collaboration_conflict_moderation
  - Final answer is extracted from last line "Final answer: <label>"

Prompt uses Ranga & Etzkowitz (2013) concepts in the step descriptions:
  - "market or non-market transfer of technology"
  - "fills a gap because the other sphere is weak — taking the role of the other"
  - "exercises convening power, coordinating top-down and bottom-up processes"
  - "turns diverging interests into convergence"

Run:
    export ANTHROPIC_API_KEY=...
    python Experiments/Relation/claude_cot/run.py

Requires:
    pip install anthropic scikit-learn
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    RELATION_LABELS,
    load_relation_eval,
    mark_entities_typed,
    save_outputs,
)

MODEL = "claude-3-5-sonnet-20241022"

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


def _parse_label(response_text: str) -> str:
    """Extract label from last 'Final answer:' line."""
    lines = response_text.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.lower().startswith("final answer"):
            candidate = re.sub(r"^final answer[:\s]*", "", line, flags=re.IGNORECASE).strip()
            candidate = candidate.strip("*").strip()
            if candidate in RELATION_LABELS:
                return candidate
            # fuzzy match — check if any label is contained
            for lbl in RELATION_LABELS:
                if lbl in candidate:
                    return lbl
    # fallback: check last non-empty line
    for line in reversed(lines):
        line = line.strip().strip("*")
        if line in RELATION_LABELS:
            return line
    return "no_explicit_relation"


def predict(entries: list[dict]) -> tuple[list[str], list[str], list[str]]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    true_labels, pred_labels, reasoning_traces = [], [], []

    for i, entry in enumerate(entries):
        e1   = entry["entity_1"]
        h1   = entry["h1"]
        e2   = entry["entity_2"]
        h2   = entry["h2"]
        text = entry.get("sentence") or entry.get("central_sent_text", "")

        # Mark entities in the sentence for the prompt
        sentence = mark_entities_typed(text, e1, h1, e2, h2)

        prompt = USER_TEMPLATE.format(sentence=sentence, h1=h1, e1=e1, h2=h2, e2=e2)

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text
        pred = _parse_label(response_text)

        true_labels.append(entry["true_relation"])
        pred_labels.append(pred)
        reasoning_traces.append(response_text)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(entries)} done")

    return true_labels, pred_labels, reasoning_traces


def main() -> None:
    entries = load_relation_eval()
    print(f"Loaded {len(entries)} labeled relation examples")
    print(f"Model: {MODEL}")

    true_labels, pred_labels, traces = predict(entries)

    predictions = [
        {
            "id":        i,
            "true":      t,
            "pred":      p,
            "text":      entries[i].get("sentence") or entries[i].get("central_sent_text", ""),
            "reasoning": traces[i],
        }
        for i, (t, p) in enumerate(zip(true_labels, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, true_labels, pred_labels, RELATION_LABELS)


if __name__ == "__main__":
    main()
