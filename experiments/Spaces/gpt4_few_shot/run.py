"""S3 — GPT-4 few-shot in-context learning for space classification.

Paper: Brown, T.B., et al. (2020). Language Models are Few-Shot Learners.
       NeurIPS 2020.

Method (from paper):
  - In-context learning: prepend k labeled examples to the prompt
  - No weight updates — all learning happens in context
  - Key design choices: consistent example format, all classes covered, order matters
  - k=3 per class (12 total), preferring confidence=="high" examples

Data split:
  - Demo pool: first 50 entries of annotation_spaces.json with true_space filled
  - Eval set:  final 50 entries of annotation_spaces.json

Space definitions in system prompt (Ranga & Etzkowitz 2013 + public_space extension):
  knowledge_space — knowledge generation, diffusion and use; R&D; strengthening the knowledge base
  innovation_space — hybrid orgs; tech transfer, IP creation; developing innovative firms
  consensus_space — blue-sky thinking; stakeholder dialogue; governance; proposals for knowledge-based regime
  public_space — civil society; ethics; equity; democratic oversight; public trust [QH extension]

Classify the ACTIVITY described, not the actors.

Run:
    export OPENAI_API_KEY=...
    python Experiments/Spaces/gpt4_few_shot/run.py

Requires:
    pip install openai scikit-learn
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    SPACE_LABELS,
    load_spaces_eval,
    save_outputs,
)

MODEL       = "gpt-4o"
K_PER_CLASS = 3   # demos per class = 12 total (Brown et al. 2020)

SYSTEM_PROMPT = """\
You classify sentences from national quantum technology policy documents into Triple \
Helix innovation spaces. Classify the ACTIVITY described in the sentence, not the actors.

Space definitions:
- knowledge_space: competencies of knowledge generation, diffusion and use; creating and \
developing knowledge resources to strengthen the knowledge base and avoid duplication of \
research efforts (Ranga & Etzkowitz 2013)
- innovation_space: competencies of hybrid organizations and entrepreneurial individuals; \
developing innovative firms, creating intellectual and entrepreneurial potential, transferring \
technology and creating IP (Ranga & Etzkowitz 2013)
- consensus_space: bringing together Triple Helix components for blue-sky thinking, discussing \
proposals for advancement toward a knowledge-based regime; turning diverging interests into \
convergence through governance and dialogue (Ranga & Etzkowitz 2013)
- public_space: civil society, media, and culture as innovation actors; public engagement, trust, \
and awareness; ethics, equity, and democratic oversight; innovation culture and cultural values; \
creative industries and the creative class; gender diversity in STEM; public legitimation of \
knowledge and technology policy; open-source movements and user communities — fourth helix \
(Carayannis & Campbell 2009; Quadruple Helix extension of R&E 2013)

Output only the label, nothing else."""


def _select_demos(pool: list[dict]) -> list[dict]:
    """Select k=3 per class, preferring confidence=='high'."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for e in pool:
        space = e.get("true_space", "")
        if space in SPACE_LABELS:
            by_class[space].append(e)

    selected = []
    for label in SPACE_LABELS:
        candidates = sorted(
            by_class[label],
            key=lambda x: 0 if x.get("confidence") == "high" else 1
        )
        selected.extend(candidates[:K_PER_CLASS])
    return selected


def _build_messages(demos: list[dict], test_sentence: str) -> list[dict]:
    user_parts = []
    for demo in demos:
        user_parts.append(f"Sentence: {demo['sentence']}\nSpace: {demo['true_space']}")
    user_parts.append(f"Sentence: {test_sentence}\nSpace:")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "\n\n".join(user_parts)},
    ]


def _parse_label(text: str) -> str:
    text = text.strip().lower()
    for lbl in SPACE_LABELS:
        if text.startswith(lbl) or text == lbl:
            return lbl
    for lbl in SPACE_LABELS:
        if lbl in text:
            return lbl
    return "knowledge_space"  # safest fallback for spaces


def main() -> None:
    from openai import OpenAI

    all_entries = load_spaces_eval()
    print(f"Loaded {len(all_entries)} labeled space examples")

    # Split: first 50 = demo pool, last 50 = eval set
    demo_pool    = all_entries[:50]
    eval_entries = all_entries[50:]
    print(f"Demo pool: {len(demo_pool)}  |  Eval set: {len(eval_entries)}")

    demos = _select_demos(demo_pool)
    print(f"Selected {len(demos)} demos ({K_PER_CLASS} per class)")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    eval_true, pred_labels = [], []

    for i, entry in enumerate(eval_entries):
        messages = _build_messages(demos, entry["sentence"])
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=32,
            temperature=0.0,
        )
        pred = _parse_label(response.choices[0].message.content or "")
        eval_true.append(entry["true_space"])
        pred_labels.append(pred)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(eval_entries)} done")

    predictions = [
        {"id": i, "true": t, "pred": p, "text": eval_entries[i]["sentence"]}
        for i, (t, p) in enumerate(zip(eval_true, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, eval_true, pred_labels, SPACE_LABELS)


if __name__ == "__main__":
    main()
