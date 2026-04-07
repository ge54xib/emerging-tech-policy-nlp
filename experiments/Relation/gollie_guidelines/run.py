"""R4 — GoLLIE: annotation guidelines as Python code for zero-shot IE.

Paper: Sainz, O., et al. (2024). GoLLIE: Annotation Guidelines improve
       Zero-Shot Information Extraction. ICLR 2024. (arXiv:2310.03668)

Method (from paper):
  - Encode annotation guidelines as Python @dataclass definitions with docstrings
  - Feed the class definitions + sentence into a code LLM as a continuation prompt
  - The model "completes" the Python list: result = [RelationClass(arg1=..., arg2=...)]
  - Parse the output to extract class name → relation label
  - Docstrings are written from the codebook (Ranga & Etzkowitz 2013 language)

Key design (paper-faithful):
  - Each relation is a @dataclass with typed fields and a docstring as the guideline
  - The prompt ends with `result: List = [` so the model must continue with valid Python
  - Parse: find instantiated class names in the output; map to relation labels
  - If the model instantiates NoExplicitRelation or nothing matches → no_explicit_relation

Run:
    export OPENAI_API_KEY=...
    python Experiments/Relation/gollie_guidelines/run.py

Requires:
    pip install openai scikit-learn
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    RELATION_LABELS,
    load_relation_eval,
    save_outputs,
)

MODEL = "gpt-4o"

# ── Annotation guidelines as Python code (GoLLIE format) ─────────────────────
# Docstrings combine codebook language with Ranga & Etzkowitz 2013 wording.
GUIDELINES = '''\
from dataclasses import dataclass
from typing import List

@dataclass
class TechnologyTransfer:
    """Technology transfer via market or non-market interactions is the core activity
    of an innovation system. Universities, industry and government generate and transfer
    technology, knowledge, and intellectual property.
    Signal words: transfer, license, patent, spin-off, commercialise, deliver, deploy."""
    source: str      # actor transferring technology or knowledge
    recipient: str   # actor receiving technology or knowledge

@dataclass
class CollaborativeLeadership:
    """An innovation organizer — individual or institutional — exercises convening power
    to bring the leadership of the spheres together. They coordinate a mix of top-down
    and bottom-up processes, bridge gaps, and generate consensus.
    Signal words: led, organized, coordinated, convened, established, launched, chaired."""
    organizer: str   # actor that exercises convening power or leads
    partner: str     # actor that is brought together or participates

@dataclass
class Substitution:
    """Institutional spheres fill gaps that emerge when another sphere is weak. One actor
    takes the role of the other — e.g. government providing venture capital (a traditional
    industry task), or universities engaging in firm formation (a traditional industry role).
    Signal words: fill, provide, take on, act as, step in, replace, substitute."""
    filler: str      # actor filling the gap of the weaker sphere
    replaced: str    # actor whose role is being filled

@dataclass
class CollaborationConflictModeration:
    """Triadic entities have a higher potential for turning tension and conflict of interest
    into convergence and confluence of interest than dyadic relationships. Task conflict
    (constructive) is moderated into win-win outcomes.
    Signal words: moderate, mediate, resolve, bridge, align, tension, conflict, dialogue."""
    moderator: str   # actor doing the moderating
    parties: str     # actors whose conflict or tension is being moderated

@dataclass
class Networking:
    """Formal and informal networks at national, regional and international level are a
    manifestation of the collective nature of science, technology and innovation.
    Signal words: network, consortium, partnership, alliance, cooperation, joint, collaborate."""
    actor1: str
    actor2: str

@dataclass
class NoExplicitRelation:
    """No Triple Helix relationship between the two actors is described in this sentence."""
    actor1: str
    actor2: str
'''

# Map class names → relation labels
CLASS_TO_LABEL = {
    "TechnologyTransfer":             "technology_transfer",
    "CollaborativeLeadership":        "collaborative_leadership",
    "Substitution":                   "substitution",
    "CollaborationConflictModeration":"collaboration_conflict_moderation",
    "Networking":                     "networking",
    "NoExplicitRelation":             "no_explicit_relation",
}


def _build_prompt(text: str, e1: str, h1: str, e2: str, h2: str) -> str:
    return (
        f"{GUIDELINES}\n"
        f'# Text: "{text}"\n'
        f"# Entities of interest: {e1} ({h1}), {e2} ({h2})\n"
        f"# Extract all relations involving these two entities:\n"
        f"result: List = ["
    )


def _parse_output(completion: str) -> str:
    """Extract the first instantiated class name from the model output."""
    # The model should output something like: TechnologyTransfer(source="X", recipient="Y")]
    class_pattern = re.compile(r"\b(" + "|".join(CLASS_TO_LABEL.keys()) + r")\s*\(")
    match = class_pattern.search(completion)
    if match:
        class_name = match.group(1)
        return CLASS_TO_LABEL.get(class_name, "no_explicit_relation")
    return "no_explicit_relation"


def predict(entries: list[dict]) -> tuple[list[str], list[str], list[str]]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    true_labels, pred_labels, raw_outputs = [], [], []

    for i, entry in enumerate(entries):
        e1   = entry["entity_1"]
        h1   = entry["h1"]
        e2   = entry["entity_2"]
        h2   = entry["h2"]
        text = entry.get("sentence") or entry.get("central_sent_text", "")

        prompt = _build_prompt(text, e1, h1, e2, h2)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Python code completion engine for information extraction. "
                        "Complete the Python list assignment `result: List = [` by instantiating "
                        "exactly one of the provided dataclass types. Output only valid Python "
                        "constructor syntax — no explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        completion = response.choices[0].message.content or ""
        pred = _parse_output(completion)

        true_labels.append(entry["true_relation"])
        pred_labels.append(pred)
        raw_outputs.append(completion)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(entries)} done")

    return true_labels, pred_labels, raw_outputs


def main() -> None:
    entries = load_relation_eval()
    print(f"Loaded {len(entries)} labeled relation examples")
    print(f"Model: {MODEL}")

    true_labels, pred_labels, raw_outputs = predict(entries)

    predictions = [
        {
            "id":         i,
            "true":       t,
            "pred":       p,
            "text":       entries[i].get("sentence") or entries[i].get("central_sent_text", ""),
            "raw_output": raw_outputs[i],
        }
        for i, (t, p) in enumerate(zip(true_labels, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, true_labels, pred_labels, RELATION_LABELS)


if __name__ == "__main__":
    main()
