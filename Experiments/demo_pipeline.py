"""Pipeline demonstration — single sentence walkthrough.

Shows all four pipeline stages on one real sentence from the corpus:

  Step 1 — NER: entity extraction with span offsets
  Step 2 — Classification: helix assignment per entity
  Step 3 — Relation: NLI confidence scores for all 5 relation types
  Step 4 — Space: NLI confidence scores for all 4 TH spaces

Example sentence (CAN_2022):
  "Over the past 10 years, the Government of Canada has provided
   $51 million to support the IQC."

Run:
    python Experiments/demo_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Config ─────────────────────────────────────────────────────────────────────

SENTENCE = (
    "Over the past 10 years, the Government of Canada has provided "
    "$51 million to support the IQC."
)

# Entities as they appear in the sentence (pre-identified from pipeline output)
# In full pipeline these come from Flair NER + manual classification (Step 1 + 2)
ENTITIES = [
    {
        "mention":  "Government of Canada",
        "label":    "ORG",
        "helix":    "government",
        "category": "national government institutions",
        "actor_type": "institutional",
    },
    {
        "mention":  "IQC",
        "label":    "ORG",
        "helix":    "academia",
        "category": "public research organisations",
        "actor_type": "institutional",
    },
]

# Relation types (Ranga & Etzkowitz 2013)
RELATION_LABELS = [
    "technology_transfer",
    "collaborative_leadership",
    "substitution",
    "networking",
    "collaboration_conflict_moderation",
    "no_explicit_relation",
]

# NLI hypothesis templates for relation (one per type, simplified)
RELATION_HYPOTHESES = {
    "technology_transfer":
        "{e1} transfers research results, funding, or intellectual property to {e2} for application.",
    "collaborative_leadership":
        "{e1} leads and directly coordinates the activities of {e2} within a cross-sector initiative.",
    "substitution":
        "{e1} fills a gap left by the absence or weakness of {e2} in the innovation system.",
    "networking":
        "{e1} and {e2} are joint members of a consortium, platform, or research network.",
    "collaboration_conflict_moderation":
        "{e1} helps resolve tensions or conflicts of interest between institutional spheres involving {e2}.",
    "no_explicit_relation":
        "{e1} and {e2} have no explicit Triple Helix relationship in this sentence.",
}

# NLI hypothesis templates for TH space
SPACE_LABELS = [
    "knowledge_space",
    "innovation_space",
    "consensus_space",
    "public_space",
    "no_explicit_space",
]

SPACE_HYPOTHESES = {
    "knowledge_space":
        "This sentence describes knowledge generation, scientific collaboration, R&D resources, or education.",
    "innovation_space":
        "This sentence describes technology transfer, commercialisation, spin-offs, IP creation, or incubation.",
    "consensus_space":
        "This sentence describes governance, policy dialogue, strategy formation, or stakeholder coordination.",
    "public_space":
        "This sentence describes public engagement, ethics, equity, civil society involvement, or societal trust in technology.",
    "no_explicit_space":
        "This sentence does not describe any specific Triple Helix innovation space activity.",
}

HELIX_COLORS = {
    "government": "\033[94m",   # blue
    "academia":   "\033[92m",   # green
    "industry":   "\033[93m",   # yellow
    "intermediary": "\033[95m", # magenta
    "civil_society": "\033[96m",# cyan
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def _highlight(sentence: str, entities: list[dict]) -> str:
    """Colour-highlight entity mentions in sentence."""
    result = sentence
    # Sort by position to avoid offset issues
    for ent in sorted(entities, key=lambda e: sentence.lower().find(e["mention"].lower()), reverse=True):
        mention = ent["mention"]
        color   = HELIX_COLORS.get(ent["helix"], "")
        pos = sentence.lower().find(mention.lower())
        if pos == -1:
            continue
        actual = sentence[pos:pos+len(mention)]
        result = result.replace(actual, f"{color}{BOLD}{actual}{RESET}", 1)
    return result


def _bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def run_demo():
    from transformers import pipeline as hf_pipeline

    # Load NLI model
    print(f"\n{BOLD}Loading NLI model (cross-encoder/nli-deberta-v3-large)...{RESET}")
    nli = hf_pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-deberta-v3-large",
        device=0,  # use GPU if available; falls back to CPU
    )

    e1 = ENTITIES[0]["mention"]
    e2 = ENTITIES[1]["mention"]
    h1 = ENTITIES[0]["helix"]
    h2 = ENTITIES[1]["helix"]

    # ── Step 1: Input sentence ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{BOLD}STEP 1 — Input Sentence{RESET}")
    print(f"{'='*65}")
    print(f"\n  {SENTENCE}")
    print(f"\n  Source: CAN_2022 — Canada's National Quantum Strategy")

    # ── Step 2: NER + Classification ────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{BOLD}STEP 2 — Named Entity Recognition + Helix Classification{RESET}")
    print(f"{'='*65}")
    print(f"\n  {_highlight(SENTENCE, ENTITIES)}\n")
    for ent in ENTITIES:
        color = HELIX_COLORS.get(ent["helix"], "")
        print(f"  {color}{BOLD}[{ent['mention']}]{RESET}")
        print(f"    NER label:    {ent['label']}")
        print(f"    Helix:        {color}{ent['helix']}{RESET}")
        print(f"    Category:     {ent['category']}")
        print(f"    Actor type:   {ent['actor_type']}")
    print(f"\n  → Pair selected for relation extraction: "
          f"{HELIX_COLORS['government']}{BOLD}{e1}{RESET} ({h1}) "
          f"↔ {HELIX_COLORS['academia']}{BOLD}{e2}{RESET} ({h2})")

    # ── Step 3: Relation extraction ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{BOLD}STEP 3 — Relation Extraction (NLI){RESET}")
    print(f"{'='*65}")
    print(f"\n  Premise (entity-masked): \"[GOV] has provided $51 million to support [ACAD].\"\n")

    hypotheses = [
        RELATION_HYPOTHESES[lbl].format(e1=e1, e2=e2)
        for lbl in RELATION_LABELS
    ]
    result = nli(SENTENCE, hypotheses, multi_label=True)
    scores = dict(zip(result["labels"], result["scores"]))

    rows = [(lbl, scores.get(RELATION_HYPOTHESES[lbl].format(e1=e1, e2=e2), 0.0))
            for lbl in RELATION_LABELS]
    rows.sort(key=lambda x: -x[1])

    for i, (lbl, score) in enumerate(rows):
        marker = f"{BOLD}← CHOSEN{RESET}" if i == 0 else ""
        print(f"  {lbl:<40} {score:.3f}  {_bar(score)}  {marker}")

    winner_rel = rows[0][0]
    print(f"\n  → Assigned relation: {BOLD}{winner_rel}{RESET} (confidence: {rows[0][1]:.3f})")

    # ── Step 4: TH Space ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{BOLD}STEP 4 — TH Space Classification (NLI){RESET}")
    print(f"{'='*65}")
    print(f"\n  Context sentence (full): \"{SENTENCE}\"\n")

    space_hyps = [SPACE_HYPOTHESES[lbl] for lbl in SPACE_LABELS]
    result_s   = nli(SENTENCE, space_hyps, multi_label=True)
    space_scores = dict(zip(result_s["labels"], result_s["scores"]))

    space_rows = [(lbl, space_scores.get(SPACE_HYPOTHESES[lbl], 0.0)) for lbl in SPACE_LABELS]
    space_rows.sort(key=lambda x: -x[1])

    for i, (lbl, score) in enumerate(space_rows):
        marker = f"{BOLD}← CHOSEN{RESET}" if i == 0 else ""
        print(f"  {lbl:<40} {score:.3f}  {_bar(score)}  {marker}")

    winner_space = space_rows[0][0]
    print(f"\n  → Assigned space: {BOLD}{winner_space}{RESET} (confidence: {space_rows[0][1]:.3f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{'='*65}")
    print(f"\n  Sentence:  \"{SENTENCE[:70]}...\"")
    print(f"  Entity 1:  {e1}  [{h1}]")
    print(f"  Entity 2:  {e2}  [{h2}]")
    print(f"  Relation:  {BOLD}{winner_rel}{RESET}")
    print(f"  TH Space:  {BOLD}{winner_space}{RESET}")
    print()


if __name__ == "__main__":
    run_demo()
