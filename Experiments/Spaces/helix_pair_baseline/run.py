"""S_baseline — Static helix-pair → TH space mapping (theoretical baseline).

No model required. Assigns space purely from the (h1, h2) helix pair using
the static TH_SPACE_MAP from Ranga & Etzkowitz (2013):

  knowledge_space  — academia–academia, academia–government
  innovation_space — academia–industry, academia–intermediary, industry–intermediary
  consensus_space  — government–industry, government–intermediary
  public_space     — any pair involving civil_society

This is the same mapping used in RQ2/RQ3 and serves as a deterministic
upper-bound reference for what can be achieved without any classifier.

Run:
    python Experiments/Spaces/helix_pair_baseline/run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    SPACE_LABELS,
    load_spaces_eval,
    save_outputs,
)

SPACE_LABELS = ["knowledge_space", "innovation_space", "consensus_space", "public_space", "no_explicit_space"]

PAIR_TO_SPACE = {
    frozenset({"academia", "academia"}):      "knowledge_space",
    frozenset({"academia", "government"}):    "knowledge_space",
    frozenset({"academia", "industry"}):      "innovation_space",
    frozenset({"academia", "intermediary"}):  "innovation_space",
    frozenset({"industry", "intermediary"}):  "innovation_space",
    frozenset({"government", "industry"}):    "consensus_space",
    frozenset({"government", "intermediary"}):"consensus_space",
}


def _predict_space(entities: list[dict]) -> str:
    helices = {e.get("helix", "") for e in entities if e.get("helix")}
    if "civil_society" in helices:
        return "public_space"
    for h1 in helices:
        for h2 in helices:
            space = PAIR_TO_SPACE.get(frozenset({h1, h2}))
            if space:
                return space
    return "no_explicit_space"  # no matching pair found


def main() -> None:
    eval_entries = load_spaces_eval()
    print(f"Loaded {len(eval_entries)} labeled space examples")

    true_labels, pred_labels = [], []

    for entry in eval_entries:
        entities = entry.get("entities", [])
        pred = _predict_space(entities)
        true_labels.append(entry["true_space"])
        pred_labels.append(pred)

    predictions = [
        {"id": i, "true": t, "pred": p, "text": eval_entries[i]["sentence"]}
        for i, (t, p) in enumerate(zip(true_labels, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, true_labels, pred_labels, SPACE_LABELS)


if __name__ == "__main__":
    main()
