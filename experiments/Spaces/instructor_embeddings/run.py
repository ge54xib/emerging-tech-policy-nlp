"""S2 — INSTRUCTOR: task-instruction-aware embeddings + prototype classification.

Paper: Su, H., et al. (2023). One Embedder, Any Task: Instruction-Finetuned
       Text Embeddings. ACL 2023.

Method (from paper):
  - Prepend a task instruction to each sentence before encoding:
      [["Represent the {domain} sentence for {task}:", sentence]]
  - Same instruction applied to BOTH training prototypes and test sentences
  - Classify by cosine similarity to per-class prototype (mean of training embeddings)
  - No gradient updates — purely inference-time instruction tuning

Task instruction: "Represent the innovation policy sentence for classifying
                  its Triple Helix space:"

Training data: spaces_labels.json (25 per class) → class prototypes (mean embedding)
               No fine-tuning; embedding model weights are frozen.

Space definitions (Ranga & Etzkowitz 2013):
  knowledge_space  — knowledge generation, diffusion and use; R&D resources
  innovation_space — hybrid orgs; tech transfer, IP, firm formation
  consensus_space  — blue-sky thinking; stakeholder dialogue; governance
  public_space     — civil society, media and culture as innovation actors; public trust, ethics, equity, democratic oversight; innovation culture; creative industries; gender diversity; public legitimation of knowledge (QH extension)

Run:
    python Experiments/Spaces/instructor_embeddings/run.py

Requires:
    pip install InstructorEmbedding scikit-learn
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    SPACE_LABELS,
    load_spaces_eval,
    save_outputs,
)

_REPO_ROOT  = Path(__file__).parent.parent.parent.parent
REVIEW_FILE = Path(__file__).parent.parent / "spaces_labels.json"

INSTRUCTOR_MODEL = "hkunlp/instructor-xl"
TASK_INSTRUCTION = "Represent the innovation policy sentence for classifying its Triple Helix space:"


def _load_training_data() -> dict[str, list[str]]:
    """Return {label: [sentences]} from spaces_labels.json."""
    entries = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    by_label: dict[str, list[str]] = {lbl: [] for lbl in SPACE_LABELS}
    for e in entries:
        space    = e.get("space", "")
        sentence = e.get("central_sentence") or e.get("sentence", "")
        if space in by_label and sentence.strip():
            by_label[space].append(sentence.strip())
    for lbl, sents in by_label.items():
        print(f"  {lbl}: {len(sents)} training sentences")
    return by_label


def main() -> None:
    from InstructorEmbedding import INSTRUCTOR

    eval_entries = load_spaces_eval()
    print(f"Loaded {len(eval_entries)} labeled space examples")

    train_data = _load_training_data()

    print(f"Loading INSTRUCTOR model: {INSTRUCTOR_MODEL}")
    model = INSTRUCTOR(INSTRUCTOR_MODEL)

    # Build per-class prototypes from training sentences
    print("Building class prototypes...")
    prototypes: dict[str, np.ndarray] = {}
    for label, sentences in train_data.items():
        if not sentences:
            continue
        inputs = [[TASK_INSTRUCTION, s] for s in sentences]
        embeddings = model.encode(inputs, batch_size=32, show_progress_bar=False)
        prototypes[label] = embeddings.mean(axis=0)

    # Classify eval sentences by cosine similarity to nearest prototype
    eval_texts = [e["sentence"] for e in eval_entries]
    eval_true  = [e["true_space"] for e in eval_entries]

    print("Encoding eval sentences...")
    eval_inputs    = [[TASK_INSTRUCTION, s] for s in eval_texts]
    eval_embeddings = model.encode(eval_inputs, batch_size=32, show_progress_bar=True)

    # Normalise for cosine similarity
    eval_embeddings  = eval_embeddings / (np.linalg.norm(eval_embeddings, axis=1, keepdims=True) + 1e-8)
    proto_matrix     = np.array([prototypes[lbl] for lbl in SPACE_LABELS])
    proto_matrix     = proto_matrix / (np.linalg.norm(proto_matrix, axis=1, keepdims=True) + 1e-8)

    sims         = eval_embeddings @ proto_matrix.T          # (N, 4)
    pred_indices = sims.argmax(axis=1)
    pred_labels  = [SPACE_LABELS[i] for i in pred_indices]

    predictions = [
        {"id": i, "true": t, "pred": p, "text": eval_texts[i]}
        for i, (t, p) in enumerate(zip(eval_true, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, eval_true, pred_labels, SPACE_LABELS)


if __name__ == "__main__":
    main()
