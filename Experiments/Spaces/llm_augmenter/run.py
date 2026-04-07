"""S5 — LLM Augmentation: targeted synthetic data for minority class (public_space).

Paper: Truveta Research (2025). Structured LLM Augmentation for Imbalanced Text
       Classification. MedInfo / arXiv 2025.

Method (from paper):
  Stage 1 — Targeted augmentation for minority class via structured LLM prompts:
    - Constrained by: (a) class definition, (b) entity types present,
      (c) required signal words
    - Generate 75 synthetic public_space sentences using GPT-4
    - Entity pairs sampled from civil_society entities in entities_classified.jsonl

  Stage 2 — Train classifier on augmented dataset:
    - Combine spaces_review.json (real) + synthetic_public_space.json (synthetic)
    - Train FastFit (same config as S4 for clean ablation) on augmented set
    - Evaluate on annotation_spaces.json (human-annotated gold standard)

Key contribution: structured prompts with entity-type and signal-word constraints
produce more diverse and label-faithful synthetic sentences than unconstrained
generation — especially valuable for rare labels in imbalanced corpora.

Run:
    export OPENAI_API_KEY=...
    python Experiments/Spaces/llm_augmenter/run.py

Requires:
    pip install openai fast-fit datasets scikit-learn
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    SPACE_LABELS,
    load_spaces_eval,
    save_outputs,
)

_REPO_ROOT       = Path(__file__).parent.parent.parent.parent
REVIEW_FILE      = _REPO_ROOT / "data/processed/step3/spaces_review.json"
ENTITIES_FILE    = _REPO_ROOT / "data/processed/step2/entities_classified.jsonl"
SYNTHETIC_FILE   = Path(__file__).parent / "synthetic_public_space.json"

GPT_MODEL        = "gpt-4o"
SYNTHETIC_TARGET = 75      # synthetic public_space sentences to generate
FASTFIT_MODEL    = "roberta-base"
NUM_TRAIN_EPOCHS = 10      # paper recommendation for few-shot regime
NUM_ITERATIONS   = 5       # repetitions per batch (paper default)
LEARNING_RATE    = 3e-5
BATCH_SIZE       = 32

# Signal words for public_space that guide constrained generation
PUBLIC_SIGNAL_WORDS = [
    "public", "society", "citizen", "ethics", "equity",
    "inclusion", "trust", "democratic", "governance",
    "awareness", "engagement", "transparency", "accountability",
]

# Other helices to pair with civil_society for diversity
PAIRING_HELICES = ["government", "industry", "academia", "intermediary"]


# ── Stage 1: Civil society entity sampling ────────────────────────────────────

def _load_civil_society_entities() -> list[dict]:
    """Return list of {entity, helix} for unique civil_society actors."""
    seen: set[str] = set()
    result: list[dict] = []
    with open(ENTITIES_FILE, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if (
                e.get("level_5_helix", "").lower() == "civil society"
                and e.get("status") == "entity"
            ):
                raw  = e.get("entity_label") or e.get("mention", "")
                name = raw.split(";")[0].strip()
                if name and name not in seen and len(name) <= 80:
                    seen.add(name)
                    result.append({"entity": name, "helix": "civil_society"})
    return result


def _load_other_entities() -> dict[str, list[str]]:
    """Return {helix: [entity_names]} for non-civil_society helices (sample pool)."""
    by_helix: dict[str, list[str]] = {h: [] for h in PAIRING_HELICES}
    seen: set[str] = set()
    with open(ENTITIES_FILE, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            helix = e.get("level_5_helix", "").lower().replace(" ", "_")
            if helix in by_helix and e.get("status") == "entity":
                raw  = e.get("entity_label") or e.get("mention", "")
                name = raw.split(";")[0].strip()
                if name and name not in seen and len(name) <= 80:
                    seen.add(name)
                    by_helix[helix].append(name)
    return by_helix


def _sample_entity_pairs(
    civil: list[dict],
    others: dict[str, list[str]],
    n: int,
) -> list[tuple[str, str, str, str]]:
    """
    Sample n (entity1, helix1, entity2, helix2) pairs each involving
    at least one civil_society actor.
    """
    random.seed(42)
    pairs: list[tuple[str, str, str, str]] = []
    helices = [h for h, lst in others.items() if lst]

    for _ in range(n):
        cs_entity = random.choice(civil)["entity"]
        partner_h = random.choice(helices)
        partner_e = random.choice(others[partner_h])
        # Randomly decide order so civil_society isn't always entity_1
        if random.random() < 0.5:
            pairs.append((cs_entity, "civil_society", partner_e, partner_h))
        else:
            pairs.append((partner_e, partner_h, cs_entity, "civil_society"))

    return pairs


# ── Stage 1: GPT-4 augmentation ───────────────────────────────────────────────

_AUGMENT_SYSTEM = (
    "You generate sentences for annotated policy document training data. "
    "Each sentence must be plausible as a direct excerpt from a national "
    "quantum technology strategy document."
)

_AUGMENT_USER = """\
Write a single sentence from a national quantum technology policy document that:
1. Involves {entity_1} ({helix_1}) and {entity_2} ({helix_2}) as actors
2. Describes civil society engagement, public awareness, ethics, equity, \
or democratic oversight in the context of quantum technology
3. Uses at least one of these signal words: {signal_words}

Output only the sentence. No explanation, no quotes."""


def _generate_synthetic(
    pairs: list[tuple[str, str, str, str]],
    existing: list[dict],
) -> list[dict]:
    """Generate synthetic public_space sentences via GPT-4; skip already-cached."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    results = list(existing)  # start from any cached entries
    cached_count = len(existing)

    # Pairs that haven't been generated yet
    to_generate = pairs[cached_count:]
    print(f"Generating {len(to_generate)} new synthetic sentences "
          f"({cached_count} already cached)...")

    for i, (e1, h1, e2, h2) in enumerate(to_generate):
        signal_sample = ", ".join(random.sample(PUBLIC_SIGNAL_WORDS, k=4))
        user_msg = _AUGMENT_USER.format(
            entity_1=e1, helix_1=h1,
            entity_2=e2, helix_2=h2,
            signal_words=signal_sample,
        )
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": _AUGMENT_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=128,
            temperature=0.8,   # diversity for augmentation
        )
        sentence = (response.choices[0].message.content or "").strip().strip('"')
        results.append({
            "sentence":   sentence,
            "true_space": "public_space",
            "source":     "synthetic",
            "entity_1":   e1,
            "helix_1":    h1,
            "entity_2":   e2,
            "helix_2":    h2,
        })

        if (i + 1) % 15 == 0:
            # Save incrementally to guard against interruption
            SYNTHETIC_FILE.parent.mkdir(parents=True, exist_ok=True)
            SYNTHETIC_FILE.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  {cached_count + i + 1}/{len(pairs)} done (saved)")

    return results


# ── Stage 2: FastFit on augmented data ────────────────────────────────────────

def _load_training_data(synthetic: list[dict]) -> tuple[list[str], list[str]]:
    """Combine real spaces_review.json + synthetic public_space sentences."""
    texts, labels = [], []

    # Real annotated training data
    if REVIEW_FILE.exists():
        entries = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
        for e in entries:
            space    = e.get("space") or e.get("true_space", "")
            sentence = e.get("central_sentence") or e.get("sentence", "")
            if space in SPACE_LABELS and sentence.strip():
                texts.append(sentence.strip())
                labels.append(space)

    # Synthetic public_space sentences
    for e in synthetic:
        if e.get("sentence", "").strip():
            texts.append(e["sentence"].strip())
            labels.append("public_space")

    print(f"Training examples: {len(texts)} total (after augmentation)")
    for lbl, cnt in Counter(labels).items():
        print(f"  {lbl}: {cnt}")
    return texts, labels


def main() -> None:
    from datasets import Dataset
    from fastfit import FastFitTrainer

    # ── Stage 1: generate or load synthetic public_space data ────────────────
    civil  = _load_civil_society_entities()
    others = _load_other_entities()
    print(f"Civil society entities available: {len(civil)}")
    for h, lst in others.items():
        print(f"  {h}: {len(lst)} entities")

    pairs = _sample_entity_pairs(civil, others, SYNTHETIC_TARGET)

    # Load cached synthetic data (incremental generation support)
    existing: list[dict] = []
    if SYNTHETIC_FILE.exists():
        existing = json.loads(SYNTHETIC_FILE.read_text(encoding="utf-8"))
        print(f"Loaded {len(existing)} cached synthetic sentences from {SYNTHETIC_FILE.name}")

    if len(existing) < SYNTHETIC_TARGET:
        synthetic = _generate_synthetic(pairs, existing)
        # Final save
        SYNTHETIC_FILE.parent.mkdir(parents=True, exist_ok=True)
        SYNTHETIC_FILE.write_text(
            json.dumps(synthetic, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved {len(synthetic)} synthetic sentences → {SYNTHETIC_FILE}")
    else:
        synthetic = existing[:SYNTHETIC_TARGET]
        print(f"Using {len(synthetic)} cached synthetic sentences")

    # ── Stage 2: train FastFit on augmented data ──────────────────────────────
    eval_entries = load_spaces_eval()
    print(f"Loaded {len(eval_entries)} labeled space examples")

    train_texts, train_labels = _load_training_data(synthetic)

    label2id = {lbl: i for i, lbl in enumerate(SPACE_LABELS)}
    id2label  = {i: lbl for lbl, i in label2id.items()}

    train_ds = Dataset.from_dict({
        "text":  train_texts,
        "label": [label2id[l] for l in train_labels],
    })

    eval_texts = [e["sentence"] for e in eval_entries]
    eval_true  = [e["true_space"] for e in eval_entries]
    eval_ds    = Dataset.from_dict({
        "text":  eval_texts,
        "label": [label2id[l] for l in eval_true],
    })

    trainer = FastFitTrainer(
        model_name_or_path=FASTFIT_MODEL,
        label_column_name="label",
        text_column_name="text",
        num_train_epochs=NUM_TRAIN_EPOCHS,
        num_iterations=NUM_ITERATIONS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        train_dataset=train_ds,
        validation_dataset=eval_ds,
        id2label=id2label,
        label2id=label2id,
    )

    model = trainer.train()

    predictions_raw = model.predict(eval_texts)
    pred_labels = [
        id2label[p] if isinstance(p, int) else p
        for p in predictions_raw
    ]

    predictions = [
        {"id": i, "true": t, "pred": p, "text": eval_texts[i]}
        for i, (t, p) in enumerate(zip(eval_true, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, eval_true, pred_labels, SPACE_LABELS)


if __name__ == "__main__":
    main()
