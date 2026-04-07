"""Few-shot TH space classification using SetFit (Tunstall et al. 2022).

Workflow
--------
Stage 1 — Sample and review:
    python -m src.pipeline.spaces_setfit --sample

    Writes data/processed/step3/spaces_labels.json with cross-helix sentences
    stratified by helix-pair TH_SPACE_MAP. Each entry has:
      - ``sentence``         ±1 sentence window  (for human review context)
      - ``central_sentence`` central sentence only (what SetFit trains/predicts on)
      - ``space``            annotator label (fill in)
    Valid labels: knowledge_space, innovation_space, consensus_space, public_space

Stage 2 — Train SetFit:
    python -m src.pipeline.spaces_setfit --train

    Reads spaces_labels.json, takes entries where use="true", balances to
    minority class size, trains SetFit on central_sentence text only.
    Saves model to data/processed/step3/setfit_spaces_model/.

Stage 3 — Predict:
    python -m src.pipeline.spaces_setfit --predict

    Loads trained model, classifies all unique central sentences in
    cooccurrence.jsonl, writes cooccurrence_setfit.jsonl with th_space_setfit.

Note: requires step3 to have been run with central_sent_text in output.

Reference
---------
Tunstall, L. et al. (2022). Efficient Few-Shot Learning Without Prompts.
arXiv:2209.11055.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SPACE_LABELS = ["knowledge_space", "innovation_space", "consensus_space", "public_space"]
N_PER_SPACE = 25
RANDOM_SEED = 42
SETFIT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"

PAIR_TO_SPACE = {
    frozenset({"academia", "government"}):     "knowledge_space",
    frozenset({"academia", "industry"}):       "innovation_space",
    frozenset({"academia", "intermediary"}):   "innovation_space",
    frozenset({"industry", "intermediary"}):   "innovation_space",
    frozenset({"government", "industry"}):     "consensus_space",
    frozenset({"government", "intermediary"}): "consensus_space",
}

PUBLIC_KEYWORDS = [
    "public", "society", "citizen", "ethics", "ethical", "responsible",
    "equity", "inclusion", "trust", "awareness", "engagement", "societal",
    "democratic", "transparency", "benefit", "diversity",
]


def _pair_space(h1: str, h2: str) -> str:
    if "civil_society" in (h1, h2):
        return "public_space"
    return PAIR_TO_SPACE.get(frozenset({h1, h2}), "")


def _config():
    from src import config
    return config


def _review_path() -> Path:
    return Path(__file__).parent.parent.parent / "Experiments" / "Spaces" / "spaces_labels.json"


def _model_path() -> Path:
    return _config().STEP3_DIR / "setfit_spaces_model"


def _output_path() -> Path:
    return _config().STEP3_DIR / "cooccurrence_setfit.jsonl"


# ---------------------------------------------------------------------------
# Stage 1: sample → spaces_labels.json
# ---------------------------------------------------------------------------

def sample_cmd() -> None:
    cfg = _config()

    if not cfg.FILE_COOCCURRENCE.exists():
        raise FileNotFoundError(f"Missing co-occurrence file: {cfg.FILE_COOCCURRENCE}")

    # Exclude evaluation sentences to prevent data leakage
    _eval_path = Path(__file__).parent.parent.parent / "evaluation" / "annotation.json"
    eval_sids: set[tuple] = set()
    if _eval_path.exists():
        for e in json.loads(_eval_path.read_text(encoding="utf-8")):
            eval_sids.add((str(e.get("doc_id", "")), int(e.get("paragraph_id", -1)), int(e.get("sentence_id", -1))))
        print(f"[SetFit] Excluding {len(eval_sids)} evaluation sentences from training pool.")

    print("[SetFit] Scanning co-occurrence file …")

    # First pass: collect entities (cross-helix only) and sentence texts per key
    # Store (canonical_name, mention_text, helix) so we can check either against sentence
    entities_by_key: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
    central_by_key: dict[tuple, str] = {}
    window_by_key: dict[tuple, str] = {}
    cross_helix_keys: set[tuple] = set()
    rows_all: list[dict] = []

    with cfg.FILE_COOCCURRENCE.open(encoding="utf-8") as fh:
        for line in fh:
            token = line.strip()
            if not token:
                continue
            row = json.loads(token)
            rows_all.append(row)
            key = (str(row["doc_id"]), int(row["paragraph_id"]), int(row["sentence_id"]))

            if key not in window_by_key:
                window_by_key[key] = row.get("sent_text", "")
                central_by_key[key] = row.get("central_sent_text") or row.get("sent_text", "")

            if row.get("h1") != row.get("h2"):
                cross_helix_keys.add(key)
                for ef, af, hf in [("entity_1", "actor_1", "h1"), ("entity_2", "actor_2", "h2")]:
                    ent = row.get(ef, "")
                    actor = row.get(af, {}) or {}
                    mention = actor.get("mention_text", "") or ent
                    helix = row.get(hf, "")
                    if ent:
                        entities_by_key[key].append((ent, mention, helix))

    def _entities_in_sentence(key: tuple, sent_lower: str) -> list[dict]:
        seen_entities = set()
        result = []
        for ent, mention, helix in entities_by_key[key]:
            if ent.lower() in sent_lower or mention.lower() in sent_lower:
                if ent not in seen_entities:
                    seen_entities.add(ent)
                    result.append({"entity": ent, "helix": helix})
        return result

    # Deduplicate, stratify by helix-pair space
    seen: set[tuple] = set()
    by_space: dict[str, list[dict]] = defaultdict(list)

    for row in rows_all:
        key = (str(row["doc_id"]), int(row["paragraph_id"]), int(row["sentence_id"]))
        if key in seen or key not in cross_helix_keys:
            continue
        seen.add(key)

        central = central_by_key.get(key, "")
        if len(central) < 20:
            continue

        space_bucket = _pair_space(row.get("h1", ""), row.get("h2", ""))
        if not space_bucket:
            continue

        if key in eval_sids:
            continue  # exclude evaluation sentences

        central_lower = central.lower()
        entities_in_sent = _entities_in_sentence(key, central_lower)
        if not entities_in_sent:
            continue  # skip if no entities appear in the central sentence

        by_space[space_bucket].append({
            "doc_id": str(row["doc_id"]),
            "paragraph_id": int(row["paragraph_id"]),
            "sentence_id": int(row["sentence_id"]),
            "central_sentence": central,   # used for training
            "sentence": central,           # central sentence only (clean)
            "entities": entities_in_sent,
            "space": "",                   # annotator label
        })

    # For public_space: also add keyword-matched sentences not already captured
    already_keys = {(e["doc_id"], e["paragraph_id"], e["sentence_id"])
                    for entries in by_space.values() for e in entries}
    for key in cross_helix_keys:
        if key in seen:
            continue
        if key in eval_sids:
            continue
        central = central_by_key.get(key, "")
        if not any(kw in central.lower() for kw in PUBLIC_KEYWORDS):
            continue
        if (key[0], key[1], key[2]) in already_keys:
            continue

        central_lower = central.lower()
        entities_in_sent = _entities_in_sentence(key, central_lower)
        if not entities_in_sent:
            continue  # skip if no entities appear in the central sentence

        by_space["public_space"].append({
            "doc_id": key[0],
            "paragraph_id": key[1],
            "sentence_id": key[2],
            "central_sentence": central,
            "sentence": central,
            "entities": entities_in_sent,
            "space": "",
        })

    # Load existing annotations to carry over manual labels
    review_path = _review_path()
    existing_labels: dict[tuple, str] = {}
    existing_confidence: dict[tuple, str] = {}
    if review_path.exists():
        for e in json.loads(review_path.read_text(encoding="utf-8")):
            if e.get("space"):
                sid = (str(e.get("doc_id", "")), int(e.get("paragraph_id", -1)), int(e.get("sentence_id", -1)))
                existing_labels[sid] = e["space"]
                existing_confidence[sid] = e.get("confidence", "")

    random.seed(RANDOM_SEED)
    samples: list[dict] = []
    carried_over = 0
    for space in SPACE_LABELS:
        pool = by_space[space]
        random.shuffle(pool)
        taken = pool[:N_PER_SPACE]
        for entry in taken:
            sid = (str(entry["doc_id"]), int(entry["paragraph_id"]), int(entry["sentence_id"]))
            if sid in existing_labels:
                entry["space"] = existing_labels[sid]
                entry["confidence"] = existing_confidence.get(sid, "")
                carried_over += 1
        samples.extend(taken)
        print(f"  {space}: {len(pool)} candidates → sampled {len(taken)}")

    review_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SetFit] Wrote {len(samples)} entries to {review_path}")
    if carried_over:
        print(f"[SetFit] Carried over {carried_over} existing annotations.")
    print('Fill in the "space" field for any unlabeled entries.')
    print(f"Valid labels: {', '.join(SPACE_LABELS)}")


# ---------------------------------------------------------------------------
# Stage 2: train from spaces_labels.json (use="true" entries only)
# ---------------------------------------------------------------------------

def train_cmd() -> None:
    try:
        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("setfit and datasets are required. Install with: pip install setfit") from exc

    from collections import Counter

    cfg = _config()
    review_path = _review_path()
    if not review_path.exists():
        raise FileNotFoundError(f"Review file not found: {review_path}\nRun --sample first.")

    by_label: dict[int, list[str]] = {i: [] for i in range(len(SPACE_LABELS))}

    # Manual training set (spaces_labels.json) — all entries included
    raw = json.loads(review_path.read_text(encoding="utf-8"))
    manual_count = 0
    for entry in raw:
        space = str(entry.get("space", "")).strip()
        if space not in SPACE_LABELS:
            continue
        text = str(entry.get("central_sentence") or entry.get("sentence", "")).strip()
        if not text:
            continue
        by_label[SPACE_LABELS.index(space)].append(text)
        manual_count += 1
    print(f"[SetFit] Manual training set: {manual_count} entries")

    # LLM-annotated set (spaces_llm_review.json) — all entries used directly
    # Following Gilardi et al. (2023): LLM labels replace crowd-worker annotations
    llm_path = cfg.STEP3_DIR / "spaces_llm_review.json"
    llm_count = 0
    if llm_path.exists():
        llm_raw = json.loads(llm_path.read_text(encoding="utf-8"))
        for entry in llm_raw:
            space = str(entry.get("llm_space", "")).strip()
            if space not in SPACE_LABELS:
                continue
            text = str(entry.get("sentence", "")).strip()
            if not text:
                continue
            by_label[SPACE_LABELS.index(space)].append(text)
            llm_count += 1
        print(f"[SetFit] LLM-annotated set (Gilardi et al. 2023): {llm_count} entries")
    else:
        print("[SetFit] No spaces_llm_review.json found — using manual set only")

    if any(len(v) == 0 for v in by_label.values()):
        missing = [SPACE_LABELS[i] for i, v in by_label.items() if not v]
        raise RuntimeError(f"No training examples for: {missing}.")

    # Balance to minority class
    min_count = min(len(v) for v in by_label.values())
    random.seed(RANDOM_SEED)
    texts: list[str] = []
    labels: list[int] = []
    for i in range(len(SPACE_LABELS)):
        pool = by_label[i][:]
        random.shuffle(pool)
        selected = pool[:min_count]
        texts.extend(selected)
        labels.extend([i] * len(selected))

    counts = Counter(labels)
    print(f"[SetFit] Training on {len(texts)} balanced examples ({min_count} per class):")
    for i, space in enumerate(SPACE_LABELS):
        print(f"  {space}: {counts[i]}")

    dataset = Dataset.from_dict({"text": texts, "label": labels})

    print(f"[SetFit] Loading base model '{SETFIT_MODEL_ID}' …")
    model = SetFitModel.from_pretrained(SETFIT_MODEL_ID, labels=SPACE_LABELS)

    args = TrainingArguments(batch_size=16, num_epochs=1, seed=RANDOM_SEED)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    print("[SetFit] Training …")
    trainer.train()

    model_path = _model_path()
    model.save_pretrained(str(model_path))
    print(f"[SetFit] Model saved to {model_path}")


# ---------------------------------------------------------------------------
# Stage 3: predict using central_sent_text
# ---------------------------------------------------------------------------

def predict_cmd() -> None:
    try:
        from setfit import SetFitModel
    except ImportError as exc:
        raise ImportError("setfit is required. Install with: pip install setfit") from exc

    cfg = _config()
    model_path = _model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nRun --train first.")

    print(f"[SetFit] Loading model from {model_path} …")
    model = SetFitModel.from_pretrained(str(model_path))

    print("[SetFit] Loading sentences from co-occurrence file …")
    rows: list[dict] = []
    sentence_cache: dict[tuple, str] = {}

    with cfg.FILE_COOCCURRENCE.open(encoding="utf-8") as fh:
        for line in fh:
            token = line.strip()
            if not token:
                continue
            row = json.loads(token)
            rows.append(row)
            key = (str(row["doc_id"]), int(row["paragraph_id"]), int(row["sentence_id"]))
            if key not in sentence_cache:
                # Prefer central sentence; fall back to ±1 window
                sentence_cache[key] = row.get("central_sent_text") or row.get("sent_text", "")

    unique_keys = list(sentence_cache.keys())
    unique_texts = [sentence_cache[k] for k in unique_keys]

    print(f"[SetFit] Predicting {len(unique_texts)} unique sentences …")
    raw_predictions = model.predict(unique_texts)

    space_by_key: dict[tuple, str] = {}
    for key, pred in zip(unique_keys, raw_predictions):
        if isinstance(pred, str):
            space_by_key[key] = pred
        else:
            idx = int(pred)
            space_by_key[key] = SPACE_LABELS[idx] if 0 <= idx < len(SPACE_LABELS) else ""

    from collections import Counter
    space_counter: Counter = Counter()
    output_path = _output_path()

    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            key = (str(row["doc_id"]), int(row["paragraph_id"]), int(row["sentence_id"]))
            setfit_space = space_by_key.get(key, "")
            row["th_space_setfit"] = setfit_space
            space_counter[setfit_space] += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    print(f"[SetFit] Wrote {output_path} ({total} rows)")
    print("[SetFit] Space distribution (SetFit, central sentence):")
    for space in SPACE_LABELS:
        count = space_counter[space]
        print(f"  {space}: {count} ({count / total * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Few-shot TH space classification with SetFit (Tunstall et al. 2022)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true",
                       help="Sample sentences → spaces_labels.json")
    group.add_argument("--train", action="store_true",
                       help="Train on use=true entries → setfit_spaces_model/")
    group.add_argument("--predict", action="store_true",
                       help="Predict all pairs → cooccurrence_setfit.jsonl")
    args = parser.parse_args()

    if args.sample:
        sample_cmd()
    elif args.train:
        train_cmd()
    elif args.predict:
        predict_cmd()


if __name__ == "__main__":
    main()
