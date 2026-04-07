"""Isolated SetFit trial for TH space classification.

This is a standalone experiment script that mirrors the repository's SetFit
workflow, but stores all trial artifacts in experiments/spaces_setfit_trial/.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPACE_LABELS = [
    "knowledge_space",
    "innovation_space",
    "consensus_space",
    "public_space",
]

DEFAULT_SEED = 42
DEFAULT_N_PER_SPACE = 25
DEFAULT_MIN_SENTENCE_CHARS = 30
DEFAULT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_cooccurrence_path() -> Path:
    return repo_root() / "data" / "processed" / "step3" / "cooccurrence.jsonl"


def default_alignment_path() -> Path:
    return repo_root() / "data" / "processed" / "step3" / "paragraph_actor_alignment.jsonl"


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            token = line.strip()
            if not token:
                continue
            yield json.loads(token)


def _normalize_pred(pred: Any) -> str:
    if isinstance(pred, str):
        return pred
    try:
        idx = int(pred)
    except Exception:
        return ""
    if 0 <= idx < len(SPACE_LABELS):
        return SPACE_LABELS[idx]
    return ""


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def _split_sentences(text: str) -> list[str]:
    raw = " ".join(str(text or "").replace("\n", " ").split())
    if not raw:
        return []
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+", raw) if part.strip()]
    if not chunks:
        return [raw]
    return chunks


def _focus_sentence_from_window(window_text: str, sentence_id: int) -> str:
    chunks = _split_sentences(window_text)
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]
    if len(chunks) >= 3:
        return chunks[1]
    return chunks[0] if sentence_id <= 1 else chunks[1]


def _best_sentence_for_entities(window_text: str, sentence_id: int, entity_1: str, entity_2: str) -> str:
    chunks = _split_sentences(window_text)
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    base = _focus_sentence_from_window(window_text, sentence_id)
    base_idx = 0
    for idx, chunk in enumerate(chunks):
        if chunk == base:
            base_idx = idx
            break

    n1 = _norm(entity_1)
    n2 = _norm(entity_2)
    if not n1 and not n2:
        return base

    scored: list[tuple[int, int, str]] = []
    for idx, chunk in enumerate(chunks):
        n_chunk = _norm(chunk)
        score = 0
        if n1 and n1 in n_chunk:
            score += 2
        if n2 and n2 in n_chunk:
            score += 2
        scored.append((score, idx, chunk))

    best_score = max(item[0] for item in scored)
    if best_score <= 0:
        return base

    best = [item for item in scored if item[0] == best_score]
    best.sort(key=lambda item: abs(item[1] - base_idx))
    return best[0][2]


def _load_alignment_mentions(alignment_path: Path) -> dict[tuple[str, int, int, int], str]:
    lookup: dict[tuple[str, int, int, int], str] = {}
    if not alignment_path.exists():
        return lookup
    for row in _iter_jsonl(alignment_path):
        doc_id = str(row.get("doc_id", "")).strip()
        para_id = _to_int(row.get("paragraph_id", -1), -1)
        sent_id = _to_int(row.get("sentence_id", -1), -1)
        ent_id = _to_int(row.get("entity_id", 0), 0)
        mention = str(row.get("mention_text", "")).strip()
        if not doc_id or para_id < 0 or sent_id < 0 or ent_id <= 0 or not mention:
            continue
        key = (doc_id, para_id, sent_id, ent_id)
        current = lookup.get(key, "")
        if len(mention) > len(current):
            lookup[key] = mention
    return lookup


def _annotation_entities(row: dict[str, Any], mention_lookup: dict[tuple[str, int, int, int], str]) -> tuple[str, str]:
    doc_id = str(row.get("doc_id", "")).strip()
    para_id = _to_int(row.get("paragraph_id", -1), -1)
    sent_id = _to_int(row.get("sentence_id", -1), -1)
    e1_id = _to_int(row.get("entity_id_1", 0), 0)
    e2_id = _to_int(row.get("entity_id_2", 0), 0)
    e1 = mention_lookup.get((doc_id, para_id, sent_id, e1_id), str(row.get("entity_1", "")).strip())
    e2 = mention_lookup.get((doc_id, para_id, sent_id, e2_id), str(row.get("entity_2", "")).strip())
    return e1, e2


def sample_cmd(
    *,
    cooccurrence_path: Path,
    alignment_path: Path,
    annotation_path: Path,
    n_per_space: int,
    seed: int,
    min_sentence_chars: int,
) -> None:
    if not cooccurrence_path.exists():
        raise FileNotFoundError(f"Missing co-occurrence file: {cooccurrence_path}")

    print(f"[SetFit Trial] Scanning {cooccurrence_path} ...")
    mention_lookup = _load_alignment_mentions(alignment_path)
    by_space: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()

    for row in _iter_jsonl(cooccurrence_path):
        doc_id = str(row.get("doc_id", ""))
        para_id = int(row.get("paragraph_id", -1))
        sent_id = int(row.get("sentence_id", -1))
        if not doc_id or para_id < 0 or sent_id < 0:
            continue

        key = (doc_id, para_id, sent_id)
        if key in seen:
            continue
        seen.add(key)

        space = str(row.get("th_space", "")).strip()
        if space not in SPACE_LABELS:
            continue

        entity_1, entity_2 = _annotation_entities(row, mention_lookup)
        sentence = _best_sentence_for_entities(
            row.get("sent_text", ""),
            sent_id,
            entity_1,
            entity_2,
        )
        if len(sentence) < min_sentence_chars:
            continue

        by_space[space].append(
            {
                "doc_id": doc_id,
                "paragraph_id": para_id,
                "sentence_id": sent_id,
                "sentence": sentence,
                "entity_1": entity_1,
                "h1": row.get("h1", ""),
                "entity_2": entity_2,
                "h2": row.get("h2", ""),
                "nli_predicted_space": space,
                "space": "",
            }
        )

    random.seed(seed)
    samples: list[dict[str, Any]] = []
    print("[SetFit Trial] Sampling ...")
    for space in SPACE_LABELS:
        candidates = by_space[space]
        random.shuffle(candidates)
        taken = candidates[:n_per_space]
        samples.extend(taken)
        print(f"  {space}: {len(candidates)} candidates -> sampled {len(taken)}")

    random.shuffle(samples)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[SetFit Trial] Wrote {len(samples)} rows to: {annotation_path}")
    print("[SetFit Trial] Fill in the 'space' field for each row.")
    print(f"[SetFit Trial] Valid labels: {', '.join(SPACE_LABELS)}")


def train_cmd(
    *,
    annotation_path: Path,
    model_path: Path,
    model_id: str,
    seed: int,
    bootstrap_from_nli: bool,
) -> None:
    try:
        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies for SetFit training. "
            "Install with: pip install -r experiments/spaces_setfit_trial/requirements.txt"
        ) from exc

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}\nRun --sample first."
        )

    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"Invalid annotation format in {annotation_path}: expected JSON array.")

    texts: list[str] = []
    labels: list[int] = []
    bootstrapped = 0
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        space = str(entry.get("space", "")).strip()
        if space not in SPACE_LABELS and bootstrap_from_nli:
            fallback = str(entry.get("nli_predicted_space", "")).strip()
            if fallback in SPACE_LABELS:
                space = fallback
                bootstrapped += 1
        sentence_chunks = _split_sentences(entry.get("sentence", ""))
        sentence = sentence_chunks[0] if sentence_chunks else ""
        if not sentence or space not in SPACE_LABELS:
            continue
        texts.append(sentence)
        labels.append(SPACE_LABELS.index(space))

    if len(texts) < len(SPACE_LABELS):
        raise RuntimeError(
            f"Only {len(texts)} annotated rows found, need at least {len(SPACE_LABELS)} "
            "(one per class)."
        )

    counts = Counter(labels)
    print(f"[SetFit Trial] Training on {len(texts)} annotated rows:")
    for idx, space in enumerate(SPACE_LABELS):
        print(f"  {space}: {counts[idx]}")
    if bootstrap_from_nli:
        print(f"[SetFit Trial] Bootstrapped labels from nli_predicted_space: {bootstrapped}")

    dataset = Dataset.from_dict({"text": texts, "label": labels})

    print(f"[SetFit Trial] Loading base model: {model_id}")
    model = SetFitModel.from_pretrained(model_id, labels=SPACE_LABELS)

    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )
    print("[SetFit Trial] Training ...")
    trainer.train()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_path))
    print(f"[SetFit Trial] Saved model to: {model_path}")


def predict_cmd(*, cooccurrence_path: Path, model_path: Path, output_path: Path) -> None:
    try:
        from setfit import SetFitModel
    except ImportError as exc:
        raise ImportError(
            "Missing SetFit dependency for prediction. "
            "Install with: pip install -r experiments/spaces_setfit_trial/requirements.txt"
        ) from exc

    if not cooccurrence_path.exists():
        raise FileNotFoundError(f"Missing co-occurrence file: {cooccurrence_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nRun --train first.")

    print(f"[SetFit Trial] Loading model from: {model_path}")
    model = SetFitModel.from_pretrained(str(model_path))

    rows: list[dict[str, Any]] = []
    sentence_cache: dict[tuple[str, int, int], str] = {}
    for row in _iter_jsonl(cooccurrence_path):
        rows.append(row)
        doc_id = str(row.get("doc_id", ""))
        para_id = int(row.get("paragraph_id", -1))
        sent_id = int(row.get("sentence_id", -1))
        if not doc_id or para_id < 0 or sent_id < 0:
            continue
        key = (doc_id, para_id, sent_id)
        if key not in sentence_cache:
            sentence_cache[key] = _focus_sentence_from_window(row.get("sent_text", ""), sent_id)

    unique_keys = list(sentence_cache.keys())
    unique_texts = [sentence_cache[k] for k in unique_keys]
    print(f"[SetFit Trial] Predicting {len(unique_texts)} unique sentences ...")

    raw_preds = model.predict(unique_texts)
    space_by_key: dict[tuple[str, int, int], str] = {}
    for key, pred in zip(unique_keys, raw_preds):
        space_by_key[key] = _normalize_pred(pred)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counter: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            doc_id = str(row.get("doc_id", ""))
            para_id = int(row.get("paragraph_id", -1))
            sent_id = int(row.get("sentence_id", -1))
            key = (doc_id, para_id, sent_id)
            pred_space = space_by_key.get(key, "")
            row["th_space_setfit"] = pred_space
            counter[pred_space] += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    print(f"[SetFit Trial] Wrote {total} rows to: {output_path}")
    print("[SetFit Trial] Space distribution:")
    for space in SPACE_LABELS:
        count = counter[space]
        share = (count / total * 100.0) if total else 0.0
        print(f"  {space}: {count} ({share:.1f}%)")


def repair_annotation_cmd(
    *,
    cooccurrence_path: Path,
    alignment_path: Path,
    annotation_path: Path,
    min_sentence_chars: int,
) -> None:
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    if not cooccurrence_path.exists():
        raise FileNotFoundError(f"Missing co-occurrence file: {cooccurrence_path}")

    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"Invalid annotation format in {annotation_path}: expected JSON array.")

    mention_lookup = _load_alignment_mentions(alignment_path)
    coocc_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in _iter_jsonl(cooccurrence_path):
        key = (
            str(row.get("doc_id", "")).strip(),
            _to_int(row.get("paragraph_id", -1), -1),
            _to_int(row.get("sentence_id", -1), -1),
        )
        if key not in coocc_by_key:
            coocc_by_key[key] = row

    fixed = 0
    missing = 0
    short = 0
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        key = (
            str(entry.get("doc_id", "")).strip(),
            _to_int(entry.get("paragraph_id", -1), -1),
            _to_int(entry.get("sentence_id", -1), -1),
        )
        row = coocc_by_key.get(key)
        if row is None:
            missing += 1
            continue

        entity_1, entity_2 = _annotation_entities(row, mention_lookup)
        sentence = _best_sentence_for_entities(row.get("sent_text", ""), key[2], entity_1, entity_2)
        if len(sentence) < min_sentence_chars:
            short += 1

        changed = (
            str(entry.get("sentence", "")) != sentence
            or str(entry.get("entity_1", "")) != entity_1
            or str(entry.get("entity_2", "")) != entity_2
        )
        entry["sentence"] = sentence
        entry["entity_1"] = entity_1
        entry["entity_2"] = entity_2
        if changed:
            fixed += 1

    annotation_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SetFit Trial] Repaired annotation file: {annotation_path}")
    print(f"[SetFit Trial] Rows updated: {fixed}")
    print(f"[SetFit Trial] Missing co-occurrence keys: {missing}")
    print(f"[SetFit Trial] Rows below min sentence length ({min_sentence_chars}): {short}")


def parse_args() -> argparse.Namespace:
    default_workdir = script_dir() / "artifacts"
    parser = argparse.ArgumentParser(
        description="Isolated SetFit trial for TH space classification."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="Create annotation sample JSON.")
    group.add_argument("--train", action="store_true", help="Train SetFit model.")
    group.add_argument("--predict", action="store_true", help="Predict spaces and write JSONL.")
    group.add_argument("--repair-annotation", action="store_true", help="Repair sentence/entity fields in annotation JSON.")

    parser.add_argument(
        "--cooccurrence",
        type=Path,
        default=default_cooccurrence_path(),
        help="Path to cooccurrence.jsonl.",
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=default_alignment_path(),
        help="Path to paragraph_actor_alignment.jsonl (for mention-level entity text).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=default_workdir,
        help="Directory for trial artifacts.",
    )
    parser.add_argument("--annotation-path", type=Path, default=None, help="Override annotation JSON path.")
    parser.add_argument("--model-path", type=Path, default=None, help="Override model directory path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Override prediction output JSONL path.")
    parser.add_argument("--n-per-space", type=int, default=DEFAULT_N_PER_SPACE, help="Rows sampled per space.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--min-sentence-chars",
        type=int,
        default=DEFAULT_MIN_SENTENCE_CHARS,
        help="Minimum sentence length for sampling.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Base sentence-transformer model for SetFit training.",
    )
    parser.add_argument(
        "--bootstrap-from-nli",
        action="store_true",
        help="For trial runs only: if 'space' is empty, use 'nli_predicted_space' as label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workdir = args.workdir
    annotation_path = args.annotation_path or (workdir / "spaces_annotation.json")
    model_path = args.model_path or (workdir / "setfit_spaces_model")
    output_path = args.output_path or (workdir / "cooccurrence_setfit.jsonl")

    if args.sample:
        sample_cmd(
            cooccurrence_path=args.cooccurrence,
            alignment_path=args.alignment,
            annotation_path=annotation_path,
            n_per_space=args.n_per_space,
            seed=args.seed,
            min_sentence_chars=args.min_sentence_chars,
        )
    elif args.train:
        train_cmd(
            annotation_path=annotation_path,
            model_path=model_path,
            model_id=args.model_id,
            seed=args.seed,
            bootstrap_from_nli=args.bootstrap_from_nli,
        )
    elif args.predict:
        predict_cmd(
            cooccurrence_path=args.cooccurrence,
            model_path=model_path,
            output_path=output_path,
        )
    elif args.repair_annotation:
        repair_annotation_cmd(
            cooccurrence_path=args.cooccurrence,
            alignment_path=args.alignment,
            annotation_path=annotation_path,
            min_sentence_chars=args.min_sentence_chars,
        )


if __name__ == "__main__":
    main()
