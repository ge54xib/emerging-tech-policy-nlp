"""Step 1: Flair NER (ORG + PER) with numeric document/entity IDs."""

from __future__ import annotations

import json
import re

from tqdm import tqdm

from src import config, utils


def _clean_mention(value: str) -> str:
    """Collapse whitespace in an entity surface form."""
    return re.sub(r"\s+", " ", (value or "")).strip()


def _canonical_mention(value: str) -> str:
    """Return a lowercased, whitespace-normalised form used as a deduplication key."""
    return utils.normalize_text(value)


def _batched(items: list, batch_size: int):
    """Yield successive slices of *items* of at most *batch_size* elements."""
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def _sentence_text(sentence) -> str:
    """Extract plain text from a Flair Sentence, trying multiple API variants."""
    if hasattr(sentence, "to_original_text"):
        try:
            return str(sentence.to_original_text())
        except Exception:
            pass
    if hasattr(sentence, "to_plain_string"):
        try:
            return str(sentence.to_plain_string())
        except Exception:
            pass
    return str(getattr(sentence, "text", ""))


def _resolve_offsets(doc_text: str, sentence, span) -> tuple[int, int] | None:
    """Map Flair span positions back to absolute character offsets in *doc_text*.

    Returns (start_char, end_char) on success, or None if the span cannot be
    located in the document text.
    """
    if not hasattr(span, "start_position") or not hasattr(span, "end_position"):
        return None

    start = int(span.start_position)
    end = int(span.end_position)
    if start < 0 or end <= start:
        return None

    sentence_start = int(getattr(sentence, "start_position", 0) or 0)
    sentence_text = _sentence_text(sentence)
    sentence_len = len(sentence_text)
    mention = _clean_mention(getattr(span, "text", ""))

    candidates: list[tuple[int, int]] = []
    if sentence_len and 0 <= start <= end <= sentence_len:
        candidates.append((sentence_start + start, sentence_start + end))
    candidates.append((start, end))

    for cand_start, cand_end in candidates:
        if not (0 <= cand_start < cand_end <= len(doc_text)):
            continue
        surface = _clean_mention(doc_text[cand_start:cand_end])
        if not surface:
            continue
        if surface == mention or mention in surface or surface in mention:
            return cand_start, cand_end

    for cand_start, cand_end in candidates:
        if 0 <= cand_start < cand_end <= len(doc_text):
            return cand_start, cand_end
    return None


def run() -> None:
    print(">>> STEP 1: Flair NER (ORG + PER, numeric IDs)")

    # ── Detect already-processed docs ────────────────────────────────────────
    existing_doc_ids: dict[str, int] = {}  # doc_name -> doc_id
    if config.FILE_NER_OUTPUT.exists():
        with open(config.FILE_NER_OUTPUT, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    dn = r.get("doc_name", "")
                    di = r.get("doc_id")
                    if dn and di and dn not in existing_doc_ids:
                        existing_doc_ids[dn] = int(di)
                except Exception:
                    pass
    next_doc_id = max(existing_doc_ids.values(), default=0) + 1

    files = sorted(config.INPUT_DIR.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {config.INPUT_DIR}")
        return

    new_files = [f for f in files if f.stem not in existing_doc_ids]
    if not new_files:
        print(f"[OK] Step 1: all {len(files)} docs already processed, nothing to do.")
        return

    print(f"[INFO] Skipping {len(existing_doc_ids)} already-processed docs, running NER on {len(new_files)} new docs.")

    try:
        from flair.models import SequenceTagger
        from flair.splitter import SegtokSentenceSplitter

        tagger = SequenceTagger.load(config.FLAIR_NER_MODEL)
        splitter = SegtokSentenceSplitter()
    except Exception as exc:
        print(f"Error loading Flair model: {exc}")
        return

    written = 0
    with open(config.FILE_NER_OUTPUT, "a", encoding="utf-8") as out:
        for path in tqdm(new_files, desc="NER Docs"):
            doc_name = path.stem
            doc_num = next_doc_id
            next_doc_id += 1
            text = path.read_text(encoding="utf-8", errors="ignore")[: config.MAX_CHARS_PER_DOC]
            sentences = splitter.split(text)

            entity_id_by_canonical: dict[str, int] = {}
            next_entity_id = 1
            next_mention_id = 1

            for sentence_batch in _batched(sentences, config.FLAIR_NER_BATCH_SIZE):
                tagger.predict(sentence_batch, mini_batch_size=config.FLAIR_NER_BATCH_SIZE)
                for sentence in sentence_batch:
                    for span in sentence.get_spans("ner"):
                        label = span.get_label("ner").value
                        if label not in {"ORG", "PER"}:
                            continue

                        mention = _clean_mention(span.text)
                        canonical = _canonical_mention(mention)
                        if not mention or not canonical:
                            continue

                        offsets = _resolve_offsets(text, sentence, span)
                        if offsets is None:
                            continue
                        start_char, end_char = offsets

                        entity_id = entity_id_by_canonical.get(canonical)
                        if entity_id is None:
                            entity_id = next_entity_id
                            entity_id_by_canonical[canonical] = entity_id
                            next_entity_id += 1

                        mention_id = next_mention_id
                        next_mention_id += 1

                        rec = {
                            "doc_id": doc_num,
                            "doc_name": doc_name,
                            "entity_id": entity_id,
                            "mention_id": mention_id,
                            "entity_key": f"{doc_num}:{entity_id}",
                            "mention": mention,
                            "canonical_mention": canonical,
                            "label": label,
                            "start": start_char,
                            "end": end_char,
                            "start_char": start_char,
                            "end_char": end_char,
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1

    print(f"Model: {config.FLAIR_NER_MODEL}")
    print(f"Done. Appended {written} NER entities for {len(new_files)} new docs to {config.FILE_NER_OUTPUT}")


if __name__ == "__main__":
    run()
