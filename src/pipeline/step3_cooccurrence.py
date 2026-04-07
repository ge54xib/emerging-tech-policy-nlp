"""Step 3: Co-occurrence extraction (sentence splitting + entity pair building).

Builds entity co-occurrence pairs at sentence level from NER mentions and
helix classifications. NLI scoring is a separate experiment (see Experiments/).

Output: cooccurrence.jsonl with sentence text and entity pair metadata.

Builds on:
- Step 1 NER mentions (mention spans)
- Step 2 classifications (new schema: status + level_1..level_5)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

from src import config, utils
from src.utils import normalize_helix, to_int, to_str


NON_HELIX_SPHERES = {"", "unknown"}


def _actor_level_from_ner_label(label: str) -> str:
    """Map a Flair NER label ('ORG' / 'PER') to a QH actor level string."""
    raw = to_str(label).strip().upper()
    if raw == "ORG":
        return "institutional"
    if raw == "PER":
        return "individual"
    return "unknown"


def _load_step1_mentions() -> tuple[dict[str, list[dict]], dict[str, int]]:
    if not config.STEP1_NER_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {config.STEP1_NER_PATH}. Run step1 first."
        )

    mentions_by_doc: dict[str, list[dict]] = defaultdict(list)
    doc_numeric_by_name: dict[str, int] = {}
    for row in utils.load_jsonl(config.STEP1_NER_PATH):
        doc_name = to_str(row.get("doc_name", "")).strip()
        if not doc_name:
            continue
        doc_id = to_int(row.get("doc_id", 0), 0)
        entity_id = to_int(row.get("entity_id", 0), 0)
        start_char = to_int(row.get("start_char", row.get("start", -1)), -1)
        end_char = to_int(row.get("end_char", row.get("end", -1)), -1)
        if not entity_id or start_char < 0 or end_char <= start_char:
            continue

        doc_numeric_by_name.setdefault(doc_name, doc_id)
        mentions_by_doc[doc_name].append(
            {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "entity_id": entity_id,
                "mention_id": to_int(row.get("mention_id", 0), 0),
                "mention": to_str(row.get("mention", "")).strip(),
                "canonical_mention": to_str(row.get("canonical_mention", "")).strip(),
                "label": to_str(row.get("label", "")).strip().upper(),
                "start_char": start_char,
                "end_char": end_char,
                "source": "ner",
            }
        )

    for mentions in mentions_by_doc.values():
        mentions.sort(
            key=lambda m: (
                m.get("start_char", 10**9),
                m.get("end_char", 10**9),
                m.get("mention_id", 10**9),
            )
        )
    return mentions_by_doc, doc_numeric_by_name


def _normalize_label_row(row: dict, *, actor_level_hint: str = "unknown") -> dict:
    """Normalise a Step-2 classification row into the fields needed by Step 3."""
    status = to_str(row.get("status", "entity")).strip().lower()
    if not status:
        status = "entity"

    if status != "entity":
        return {
            "entity": to_str(row.get("entity", "")).strip(),
            "entity_label": None,
            "status": status,
            "level_1_actor_type": None,
            "level_2_sphere_boundary": None,
            "level_3_exact_category": None,
            "level_4_innovation_type": None,
            "level_5_helix": None,

            "confidence": None,
            "website_url": None,
            "website_about_page": None,
            "local_name": None,
            "english_name": None,
            "abbreviation": None,
            "local_abbreviation": None,
            "notes": to_str(row.get("notes", "")).strip(),
            "checked": bool(row.get("checked", False) or row.get("manual_checked", False)),
        }

    level_1 = to_str(row.get("level_1_actor_type", "")).strip() or actor_level_hint
    return {
        "entity": to_str(row.get("entity", "")).strip(),
        "entity_label": to_str(row.get("entity_label", "")).strip(),
        "status": status,
        "level_1_actor_type": level_1,
        "level_2_sphere_boundary": to_str(row.get("level_2_sphere_boundary", "")).strip(),
        "level_3_exact_category": to_str(row.get("level_3_exact_category", "")).strip(),
        "level_4_innovation_type": to_str(row.get("level_4_innovation_type", "")).strip(),
        "level_5_helix": normalize_helix(to_str(row.get("level_5_helix", "")).strip()),

        "confidence": to_str(row.get("confidence", "")).strip(),
        "website_url": to_str(row.get("website_url", "")).strip()
        or to_str(row.get("website_about_url", "")).strip(),
        "website_about_page": to_str(row.get("website_about_page", "")).strip(),
        "local_name": to_str(row.get("local_name", "")).strip(),
        "english_name": to_str(row.get("english_name", "")).strip(),
        "abbreviation": to_str(row.get("abbreviation", "")).strip(),
        "local_abbreviation": to_str(row.get("local_abbreviation", "")).strip(),
        "notes": to_str(row.get("notes", "")).strip(),
        "checked": bool(row.get("checked", False) or row.get("manual_checked", False)),
    }


def _build_classification_lookup_from_classified() -> tuple[dict[tuple[int, int], dict], int]:
    if not config.STEP2_CLASSIFIED_PATH.exists():
        return {}, 0

    lookup: dict[tuple[int, int], dict] = {}
    non_empty_sphere = 0
    for row in utils.load_jsonl(config.STEP2_CLASSIFIED_PATH):
        doc_id = to_int(row.get("doc_id", 0), 0)
        entity_id = to_int(row.get("entity_id", 0), 0)
        if not doc_id or not entity_id:
            continue
        actor_level = _actor_level_from_ner_label(to_str(row.get("label", "")))
        normalized = _normalize_label_row(row, actor_level_hint=actor_level)
        lookup[(doc_id, entity_id)] = normalized
        if normalized.get("level_5_helix") not in NON_HELIX_SPHERES | {None}:
            non_empty_sphere += 1
    return lookup, non_empty_sphere


def _build_classification_lookup_from_manual_labels() -> tuple[dict[tuple[int, int], dict], int]:
    if not config.STEP2_MANUAL_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Missing manual labels: {config.STEP2_MANUAL_LABELS_PATH}. "
            "Run step2 or the manual classification UI first."
        )
    payload = json.loads(config.STEP2_MANUAL_LABELS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(
            f"Invalid labels format in {config.STEP2_MANUAL_LABELS_PATH}: expected JSON array."
        )

    lookup: dict[tuple[int, int], dict] = {}
    non_empty_sphere = 0
    for row in payload:
        doc_id = to_int(row.get("doc_id", 0), 0)
        entity_id = to_int(row.get("entity_id", 0), 0)
        if not doc_id or not entity_id:
            continue
        normalized = _normalize_label_row(row)
        lookup[(doc_id, entity_id)] = normalized
        if normalized.get("level_5_helix") not in NON_HELIX_SPHERES | {None}:
            non_empty_sphere += 1
    return lookup, non_empty_sphere


def _hydrate_step2_classified(
    mentions_by_doc: dict[str, list[dict]], classification_lookup: dict[tuple[int, int], dict]
) -> None:
    """Write a refreshed Step2 classified JSONL from Step1 mentions + labels."""

    out_rows: list[dict] = []
    for doc_name in sorted(mentions_by_doc):
        for mention in mentions_by_doc[doc_name]:
            row = dict(mention)
            key = (int(mention["doc_id"]), int(mention["entity_id"]))
            cls = classification_lookup.get(key, {})

            row["entity_key"] = f"{key[0]}:{key[1]}"
            row["start"] = mention["start_char"]
            row["end"] = mention["end_char"]

            status = to_str(cls.get("status", "entity")).strip() or "entity"
            is_entity = status == "entity"

            row["entity"] = to_str(cls.get("entity", "")) or row.get("canonical_mention", "")
            row["entity_label"] = cls.get("entity_label")
            row["status"] = status
            row["level_1_actor_type"] = (
                cls.get("level_1_actor_type") if is_entity else None
            ) or (_actor_level_from_ner_label(to_str(mention.get("label", ""))) if is_entity else None)
            row["level_2_sphere_boundary"] = cls.get("level_2_sphere_boundary") if is_entity else None
            row["level_3_exact_category"] = cls.get("level_3_exact_category") if is_entity else None
            row["level_4_innovation_type"] = cls.get("level_4_innovation_type") if is_entity else None
            row["level_5_helix"] = cls.get("level_5_helix") if is_entity else None
            row["confidence"] = cls.get("confidence") if is_entity else None
            row["website_url"] = cls.get("website_url") if is_entity else None
            row["website_about_page"] = cls.get("website_about_page") if is_entity else None
            row["local_name"] = cls.get("local_name") if is_entity else None
            row["english_name"] = cls.get("english_name") if is_entity else None
            row["abbreviation"] = cls.get("abbreviation") if is_entity else None
            row["local_abbreviation"] = cls.get("local_abbreviation") if is_entity else None
            row["notes"] = cls.get("notes", "")
            row["checked"] = bool(cls.get("checked", False) or cls.get("manual_checked", False))

            out_rows.append(row)

    utils.write_jsonl(config.STEP2_CLASSIFIED_PATH, out_rows)



def _build_sentence_splitter():
    """Load spaCy with only the sentence segmenter enabled (fast)."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
        if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception as exc:
        print(f"[WARN] spaCy sentence splitter unavailable ({exc}). Falling back to paragraph-level.")
        return None


_BULLET_RE = re.compile(r"(?m)^[ \t]*[-•·\*]\s+")


def _pre_split_bullets(para_text: str) -> list[str]:
    """Split paragraph on bullet-list markers before sentence tokenization.

    Returns a list of text chunks — each bullet item becomes its own chunk,
    which is then treated as a separate sentence by the sentence splitter.
    Non-bullet text is returned as a single chunk.
    """
    # Find all bullet positions
    splits = [m.start() for m in _BULLET_RE.finditer(para_text)]
    if not splits:
        return [para_text]
    chunks: list[str] = []
    # Text before first bullet
    if splits[0] > 0:
        pre = para_text[: splits[0]].strip()
        if pre:
            chunks.append(pre)
    for i, start in enumerate(splits):
        end = splits[i + 1] if i + 1 < len(splits) else len(para_text)
        chunk = _BULLET_RE.sub("", para_text[start:end]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [para_text]


def _sentence_spans(para_text: str, para_start_char: int, nlp) -> list[dict]:
    """Split paragraph text into sentence spans with doc-level char offsets.

    Pre-splits on bullet-list markers so each bullet item becomes its own
    sentence, avoiding multi-line inputs that degrade classifier quality.
    """
    if nlp is None:
        return [{"sentence_id": 1, "start_char": para_start_char, "end_char": para_start_char + len(para_text), "text": para_text}]

    # Pre-split on bullet markers, then run spaCy on each chunk
    chunks = _pre_split_bullets(para_text)
    sentences = []
    sid = 1
    cursor = para_start_char
    for chunk in chunks:
        doc = nlp(chunk)
        for sent in doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            sentences.append({
                "sentence_id": sid,
                "start_char": cursor + sent.start_char,
                "end_char": cursor + sent.end_char,
                "text": text,
            })
            sid += 1
        cursor += len(chunk) + 1  # +1 for the stripped newline/space between chunks
    return sentences if sentences else [{"sentence_id": 1, "start_char": para_start_char, "end_char": para_start_char + len(para_text), "text": para_text}]


def _paragraph_spans(text: str) -> list[dict]:
    """Split document into paragraph blocks with character offsets."""

    if not text:
        return []

    bounds: list[tuple[int, int]] = []
    cursor = 0
    for match in re.finditer(r"\n\s*\n+", text):
        bounds.append((cursor, match.start()))
        cursor = match.end()
    bounds.append((cursor, len(text)))

    paragraphs: list[dict] = []
    paragraph_id = 1
    for raw_start, raw_end in bounds:
        raw = text[raw_start:raw_end]
        if not raw:
            continue
        lead_trim = len(raw) - len(raw.lstrip())
        tail_trim = len(raw) - len(raw.rstrip())
        start_char = raw_start + lead_trim
        end_char = raw_end - tail_trim
        if end_char <= start_char:
            continue
        paragraph_text = text[start_char:end_char]
        paragraphs.append(
            {
                "paragraph_id": paragraph_id,
                "start_char": start_char,
                "end_char": end_char,
                "text": paragraph_text,
            }
        )
        paragraph_id += 1
    return paragraphs


def _find_paragraph_id(paragraphs: list[dict], start_char: int, end_char: int) -> int | None:
    if not paragraphs:
        return None
    midpoint = (start_char + end_char) // 2
    for paragraph in paragraphs:
        if paragraph["start_char"] <= midpoint < paragraph["end_char"]:
            return int(paragraph["paragraph_id"])
    return None


def _doc_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in sorted(config.STEP0_TEXT_DIR.glob("*.txt")):
        texts[path.stem] = path.read_text(encoding="utf-8", errors="ignore")[: config.MAX_DOC_CHARS]
    if not texts:
        raise FileNotFoundError(
            f"No text files found in {config.STEP0_TEXT_DIR}. Run step0 first."
        )
    return texts


def _country_year_from_doc_name(doc_name: str) -> tuple[str, str]:
    parts = to_str(doc_name).split("_")
    country = parts[0].upper() if parts else "UNK"
    year = parts[1] if len(parts) > 1 else ""
    return country, year


def run() -> None:
    print(">>> STEP 3: Co-occurrence extraction (sentence splitting + entity pairs)")

    texts = _doc_texts()
    mentions_by_doc, doc_numeric_by_name = _load_step1_mentions()

    classified_lookup, sphere_count = _build_classification_lookup_from_classified()
    source = "step2_classified"
    if sphere_count == 0:
        classified_lookup, sphere_count = _build_classification_lookup_from_manual_labels()
        source = "manual_labels"
        _hydrate_step2_classified(mentions_by_doc, classified_lookup)
        print(
            "[INFO] STEP2 classified file had no helix assignments. "
            "Rebuilt it from manual/AI labels."
        )

    if sphere_count == 0:
        raise RuntimeError(
            "No classified helix values found after loading labels. "
            f"Check {config.STEP2_MANUAL_LABELS_PATH}."
        )

    print(f"[INFO] Classification source for Step3: {source} ({sphere_count} labeled entities)")

    sentence_splitter = _build_sentence_splitter()
    if sentence_splitter is not None:
        print("[INFO] Sentence-level co-occurrence enabled (spaCy en_core_web_sm)")
    else:
        print("[INFO] Falling back to paragraph-level co-occurrence")

    # ── Detect already-processed docs ────────────────────────────────────────
    existing_doc_names: set[str] = set()
    if config.FILE_PARAGRAPHS.exists():
        import json as _json
        with config.FILE_PARAGRAPHS.open(encoding="utf-8") as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _r = _json.loads(_line)
                    _dn = str(_r.get("doc_id", "")).strip()
                    if _dn:
                        existing_doc_names.add(_dn)
                except Exception:
                    pass

    new_texts = {k: v for k, v in texts.items() if k not in existing_doc_names}
    if not new_texts:
        print(f"[OK] Step 3: all {len(texts)} docs already processed, nothing to do.")
        return

    print(f"[INFO] Skipping {len(existing_doc_names)} already-processed docs, processing {len(new_texts)} new docs.")

    total_paragraphs = 0
    total_alignment_rows = 0
    total_coocc_rows = 0
    pair_counter: Counter[str] = Counter()

    with (
        config.FILE_PARAGRAPHS.open("a", encoding="utf-8") as paragraphs_out,
        config.FILE_ALIGNMENT.open("a", encoding="utf-8") as alignment_out,
        config.FILE_COOCCURRENCE.open("a", encoding="utf-8") as coocc_out,
    ):
        for doc_name in tqdm(sorted(new_texts), desc="Co-occurrence Extraction"):
            text = new_texts[doc_name]
            paragraphs = _paragraph_spans(text)
            country, year = _country_year_from_doc_name(doc_name)
            para_text_by_id: dict[int, str] = {
                p["paragraph_id"]: p["text"] for p in paragraphs
            }
            para_start_by_id: dict[int, int] = {
                p["paragraph_id"]: p["start_char"] for p in paragraphs
            }
            for paragraph in paragraphs:
                paragraph_row = {
                    "doc_id": doc_name,
                    "paragraph_id": paragraph["paragraph_id"],
                    "start_char": paragraph["start_char"],
                    "end_char": paragraph["end_char"],
                    "country": country,
                    "year": year,
                    "text": paragraph["text"],
                }
                paragraphs_out.write(json.dumps(paragraph_row, ensure_ascii=False) + "\n")
            total_paragraphs += len(paragraphs)

            doc_mentions = mentions_by_doc.get(doc_name, [])

            # Build sentence spans per paragraph
            # Key: (paragraph_id, sentence_id) -> sentence info
            sentence_text_by_key: dict[tuple[int, int], str] = {}
            sentence_start_by_key: dict[tuple[int, int], int] = {}
            # For each paragraph, compute sentences once
            sentences_by_para: dict[int, list[dict]] = {}
            for paragraph in paragraphs:
                pid = paragraph["paragraph_id"]
                sents = _sentence_spans(paragraph["text"], paragraph["start_char"], sentence_splitter)
                sentences_by_para[pid] = sents
                for s in sents:
                    key = (pid, s["sentence_id"])
                    sentence_text_by_key[key] = s["text"]
                    sentence_start_by_key[key] = s["start_char"]

            # entities_per_sentence: (para_id, sentence_id) -> {entity_id -> entity_dict}
            entities_per_sentence: dict[tuple[int, int], dict[int, dict]] = defaultdict(dict)

            for mention in doc_mentions:
                m_start = to_int(mention.get("start_char", -1), -1)
                m_end = to_int(mention.get("end_char", -1), -1)
                para_id = _find_paragraph_id(paragraphs, m_start, m_end)
                if para_id is None:
                    continue

                # Find which sentence contains the mention midpoint
                m_mid = (m_start + m_end) // 2
                sentence_id = 1
                for s in sentences_by_para.get(para_id, []):
                    if s["start_char"] <= m_mid < s["end_char"]:
                        sentence_id = s["sentence_id"]
                        break

                key = (to_int(mention.get("doc_id", 0), 0), to_int(mention.get("entity_id", 0), 0))
                cls = classified_lookup.get(key, {})
                status = to_str(cls.get("status", "entity")).strip() or "entity"
                helix = normalize_helix(to_str(cls.get("level_5_helix", "")).strip())
                actor_level = to_str(cls.get("level_1_actor_type", "")).strip()
                if not actor_level and status == "entity":
                    actor_level = _actor_level_from_ner_label(to_str(mention.get("label", "")))
                if not actor_level:
                    actor_level = "unknown"

                entity_name = (
                    to_str(cls.get("entity", "")).strip()
                    or to_str(mention.get("canonical_mention", "")).strip()
                    or to_str(mention.get("mention", "")).strip()
                )
                alignment_row = {
                    "doc_id": doc_name,
                    "paragraph_id": para_id,
                    "sentence_id": sentence_id,
                    "entity_id": key[1],
                    "entity": entity_name,
                    "mention_text": to_str(mention.get("mention", "")).strip(),
                    "mention_start_char": m_start,
                    "mention_end_char": m_end,
                    "mention_source": to_str(mention.get("source", "")).strip() or "ner",
                    "status": status,
                    "level_1_actor_type": actor_level,
                    "level_5_helix": helix,
                    "country": country,
                    "year": year,
                }
                alignment_out.write(json.dumps(alignment_row, ensure_ascii=False) + "\n")
                total_alignment_rows += 1

                if status != "entity":
                    continue
                if helix in NON_HELIX_SPHERES:
                    continue
                sent_key = (para_id, sentence_id)
                if key[1] not in entities_per_sentence[sent_key]:
                    entities_per_sentence[sent_key][key[1]] = {
                        "entity_id": key[1],
                        "entity": entity_name,
                        "actor_level": actor_level,
                        "helix": helix,
                        "mention_start_char": m_start,
                        "mention_end_char": m_end,
                    }

            for (para_id, sentence_id), entity_map in entities_per_sentence.items():
                entities = sorted(entity_map.values(), key=lambda row: row["entity_id"])
                if len(entities) < 2:
                    continue
                sent_key = (para_id, sentence_id)

                # Use only the central sentence — no ±1 window
                para_sents = sentences_by_para.get(para_id, [])
                sent_by_id = {s["sentence_id"]: s["text"] for s in para_sents}
                central_sent_text = sent_by_id.get(sentence_id, "")

                for left, right in combinations(entities, 2):
                    h1 = to_str(left.get("helix", "")).strip()
                    h2 = to_str(right.get("helix", "")).strip()
                    pair = "–".join(sorted([h1, h2]))
                    pair_counter[pair] += 1

                    coocc_row = {
                        "doc_id": doc_name,
                        "paragraph_id": para_id,
                        "sentence_id": sentence_id,
                        "sent_text": central_sent_text,
                        "central_sent_text": central_sent_text,
                        "country": country,
                        "year": year,
                        "entity_id_1": left["entity_id"],
                        "entity_1": left["entity"],
                        "h1": h1,
                        "actor_1": {
                            "entity_id": left["entity_id"],
                            "entity": left["entity"],
                            "helix": h1,
                            "actor_level": left["actor_level"],
                            "mention_start_char": left["mention_start_char"],
                            "mention_end_char": left["mention_end_char"],
                        },
                        "entity_id_2": right["entity_id"],
                        "entity_2": right["entity"],
                        "h2": h2,
                        "actor_2": {
                            "entity_id": right["entity_id"],
                            "entity": right["entity"],
                            "helix": h2,
                            "actor_level": right["actor_level"],
                            "mention_start_char": right["mention_start_char"],
                            "mention_end_char": right["mention_end_char"],
                        },
                        "pair": pair,
                    }
                    coocc_out.write(json.dumps(coocc_row, ensure_ascii=False) + "\n")
                    total_coocc_rows += 1

    print(f"[OK] Wrote paragraphs: {config.FILE_PARAGRAPHS} ({total_paragraphs} rows)")
    print(f"[OK] Wrote paragraph alignment: {config.FILE_ALIGNMENT} ({total_alignment_rows} rows)")
    print(f"[OK] Wrote co-occurrence pairs: {config.FILE_COOCCURRENCE} ({total_coocc_rows} rows)")
    print("[INFO] Run NLI experiments separately: Experiments/Relation/nli_pipeline/run.py")
    if pair_counter:
        top = ", ".join(f"{pair}={count}" for pair, count in pair_counter.most_common(10))
        print(f"[INFO] Top helix pairs: {top}")


if __name__ == "__main__":
    run()
