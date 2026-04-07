"""Step 2: Apply manual labels in the new Level-1..Level-5 schema.

This module intentionally drops legacy fields such as:
- old status/sphere naming and fallback fields
- deprecated decision-logic compatibility fields

Quadruple Helix classification schema (Ranga & Etzkowitz 2013)
--------------------------------------------------------------
Level 1  actor_type         institutional | individual
Level 2  sphere_boundary    single-sphere | multi-sphere (hybrid)
Level 3  exact_category     fine-grained type (e.g. "national government institutions")
Level 4  innovation_type    R&D | Non R&D | Both
Level 5  helix              government | industry | academia |
                            civil_society | intermediary
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src import config, utils


ORG_LABEL_FIELDS = (
    "doc_id",
    "doc_name",
    "entity_id",
    "entity",
    "canonical_mention",
    "stable_actor_key",
    "entity_label",
    "status",
    "excluded_reason",
    "level_1_actor_type",
    "level_2_sphere_boundary",
    "level_2_evidence_type",
    "level_3_exact_category",
    "level_3_evidence_type",
    "level_3_evidence_operating_scope",
    "level_3_evidence_linkedin_url",
    "level_3_evidence_linkedin_hiring_growth_2y",
    "level_3_evidence_linkedin_employees_2y_ago",
    "level_3_evidence_linkedin_employees_today",
    "level_4_innovation_type",
    "level_4_evidence_type",
    "level_5_helix",
    "level_5_evidence_type",
    "occurrence_sentence",
    "context_window_400",
    "evidence_snippet_r_and_d",
    "evidence_snippet_non_r_and_d",
    "website_url",
    "website_about_page",
    "institution_origin_scope",
    "institution_origin_country",
    "local_name",
    "english_name",
    "abbreviation",
    "local_abbreviation",
    "notes",
    "checked",
    "final_review_at",
)

PER_LABEL_FIELDS = (
    "doc_id",
    "doc_name",
    "entity_id",
    "entity",
    "canonical_mention",
    "stable_actor_key",
    "entity_label",
    "status",
    "excluded_reason",
    "level_1_actor_type",
    "level_2_sphere_boundary",
    "level_2_evidence_type",
    "level_3_exact_category",
    "level_3_evidence_type",
    "level_4_innovation_type",
    "level_4_evidence_type",
    "level_5_helix",
    "level_5_evidence_type",
    "occurrence_sentence",
    "context_window_400",
    "evidence_snippet_r_and_d",
    "evidence_snippet_non_r_and_d",
    "website_url",
    "website_about_page",
    "name",
    "surname",
    "current_individual_affiliation_local_name",
    "current_individual_affiliation_english_name",
    "current_individual_affiliation_local_abbreviation",
    "current_individual_affiliation_english_abbreviation",
    "strategy_individual_affiliation_local_name",
    "strategy_individual_affiliation_english_name",
    "strategy_individual_affiliation_local_abbreviation",
    "strategy_individual_affiliation_english_abbreviation",
    "secondary_strategy_individual_affiliation_local_name",
    "secondary_strategy_individual_affiliation_english_name",
    "secondary_strategy_individual_affiliation_local_abbreviation",
    "secondary_strategy_individual_affiliation_english_abbreviation",
    "notes",
    "checked",
    "final_review_at",
)


def _label_fields_for_type(actor_type: str) -> tuple:
    """Return the ordered field tuple for the given actor type."""
    if str(actor_type).strip().lower() == "individual":
        return PER_LABEL_FIELDS
    return ORG_LABEL_FIELDS

NON_ENTITY_STATUSES = {
    "historical_individual",
    "ner_error",
    "not_actor",
    "not_specific",
    "non_entity_phrase",
    "unclear",
}
LEGACY_AFFILIATION_FIELD_MAP = {
    "individual_affiliation_local_name": "strategy_individual_affiliation_local_name",
    "individual_affiliation_english_name": "strategy_individual_affiliation_english_name",
    "individual_affiliation_local_abbreviation": "strategy_individual_affiliation_local_abbreviation",
    "individual_affiliation_english_abbreviation": "strategy_individual_affiliation_english_abbreviation",
    "institution_local_name": "strategy_individual_affiliation_local_name",
    "institution_english_name": "strategy_individual_affiliation_english_name",
    "institution_local_abbreviation": "strategy_individual_affiliation_local_abbreviation",
    "institution_english_abbreviation": "strategy_individual_affiliation_english_abbreviation",
}


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _canonicalize_label_row(row: dict) -> dict:
    """Return a copy of *row* with all schema fields present and correctly typed."""
    actor_type = str(row.get("level_1_actor_type", "")).strip().lower()
    fields = _label_fields_for_type(actor_type)
    ordered: dict = {}
    source = dict(row)
    for field in fields:
        if field in {"doc_id", "entity_id"}:
            default = 0
        elif field == "checked":
            default = False
        else:
            default = ""
        value = source.get(field, default)
        if field == "checked":
            value = bool(value)
        ordered[field] = value
    return ordered


def _canonicalize_label_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(_canonicalize_label_row(row))
    return out


def _to_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _context_window(text: str, start_char: int, end_char: int, window_chars: int = 400) -> str:
    """Extract up to *window_chars* characters centred on the entity span."""
    if not text:
        return ""

    n = len(text)
    if n <= window_chars:
        return " ".join(text.split())

    if start_char < 0 or end_char <= start_char:
        return " ".join(text[:window_chars].split())

    center = start_char + max(1, end_char - start_char) // 2
    half = window_chars // 2
    left = max(0, center - half)
    right = min(n, left + window_chars)
    if right - left < window_chars:
        left = max(0, right - window_chars)

    return " ".join(text[left:right].split())


def _affiliation_value(row: dict, new_field: str) -> str:
    """Return affiliation value, falling back to legacy field names if needed."""
    value = _to_text(row.get(new_field, ""))
    if value:
        return value
    for old_field, mapped_new in LEGACY_AFFILIATION_FIELD_MAP.items():
        if mapped_new == new_field:
            return _to_text(row.get(old_field, ""))
    return ""


def _canonical_mention(value: str) -> str:
    return utils.normalize_text(value)


def _stable_actor_key(doc_name: str, canonical_mention: str) -> str:
    doc = _to_text(doc_name)
    canonical = _canonical_mention(canonical_mention)
    if not doc or not canonical:
        return ""
    return f"{doc}::{canonical}"


def _actor_level_from_labels(labels: set[str]) -> str:
    normalized = {str(label or "").upper() for label in labels if str(label or "").strip()}
    if "ORG" in normalized and "PER" not in normalized:
        return "institutional"
    if "PER" in normalized and "ORG" not in normalized:
        return "individual"
    if "ORG" in normalized:
        return "institutional"
    if "PER" in normalized:
        return "individual"
    return "unknown"


def _require_step1_numeric_ids(rows: list[dict]) -> None:
    for idx, row in enumerate(rows, start=1):
        try:
            int(row.get("doc_id"))
            int(row.get("entity_id"))
        except Exception as exc:
            raise RuntimeError(
                "Step1 output is in old format. Please rerun step1 so doc_id/entity_id are numeric. "
                f"First invalid row index: {idx}."
            ) from exc


def _group_entities(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], dict] = defaultdict(
        lambda: {
            "doc_id": 0,
            "doc_name": "",
            "entity_id": 0,
            "canonical_mention": "",
            "mentions": set(),
            "ner_labels": set(),
            "first_start_char": -1,
            "first_end_char": -1,
        }
    )

    for row in rows:
        doc_id = int(row.get("doc_id"))
        entity_id = int(row.get("entity_id"))
        mention = _to_text(row.get("mention", ""))
        canonical = _to_text(row.get("canonical_mention", "")) or _canonical_mention(mention)
        if not canonical:
            continue

        key = (doc_id, entity_id)
        item = grouped[key]
        doc_name = _to_text(row.get("doc_name", ""))
        item["doc_id"] = doc_id
        item["doc_name"] = doc_name
        item["entity_id"] = entity_id
        item["canonical_mention"] = canonical
        if mention:
            item["mentions"].add(mention)
        label = _to_text(row.get("label", "")).upper()
        if label:
            item["ner_labels"].add(label)
        start_char = _to_int(row.get("start_char", row.get("start", -1)), -1)
        end_char = _to_int(row.get("end_char", row.get("end", -1)), -1)
        if start_char >= 0 and end_char > start_char:
            existing_start = _to_int(item.get("first_start_char", -1), -1)
            if existing_start < 0 or start_char < existing_start:
                item["first_start_char"] = start_char
                item["first_end_char"] = end_char

    entities: list[dict] = []
    for key in sorted(grouped):
        item = grouped[key]
        entities.append(
            {
                "doc_id": item["doc_id"],
                "doc_name": item["doc_name"],
                "entity_id": item["entity_id"],
                "canonical_mention": item["canonical_mention"],
                "mentions": sorted(item["mentions"]),
                "ner_labels": sorted(item["ner_labels"]),
                "first_start_char": _to_int(item.get("first_start_char", -1), -1),
                "first_end_char": _to_int(item.get("first_end_char", -1), -1),
            }
        )
    return entities


def _build_template(rows: list[dict]) -> list[dict]:
    entities = _group_entities(rows)
    records: list[dict] = []
    doc_text_cache: dict[str, str] = {}
    for entity in entities:
        canonical = entity["canonical_mention"]
        doc_name = entity["doc_name"]
        if doc_name not in doc_text_cache:
            doc_path = config.STEP0_TEXT_DIR / f"{doc_name}.txt"
            doc_text_cache[doc_name] = utils.read_text(doc_path) if doc_path.exists() else ""
        context_window_400 = _context_window(
            doc_text_cache.get(doc_name, ""),
            _to_int(entity.get("first_start_char", -1), -1),
            _to_int(entity.get("first_end_char", -1), -1),
            window_chars=400,
        )
        actor_level = _actor_level_from_labels(set(entity["ner_labels"]))
        stable_key = _stable_actor_key(doc_name, canonical)
        base: dict = {
            "doc_id": entity["doc_id"],
            "doc_name": doc_name,
            "entity_id": entity["entity_id"],
            "entity": canonical,
            "canonical_mention": canonical,
            "stable_actor_key": stable_key,
            "entity_label": "",
            "status": "entity",
            "excluded_reason": "",
            "level_1_actor_type": actor_level,
            "level_2_sphere_boundary": "",
            "level_2_evidence_type": "",
            "level_3_exact_category": "",
            "level_3_evidence_type": "",
            "level_4_innovation_type": "",
            "level_4_evidence_type": "",
            "level_5_helix": "",
            "level_5_evidence_type": "",

            "occurrence_sentence": "",
            "context_window_400": context_window_400,

            "evidence_snippet_r_and_d": "",
            "evidence_snippet_non_r_and_d": "",
            "website_url": "",
            "website_about_page": "",
            "notes": "",
            "checked": False,
            "final_review_at": "",
            "mentions": entity["mentions"],
            "ner_labels": entity["ner_labels"],
        }
        if actor_level == "individual":
            base.update({
                "name": "",
                "surname": "",
                "current_individual_affiliation_local_name": "",
                "current_individual_affiliation_english_name": "",
                "current_individual_affiliation_local_abbreviation": "",
                "current_individual_affiliation_english_abbreviation": "",
                "strategy_individual_affiliation_local_name": "",
                "strategy_individual_affiliation_english_name": "",
                "strategy_individual_affiliation_local_abbreviation": "",
                "strategy_individual_affiliation_english_abbreviation": "",
                "secondary_strategy_individual_affiliation_local_name": "",
                "secondary_strategy_individual_affiliation_english_name": "",
                "secondary_strategy_individual_affiliation_local_abbreviation": "",
                "secondary_strategy_individual_affiliation_english_abbreviation": "",
            })
        else:
            base.update({
                "level_3_evidence_operating_scope": "",
                "level_3_evidence_linkedin_url": "",
                "level_3_evidence_linkedin_hiring_growth_2y": "",
                "level_3_evidence_linkedin_employees_2y_ago": "",
                "level_3_evidence_linkedin_employees_today": "",
                "institution_origin_scope": "",
                "institution_origin_country": "",
                "local_name": "",
                "english_name": "",
                "abbreviation": "",
                "local_abbreviation": "",
            })
        records.append(base)
    return records


_SEED_PASSTHROUGH_FIELDS = {
    "context_window_400",
    "excluded_reason", "final_review_at", "entity", "canonical_mention",
    "stable_actor_key", "doc_name",
}


def _labels_seed_from_template(template_rows: list[dict]) -> list[dict]:
    seed: list[dict] = []
    for row in template_rows:
        actor_type = _to_text(row.get("level_1_actor_type", "")).strip().lower()
        fields = _label_fields_for_type(actor_type)
        entry: dict = {}
        for field in fields:
            if field == "doc_id":
                entry[field] = int(row.get("doc_id", 0) or 0)
            elif field == "entity_id":
                entry[field] = int(row.get("entity_id", 0) or 0)
            elif field == "checked":
                entry[field] = bool(row.get("checked", False) or row.get("manual_checked", False))
            elif field == "status":
                entry[field] = _to_text(row.get("status", "entity")) or "entity"
            elif field == "level_1_actor_type":
                entry[field] = actor_type or "unknown"
            elif field in _SEED_PASSTHROUGH_FIELDS:
                entry[field] = _to_text(row.get(field, ""))
            else:
                entry[field] = ""
        seed.append(entry)
    return seed


def _normalize_status(value: str) -> str:
    status = _to_text(value).lower()
    return status if status else "entity"


def _load_legacy_entity_lookup() -> dict[tuple[int, int], dict[str, str]]:
    path = config.STEP2_CLASSIFIED_PATH
    if not path.exists():
        return {}

    lookup: dict[tuple[int, int], dict[str, str]] = {}
    for row in utils.load_jsonl(path):
        try:
            key = (int(row.get("doc_id", 0) or 0), int(row.get("entity_id", 0) or 0))
        except Exception:
            continue
        if not key[0] or not key[1]:
            continue
        doc_name = _to_text(row.get("doc_name", ""))
        canonical = _to_text(row.get("canonical_mention", "")) or _canonical_mention(
            _to_text(row.get("entity", ""))
        )
        lookup[key] = {
            "doc_name": doc_name,
            "canonical_mention": canonical,
            "stable_actor_key": _stable_actor_key(doc_name, canonical),
        }
    return lookup


def _normalize_label_row(
    row: dict,
    key: tuple[int, int],
    template_row: dict | None = None,
    legacy_lookup: dict[tuple[int, int], dict[str, str]] | None = None,
) -> dict:
    legacy_row = (legacy_lookup or {}).get(key, {})
    row_doc_name = _to_text(row.get("doc_name", ""))
    template_doc_name = _to_text((template_row or {}).get("doc_name", ""))
    legacy_doc_name = _to_text(legacy_row.get("doc_name", ""))
    doc_name = row_doc_name or template_doc_name or legacy_doc_name

    row_canonical = _to_text(row.get("canonical_mention", ""))
    template_canonical = _to_text((template_row or {}).get("canonical_mention", ""))
    legacy_canonical = _to_text(legacy_row.get("canonical_mention", ""))
    canonical_mention = (
        row_canonical
        or template_canonical
        or legacy_canonical
        or _canonical_mention(_to_text(row.get("entity", "")))
    )

    stable_actor_key = _to_text(row.get("stable_actor_key", "")) or _stable_actor_key(
        doc_name, canonical_mention
    )
    template_context_window_400 = _to_text((template_row or {}).get("context_window_400", ""))

    cleaned = {
        "doc_id": int(row.get("doc_id", key[0]) or key[0]),
        "doc_name": doc_name,
        "entity_id": int(row.get("entity_id", key[1]) or key[1]),
        "entity": _to_text(row.get("entity", "")),
        "canonical_mention": canonical_mention,
        "stable_actor_key": stable_actor_key,
        "entity_label": _to_text(row.get("entity_label", "")),
        "status": _normalize_status(row.get("status", "")),
        "level_1_actor_type": _to_text(row.get("level_1_actor_type", "")),
        "level_2_sphere_boundary": _to_text(row.get("level_2_sphere_boundary", "")),
        "level_2_evidence_type": _to_text(row.get("level_2_evidence_type", "")),
        "level_3_exact_category": _to_text(row.get("level_3_exact_category", "")),
        "level_3_evidence_type": _to_text(row.get("level_3_evidence_type", "")),
        "level_3_evidence_operating_scope": _to_text(row.get("level_3_evidence_operating_scope", "")),
        "level_3_evidence_linkedin_url": _to_text(row.get("level_3_evidence_linkedin_url", "")),
        "level_3_evidence_linkedin_hiring_growth_2y": _to_text(
            row.get("level_3_evidence_linkedin_hiring_growth_2y", "")
        ),
        "level_3_evidence_linkedin_employees_2y_ago": _to_text(
            row.get("level_3_evidence_linkedin_employees_2y_ago", "")
        ),
        "level_3_evidence_linkedin_employees_today": _to_text(
            row.get("level_3_evidence_linkedin_employees_today", "")
        ),
        "level_4_innovation_type": _to_text(row.get("level_4_innovation_type", "")),
        "level_4_evidence_type": _to_text(row.get("level_4_evidence_type", "")),
        "level_5_helix": _to_text(row.get("level_5_helix", "")),
        "level_5_evidence_type": _to_text(row.get("level_5_evidence_type", "")),

        "occurrence_sentence": _to_text(row.get("occurrence_sentence", "")),
        "context_window_400": _to_text(row.get("context_window_400", "")) or template_context_window_400,

        "evidence_snippet_r_and_d": _to_text(row.get("evidence_snippet_r_and_d", "")),
        "evidence_snippet_non_r_and_d": _to_text(row.get("evidence_snippet_non_r_and_d", "")),
        "website_url": _to_text(row.get("website_url", "")) or _to_text(row.get("website_about_url", "")),
        "website_about_page": _to_text(row.get("website_about_page", "")),
        "institution_origin_scope": (
            _to_text(row.get("institution_origin_scope", "")).lower()
            if _to_text(row.get("institution_origin_scope", "")).lower() in {"domestic", "foreign"}
            else ""
        ),
        "institution_origin_country": _to_text(row.get("institution_origin_country", "")),
        "name": _to_text(row.get("name", "")),
        "surname": _to_text(row.get("surname", "")),
        "local_name": _to_text(row.get("local_name", "")),
        "english_name": _to_text(row.get("english_name", "")),
        "abbreviation": _to_text(row.get("abbreviation", "")),
        "local_abbreviation": _to_text(row.get("local_abbreviation", "")),
        "current_individual_affiliation_local_name": _to_text(
            row.get("current_individual_affiliation_local_name", "")
        ),
        "current_individual_affiliation_english_name": _to_text(
            row.get("current_individual_affiliation_english_name", "")
        ),
        "current_individual_affiliation_local_abbreviation": _to_text(
            row.get("current_individual_affiliation_local_abbreviation", "")
        ),
        "current_individual_affiliation_english_abbreviation": _to_text(
            row.get("current_individual_affiliation_english_abbreviation", "")
        ),
        "strategy_individual_affiliation_local_name": _affiliation_value(
            row, "strategy_individual_affiliation_local_name"
        ),
        "strategy_individual_affiliation_english_name": _affiliation_value(
            row, "strategy_individual_affiliation_english_name"
        ),
        "strategy_individual_affiliation_local_abbreviation": _affiliation_value(
            row, "strategy_individual_affiliation_local_abbreviation"
        ),
        "strategy_individual_affiliation_english_abbreviation": _affiliation_value(
            row, "strategy_individual_affiliation_english_abbreviation"
        ),
        "secondary_strategy_individual_affiliation_local_name": _to_text(
            row.get("secondary_strategy_individual_affiliation_local_name", "")
        ),
        "secondary_strategy_individual_affiliation_english_name": _to_text(
            row.get("secondary_strategy_individual_affiliation_english_name", "")
        ),
        "secondary_strategy_individual_affiliation_local_abbreviation": _to_text(
            row.get("secondary_strategy_individual_affiliation_local_abbreviation", "")
        ),
        "secondary_strategy_individual_affiliation_english_abbreviation": _to_text(
            row.get("secondary_strategy_individual_affiliation_english_abbreviation", "")
        ),
        "notes": _to_text(row.get("notes", "")),
        "checked": bool(row.get("checked", False) or row.get("manual_checked", False)),
        "excluded_reason": _to_text(row.get("excluded_reason", "")),
        "final_review_at": _to_text(row.get("final_review_at", ""))[:10],
    }

    if cleaned["status"] in NON_ENTITY_STATUSES:
        cleaned["level_1_actor_type"] = ""
        cleaned["level_2_sphere_boundary"] = ""
        cleaned["level_2_evidence_type"] = ""
        cleaned["level_3_exact_category"] = ""
        cleaned["level_3_evidence_type"] = ""
        cleaned["level_3_evidence_operating_scope"] = ""
        cleaned["level_3_evidence_linkedin_url"] = ""
        cleaned["level_3_evidence_linkedin_hiring_growth_2y"] = ""
        cleaned["level_3_evidence_linkedin_employees_2y_ago"] = ""
        cleaned["level_3_evidence_linkedin_employees_today"] = ""
        cleaned["level_4_innovation_type"] = ""
        cleaned["level_4_evidence_type"] = ""
        cleaned["level_5_helix"] = ""
        cleaned["level_5_evidence_type"] = ""
        cleaned["institution_origin_scope"] = ""
        cleaned["institution_origin_country"] = ""

    return cleaned


def _load_manual_labels(
    template_rows: list[dict],
) -> tuple[dict[str, dict], dict[tuple[int, int], dict]]:
    path = config.STEP2_MANUAL_LABELS_PATH
    if not path.exists():
        _write_json(path, _canonicalize_label_rows(_labels_seed_from_template(template_rows)))
        raise FileNotFoundError(
            f"Manual labels file was created at {path}. Fill it and run step2 again."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Manual labels file must be a JSON array.")

    legacy_lookup = _load_legacy_entity_lookup()
    template_by_pair: dict[tuple[int, int], dict] = {
        (int(row["doc_id"]), int(row["entity_id"])): row for row in template_rows
    }
    template_by_stable: dict[str, dict] = {
        _to_text(row.get("stable_actor_key", "")): row
        for row in template_rows
        if _to_text(row.get("stable_actor_key", ""))
    }

    label_map_by_stable: dict[str, dict] = {}
    label_map_by_pair: dict[tuple[int, int], dict] = {}
    normalized_rows: list[dict] = []
    for row in data:
        try:
            doc_id = int(row.get("doc_id"))
            entity_id = int(row.get("entity_id"))
        except Exception:
            doc_id = 0
            entity_id = 0

        key = (doc_id, entity_id)
        cleaned = _normalize_label_row(
            row,
            key,
            template_row=template_by_pair.get(key),
            legacy_lookup=legacy_lookup,
        )

        stable_key = _to_text(cleaned.get("stable_actor_key", ""))
        template_match = template_by_stable.get(stable_key)
        if template_match:
            cleaned["doc_id"] = int(template_match["doc_id"])
            cleaned["doc_name"] = _to_text(template_match.get("doc_name", ""))
            cleaned["entity_id"] = int(template_match["entity_id"])
            cleaned["canonical_mention"] = _to_text(template_match.get("canonical_mention", ""))
            cleaned["stable_actor_key"] = _to_text(template_match.get("stable_actor_key", ""))
            if not cleaned.get("entity"):
                cleaned["entity"] = _to_text(template_match.get("entity", ""))

        pair_key = (int(cleaned["doc_id"]), int(cleaned["entity_id"]))
        if pair_key[0] and pair_key[1]:
            label_map_by_pair[pair_key] = cleaned
        stable_key = _to_text(cleaned.get("stable_actor_key", ""))
        if stable_key:
            label_map_by_stable[stable_key] = cleaned
        normalized_rows.append(cleaned)

    if not label_map_by_stable and not label_map_by_pair:
        raise RuntimeError(f"No valid manual labels found in {path}.")

    # Append seed entries for any template rows not yet in the labels file.
    missing_template_rows = [
        row for row in template_rows
        if _to_text(row.get("stable_actor_key", "")) not in label_map_by_stable
    ]
    if missing_template_rows:
        new_seeds = _canonicalize_label_rows(_labels_seed_from_template(missing_template_rows))
        normalized_rows.extend(new_seeds)
        _write_json(path, _canonicalize_label_rows(normalized_rows))
        preview = ", ".join(
            f"{_to_text(r.get('doc_name', ''))}::{_to_text(r.get('entity', ''))}"
            for r in missing_template_rows[:20]
        )
        raise FileNotFoundError(
            f"Appended {len(missing_template_rows)} new seed entries to {path}. "
            f"Fill them in and run step 2 again. Examples: {preview}"
        )

    # Drop orphaned labels (entities no longer present in step 1 output).
    pruned_rows = [
        r for r in normalized_rows
        if _to_text(r.get("stable_actor_key", "")) in template_by_stable
    ]
    pruned_count = len(normalized_rows) - len(pruned_rows)
    if pruned_count:
        print(f"[INFO] Pruned {pruned_count} orphaned label(s) not present in current step 1 output.")

    # Keep labels file in canonical new schema.
    _write_json(path, _canonicalize_label_rows(pruned_rows))
    return label_map_by_stable, label_map_by_pair


def _validate_complete_manual_labels(template_rows: list[dict], label_map_by_stable: dict[str, dict]) -> None:
    missing = [
        row
        for row in template_rows
        if _to_text(row.get("stable_actor_key", "")) not in label_map_by_stable
    ]
    if missing:
        preview = ", ".join(
            f"{_to_text(row.get('doc_name', ''))}::{_to_text(row.get('entity', ''))}" for row in missing[:20]
        )
        raise RuntimeError(
            "Manual classification is incomplete. "
            f"Missing labels for {len(missing)} stable actors. Examples: {preview}"
        )


def _apply_labels(
    rows: list[dict],
    label_map_by_stable: dict[str, dict],
    label_map_by_pair: dict[tuple[int, int], dict],
) -> list[dict]:
    labeled_rows: list[dict] = []
    unresolved: set[str] = set()
    for row in rows:
        output_row = dict(row)
        doc_id = int(output_row.get("doc_id"))
        entity_id = int(output_row.get("entity_id"))
        doc_name = _to_text(output_row.get("doc_name", ""))
        canonical = _to_text(output_row.get("canonical_mention", "")) or _canonical_mention(
            _to_text(output_row.get("mention", ""))
        )
        stable_key = _stable_actor_key(doc_name, canonical)

        label_row = label_map_by_stable.get(stable_key)
        if label_row is None:
            label_row = label_map_by_pair.get((doc_id, entity_id))
        if label_row is None:
            unresolved.add(stable_key or f"{doc_id}:{entity_id}")
            continue

        output_row["doc_id"] = doc_id
        output_row["entity_id"] = entity_id
        output_row["entity_key"] = f"{doc_id}:{entity_id}"
        output_row["entity"] = label_row.get("entity") or output_row.get("canonical_mention", "")

        actor_type = _to_text(label_row.get("level_1_actor_type", "")).strip().lower()
        for field in _label_fields_for_type(actor_type):
            if field in {"doc_id", "entity_id", "entity"}:
                continue
            output_row[field] = label_row.get(field)

        labeled_rows.append(output_row)

    if unresolved:
        preview = ", ".join(sorted(unresolved)[:20])
        raise RuntimeError(
            "Could not map some Step1 rows to manual labels after stable-key matching. "
            f"Unresolved examples: {preview}"
        )
    return labeled_rows


def run() -> None:
    print(">>> STEP 2: Manual actor classification (new schema)")

    if not config.STEP1_NER_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {config.STEP1_NER_PATH}. Run step1 first.")

    rows = list(utils.load_jsonl(config.STEP1_NER_PATH))
    if not rows:
        raise RuntimeError(f"Step 1 output is empty: {config.STEP1_NER_PATH}")

    _require_step1_numeric_ids(rows)

    # ── Prune orphaned labels (always runs, even on incremental re-runs) ─────
    if config.STEP2_MANUAL_LABELS_PATH.exists():
        full_template = _build_template(rows)
        _load_manual_labels(full_template)

    # ── Detect already-classified docs ───────────────────────────────────────
    existing_doc_names: set[str] = set()
    if config.STEP2_CLASSIFIED_PATH.exists():
        for row in utils.load_jsonl(config.STEP2_CLASSIFIED_PATH):
            dn = _to_text(row.get("doc_name", ""))
            if dn:
                existing_doc_names.add(dn)

    new_rows = [r for r in rows if _to_text(r.get("doc_name", "")) not in existing_doc_names]
    existing_rows = [r for r in rows if _to_text(r.get("doc_name", "")) in existing_doc_names]

    if not new_rows:
        print(f"[OK] Step 2: all docs already classified, nothing to do.")
        return

    new_doc_names = sorted({_to_text(r.get("doc_name", "")) for r in new_rows})
    print(f"[INFO] Skipping {len(existing_doc_names)} already-classified docs.")
    print(f"[INFO] Processing {len(new_doc_names)} new docs: {new_doc_names}")

    # Build template only for new docs — used to seed manual_classification_labels.json
    template = _build_template(new_rows)

    label_map_by_stable, label_map_by_pair = _load_manual_labels(template)

    # Only validate completeness for new docs
    _validate_complete_manual_labels(template, label_map_by_stable)

    labeled_rows = _apply_labels(new_rows, label_map_by_stable, label_map_by_pair)

    # Append to existing classified file (existing entries untouched)
    with open(config.STEP2_CLASSIFIED_PATH, "a", encoding="utf-8") as fh:
        for row in labeled_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Appended {len(labeled_rows)} labeled entities for {len(new_doc_names)} new docs.")


if __name__ == "__main__":
    run()
