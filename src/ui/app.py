"""Streamlit UI for manual Quadruple Helix classification of NLP-extracted entities."""
from __future__ import annotations

import html as _html
import json

import streamlit as st

from src import config

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

STATUSES = [
    "", "entity", "not_actor", "ner_error", "not_specific",
    "non_entity_phrase", "unclear", "historical_individual",
]
NON_ENTITY_STATUSES = {
    "historical_individual", "ner_error", "not_actor",
    "not_specific", "non_entity_phrase", "unclear",
}
ACTOR_TYPES = ["institutional", "individual"]
SPHERE_BOUNDS = ["", "single-sphere", "multi-sphere"]
CATEGORIES = [
    "",
    "Large Enterprises",
    "business support institutions",
    "corporate labs",
    "entrepreneurial_scientist",
    "financial support institutions",
    "industrial liaison offices",
    "innovation_organizer",
    "joint research centers",
    "media and cultural Institutions",
    "national government institutions",
    "non-governmental and non-profit sector",
    "research institutes",
    "small and medium sized enterprises",
    "start ups",
    "sub-national government institutions",
    "supranational government institutions",
    "technology transfer offices",
    "universities",
    "user communities",
]
INNOVATION_TYPES = ["R&D", "Non R&D", "Both"]
HELICES = ["", "government", "industry", "academia", "civil_society", "intermediary"]
ORIGIN_SCOPES = ["", "domestic", "foreign"]

LABELS_PATH = config.STEP2_MANUAL_LABELS_PATH

# All form-editable fields (not doc-specific meta fields)
FORM_FIELDS: list[str] = [
    "status", "excluded_reason",
    "level_1_actor_type",
    "level_2_sphere_boundary", "level_2_evidence_type",
    "level_3_exact_category", "level_3_evidence_type",
    "level_3_evidence_operating_scope",
    "level_3_evidence_linkedin_url",
    "level_3_evidence_linkedin_hiring_growth_2y",
    "level_3_evidence_linkedin_employees_2y_ago",
    "level_3_evidence_linkedin_employees_today",
    "level_4_innovation_type", "level_4_evidence_type",
    "level_5_helix", "level_5_evidence_type",
    "evidence_snippet_r_and_d", "evidence_snippet_non_r_and_d",
    "website_url", "website_about_page",
    "institution_origin_scope", "institution_origin_country",
    "local_name", "english_name", "abbreviation", "local_abbreviation",
    "name", "surname",
    "strategy_individual_affiliation_local_name",
    "strategy_individual_affiliation_english_name",
    "strategy_individual_affiliation_local_abbreviation",
    "strategy_individual_affiliation_english_abbreviation",
    "secondary_strategy_individual_affiliation_local_name",
    "secondary_strategy_individual_affiliation_english_name",
    "secondary_strategy_individual_affiliation_local_abbreviation",
    "secondary_strategy_individual_affiliation_english_abbreviation",
    "current_individual_affiliation_local_name",
    "current_individual_affiliation_english_name",
    "current_individual_affiliation_local_abbreviation",
    "current_individual_affiliation_english_abbreviation",
    "notes", "checked",
]

# Fields to copy on auto-fill (exclude doc-specific and review fields)
AUTOFILL_FIELDS: list[str] = [
    f for f in FORM_FIELDS if f not in {"notes", "checked", "excluded_reason"}
]


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def _load_entries() -> list[dict]:
    if not LABELS_PATH.exists():
        st.error(
            f"Labels file not found: `{LABELS_PATH}`\n\n"
            "Run `python run.py pipeline --step 2` first to create the seed file."
        )
        st.stop()
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def _save_entries(entries: list[dict]) -> None:
    LABELS_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ──────────────────────────────────────────────────────────────────────────────
# State helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_filled(e: dict) -> bool:
    status = str(e.get("status", "")).strip()
    if not status:
        return False
    if status in NON_ENTITY_STATUSES:
        return True
    return bool(str(e.get("level_5_helix", "")).strip())


def _build_filtered(
    entries: list[dict],
    selected_doc: str | None,
    show_unfilled: bool,
) -> list[tuple[int, dict]]:
    return [
        (i, e)
        for i, e in enumerate(entries)
        if (selected_doc is None or e.get("doc_name") == selected_doc)
        and (not show_unfilled or not _is_filled(e))
    ]


def _sanitize(field: str, val: str) -> str:
    """Return a valid value for constrained fields; keep free-text fields as-is."""
    if field == "level_5_helix" and val not in HELICES:
        return ""
    if field == "level_2_sphere_boundary" and val not in SPHERE_BOUNDS:
        return ""
    if field == "level_3_exact_category" and val not in CATEGORIES:
        return ""
    if field == "level_4_innovation_type" and val not in INNOVATION_TYPES:
        return INNOVATION_TYPES[0]
    if field == "status" and val not in STATUSES:
        return ""
    if field == "institution_origin_scope" and val not in ORIGIN_SCOPES:
        return ""
    if field == "level_1_actor_type" and val not in ACTOR_TYPES:
        return ACTOR_TYPES[0]
    return val


def _load_form(entry: dict) -> None:
    """Pre-populate session state form keys from entry values (run before widgets)."""
    for field in FORM_FIELDS:
        raw = entry.get(field, "")
        if field == "checked":
            st.session_state[f"f_{field}"] = bool(raw)
        else:
            v = str(raw) if raw is not None else ""
            st.session_state[f"f_{field}"] = _sanitize(field, v)


def _apply_autofill(source: dict) -> None:
    """Copy classification fields from a source entry into session state."""
    for field in AUTOFILL_FIELDS:
        raw = source.get(field, "")
        v = str(raw) if raw is not None else ""
        st.session_state[f"f_{field}"] = _sanitize(field, v)


def _autofill_candidates(
    canonical: str, doc_name: str, entries: list[dict]
) -> list[dict]:
    return [
        e for e in entries
        if e.get("canonical_mention", "") == canonical
        and e.get("doc_name", "") != doc_name
        and str(e.get("level_5_helix", "")).strip()
        and e.get("status", "") == "entity"
    ]


def _read_form() -> dict:
    """Read all FORM_FIELDS from session state into a plain dict."""
    out: dict = {}
    for field in FORM_FIELDS:
        val = st.session_state.get(f"f_{field}", "")
        out[field] = bool(val) if field == "checked" else (str(val) if val is not None else "")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

def _do_save(entries: list[dict], global_idx: int, current_filtered_idx: int) -> None:
    form_data = _read_form()
    entries[global_idx].update(form_data)
    _save_entries(entries)

    # Rebuild filtered to find next position
    new_filtered = _build_filtered(
        entries,
        st.session_state.get("_sel_doc"),
        st.session_state.get("_show_unfilled", True),
    )
    next_idx = min(current_filtered_idx, max(0, len(new_filtered) - 1))
    st.session_state.current_index = next_idx
    st.session_state.nav_changed = True
    st.session_state._saved = True
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Form section renderers  (called inside col_right context)
# ──────────────────────────────────────────────────────────────────────────────

def _section_a_core() -> None:
    st.markdown("#### A — Core Classification")

    st.radio(
        "Level 1 — Actor Type",
        options=ACTOR_TYPES,
        horizontal=True,
        key="f_level_1_actor_type",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Level 2 — Sphere Boundary", options=SPHERE_BOUNDS,
                     key="f_level_2_sphere_boundary")
    with c2:
        st.selectbox("Level 3 — Exact Category", options=CATEGORIES,
                     key="f_level_3_exact_category")

    c3, c4 = st.columns(2)
    with c3:
        st.radio("Level 4 — Innovation Type", options=INNOVATION_TYPES,
                 horizontal=True, key="f_level_4_innovation_type")
    with c4:
        st.selectbox("Level 5 — Helix", options=HELICES, key="f_level_5_helix")


def _section_b_evidence() -> None:
    st.markdown("#### B — Evidence")

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.text_input("Level 2 evidence type", key="f_level_2_evidence_type")
    with ec2:
        st.text_input("Level 3 evidence type", key="f_level_3_evidence_type")
    with ec3:
        st.text_input("Operating scope", key="f_level_3_evidence_operating_scope")

    ec4, ec5 = st.columns(2)
    with ec4:
        st.text_input("Level 4 evidence type", key="f_level_4_evidence_type")
    with ec5:
        st.text_input("Level 5 evidence type", key="f_level_5_evidence_type")

    sn1, sn2 = st.columns(2)
    with sn1:
        st.text_area("Evidence snippet — R&D", key="f_evidence_snippet_r_and_d", height=70)
    with sn2:
        st.text_area("Evidence snippet — Non R&D",
                     key="f_evidence_snippet_non_r_and_d", height=70)


def _section_c_institutional() -> None:
    st.markdown("#### C — Institution Identity")

    ic1, ic2 = st.columns(2)
    with ic1:
        st.text_input("Local name", key="f_local_name")
        st.text_input("Abbreviation", key="f_abbreviation")
    with ic2:
        st.text_input("English name", key="f_english_name")
        st.text_input("Local abbreviation", key="f_local_abbreviation")

    oc1, oc2 = st.columns(2)
    with oc1:
        st.selectbox("Origin scope", options=ORIGIN_SCOPES,
                     key="f_institution_origin_scope")
    with oc2:
        st.text_input("Origin country (ISO code)", key="f_institution_origin_country")

    wc1, wc2 = st.columns(2)
    with wc1:
        st.text_input("Website URL", key="f_website_url")
    with wc2:
        st.text_input("Website about page", key="f_website_about_page")

    st.markdown("**LinkedIn data**")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.text_input("LinkedIn URL", key="f_level_3_evidence_linkedin_url")
        st.text_input("Hiring growth 2y", key="f_level_3_evidence_linkedin_hiring_growth_2y")
    with lc2:
        st.text_input("Employees 2y ago", key="f_level_3_evidence_linkedin_employees_2y_ago")
        st.text_input("Employees today", key="f_level_3_evidence_linkedin_employees_today")


def _section_c_individual() -> None:
    st.markdown("#### C — Individual Identity")

    nc1, nc2 = st.columns(2)
    with nc1:
        st.text_input("Name", key="f_name")
    with nc2:
        st.text_input("Surname", key="f_surname")

    wc1, wc2 = st.columns(2)
    with wc1:
        st.text_input("Website URL", key="f_website_url")
    with wc2:
        st.text_input("Website about page", key="f_website_about_page")

    st.markdown("**Strategy affiliation**")
    sa1, sa2 = st.columns(2)
    with sa1:
        st.text_input("Local name",
                      key="f_strategy_individual_affiliation_local_name")
        st.text_input("Local abbreviation",
                      key="f_strategy_individual_affiliation_local_abbreviation")
    with sa2:
        st.text_input("English name",
                      key="f_strategy_individual_affiliation_english_name")
        st.text_input("English abbreviation",
                      key="f_strategy_individual_affiliation_english_abbreviation")

    st.markdown("**Secondary affiliation**")
    sb1, sb2 = st.columns(2)
    with sb1:
        st.text_input("Local name",
                      key="f_secondary_strategy_individual_affiliation_local_name")
        st.text_input("Local abbreviation",
                      key="f_secondary_strategy_individual_affiliation_local_abbreviation")
    with sb2:
        st.text_input("English name",
                      key="f_secondary_strategy_individual_affiliation_english_name")
        st.text_input("English abbreviation",
                      key="f_secondary_strategy_individual_affiliation_english_abbreviation")

    st.markdown("**Current affiliation**")
    ca1, ca2 = st.columns(2)
    with ca1:
        st.text_input("Local name",
                      key="f_current_individual_affiliation_local_name")
        st.text_input("Local abbreviation",
                      key="f_current_individual_affiliation_local_abbreviation")
    with ca2:
        st.text_input("English name",
                      key="f_current_individual_affiliation_english_name")
        st.text_input("English abbreviation",
                      key="f_current_individual_affiliation_english_abbreviation")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="QH Classification",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Session state init ─────────────────────────────────────────────────
    if "entries" not in st.session_state:
        st.session_state.entries = _load_entries()
        st.session_state.nav_changed = True
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "nav_changed" not in st.session_state:
        st.session_state.nav_changed = True
    if "_sel_doc" not in st.session_state:
        st.session_state._sel_doc = None
    if "_show_unfilled" not in st.session_state:
        st.session_state._show_unfilled = True

    entries: list[dict] = st.session_state.entries
    selected_doc: str | None = st.session_state._sel_doc
    show_unfilled: bool = st.session_state._show_unfilled

    # ── Build filtered list ────────────────────────────────────────────────
    filtered = _build_filtered(entries, selected_doc, show_unfilled)
    n = len(filtered)

    cur = max(0, min(st.session_state.current_index, n - 1)) if n > 0 else 0
    if cur != st.session_state.current_index:
        st.session_state.current_index = cur

    # ── Load form for current entry (BEFORE any widgets render) ───────────
    if st.session_state.nav_changed and n > 0:
        _load_form(filtered[cur][1])
        st.session_state.nav_changed = False

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("QH Classifier")

        all_docs = sorted({e.get("doc_name", "") for e in entries if e.get("doc_name")})
        doc_opts = ["(all documents)"] + all_docs
        cur_doc_label = selected_doc if selected_doc else "(all documents)"
        doc_idx = doc_opts.index(cur_doc_label) if cur_doc_label in doc_opts else 0

        new_doc_label = st.selectbox("Document", options=doc_opts, index=doc_idx)
        new_doc = None if new_doc_label == "(all documents)" else new_doc_label
        if new_doc != selected_doc:
            st.session_state._sel_doc = new_doc
            st.session_state.current_index = 0
            st.session_state.nav_changed = True
            st.rerun()

        # Progress bar
        if selected_doc:
            doc_entries = [e for e in entries if e.get("doc_name") == selected_doc]
            doc_filled = sum(1 for e in doc_entries if _is_filled(e))
            st.progress(
                doc_filled / max(len(doc_entries), 1),
                text=f"{doc_filled} / {len(doc_entries)} filled",
            )
        else:
            total_filled = sum(1 for e in entries if _is_filled(e))
            st.progress(
                total_filled / max(len(entries), 1),
                text=f"{total_filled} / {len(entries)} filled",
            )

        new_unfilled = st.checkbox("Show unfilled only", value=show_unfilled)
        if new_unfilled != show_unfilled:
            st.session_state._show_unfilled = new_unfilled
            st.session_state.current_index = 0
            st.session_state.nav_changed = True
            st.rerun()

        st.markdown(f"**{cur + 1} / {n}** entries shown")

        nc1, nc2 = st.columns(2)
        with nc1:
            if st.button("← Prev", use_container_width=True, disabled=(cur <= 0 or n == 0)):
                st.session_state.current_index -= 1
                st.session_state.nav_changed = True
                st.rerun()
        with nc2:
            if st.button("Next →", use_container_width=True,
                         disabled=(cur >= n - 1 or n == 0)):
                st.session_state.current_index += 1
                st.session_state.nav_changed = True
                st.rerun()

        st.divider()
        total_all = len(entries)
        total_filled_all = sum(1 for e in entries if _is_filled(e))
        st.metric("Total entries", total_all)
        sm1, sm2 = st.columns(2)
        with sm1:
            st.metric("Filled", total_filled_all)
        with sm2:
            st.metric("Unfilled", total_all - total_filled_all)

        st.divider()
        if st.button("↺ Reload from disk", use_container_width=True):
            del st.session_state["entries"]
            st.session_state.nav_changed = True
            st.rerun()

    # ── Save confirmation banner ───────────────────────────────────────────
    if st.session_state.get("_saved", False):
        st.session_state._saved = False
        st.success("Entry saved!", icon="✅")

    # ── No entries ─────────────────────────────────────────────────────────
    if n == 0:
        if show_unfilled:
            st.success("All entries classified for the current selection!")
        else:
            st.info("No entries match the current filter.")
        return

    global_idx, entry = filtered[cur]

    # ── Page header ────────────────────────────────────────────────────────
    entity_name = str(entry.get("entity", entry.get("canonical_mention", "")))
    entity_label = str(entry.get("entity_label", "ORG"))
    doc_name = str(entry.get("doc_name", ""))

    st.markdown(
        f"<h3 style='margin-bottom:2px'>{_html.escape(entity_name)}"
        f"&nbsp;&nbsp;<span style='font-size:0.65em;background:#EEEEEE;"
        f"padding:2px 8px;border-radius:4px;font-family:monospace;color:#555'>"
        f"{entity_label}</span></h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"📄 {doc_name}  ·  entity_id={entry.get('entity_id', '')}  ·  "
        f"canonical: {entry.get('canonical_mention', '')}"
    )

    col_left, col_right = st.columns([1, 1.5])

    # ── LEFT COLUMN ────────────────────────────────────────────────────────
    with col_left:

        # Context window (400 chars)
        ctx = str(entry.get("context_window_400", "")).strip()
        occ = str(entry.get("occurrence_sentence", "")).strip()

        if ctx:
            ctx_esc = _html.escape(ctx)
            name_esc = _html.escape(entity_name)
            ctx_highlighted = ctx_esc.replace(
                name_esc, f"<strong>{name_esc}</strong>"
            )
            st.markdown(
                f"<div style='background:#F7F7F5;border-left:4px solid #CCCCCC;"
                f"padding:10px 14px;border-radius:0 6px 6px 0;"
                f"font-size:0.87em;line-height:1.6;font-family:serif;"
                f"word-break:break-word;'>{ctx_highlighted}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("*(no context window available)*")

        if occ:
            occ_bold = occ.replace(entity_name, f"**{entity_name}**")
            st.markdown(f"*Sentence:* {occ_bold}")

        # Auto-fill banner
        canonical = str(entry.get("canonical_mention", ""))
        candidates = _autofill_candidates(canonical, doc_name, entries)
        if candidates:
            st.markdown("---")
            st.markdown(f"**⚡ Auto-fill** available from {len(candidates)} doc(s):")
            for cand in candidates[:5]:
                cand_doc = str(cand.get("doc_name", ""))
                cand_helix = str(cand.get("level_5_helix", ""))
                btn_label = f"{cand_doc[:42]}  [{cand_helix}]"
                if st.button(btn_label, key=f"af_{global_idx}_{cand_doc}",
                             use_container_width=True):
                    _apply_autofill(cand)
                    st.rerun()

        # Status
        st.markdown("---")
        st.selectbox("Status", options=STATUSES, key="f_status")
        current_status = str(st.session_state.get("f_status", ""))

        # Non-entity path: remaining fields + save stay in left column
        if current_status and current_status != "entity":
            st.text_input("Excluded reason", key="f_excluded_reason")
            st.text_area("Notes", key="f_notes", height=80)
            st.divider()
            if st.button("💾 Save", type="primary", use_container_width=True,
                         key="save_left"):
                _do_save(entries, global_idx, cur)

    # ── RIGHT COLUMN ───────────────────────────────────────────────────────
    with col_right:
        current_status = str(st.session_state.get("f_status", ""))

        if not current_status:
            st.info("Set a status in the left panel to start classifying.")
            return

        if current_status != "entity":
            return  # non-entity fields are handled in the left column

        # Section A: Core classification
        _section_a_core()

        # Section B: Evidence
        _section_b_evidence()

        # Section C: Identity (institutional vs individual)
        actor_type = str(st.session_state.get("f_level_1_actor_type", "institutional"))
        if actor_type == "individual":
            _section_c_individual()
        else:
            _section_c_institutional()

        # Section D: Notes & Review
        st.markdown("#### D — Notes & Review")
        nd1, nd2 = st.columns([4, 1])
        with nd1:
            st.text_area("Notes", key="f_notes", height=80)
        with nd2:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            st.checkbox("Checked", key="f_checked")

        st.divider()
        if st.button("💾 Save & Next", type="primary", use_container_width=True,
                     key="save_right"):
            _do_save(entries, global_idx, cur)


if __name__ == "__main__":
    main()
