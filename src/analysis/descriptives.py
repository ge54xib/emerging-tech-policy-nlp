"""Descriptives deliverable: corpus overview + actor summary for methods chapter.

Outputs:
- descriptives_corpus_overview.csv  -- one row per document
- descriptives_actor_summary.csv    -- one row per helix (global breakdown)
- descriptives_corpus_figure.png    -- 2-panel: paragraphs per doc + helix distribution per doc
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from math import log
from pathlib import Path

from src import config, utils
from src.analysis._helpers import thesis_style
from src.utils import country_from_doc, normalize_helix, to_int, to_str, year_from_doc


HELIXES = ["government", "industry", "academia", "civil_society", "intermediary"]
HELIX_COLORS = {
    "government":    "#2166ac",
    "industry":      "#d6604d",
    "academia":      "#4dac26",
    "civil_society": "#8073ac",
    "intermediary":  "#f4a582",
}

# Region groupings (for color coding in corpus figure)
REGION_MAP = {
    "GER": "Europe", "NLD": "Europe", "SVN": "Europe", "ESP": "Europe",
    "GBR": "Europe", "CZE": "Europe", "DNK": "Europe", "ITA": "Europe",
    "FIN": "Europe", "FRA": "Europe",
    "USA": "Americas", "CAN": "Americas",
    "JPN": "Asia-Pacific", "KOR": "Asia-Pacific", "AUS": "Asia-Pacific",
    "IRL": "Middle East",
}
REGION_COLORS = {
    "Europe":       "#4393c3",
    "Americas":     "#d6604d",
    "Asia-Pacific": "#4dac26",
    "Middle East":  "#f4a582",
}


def _hbi(shares: dict[str, float]) -> float:
    """Helix Balance Index: normalised Shannon entropy over helix shares (0 = monopoly, 1 = uniform)."""
    entropy = 0.0
    for p in shares.values():
        if p > 0:
            entropy -= p * log(p)
    n = len(shares)
    return (entropy / log(n)) if n > 1 and sum(shares.values()) > 0 else 0.0


def _load_plot_dependencies():
    """Lazily import matplotlib and numpy; raises RuntimeError with install hint on failure."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        thesis_style()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plot dependencies missing. Install with: pip install -r requirements.txt"
        ) from exc
    return plt, np


def run() -> None:
    print(">>> ANALYSIS: Descriptives")

    # ── Load paragraphs and count total sentences (same spaCy splitter as step3) ──
    paragraphs_by_doc: dict[str, int] = defaultdict(int)
    sentences_by_doc: dict[str, int] = defaultdict(int)
    if config.FILE_PARAGRAPHS.exists():
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
            if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        except Exception:
            nlp = None
        for row in utils.load_jsonl(config.FILE_PARAGRAPHS):
            doc = to_str(row.get("doc_id", "")).strip()
            if not doc:
                continue
            paragraphs_by_doc[doc] += 1
            text = to_str(row.get("text", "")).strip()
            if not text:
                continue
            if nlp is not None:
                sentences_by_doc[doc] += sum(1 for s in nlp(text).sents if s.text.strip())
            else:
                sentences_by_doc[doc] += 1  # fallback: treat paragraph as one sentence

    # ── Load sentences with actors (alignment) and with co-occurrences ───────
    actor_sentences_by_doc: dict[str, set] = defaultdict(set)
    if config.FILE_ALIGNMENT.exists():
        for row in utils.load_jsonl(config.FILE_ALIGNMENT):
            doc = to_str(row.get("doc_id", "")).strip()
            if doc:
                actor_sentences_by_doc[doc].add(
                    (to_int(row.get("paragraph_id", 0), 0), to_int(row.get("sentence_id", 0), 0))
                )

    cooccur_sentences_by_doc: dict[str, set] = defaultdict(set)
    if config.FILE_COOCCURRENCE.exists():
        for row in utils.load_jsonl(config.FILE_COOCCURRENCE):
            doc = to_str(row.get("doc_id", "")).strip()
            if doc:
                cooccur_sentences_by_doc[doc].add(
                    (to_int(row.get("paragraph_id", 0), 0), to_int(row.get("sentence_id", 0), 0))
                )

    # ── Load NER entities (unique entity_ids, consistent with classified) ────
    ner_unique: dict[str, set] = defaultdict(set)
    if config.STEP1_NER_PATH.exists():
        for row in utils.load_jsonl(config.STEP1_NER_PATH):
            doc = to_str(row.get("doc_name", "")).strip()
            entity_id = to_int(row.get("entity_id", 0), 0)
            if doc and entity_id:
                ner_unique[doc].add(entity_id)
    ner_by_doc: dict[str, int] = {doc: len(ids) for doc, ids in ner_unique.items()}

    # ── Load classified entities ─────────────────────────────────────────────
    classified_by_doc: dict[str, int] = defaultdict(int)
    helix_by_doc: dict[str, Counter] = defaultdict(Counter)
    rd_total: Counter = Counter()
    sphere_total: Counter = Counter()
    actor_type_total: Counter = Counter()
    helix_total: Counter = Counter()

    if config.STEP2_CLASSIFIED_PATH.exists():
        rows_classified = list(utils.load_jsonl(config.STEP2_CLASSIFIED_PATH))
        seen: set[tuple] = set()
        for row in rows_classified:
            status = to_str(row.get("status", "")).strip().lower()
            if status != "entity":
                continue
            key = (
                to_int(row.get("doc_id", 0), 0),
                to_int(row.get("entity_id", 0), 0),
            )
            if key in seen:
                continue
            seen.add(key)

            doc = to_str(row.get("doc_name", "")).strip()
            helix = normalize_helix(row.get("level_5_helix", ""))
            if helix not in HELIXES:
                continue

            classified_by_doc[doc] += 1
            helix_by_doc[doc][helix] += 1
            helix_total[helix] += 1

            rd = to_str(row.get("level_4_innovation_type", "")).strip()
            rd_total[rd if rd else "Unknown"] += 1

            sb = to_str(row.get("level_2_sphere_boundary", "")).strip()
            sphere_total[sb if sb else "Unknown"] += 1

            at = to_str(row.get("level_1_actor_type", "")).strip().lower()
            if "individual" in at and "institution" not in at:
                actor_type_total["Individual"] += 1
            else:
                actor_type_total["Institutional"] += 1

    # ── Build corpus overview ─────────────────────────────────────────────────
    # All known docs (from paragraphs, NER, or classified)
    all_docs: set[str] = (
        set(paragraphs_by_doc) | set(ner_by_doc) | set(helix_by_doc)
    )
    # Remove generic DOC_N placeholders
    all_docs = {d for d in all_docs if not d.startswith("DOC_")}

    corpus_rows = []
    for doc in sorted(all_docs):
        country = country_from_doc(doc)
        year = year_from_doc(doc)
        n_paragraphs = paragraphs_by_doc.get(doc, 0)
        n_ner = ner_by_doc.get(doc, 0)
        n_classified = classified_by_doc.get(doc, 0)
        helix_counts = helix_by_doc.get(doc, Counter())
        total_h = sum(helix_counts.values()) or 1
        shares = {h: helix_counts.get(h, 0) / total_h for h in HELIXES}
        hbi = _hbi(shares)
        corpus_rows.append({
            "doc_name": doc,
            "country": country,
            "year": year,
            "paragraphs": n_paragraphs,
            "entities_ner": n_ner,
            "entities_classified": n_classified,
            "gov": helix_counts.get("government", 0),
            "ind": helix_counts.get("industry", 0),
            "acad": helix_counts.get("academia", 0),
            "cs": helix_counts.get("civil_society", 0),
            "inter": helix_counts.get("intermediary", 0),
            "hbi": hbi,
        })

    # Sort by year, then country
    corpus_rows.sort(key=lambda r: (r["year"], r["country"]))

    # ── Build actor summary ───────────────────────────────────────────────────
    total_entities = sum(helix_total.values()) or 1
    total_rd = sum(rd_total.values()) or 1
    total_sphere = sum(sphere_total.values()) or 1
    total_actor = sum(actor_type_total.values()) or 1

    actor_summary_rows = []
    for h in HELIXES:
        count = helix_total.get(h, 0)
        countries_present = sum(
            1 for r in corpus_rows
            if r["country"] != "UNK" and helix_by_doc.get(r["doc_name"], Counter()).get(h, 0) > 0
        )
        actor_summary_rows.append({
            "helix": h,
            "count": count,
            "share": count / total_entities,
            "countries_present": countries_present,
        })

    # Global R&D / sphere / actor type
    rd_share = (rd_total.get("R&D", 0) + rd_total.get("Both", 0)) / total_rd
    non_rd_share = rd_total.get("Non R&D", 0) / total_rd
    multi_sphere_share = sphere_total.get("Multi-Sphere", 0) / total_sphere
    individual_share = actor_type_total.get("Individual", 0) / total_actor

    # ── Write CSVs ───────────────────────────────────────────────────────────
    try:
        corpus_fields = [
            "Country", "Year", "Doc_Name", "Paragraphs", "Entities_NER",
            "Entities_Classified", "Gov", "Ind", "Acad", "CS", "Int", "HBI",
        ]
        with open(config.ANALYSIS_DESCRIPTIVES_CORPUS_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=corpus_fields)
            writer.writeheader()
            for r in corpus_rows:
                writer.writerow({
                    "Country": r["country"],
                    "Year": r["year"],
                    "Doc_Name": r["doc_name"],
                    "Paragraphs": r["paragraphs"],
                    "Entities_NER": r["entities_ner"],
                    "Entities_Classified": r["entities_classified"],
                    "Gov": r["gov"],
                    "Ind": r["ind"],
                    "Acad": r["acad"],
                    "CS": r["cs"],
                    "Int": r["inter"],
                    "HBI": f"{r['hbi']:.4f}",
                })
        print(f"[OK] Wrote: {config.ANALYSIS_DESCRIPTIVES_CORPUS_CSV}")
    except Exception as exc:
        print(f"[WARN] Corpus CSV skipped: {exc}")

    # ── Per-country summary CSV ───────────────────────────────────────────────
    try:
        country_map: dict[str, dict] = {}
        for r in corpus_rows:
            c = r["country"]
            if c not in country_map:
                country_map[c] = {
                    "country": c,
                    "years": set(),
                    "documents": 0,
                    "sentences": 0,
                    "entities_ner": 0,
                    "entities_classified": 0,
                    "sentences_with_actors": 0,
                    "sentences_cooccurrence": 0,
                }
            e = country_map[c]
            e["years"].add(r["year"])
            e["documents"] += 1
            e["sentences"] += sentences_by_doc.get(r["doc_name"], 0)
            e["entities_ner"] += r["entities_ner"]
            e["entities_classified"] += r["entities_classified"]
            e["sentences_with_actors"] += len(actor_sentences_by_doc.get(r["doc_name"], set()))
            e["sentences_cooccurrence"] += len(cooccur_sentences_by_doc.get(r["doc_name"], set()))

        country_rows = sorted(country_map.values(), key=lambda x: min(x["years"]))
        country_fields = ["Country", "Year(s)", "Documents", "Sentences"]
        with open(config.ANALYSIS_DESCRIPTIVES_COUNTRY_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=country_fields)
            writer.writeheader()
            for e in country_rows:
                years_str = ", ".join(str(y) for y in sorted(e["years"]))
                writer.writerow({
                    "Country": e["country"],
                    "Year(s)": years_str,
                    "Documents": e["documents"],
                    "Sentences": e["sentences"],
                })
        print(f"[OK] Wrote: {config.ANALYSIS_DESCRIPTIVES_COUNTRY_CSV}")
    except Exception as exc:
        print(f"[WARN] Country CSV skipped: {exc}")

    try:
        actor_fields = [
            "Helix", "Count", "Share", "Countries_Present",
        ]
        with open(config.ANALYSIS_DESCRIPTIVES_ACTORS_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=actor_fields)
            writer.writeheader()
            for r in actor_summary_rows:
                writer.writerow({
                    "Helix": r["helix"],
                    "Count": r["count"],
                    "Share": f"{r['share']:.4f}",
                    "Countries_Present": r["countries_present"],
                })
            # Global footnote rows
            writer.writerow({"Helix": "--- Global stats ---", "Count": "", "Share": "", "Countries_Present": ""})
            writer.writerow({"Helix": "R&D share (all entities)", "Count": "", "Share": f"{rd_share:.4f}", "Countries_Present": ""})
            writer.writerow({"Helix": "Non-R&D share (all entities)", "Count": "", "Share": f"{non_rd_share:.4f}", "Countries_Present": ""})
            writer.writerow({"Helix": "Multi-sphere share (all entities)", "Count": "", "Share": f"{multi_sphere_share:.4f}", "Countries_Present": ""})
            writer.writerow({"Helix": "Individual actor share (all entities)", "Count": "", "Share": f"{individual_share:.4f}", "Countries_Present": ""})
        print(f"[OK] Wrote: {config.ANALYSIS_DESCRIPTIVES_ACTORS_CSV}")
    except Exception as exc:
        print(f"[WARN] Actor summary CSV skipped: {exc}")

    # ── Visualized table 2: Actor summary per helix + global stats ───────────
    try:
        plt, np = _load_plot_dependencies()

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1]})

        # Left: per-helix table
        helix_col_labels = ["Helix", "Total\nEntities", "Share", "Countries\nPresent"]
        helix_table_data = []
        for r in actor_summary_rows:
            helix_table_data.append([
                r["helix"].replace("_", " ").title(),
                r["count"],
                f"{r['share']*100:.1f}%",
                r["countries_present"],
            ])

        ax = axes[0]
        ax.axis("off")
        tbl1 = ax.table(
            cellText=helix_table_data,
            colLabels=helix_col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl1.auto_set_font_size(False)
        tbl1.set_fontsize(10)
        tbl1.scale(1.1, 2.0)
        for col_idx in range(len(helix_col_labels)):
            tbl1[0, col_idx].set_facecolor("#2c3e50")
            tbl1[0, col_idx].set_text_props(color="white", fontweight="bold")
        for row_idx, asr in enumerate(actor_summary_rows):
            color = HELIX_COLORS.get(asr["helix"], "#cccccc") + "44"
            for col_idx in range(len(helix_col_labels)):
                tbl1[row_idx + 1, col_idx].set_facecolor(color)
        ax.set_title("Actor Summary by Helix", fontsize=11, pad=10)

        # Right: global component stats table
        global_col_labels = ["Component Dimension", "Category", "Share"]
        total_classified = sum(helix_total.values())
        global_table_data = [
            ["Actor type", "Institutional", f"{(1 - individual_share)*100:.1f}%"],
            ["Actor type", "Individual", f"{individual_share*100:.1f}%"],
            ["Innovation type", "R&D", f"{rd_share*100:.1f}%"],
            ["Innovation type", "Non-R&D", f"{non_rd_share*100:.1f}%"],
            ["Sphere boundary", "Single-Sphere", f"{(1 - multi_sphere_share)*100:.1f}%"],
            ["Sphere boundary", "Multi-Sphere (hybrid)", f"{multi_sphere_share*100:.1f}%"],
            ["Total classified", f"{total_classified} entities", "100%"],
        ]
        row_bg = ["#e8f4f8", "#d1eaf5", "#e8f5e9", "#c8e6c9", "#f3e5f5", "#e1bee7", "#f5f5f5"]

        ax = axes[1]
        ax.axis("off")
        tbl2 = ax.table(
            cellText=global_table_data,
            colLabels=global_col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(10)
        tbl2.scale(1.1, 2.0)
        for col_idx in range(len(global_col_labels)):
            tbl2[0, col_idx].set_facecolor("#2c3e50")
            tbl2[0, col_idx].set_text_props(color="white", fontweight="bold")
        for row_idx, bg in enumerate(row_bg):
            for col_idx in range(len(global_col_labels)):
                tbl2[row_idx + 1, col_idx].set_facecolor(bg)
        ax.set_title("Global Component Breakdown\n(Ranga & Etzkowitz 2013 distinctions)", fontsize=11, pad=10)

        fig.suptitle(
            "Descriptives: Classified Actor Summary\n"
            f"(N = {total_classified} entities across {len(corpus_rows)} documents, "
            f"{len(set(r['country'] for r in corpus_rows))} countries)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(config.ANALYSIS_DESCRIPTIVES_TABLE_ACTORS_PNG, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Wrote: {config.ANALYSIS_DESCRIPTIVES_TABLE_ACTORS_PNG}")
    except Exception as exc:
        print(f"[WARN] Actor table figure skipped: {exc}")


if __name__ == "__main__":
    run()
