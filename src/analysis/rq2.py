"""RQ2 deliverable: GLiREL-based helix relation classification by country.

Outputs:
- rq2_helix_pair_counts_by_country.json  (helix pair counts / density)
- rq2_helix_pair_counts_by_country.png   (heatmap, normalized by paragraphs)
- rq2_top_pairs.png                      (global pair ranking colored by TH Space)
- rq2_spaces.png                         (TH Space shares per country)
- rq2_table.csv                          (helix pair density table)
- rq2_relation_distribution.png          (NEW: global relation type frequency)
- rq2_relation_by_helix_pair.png         (NEW: relation type × helix pair heatmap)
- rq2_relation_table.csv                 (NEW: country × pair × relation type counts)
- rq2_network_edges.csv                  (NEW: entity-level network edges)
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

from src import config, utils
from src.analysis._helpers import utc_now_iso, write_json, thesis_style
from src.utils import normalize_helix, to_str


HELIXES = ["government", "industry", "academia", "civil_society", "intermediary"]
PAIR_SEP = "\u2013"  # en-dash

HELIX_COLORS = {
    "government":    "#2166ac",
    "industry":      "#d6604d",
    "academia":      "#4dac26",
    "civil_society": "#8073ac",
    "intermediary":  "#f4a582",
}

TH_SPACE_MAP: dict[str, str] = {
    f"academia{PAIR_SEP}government":      "Knowledge Space",
    f"academia{PAIR_SEP}academia":        "Knowledge Space",
    f"academia{PAIR_SEP}industry":        "Innovation Space",
    f"academia{PAIR_SEP}intermediary":    "Innovation Space",
    f"industry{PAIR_SEP}intermediary":    "Innovation Space",
    f"government{PAIR_SEP}industry":      "Consensus Space",
    f"government{PAIR_SEP}intermediary":  "Consensus Space",
    f"government{PAIR_SEP}government":    "Consensus Space",
    f"academia{PAIR_SEP}civil_society":   "Civil Society",
    f"civil_society{PAIR_SEP}government": "Civil Society",
    f"civil_society{PAIR_SEP}industry":   "Civil Society",
    f"civil_society{PAIR_SEP}intermediary": "Civil Society",
    f"civil_society{PAIR_SEP}civil_society": "Civil Society",
    f"industry{PAIR_SEP}industry":        "Internal",
    f"intermediary{PAIR_SEP}intermediary": "Internal",
}

TH_SPACE_COLORS = {
    "Knowledge Space":  "#4393c3",
    "Innovation Space": "#74c476",
    "Consensus Space":  "#fd8d3c",
    "Civil Society":    "#9e9ac8",
    "Internal":         "#bdbdbd",
}

RELATION_TYPES = [
    "technology_transfer",
    "collaboration_conflict_moderation",
    "collaborative_leadership",
    "substitution",
    "networking",
    "no_explicit_relation",
]
EXPLICIT_RELATION_TYPES = [r for r in RELATION_TYPES if r != "no_explicit_relation"]

RELATION_COLORS = {
    "technology_transfer":              "#2171b5",
    "collaboration_conflict_moderation": "#6baed6",
    "collaborative_leadership":         "#238b45",
    "substitution":                     "#d94801",
    "networking":                       "#8856a7",
    "no_explicit_relation":             "#bdbdbd",
}


def _load_jsonl_safe(path) -> list[dict]:
    """Load JSONL while skipping malformed lines (e.g., interrupted writes)."""
    rows: list[dict] = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            token = line.strip()
            if not token:
                continue
            try:
                rows.append(json.loads(token))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[WARN] Skipped {bad} malformed rows in {path}")
    return rows


def _normalize_pair(h1: str, h2: str) -> str:
    return PAIR_SEP.join(sorted([normalize_helix(h1), normalize_helix(h2)]))


def _pair_from_row(row: dict) -> str:
    explicit = to_str(row.get("pair", "")).strip()
    if explicit:
        # Re-normalize in case it uses legacy "university"
        for sep in [PAIR_SEP, "-"]:
            if sep in explicit:
                h1, h2 = explicit.split(sep, 1)
                return _normalize_pair(h1, h2)
        return explicit
    h1 = to_str(row.get("h1", "")).strip()
    h2 = to_str(row.get("h2", "")).strip()
    return _normalize_pair(h1, h2)


def _th_space(pair: str) -> str:
    return TH_SPACE_MAP.get(pair, "Civil Society" if "civil_society" in pair else "Internal")


def _all_pairs() -> list[str]:
    pairs: list[str] = []
    for i, left in enumerate(HELIXES):
        for right in HELIXES[i:]:
            pairs.append(_normalize_pair(left, right))
    return sorted(set(pairs))


def _load_plot_dependencies():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        thesis_style()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plot dependencies missing. Install with: pip install -r requirements.txt"
        ) from exc
    return plt, pd, sns


def run() -> None:
    print(">>> ANALYSIS: RQ2")

    # Prefer cooccurrence_nli.jsonl (step 4 output) over cooccurrence.jsonl (step 3 output)
    nli_path = config.FILE_COOCCURRENCE_NLI
    coocc_path = nli_path if nli_path.exists() else config.FILE_COOCCURRENCE
    if not coocc_path.exists():
        raise FileNotFoundError(f"Co-occurrence file not found: {coocc_path}")
    if nli_path.exists():
        print(f"[INFO] Using NLI-scored co-occurrences from {nli_path.name}")
    else:
        print("[WARN] cooccurrence_nli.jsonl not found — falling back to cooccurrence.jsonl (no NLI scores). Run step 4.")

    all_cooccurrence_rows = _load_jsonl_safe(coocc_path)
    if not all_cooccurrence_rows:
        raise RuntimeError("Co-occurrence file exists but contains no valid rows for RQ2.")
    # Re-apply NLI threshold from config (all_scores is stored per row, so no re-run needed).
    def _rethreshold(row: dict) -> dict:
        all_scores: dict = row.get("all_scores") or {}
        if not all_scores:
            return row
        best_rel = max(all_scores, key=lambda k: all_scores[k])
        best_score = all_scores[best_rel]
        if best_score < config.NLI_THRESHOLD:
            row = {**row, "relation_type": "no_explicit_relation", "confidence": best_score}
        else:
            row = {**row, "relation_type": best_rel, "confidence": best_score}
        return row
    all_cooccurrence_rows = [_rethreshold(r) for r in all_cooccurrence_rows]

    # Exclude same-helix pairs — Quadruple Helix relations are by definition cross-sphere.
    # cooccurrence_rows excludes same-helix pairs consistently across all figures.
    cooccurrence_rows = [
        r for r in all_cooccurrence_rows
        if normalize_helix(r.get("h1", "")) != normalize_helix(r.get("h2", ""))
        or not normalize_helix(r.get("h1", ""))
    ]

    # Load paragraph counts per country/year for density normalization
    paragraphs_by_country: dict[str, int] = defaultdict(int)
    paragraphs_by_year_rq2: dict[str, int] = defaultdict(int)
    paragraphs_per_doc: dict[str, int] = defaultdict(int)
    doc_year: dict[str, str] = {}
    if config.FILE_PARAGRAPHS.exists():
        for row in utils.load_jsonl(config.FILE_PARAGRAPHS):
            country = to_str(row.get("country", "")).strip().upper() or "UNK"
            paragraphs_by_country[country] += 1
            yr = to_str(row.get("year", "")).strip()
            if not yr:
                from src.utils import year_from_doc
                yr = year_from_doc(to_str(row.get("doc_id", "")))
            if yr:
                paragraphs_by_year_rq2[yr] += 1
            doc_id = to_str(row.get("doc_id", "")).strip()
            if doc_id:
                paragraphs_per_doc[doc_id] += 1
                if yr:
                    doc_year[doc_id] = yr

    pair_counts_by_country: dict[str, Counter] = defaultdict(Counter)
    doc_counts: dict[str, set[str]] = defaultdict(set)
    # relation_type accumulation: (country, pair) → Counter of relation types
    relation_counts_by_country_pair: dict[tuple[str, str], Counter] = defaultdict(Counter)
    # global relation type counter
    relation_counts_global: Counter = Counter()
    # relation type × helix pair counter
    relation_counts_by_pair: dict[str, Counter] = defaultdict(Counter)
    # relation type → list of model confidences (for explicit predictions)
    relation_confidence_by_type: dict[str, list[float]] = defaultdict(list)
    # temporal: year → Counter of relation types (explicit only)
    relation_counts_by_year: dict[str, Counter] = defaultdict(Counter)
    docs_by_year_rq2: dict[str, set] = defaultdict(set)
    relations_per_doc: dict[str, int] = defaultdict(int)
    total_rows = 0

    for row in cooccurrence_rows:
        total_rows += 1
        country = to_str(row.get("country", "")).strip().upper() or "UNK"
        doc_name = to_str(row.get("doc_id", "")).strip()
        if doc_name:
            doc_counts[country].add(doc_name)

        pair = _pair_from_row(row)
        if not pair or PAIR_SEP not in pair:
            continue
        h1, h2 = pair.split(PAIR_SEP, 1)
        if h1 not in HELIXES or h2 not in HELIXES:
            continue
        pair_counts_by_country[country][pair] += 1

        relation_type = to_str(row.get("relation_type", "no_explicit_relation")).strip()
        if not relation_type:
            relation_type = "no_explicit_relation"
        relation_counts_by_country_pair[(country, pair)][relation_type] += 1
        relation_counts_global[relation_type] += 1
        relation_counts_by_pair[pair][relation_type] += 1

        year = to_str(row.get("year", "")).strip()
        if year and relation_type != "no_explicit_relation":
            relation_counts_by_year[year][relation_type] += 1
            docs_by_year_rq2[year].add(country)
            if doc_name:
                relations_per_doc[doc_name] += 1

        if relation_type != "no_explicit_relation":
            try:
                relation_confidence_by_type[relation_type].append(float(row.get("confidence", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass

    # Build relation × helix pair counts (cross-helix only, consistent with all other figures)
    relation_counts_by_pair_all: dict[str, Counter] = defaultdict(Counter)
    for row in cooccurrence_rows:
        pair = _pair_from_row(row)
        if not pair or PAIR_SEP not in pair:
            continue
        h1, h2 = pair.split(PAIR_SEP, 1)
        if h1 not in HELIXES or h2 not in HELIXES:
            continue
        rel = to_str(row.get("relation_type", "no_explicit_relation")).strip() or "no_explicit_relation"
        if rel != "no_explicit_relation":
            relation_counts_by_pair_all[pair][rel] += 1

    if not pair_counts_by_country:
        raise RuntimeError("No valid helix-pair rows in co-occurrence file for RQ2.")

    pairs = _all_pairs()

    # Build density rows (count / paragraphs_per_country)
    density_rows = []
    raw_rows = []
    for country in sorted(pair_counts_by_country.keys()):
        n_paragraphs = paragraphs_by_country.get(country, 0) or 1
        for pair in pairs:
            count = pair_counts_by_country[country].get(pair, 0)
            density = count / n_paragraphs
            density_rows.append({
                "country": country,
                "pair": pair,
                "count": count,
                "density": density,
                "total_paragraphs": paragraphs_by_country.get(country, 0),
                "th_space": _th_space(pair),
                "total_documents_in_country": len(doc_counts[country]),
            })
            raw_rows.append({
                "country": country,
                "pair": pair,
                "count": count,
                "total_documents_in_country": len(doc_counts[country]),
            })

    country_totals = {
        country: sum(counter.values())
        for country, counter in sorted(pair_counts_by_country.items(), key=lambda kv: kv[0])
    }
    pair_totals_counter: Counter = Counter()
    for counter in pair_counts_by_country.values():
        pair_totals_counter.update(counter)
    pair_totals = {pair: pair_totals_counter.get(pair, 0) for pair in sorted(pair_totals_counter)}

    # ── Figure 5: Relation type × helix pair bubble matrix (explicit only, cross-helix pairs) ──
    try:
        plt, pd, sns = _load_plot_dependencies()
        all_pairs_fig5 = _all_pairs()
        explicit_totals_by_pair = {
            pair: sum(relation_counts_by_pair_all[pair].get(rel_type, 0) for rel_type in EXPLICIT_RELATION_TYPES)
            for pair in all_pairs_fig5
        }
        pair_order = [
            pair
            for pair in sorted(all_pairs_fig5, key=lambda p: explicit_totals_by_pair[p], reverse=True)
            if explicit_totals_by_pair[pair] > 0
        ]

        if pair_order:
            max_cell_count = max(
                relation_counts_by_pair_all[pair].get(rel_type, 0)
                for pair in pair_order
                for rel_type in EXPLICIT_RELATION_TYPES
            ) or 1

            fig = plt.figure(figsize=(max(16, len(pair_order) * 1.1), 12))
            gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.4], width_ratios=[1, 0.03], hspace=0.08, wspace=0.05)
            ax = fig.add_subplot(gs[0, 0])
            ax_cbar = fig.add_subplot(gs[0, 1])
            ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax)
            fig.add_subplot(gs[1, 1]).set_visible(False)

            xs: list[int] = []
            ys: list[int] = []
            sizes: list[float] = []
            shares: list[float] = []
            counts: list[int] = []

            for x_idx, pair in enumerate(pair_order):
                pair_total = explicit_totals_by_pair[pair] or 1
                for y_idx, rel_type in enumerate(EXPLICIT_RELATION_TYPES):
                    count = relation_counts_by_pair_all[pair].get(rel_type, 0)
                    if count <= 0:
                        continue
                    xs.append(x_idx)
                    ys.append(y_idx)
                    counts.append(count)
                    sizes.append(120 + (1600 * (count / max_cell_count)))
                    shares.append(count / pair_total)

            scatter = ax.scatter(
                xs,
                ys,
                s=sizes,
                c=shares,
                cmap="YlGnBu",
                vmin=0.0,
                vmax=1.0,
                alpha=0.95,
                edgecolors="white",
                linewidths=0.9,
                zorder=3,
            )
            for x_idx, y_idx, count in zip(xs, ys, counts):
                if count >= max(20, int(max_cell_count * 0.2)):
                    ax.text(x_idx, y_idx, str(count), ha="center", va="center", fontsize=8, color="#1a1a1a")

            for xline in range(len(pair_order) + 1):
                ax.axvline(xline - 0.5, color="#cccccc", linewidth=0.8, zorder=0)
            for xline in range(len(pair_order) + 1):
                ax_bottom.axvline(xline - 0.5, color="#cccccc", linewidth=0.8, zorder=0)
            for yline in range(len(EXPLICIT_RELATION_TYPES) + 1):
                ax.axhline(yline - 0.5, color="#f5f5f5", linewidth=0.8, zorder=0)

            ax.set_yticks(range(len(EXPLICIT_RELATION_TYPES)))
            ax.set_yticklabels([r.replace("_", " ") for r in EXPLICIT_RELATION_TYPES], fontsize=9)
            ax.set_xticks(range(len(pair_order)))
            ax.set_xticklabels([])
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.set_xlabel("")
            ax.set_ylabel("Relation type")
            ax.set_title("Relation Type Profile per Helix Pair", fontsize=11, pad=8)
            ax_bottom.set_title(
                "TH/QH Innovation Space per Helix Pair",
                fontsize=10, pad=6,
            )

            cbar = fig.colorbar(scatter, cax=ax_cbar)
            cbar.set_label("Share within helix pair")

            # Bubble size legend — show representative counts
            from matplotlib.lines import Line2D
            ref_counts = [
                int(max_cell_count * 0.1),
                int(max_cell_count * 0.5),
                max_cell_count,
            ]
            size_handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                       markeredgecolor="#333333", markeredgewidth=0.5,
                       markersize=(120 + 1600 * (c / max_cell_count)) ** 0.5 * 0.52,
                       label=f"n = {c}")
                for c in ref_counts
            ]
            ax.legend(
                handles=size_handles,
                title="Bubble size (raw count)",
                title_fontsize=8,
                fontsize=8,
                loc="upper right",
                framealpha=0.9,
                edgecolor="#cccccc",
            )

            pair_totals = [explicit_totals_by_pair[p] for p in pair_order]
            pair_colors = [TH_SPACE_COLORS[_th_space(p)] for p in pair_order]
            ax_bottom.bar(range(len(pair_order)), pair_totals, color=pair_colors, alpha=0.9, width=0.6)
            ax_bottom.set_xticks(range(len(pair_order)))
            ax_bottom.set_xticklabels(
                [p.replace("_", " ") for p in pair_order],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            ax_bottom.set_ylabel("Number of explicit relations", labelpad=8)
            ax_bottom.grid(axis="y", linestyle="--", alpha=0.25)

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=col, label=space) for space, col in TH_SPACE_COLORS.items()]
            ax_bottom.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize=8, title="Innovation Space")

            # Set xlim after all plotting so matplotlib doesn't reset it
            xlim = (-0.5, len(pair_order) - 0.5)
            ax.set_xlim(xlim)
            ax_bottom.set_xlim(xlim)

            fig.tight_layout(pad=2.0)
            fig.savefig(config.ANALYSIS_RQ2_RELATION_HELIX_FIGURE_PATH, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Wrote: {config.ANALYSIS_RQ2_RELATION_HELIX_FIGURE_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ2 relation × helix pair figure skipped: {exc}")

    # ── Figure 5b: Relation type × helix pair bubble matrix (bubble only, total counts bar, no space colours) ──
    try:
        plt, pd, sns = _load_plot_dependencies()
        if pair_order:
            fig2 = plt.figure(figsize=(max(16, len(pair_order) * 1.1), 10))
            gs2 = fig2.add_gridspec(2, 2, height_ratios=[3.0, 1.4], width_ratios=[1, 0.03], hspace=0.08, wspace=0.05)
            ax2 = fig2.add_subplot(gs2[0, 0])
            ax2_cbar = fig2.add_subplot(gs2[0, 1])
            ax2_bottom = fig2.add_subplot(gs2[1, 0], sharex=ax2)
            fig2.add_subplot(gs2[1, 1]).set_visible(False)

            xs2: list[int] = []
            ys2: list[int] = []
            sizes2: list[float] = []
            shares2: list[float] = []
            counts2: list[int] = []

            for x_idx, pair in enumerate(pair_order):
                pair_total = explicit_totals_by_pair[pair] or 1
                for y_idx, rel_type in enumerate(EXPLICIT_RELATION_TYPES):
                    count = relation_counts_by_pair_all[pair].get(rel_type, 0)
                    if count <= 0:
                        continue
                    xs2.append(x_idx)
                    ys2.append(y_idx)
                    counts2.append(count)
                    sizes2.append(120 + (1600 * (count / max_cell_count)))
                    shares2.append(count / pair_total)

            scatter2 = ax2.scatter(
                xs2, ys2, s=sizes2, c=shares2, cmap="YlGnBu",
                vmin=0.0, vmax=1.0, alpha=0.95,
                edgecolors="white", linewidths=0.9, zorder=3,
            )
            for x_idx, y_idx, count in zip(xs2, ys2, counts2):
                if count >= max(20, int(max_cell_count * 0.2)):
                    ax2.text(x_idx, y_idx, str(count), ha="center", va="center", fontsize=8, color="#1a1a1a")

            for xline in range(len(pair_order) + 1):
                ax2.axvline(xline - 0.5, color="#cccccc", linewidth=0.8, zorder=0)
                ax2_bottom.axvline(xline - 0.5, color="#cccccc", linewidth=0.8, zorder=0)
            for yline in range(len(EXPLICIT_RELATION_TYPES) + 1):
                ax2.axhline(yline - 0.5, color="#f5f5f5", linewidth=0.8, zorder=0)

            ax2.set_yticks(range(len(EXPLICIT_RELATION_TYPES)))
            ax2.set_yticklabels([r.replace("_", " ") for r in EXPLICIT_RELATION_TYPES], fontsize=9)
            ax2.set_xticks(range(len(pair_order)))
            ax2.set_xticklabels([])
            ax2.tick_params(axis="x", bottom=False, labelbottom=False)
            ax2.set_ylabel("Relation type")
            ax2.set_title("Relation Type Profile per Helix Pair", fontsize=11, pad=8)

            cbar2 = fig2.colorbar(scatter2, cax=ax2_cbar)
            cbar2.set_label("Share within helix pair")

            from matplotlib.lines import Line2D as _Line2D
            ref_counts2 = [int(max_cell_count * 0.1), int(max_cell_count * 0.5), max_cell_count]
            size_handles2 = [
                _Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                        markeredgecolor="#333333", markeredgewidth=0.5,
                        markersize=(120 + 1600 * (c / max_cell_count)) ** 0.5 * 0.52,
                        label=f"n = {c}")
                for c in ref_counts2
            ]
            ax2.legend(handles=size_handles2, title="Bubble size (raw count)", title_fontsize=8,
                       fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#cccccc")

            # Bottom panel: plain grey bars with total counts, no space colours
            pair_totals2 = [explicit_totals_by_pair[p] for p in pair_order]
            ax2_bottom.bar(range(len(pair_order)), pair_totals2, color="#888888", alpha=0.85, width=0.6)
            ax2_bottom.set_xticks(range(len(pair_order)))
            ax2_bottom.set_xticklabels(
                [p.replace("_", " ") for p in pair_order],
                rotation=45, ha="right", fontsize=8,
            )
            ax2_bottom.set_ylabel("Number of explicit relations", labelpad=8)
            ax2_bottom.set_title("Total explicit relations per helix pair", fontsize=10, pad=6)
            ax2_bottom.grid(axis="y", linestyle="--", alpha=0.25)

            xlim2 = (-0.5, len(pair_order) - 0.5)
            ax2.set_xlim(xlim2)
            ax2_bottom.set_xlim(xlim2)

            fig2.tight_layout(pad=2.0)
            fig2.savefig(config.ANALYSIS_RQ2_RELATION_HELIX_SIMPLE_FIGURE_PATH, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            print(f"[OK] Wrote: {config.ANALYSIS_RQ2_RELATION_HELIX_SIMPLE_FIGURE_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ2 relation × helix pair simple figure skipped: {exc}")

    # ── Figure 5c: Bubble matrix only — no bar chart ──────────────────────────
    try:
        plt, pd, sns = _load_plot_dependencies()
        if pair_order:
            fig3, (ax3, ax3_cbar_ax) = plt.subplots(
                1, 2,
                figsize=(max(14, len(pair_order) * 1.0), 5.5),
                gridspec_kw={"width_ratios": [1, 0.025], "wspace": 0.04},
            )

            xs3: list[int] = []
            ys3: list[int] = []
            sizes3: list[float] = []
            shares3: list[float] = []
            counts3: list[int] = []

            for x_idx, pair in enumerate(pair_order):
                pair_total = explicit_totals_by_pair[pair] or 1
                for y_idx, rel_type in enumerate(EXPLICIT_RELATION_TYPES):
                    count = relation_counts_by_pair_all[pair].get(rel_type, 0)
                    if count <= 0:
                        continue
                    xs3.append(x_idx)
                    ys3.append(y_idx)
                    counts3.append(count)
                    sizes3.append(120 + (1600 * (count / max_cell_count)))
                    shares3.append(count / pair_total)

            scatter3 = ax3.scatter(
                xs3, ys3, s=sizes3, c=shares3, cmap="YlGnBu",
                vmin=0.0, vmax=1.0, alpha=0.95,
                edgecolors="white", linewidths=0.9, zorder=3,
            )
            for x_idx, y_idx, count in zip(xs3, ys3, counts3):
                if count >= max(20, int(max_cell_count * 0.2)):
                    ax3.text(x_idx, y_idx, str(count), ha="center", va="center",
                             fontsize=8, color="#1a1a1a")

            for xline in range(len(pair_order) + 1):
                ax3.axvline(xline - 0.5, color="#cccccc", linewidth=0.8, zorder=0)
            for yline in range(len(EXPLICIT_RELATION_TYPES) + 1):
                ax3.axhline(yline - 0.5, color="#f5f5f5", linewidth=0.8, zorder=0)

            ax3.set_yticks(range(len(EXPLICIT_RELATION_TYPES)))
            ax3.set_yticklabels([r.replace("_", " ") for r in EXPLICIT_RELATION_TYPES])
            ax3.set_xticks(range(len(pair_order)))
            ax3.set_xticklabels(
                [p.replace("_", " ") for p in pair_order],
                rotation=45, ha="right",
            )
            ax3.set_ylabel("Relation type")
            ax3.set_title("Relation Type Profile per Helix Pair", pad=10)
            ax3.set_xlim(-0.5, len(pair_order) - 0.5)

            cbar3 = fig3.colorbar(scatter3, cax=ax3_cbar_ax)
            cbar3.set_label("Share within helix pair")

            from matplotlib.lines import Line2D as _L2D
            ref_counts3 = [int(max_cell_count * 0.1), int(max_cell_count * 0.5), max_cell_count]
            size_handles3 = [
                _L2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                     markeredgecolor="#333333", markeredgewidth=0.5,
                     markersize=(120 + 1600 * (c / max_cell_count)) ** 0.5 * 0.52,
                     label=f"n = {c}")
                for c in ref_counts3
            ]
            ax3.legend(handles=size_handles3, title="Bubble size (raw count)",
                       title_fontsize=8, fontsize=8, loc="upper right",
                       framealpha=0.9, edgecolor="#cccccc")

            fig3.tight_layout(pad=2.0)
            fig3.savefig(config.ANALYSIS_RQ2_RELATION_HELIX_BUBBLE_ONLY_PATH, dpi=300, bbox_inches="tight")
            plt.close(fig3)
            print(f"[OK] Wrote: {config.ANALYSIS_RQ2_RELATION_HELIX_BUBBLE_ONLY_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ2 bubble-only figure skipped: {exc}")

    # ── Figure 6: Country-level relation profile (heatmap country × helix pair) ─
    try:
        plt, pd, sns = _load_plot_dependencies()
        import numpy as np

        countries_sorted = sorted({c for (c, _) in relation_counts_by_country_pair})
        pair_order_detail = [
            p for p in sorted(pairs, key=lambda p: sum(
                sum(relation_counts_by_country_pair.get((c, p), Counter()).values())
                for c in countries_sorted
            ), reverse=True)
        ]

        if countries_sorted and pair_order_detail:
            # Build count matrix and dominant-relation matrix
            count_matrix = np.zeros((len(countries_sorted), len(pair_order_detail)))
            dom_matrix = np.full((len(countries_sorted), len(pair_order_detail)), -1, dtype=int)
            rel_index = {r: i for i, r in enumerate(EXPLICIT_RELATION_TYPES)}

            for ci, country in enumerate(countries_sorted):
                for pi, pair in enumerate(pair_order_detail):
                    counter = relation_counts_by_country_pair.get((country, pair), Counter())
                    explicit_total = sum(counter.get(r, 0) for r in EXPLICIT_RELATION_TYPES)
                    count_matrix[ci, pi] = explicit_total
                    if explicit_total > 0:
                        dom_rel = max(EXPLICIT_RELATION_TYPES, key=lambda r: counter.get(r, 0))
                        dom_matrix[ci, pi] = rel_index[dom_rel]

            # Color map: one color per relation type
            rel_colors = [RELATION_COLORS[r] for r in EXPLICIT_RELATION_TYPES]

            fig, ax = plt.subplots(figsize=(max(14, len(pair_order_detail) * 1.0), max(6, len(countries_sorted) * 0.5)))

            for ci in range(len(countries_sorted)):
                for pi in range(len(pair_order_detail)):
                    count = int(count_matrix[ci, pi])
                    if count == 0:
                        ax.add_patch(plt.Rectangle((pi - 0.5, ci - 0.5), 1, 1, color="#f5f5f5", zorder=1))
                    else:
                        dom_idx = dom_matrix[ci, pi]
                        color = rel_colors[dom_idx] if dom_idx >= 0 else "#dddddd"
                        alpha = 0.3 + 0.7 * (count / count_matrix.max())
                        ax.add_patch(plt.Rectangle((pi - 0.5, ci - 0.5), 1, 1, color=color, alpha=alpha, zorder=1))
                        ax.text(pi, ci, str(count), ha="center", va="center", fontsize=7, color="#1a1a1a", zorder=2)

            ax.set_xlim(-0.5, len(pair_order_detail) - 0.5)
            ax.set_ylim(-0.5, len(countries_sorted) - 0.5)
            ax.set_xticks(range(len(pair_order_detail)))
            ax.set_xticklabels([p.replace("_", " ") for p in pair_order_detail], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(countries_sorted)))
            ax.set_yticklabels(countries_sorted, fontsize=9)
            ax.set_title("RQ2: Explicit Relation Count by Country and Helix Pair\nCell color = dominant explicit relation type · intensity = explicit relation count · white = no explicit relation detected", fontsize=11)

            # Grid lines
            for x in range(len(pair_order_detail) + 1):
                ax.axvline(x - 0.5, color="#cccccc", linewidth=0.6, zorder=0)
            for y in range(len(countries_sorted) + 1):
                ax.axhline(y - 0.5, color="#cccccc", linewidth=0.6, zorder=0)

            # Legend: relation types
            from matplotlib.patches import Patch
            legend_handles = [Patch(facecolor=RELATION_COLORS[r], label=r.replace("_", " ").title()) for r in EXPLICIT_RELATION_TYPES]
            ax.legend(handles=legend_handles, title="Dominant relation type", fontsize=8,
                      loc="upper right", bbox_to_anchor=(1.0, 1.0), framealpha=0.9)

            fig.tight_layout(pad=2.0)
            fig.savefig(config.ANALYSIS_RQ2_RELATION_COUNTRY_DETAIL_PATH, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Wrote: {config.ANALYSIS_RQ2_RELATION_COUNTRY_DETAIL_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ2 country detail figure skipped: {exc}")

    # ── Figure 7: Explicit relation types over time (cumulative density) ──────
    try:
        plt, pd, sns = _load_plot_dependencies()
        import numpy as np

        years = sorted(relation_counts_by_year.keys())
        if len(years) >= 2:
            fig, ax = plt.subplots(figsize=(max(10, len(years) * 1.5), 5))

            # Cumulative counts per relation type up to each year → share from running totals
            # Normalizes by paragraphs so doc-length differences don't skew counts.
            cum_counts_rel: dict[str, float] = {r: 0.0 for r in EXPLICIT_RELATION_TYPES}
            cum_paragraphs_total = 0
            cum_share_matrix: list[list[float]] = [[] for _ in EXPLICIT_RELATION_TYPES]
            cum_density_vals: list[float] = []

            for yr in years:
                n_par = paragraphs_by_year_rq2.get(yr, 0) or 1
                cum_paragraphs_total += n_par
                for r in EXPLICIT_RELATION_TYPES:
                    # Add length-normalized contribution of this year
                    cum_counts_rel[r] += relation_counts_by_year[yr].get(r, 0) / n_par
                total_cum = sum(cum_counts_rel.values()) or 1.0
                for i, r in enumerate(EXPLICIT_RELATION_TYPES):
                    cum_share_matrix[i].append(cum_counts_rel[r] / total_cum)
                cum_density_vals.append(total_cum)

            x_pos = list(range(len(years)))
            ax.stackplot(
                x_pos,
                cum_share_matrix,
                labels=[r.replace("_", " ").title() for r in EXPLICIT_RELATION_TYPES],
                colors=[RELATION_COLORS[r] for r in EXPLICIT_RELATION_TYPES],
                alpha=0.85,
            )

            # Per-year technology transfer share: TT count / total explicit relations
            # Grounded in Ranga & Etzkowitz (2013): TT is "the core activity in an
            # innovation system" — tracking its share shows whether strategies mature
            # toward explicit knowledge-commercialization framing over time.
            tt_share_vals = []
            for yr in years:
                yr_counts = relation_counts_by_year.get(yr, {})
                yr_total = sum(yr_counts.values()) or 1
                tt_share_vals.append(yr_counts.get("technology_transfer", 0) / yr_total)

            ax2 = ax.twinx()
            ax2.plot(
                x_pos, tt_share_vals,
                color="#e34a33", linewidth=2, linestyle="--", marker="D",
                markersize=5, label="Technology transfer share (per year)",
            )
            ax2.set_ylim(0, 1.0)
            ax2.set_ylabel("Technology transfer share of explicit relations", fontsize=9)
            ax2.legend(fontsize=9, loc="upper right")

            x_labels = []
            cum_countries_rq2: set[str] = set()
            for yr in years:
                cum_countries_rq2.update(docs_by_year_rq2.get(yr, set()))
                n = len(cum_countries_rq2)
                x_labels.append(f"{yr}\nN={n}")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=7, ha="center")

            ax.set_title(
                "RQ2: Relation Type Share Over Time (Cumulative)\n"
                "(technology transfer share per year — Ranga & Etzkowitz 2013: 'core activity')",
                fontsize=12,
            )
            ax.set_xlabel("Year of strategy publication")
            ax.set_ylabel("Cumulative share of explicit relations")
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            fig.tight_layout()
            fig.savefig(config.ANALYSIS_RQ2_RELATION_TEMPORAL_PATH, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Wrote: {config.ANALYSIS_RQ2_RELATION_TEMPORAL_PATH}")
        else:
            print("[WARN] RQ2 temporal figure skipped: not enough year data points")
    except Exception as exc:
        print(f"[WARN] RQ2 temporal figure skipped: {exc}")



if __name__ == "__main__":
    run()
