"""RQ1 deliverable: actor prominence and helix balance by country (JSON + figures + CSV).

Uses the Step2 schema (`status`, `level_5_helix`, `level_1_actor_type`,
`level_2_sphere_boundary`, `level_4_innovation_type`).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from math import log

from src import config, utils
from src.analysis._helpers import utc_now_iso, write_json, thesis_style
from src.utils import country_from_doc, normalize_helix, to_int, to_str, year_from_doc


HELIXES = ["government", "industry", "academia", "civil_society", "intermediary"]
HELIX_COLORS = {
    "government":    "#2166ac",
    "industry":      "#d6604d",
    "academia":      "#4dac26",
    "civil_society": "#8073ac",
    "intermediary":  "#f4a582",
}


def _load_actor_entities() -> list[dict]:
    if config.STEP2_CLASSIFIED_PATH.exists():
        rows = list(utils.load_jsonl(config.STEP2_CLASSIFIED_PATH))
        grouped: dict[tuple[int, int], dict] = {}
        for row in rows:
            key = (to_int(row.get("doc_id", 0), 0), to_int(row.get("entity_id", 0), 0))
            if key[0] and key[1] and key not in grouped:
                grouped[key] = row
        entities = list(grouped.values())
        if any(to_str(row.get("level_5_helix", "")).strip() for row in entities):
            return entities

    if not config.STEP2_MANUAL_LABELS_PATH.exists():
        return []
    payload = json.loads(config.STEP2_MANUAL_LABELS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return payload


def _load_plot_dependencies():
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
    print(">>> ANALYSIS: RQ1")

    actor_rows = _load_actor_entities()
    if not actor_rows:
        raise RuntimeError("No Step2 actor rows available for RQ1.")

    paragraphs_by_country: dict[str, int] = defaultdict(int)
    paragraphs_by_year: dict[str, int] = defaultdict(int)
    if config.FILE_PARAGRAPHS.exists():
        for row in utils.load_jsonl(config.FILE_PARAGRAPHS):
            country = to_str(row.get("country", "")).strip().upper()
            if not country:
                country = country_from_doc(row.get("doc_id", ""))
            paragraphs_by_country[country] += 1
            yr = to_str(row.get("year", "")).strip()
            if not yr:
                yr = year_from_doc(to_str(row.get("doc_id", "")))
            if yr:
                paragraphs_by_year[yr] += 1

    counts_by_country: dict[str, Counter] = defaultdict(Counter)

    # Temporal counters (for rq1_helix_share_over_time figure)
    counts_by_year: dict[str, Counter] = defaultdict(Counter)
    docs_by_year: dict[str, set] = defaultdict(set)

    # Global component counters (for rq1_components_breakdown figure)
    rd_by_helix: dict[str, Counter] = defaultdict(Counter)
    sphere_by_helix: dict[str, Counter] = defaultdict(Counter)
    actor_type_by_helix: dict[str, Counter] = defaultdict(Counter)
    # Exact category counters (for rq1_components_detail figure)
    exact_cat_by_helix: dict[str, Counter] = defaultdict(Counter)
    exact_cat_global: Counter = Counter()
    # Origin scope (for panel 3 of breakdown figure)
    origin_by_helix: dict[str, Counter] = defaultdict(Counter)

    for row in actor_rows:
        status = to_str(row.get("status", "")).strip().lower()
        if status != "entity":
            continue
        sphere = normalize_helix(row.get("level_5_helix", ""))
        if sphere not in HELIXES:
            continue

        doc_name = to_str(row.get("doc_name", "")).strip()
        if not doc_name:
            doc_id = to_int(row.get("doc_id", 0), 0)
            doc_name = f"DOC_{doc_id}" if doc_id else ""
        country = country_from_doc(doc_name)
        counts_by_country[country][sphere] += 1

        year = year_from_doc(doc_name)
        if year:
            counts_by_year[year][sphere] += 1
            docs_by_year[year].add(country_from_doc(doc_name))

        # R&D vs Non-R&D
        rd = to_str(row.get("level_4_innovation_type", "")).strip()
        rd_by_helix[sphere][rd if rd else "Unknown"] += 1

        # Single-sphere vs Multi-sphere
        sb = to_str(row.get("level_2_sphere_boundary", "")).strip()
        sphere_by_helix[sphere][sb if sb else "Unknown"] += 1

        # Institutional vs Individual
        at = to_str(row.get("level_1_actor_type", "")).strip().lower()
        if "individual" in at and "institution" not in at:
            actor_type_by_helix[sphere]["Individual"] += 1
        else:
            actor_type_by_helix[sphere]["Institutional"] += 1

        # Exact category
        ec = to_str(row.get("level_3_exact_category", "")).strip()
        if not ec:
            ec = "other"
        exact_cat_by_helix[sphere][ec] += 1
        exact_cat_global[(ec, sphere)] += 1

        # Origin scope
        origin = to_str(row.get("institution_origin_scope", "")).strip().lower()
        if not origin:
            origin = "unknown"
        origin_by_helix[sphere][origin] += 1

    countries = sorted(counts_by_country.keys())
    if not countries:
        raise RuntimeError("No classified entity rows with helix spheres found for RQ1.")

    rows = []
    for country in countries:
        total_mentions = sum(counts_by_country[country].values())
        total_paragraphs = paragraphs_by_country.get(country, 0)
        shares = {}
        densities = {}
        for helix in HELIXES:
            count = counts_by_country[country][helix]
            shares[helix] = (count / total_mentions) if total_mentions else 0.0
            densities[helix] = (count / total_paragraphs) if total_paragraphs else 0.0

        entropy = 0.0
        for p in shares.values():
            if p > 0:
                entropy -= p * log(p)
        entropy_norm = entropy / log(len(HELIXES)) if total_mentions else 0.0

        rows.append(
            {
                "country": country,
                "total_mentions": total_mentions,
                "total_paragraphs": total_paragraphs,
                "helix_balance_index": entropy_norm,
                "shares": shares,
                "densities_per_paragraph": densities,
            }
        )

    # Sort by HBI ascending (least balanced first)
    rows.sort(key=lambda r: r["helix_balance_index"], reverse=False)

    output = {
        "id": "rq1_actor_prominence_and_balance",
        "title": "RQ1 Actor Prominence and Helix Balance by Country",
        "generated_utc": utc_now_iso(),
        "helix_values": HELIXES,
        "definition_helix_balance_index": "Normalized Shannon entropy over helix shares within each country.",
        "rows": rows,
        "rankings": {
            "by_total_mentions": [
                r["country"]
                for r in sorted(rows, key=lambda r: r["total_mentions"], reverse=True)
            ],
            "by_helix_balance_index": [r["country"] for r in rows],
        },
        "figure_path": str(config.ANALYSIS_RQ1_FIGURE_PATH),
    }

    write_json(config.ANALYSIS_RQ1_PATH, output)

    # ── Figure 1: Improved existing figure (sorted by HBI, reference line) ──
    figure_written = False
    try:
        plt, np = _load_plot_dependencies()
        countries_sorted = [r["country"] for r in rows]
        share_matrix = np.array([[r["shares"][h] for h in HELIXES] for r in rows])
        balance_values = [r["helix_balance_index"] for r in rows]

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2.4, 1.2]}
        )

        x = np.arange(len(countries_sorted))
        bottom = np.zeros(len(countries_sorted))
        colors = [HELIX_COLORS[h] for h in HELIXES]
        for idx, helix in enumerate(HELIXES):
            vals = share_matrix[:, idx]
            ax1.bar(
                x, vals, bottom=bottom,
                color=colors[idx],
                label=helix.replace("_", " ").title(),
            )
            bottom += vals
        ax1.set_title(
            "RQ1: Helix Share by Country\n(sorted by Helix Balance Index, least balanced first)",
            fontsize=11,
        )
        ax1.set_ylabel("Share of classified actors")
        ax1.set_xticks(x)
        ax1.set_xticklabels(countries_sorted, rotation=45, ha="right")
        ax1.set_ylim(0, 1.0)
        ax1.legend(loc="upper right", fontsize=9)

        bar_colors = ["#4393c3" if v >= 0.80 else "#d1e5f0" for v in balance_values]
        ax2.bar(x, balance_values, color=bar_colors)
        ax2.axhline(
            0.80, color="#e34a33", linestyle="--", linewidth=1.4,
            label="Balanced threshold (HBI = 0.80)",
        )
        ax2.set_title("RQ1: Helix Balance Index\n(normalized Shannon entropy)", fontsize=11)
        ax2.set_ylabel("Normalized entropy (HBI)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(countries_sorted, rotation=45, ha="right")
        ax2.set_ylim(0, 1.0)
        ax2.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(config.ANALYSIS_RQ1_FIGURE_PATH, dpi=300)
        plt.close(fig)
        figure_written = True
        print(f"[OK] Wrote: {config.ANALYSIS_RQ1_FIGURE_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ1 figure skipped: {exc}")

    # ── Figure 2: Components breakdown (3-panel, aggregated distinctions) ────
    try:
        plt, np = _load_plot_dependencies()
        from matplotlib.patches import Patch
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
        x = np.arange(len(HELIXES))
        helix_labels = [h.replace("_", " ").title() for h in HELIXES]
        width = 0.35

        GREY = "#b0b0b0"
        GREEN = "#4caf50"   # green — R&D and Individual
        ORANGE = "#f57c00"  # orange — Multi-sphere

        # Panel 1: R&D vs Non-R&D — stacked bar per helix
        ax = axes[0]
        rd_vals, non_rd_vals = [], []
        for h in HELIXES:
            total = sum(rd_by_helix[h].values()) or 1
            rd_count = rd_by_helix[h].get("R&D", 0) + rd_by_helix[h].get("Both", 0)
            rd_vals.append(rd_count / total)
            non_rd_vals.append(rd_by_helix[h].get("Non R&D", 0) / total)
        rd_vals = np.array(rd_vals)
        non_rd_vals = np.array(non_rd_vals)
        ax.bar(x, rd_vals, width=0.6, label="R&D", color=GREEN)
        ax.bar(x, non_rd_vals, width=0.6, bottom=rd_vals, label="Non R&D", color=GREY)
        ax.set_title("Innovation Activity", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(helix_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Share of entities")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlabel("Per-helix aggregated share", fontsize=8, color="#555555")

        # Panel 2: Single- vs Multi-sphere — pie chart (aggregate across all helixes)
        ax = axes[1]
        total_single, total_multi = 0, 0
        for h in HELIXES:
            for raw_label, count in sphere_by_helix[h].items():
                token = to_str(raw_label).strip().lower()
                token = token.replace("_", "").replace("-", "").replace(" ", "")
                if token == "singlesphere":
                    total_single += count
                elif token == "multisphere":
                    total_multi += count

        grand = (total_single + total_multi) or 1
        sizes = [total_single / grand, total_multi / grand]
        wedge_props = {"linewidth": 1.2, "edgecolor": "white"}
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            colors=[GREY, ORANGE],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=wedge_props,
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight("bold")
        ax.legend(wedges, ["Single-sphere", "Multi-sphere (hybrid)"], fontsize=9, loc="upper right")
        ax.set_title("Sphere Boundary", fontsize=10)

        # Panel 3: Institutional vs Individual — stacked bar per helix
        ax = axes[2]
        inst_vals, ind_vals = [], []
        for h in HELIXES:
            total = sum(actor_type_by_helix[h].values()) or 1
            inst_vals.append(actor_type_by_helix[h].get("Institutional", 0) / total)
            ind_vals.append(actor_type_by_helix[h].get("Individual", 0) / total)
        inst_vals = np.array(inst_vals)
        ind_vals = np.array(ind_vals)
        BLUE = "#2166ac"
        ax.bar(x, inst_vals, width=0.6, label="Institutional", color=GREY)
        ax.bar(x, ind_vals, width=0.6, bottom=inst_vals, label="Individual", color=BLUE)
        ax.set_title("Actor Type", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(helix_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Share of entities")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlabel("Per-helix aggregated share", fontsize=8, color="#555555")

        fig.suptitle(
            "RQ1 Components Breakdown by Helix",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(config.ANALYSIS_RQ1_COMPONENTS_FIGURE_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Wrote: {config.ANALYSIS_RQ1_COMPONENTS_FIGURE_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ1 components figure skipped: {exc}")

    # ── Temporal figure: cumulative helix density over time ──────────────────
    try:
        plt, np = _load_plot_dependencies()

        years = sorted(counts_by_year.keys())
        if len(years) >= 2:
            fig, ax = plt.subplots(figsize=(max(10, len(years) * 1.5), 5))

            # Cumulative counts per helix up to each year → share from running totals
            # Normalizes by paragraphs so doc-length differences don't skew counts.
            cum_counts: dict[str, float] = {h: 0.0 for h in HELIXES}
            cum_paragraphs = 0
            cum_share_matrix: list[list[float]] = [[] for _ in HELIXES]
            hbi_vals: list[float] = []

            for yr in years:
                n_par = paragraphs_by_year.get(yr, 0) or 1
                cum_paragraphs += n_par
                for h in HELIXES:
                    # Add length-normalized contribution of this year
                    cum_counts[h] += counts_by_year[yr].get(h, 0) / n_par
                total_cum = sum(cum_counts.values()) or 1.0
                for i, h in enumerate(HELIXES):
                    cum_share_matrix[i].append(cum_counts[h] / total_cum)
                entropy = 0.0
                for h in HELIXES:
                    p = cum_counts[h] / total_cum
                    if p > 0:
                        entropy -= p * log(p)
                hbi_vals.append(entropy / log(len(HELIXES)))

            x_pos = list(range(len(years)))
            ax.stackplot(
                x_pos,
                cum_share_matrix,
                labels=[h.replace("_", " ").title() for h in HELIXES],
                colors=[HELIX_COLORS[h] for h in HELIXES],
                alpha=0.85,
            )

            ax2 = ax.twinx()
            ax2.plot(
                x_pos, hbi_vals,
                color="#e34a33", linewidth=2, linestyle="--", marker="D",
                markersize=5, label="Cumulative HBI",
            )
            ax2.set_ylim(0, 1.0)
            ax2.set_ylabel("Cumulative HBI (normalized Shannon entropy)", fontsize=9)
            ax2.legend(fontsize=9, loc="lower right")

            x_labels = []
            cum_countries: set[str] = set()
            for yr in years:
                cum_countries.update(docs_by_year.get(yr, set()))
                n = len(cum_countries)
                x_labels.append(f"{yr}\nN={n}")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=7, ha="center")

            ax.set_title(
                "RQ1: Helix Share of Classified Actors Over Time (Cumulative)\n"
                "(running share from paragraph-normalized counts)",
                fontsize=12,
            )
            ax.set_xlabel("Year of strategy publication")
            ax.set_ylabel("Cumulative share of classified actors")
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            fig.tight_layout()
            fig.savefig(config.ANALYSIS_RQ1_TEMPORAL_FIGURE_PATH, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Wrote: {config.ANALYSIS_RQ1_TEMPORAL_FIGURE_PATH}")
        else:
            print("[WARN] RQ1 temporal figure skipped: not enough year data points")
    except Exception as exc:
        print(f"[WARN] RQ1 temporal figure skipped: {exc}")

    # ── CSV export ───────────────────────────────────────────────────────────
    try:
        fieldnames = [
            "Country", "Total_Entities", "Gov_Share", "Ind_Share", "Acad_Share",
            "CS_Share", "Int_Share", "HBI", "Paragraphs",
        ]
        with open(config.ANALYSIS_RQ1_CSV_PATH, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    "Country": r["country"],
                    "Total_Entities": r["total_mentions"],
                    "Gov_Share": f"{r['shares']['government']:.4f}",
                    "Ind_Share": f"{r['shares']['industry']:.4f}",
                    "Acad_Share": f"{r['shares']['academia']:.4f}",
                    "CS_Share": f"{r['shares']['civil_society']:.4f}",
                    "Int_Share": f"{r['shares']['intermediary']:.4f}",
                    "HBI": f"{r['helix_balance_index']:.4f}",
                    "Paragraphs": r["total_paragraphs"],
                })
        print(f"[OK] Wrote: {config.ANALYSIS_RQ1_CSV_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ1 CSV skipped: {exc}")

    print(f"[OK] Wrote: {config.ANALYSIS_RQ1_PATH}")



if __name__ == "__main__":
    run()
