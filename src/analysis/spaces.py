"""TH Space analysis: SetFit-predicted Triple/Quadruple Helix space distribution.

Mirrors the structure of rq2.py but uses ``th_space_setfit`` as the
primary dimension instead of helix pairs.

Outputs
-------
- spaces_by_country_heatmap.png   country × space counts (normalised by paragraphs)
- spaces_global_counts.png        global space ranking bar chart
- spaces_by_country_stacked.png   space share per country (stacked bar)
- spaces_relation_matrix.png      relation type × space bubble matrix
- spaces_table.csv                country × space density table
- spaces_summary.json             structured summary
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict

from src import config, utils
from src.analysis._helpers import utc_now_iso, write_json, thesis_style
from src.utils import normalize_helix, to_str


SPACES = ["knowledge_space", "innovation_space", "consensus_space", "public_space", "no_explicit_space"]

SPACE_LABELS = {
    "knowledge_space":   "Knowledge Space",
    "innovation_space":  "Innovation Space",
    "consensus_space":   "Consensus Space",
    "public_space":      "Public Space",
    "no_explicit_space": "No Explicit Space",
}

SPACE_COLORS = {
    "knowledge_space":   "#4393c3",
    "innovation_space":  "#74c476",
    "consensus_space":   "#fd8d3c",
    "public_space":      "#9e9ac8",
    "no_explicit_space": "#bdbdbd",
}

EXPLICIT_RELATION_TYPES = [
    "technology_transfer",
    "collaboration_conflict_moderation",
    "collaborative_leadership",
    "substitution",
    "networking",
]

RELATION_LABELS = {
    "technology_transfer":               "Tech Transfer",
    "collaboration_conflict_moderation": "Collab/Conflict",
    "collaborative_leadership":          "Collab Leadership",
    "substitution":                      "Substitution",
    "networking":                        "Networking",
}

PAIR_SEP = "\u2013"


def _load_jsonl_safe(path) -> list[dict]:
    rows: list[dict] = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
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
    print(">>> ANALYSIS: TH Spaces (SetFit)")

    # Prefer SetFit > NLI (cooccurrence_nli.jsonl) > cooccurrence.jsonl (base step 3)
    setfit_path = config.STEP3_DIR / "setfit" / "cooccurrence_setfit.jsonl"
    nli_path = config.FILE_COOCCURRENCE_NLI
    use_setfit = setfit_path.exists()
    if use_setfit:
        source_path = setfit_path
        space_field = "th_space_setfit"
        print(f"[INFO] Using SetFit space predictions from {setfit_path.name}")
    elif nli_path.exists():
        source_path = nli_path
        space_field = "th_space"
        print(f"[INFO] Using NLI space predictions from {nli_path.name}")
    elif config.FILE_COOCCURRENCE.exists():
        source_path = config.FILE_COOCCURRENCE
        space_field = "th_space"
        print("[INFO] Using base co-occurrences from cooccurrence.jsonl (no space predictions — run SetFit for best results)")
    else:
        raise FileNotFoundError(f"Co-occurrence file not found: {config.FILE_COOCCURRENCE}")

    all_rows = _load_jsonl_safe(source_path)
    if use_setfit:
        rows = [r for r in all_rows if r.get(space_field) in SPACES]
    else:
        rows = [
            r for r in all_rows
            if r.get(space_field) in SPACES
            and float(r.get("th_space_confidence") or 0) >= 0.5
        ]

    # Cross-helix only — same-helix pairs are not meaningful for TH space analysis
    rows = [
        r for r in rows
        if normalize_helix(to_str(r.get("h1", ""))) != normalize_helix(to_str(r.get("h2", "")))
        and normalize_helix(to_str(r.get("h1", "")))
    ]
    print(f"[INFO] {len(rows)} cross-helix rows with valid space predictions")
    if not rows:
        raise RuntimeError("No rows with valid th_space predictions.")

    # ── Load paragraph counts for density normalisation ────────────────────
    paragraphs_by_country: dict[str, int] = defaultdict(int)
    if config.FILE_PARAGRAPHS.exists():
        for r in utils.load_jsonl(config.FILE_PARAGRAPHS):
            country = to_str(r.get("country", "")).strip().upper() or "UNK"
            paragraphs_by_country[country] += 1

    # ── Accumulators ───────────────────────────────────────────────────────
    space_by_country: dict[str, Counter] = defaultdict(Counter)
    relation_by_space: dict[str, Counter] = defaultdict(Counter)
    space_counts_by_pair: dict[str, Counter] = defaultdict(Counter)
    confidence_by_space: dict[str, list[float]] = defaultdict(list)
    countries: set[str] = set()

    for r in rows:
        country = to_str(r.get("country", "")).strip().upper() or "UNK"
        space = to_str(r.get(space_field, "")).strip()
        relation = to_str(r.get("relation_type", "no_explicit_relation")).strip() or "no_explicit_relation"
        conf = r.get("th_space_confidence")
        h1 = normalize_helix(to_str(r.get("h1", "")))
        h2 = normalize_helix(to_str(r.get("h2", "")))
        pair = PAIR_SEP.join(sorted([h1, h2])) if h1 and h2 else ""

        countries.add(country)
        space_by_country[country][space] += 1
        if relation in EXPLICIT_RELATION_TYPES:
            relation_by_space[space][relation] += 1
        if pair:
            space_counts_by_pair[pair][space] += 1
        if isinstance(conf, (int, float)):
            confidence_by_space[space].append(float(conf))

    sorted_countries = sorted(countries)
    plt, pd, sns = _load_plot_dependencies()
    import numpy as np

    # ── Figure 1: Country × space heatmap (normalised by paragraphs) ──────
    heat_data = pd.DataFrame(
        {SPACE_LABELS[s]: [
            space_by_country[c].get(s, 0) / max(paragraphs_by_country.get(c, 1), 1)
            for c in sorted_countries
        ] for s in SPACES},
        index=sorted_countries,
    )
    fig1, ax1 = plt.subplots(figsize=(7, max(5, len(sorted_countries) * 0.45)))
    raw_counts = pd.DataFrame(
        {SPACE_LABELS[s]: [space_by_country[c].get(s, 0) for c in sorted_countries] for s in SPACES},
        index=sorted_countries,
    )
    sns.heatmap(
        heat_data,
        ax=ax1,
        cmap="YlGnBu",
        annot=raw_counts.values,
        fmt="d",
        linewidths=0.5,
        cbar_kws={"label": "Count / paragraphs"},
    )
    ax1.set_title("TH Space Counts by Country (SetFit, normalised by paragraphs)")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    fig1.tight_layout()
    fig1.savefig(config.ANALYSIS_SPACES_COUNTRY_HEATMAP_PNG)
    plt.close(fig1)
    print(f"[OK] {config.ANALYSIS_SPACES_COUNTRY_HEATMAP_PNG}")

    # ── Figure 2: Global space counts bar chart ────────────────────────────
    global_counts = {s: sum(space_by_country[c].get(s, 0) for c in sorted_countries) for s in SPACES}
    sorted_spaces = sorted(SPACES, key=lambda s: global_counts[s], reverse=True)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bars = ax2.bar(
        [SPACE_LABELS[s] for s in sorted_spaces],
        [global_counts[s] for s in sorted_spaces],
        color=[SPACE_COLORS[s] for s in sorted_spaces],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, s in zip(bars, sorted_spaces):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 str(global_counts[s]), ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Co-occurrence count")
    ax2.set_title("Global TH Space Distribution (SetFit)")
    ax2.set_xticklabels([SPACE_LABELS[s] for s in sorted_spaces], rotation=20, ha="right")
    fig2.tight_layout()
    fig2.savefig(config.ANALYSIS_SPACES_GLOBAL_PNG)
    plt.close(fig2)
    print(f"[OK] {config.ANALYSIS_SPACES_GLOBAL_PNG}")

    # ── Figure 3: Space share per country (stacked bar) ───────────────────
    fig3, ax3 = plt.subplots(figsize=(13, 5))
    country_totals = {c: sum(space_by_country[c].values()) for c in sorted_countries}
    bottoms = np.zeros(len(sorted_countries))
    x = np.arange(len(sorted_countries))

    for space in SPACES:
        shares = np.array([
            space_by_country[c].get(space, 0) / max(country_totals[c], 1)
            for c in sorted_countries
        ])
        ax3.bar(x, shares, bottom=bottoms, color=SPACE_COLORS[space],
                label=SPACE_LABELS[space], width=0.7)
        bottoms += shares

    ax3.set_xticks(x)
    ax3.set_xticklabels(sorted_countries, rotation=45, ha="right")
    ax3.set_ylabel("Share of SetFit-predicted space")
    ax3.set_title("TH Space Share per Country (SetFit)")
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.set_ylim(0, 1.05)
    fig3.tight_layout()
    fig3.savefig(config.ANALYSIS_SPACES_STACKED_PNG)
    plt.close(fig3)
    print(f"[OK] {config.ANALYSIS_SPACES_STACKED_PNG}")

    # ── Figure 4: Relation type × space bubble matrix ─────────────────────
    active_spaces = [s for s in SPACES if sum(relation_by_space[s].values()) > 0]
    if active_spaces:
        max_cell = max(
            relation_by_space[sp].get(rel, 0)
            for sp in active_spaces
            for rel in EXPLICIT_RELATION_TYPES
        ) or 1

        fig4, axes = plt.subplots(1, 2, figsize=(12, 5),
                                  gridspec_kw={"width_ratios": [1, 0.03]})
        ax4, ax_cbar = axes

        xs, ys, sizes, shares_list, counts_list = [], [], [], [], []
        for x_idx, space in enumerate(active_spaces):
            sp_total = sum(relation_by_space[space].get(r, 0) for r in EXPLICIT_RELATION_TYPES) or 1
            for y_idx, rel in enumerate(EXPLICIT_RELATION_TYPES):
                count = relation_by_space[space].get(rel, 0)
                if count <= 0:
                    continue
                xs.append(x_idx)
                ys.append(y_idx)
                counts_list.append(count)
                sizes.append(120 + 1600 * (count / max_cell))
                shares_list.append(count / sp_total)

        scatter = ax4.scatter(
            xs, ys, s=sizes, c=shares_list,
            cmap="YlGnBu", vmin=0.0, vmax=1.0,
            alpha=0.95, edgecolors="white", linewidths=0.9, zorder=3,
        )
        for xi, yi, cnt in zip(xs, ys, counts_list):
            if cnt >= max(5, int(max_cell * 0.1)):
                ax4.text(xi, yi, str(cnt), ha="center", va="center", fontsize=8, color="#1a1a1a")

        for xline in range(len(active_spaces) + 1):
            ax4.axvline(xline - 0.5, color="#cccccc", linewidth=0.8, zorder=0)
        for yline in range(len(EXPLICIT_RELATION_TYPES) + 1):
            ax4.axhline(yline - 0.5, color="#f5f5f5", linewidth=0.8, zorder=0)

        ax4.set_xticks(range(len(active_spaces)))
        ax4.set_xticklabels([SPACE_LABELS[s] for s in active_spaces], rotation=30, ha="right")
        ax4.set_yticks(range(len(EXPLICIT_RELATION_TYPES)))
        ax4.set_yticklabels([RELATION_LABELS[r] for r in EXPLICIT_RELATION_TYPES])
        ax4.set_title("Relation Type Profile per TH Space (SetFit)")
        ax4.set_ylabel("Relation type")

        fig4.colorbar(scatter, cax=ax_cbar).set_label("Share within space")
        fig4.tight_layout()
        fig4.savefig(config.ANALYSIS_SPACES_RELATION_MATRIX_PNG)
        plt.close(fig4)
        print(f"[OK] {config.ANALYSIS_SPACES_RELATION_MATRIX_PNG}")

    # ── Figure 5: Space × helix pair bubble matrix (with bottom bar) ─────────
    try:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        pair_totals_by_space = {
            pair: sum(space_counts_by_pair[pair].values())
            for pair in space_counts_by_pair
        }
        pair_order = [
            pair for pair in sorted(space_counts_by_pair.keys(),
                                    key=lambda p: pair_totals_by_space[p], reverse=True)
            if pair_totals_by_space[pair] > 0
        ]

        if pair_order:
            max_cell = max(
                space_counts_by_pair[pair].get(space, 0)
                for pair in pair_order
                for space in SPACES
            ) or 1

            # ── Figure 5a: with bottom bar ──────────────────────────────────
            fig5a = plt.figure(figsize=(max(16, len(pair_order) * 1.1), 12))
            gs5a = fig5a.add_gridspec(2, 2, height_ratios=[3.0, 1.4],
                                      width_ratios=[1, 0.03], hspace=0.08, wspace=0.05)
            ax5a = fig5a.add_subplot(gs5a[0, 0])
            ax5a_cbar = fig5a.add_subplot(gs5a[0, 1])
            ax5a_bot = fig5a.add_subplot(gs5a[1, 0], sharex=ax5a)
            fig5a.add_subplot(gs5a[1, 1]).set_visible(False)

            xs, ys, sizes, shares_list, counts_list = [], [], [], [], []
            for x_idx, pair in enumerate(pair_order):
                pair_total = pair_totals_by_space[pair] or 1
                for y_idx, space in enumerate(SPACES):
                    count = space_counts_by_pair[pair].get(space, 0)
                    if count <= 0:
                        continue
                    xs.append(x_idx)
                    ys.append(y_idx)
                    counts_list.append(count)
                    sizes.append(120 + 1600 * (count / max_cell))
                    shares_list.append(count / pair_total)

            scatter5a = ax5a.scatter(
                xs, ys, s=sizes, c=shares_list, cmap="YlGnBu",
                vmin=0.0, vmax=1.0, alpha=0.95, edgecolors="white", linewidths=0.9, zorder=3,
            )
            for xi, yi, cnt in zip(xs, ys, counts_list):
                if cnt >= max(20, int(max_cell * 0.2)):
                    ax5a.text(xi, yi, str(cnt), ha="center", va="center", fontsize=8, color="#1a1a1a")

            for xline in range(len(pair_order) + 1):
                ax5a.axvline(xline - 0.5, color="#cccccc", linewidth=0.8, zorder=0)
                ax5a_bot.axvline(xline - 0.5, color="#cccccc", linewidth=0.8, zorder=0)
            for yline in range(len(SPACES) + 1):
                ax5a.axhline(yline - 0.5, color="#f5f5f5", linewidth=0.8, zorder=0)

            ax5a.set_yticks(range(len(SPACES)))
            ax5a.set_yticklabels([SPACE_LABELS[s] for s in SPACES], fontsize=9)
            ax5a.set_xticks(range(len(pair_order)))
            ax5a.set_xticklabels([])
            ax5a.tick_params(axis="x", bottom=False, labelbottom=False)
            ax5a.set_ylabel("TH Space")
            ax5a.set_title("TH Space Profile per Helix Pair (SetFit)", fontsize=11, pad=8)

            fig5a.colorbar(scatter5a, cax=ax5a_cbar).set_label("Share within helix pair")

            ref_counts = [int(max_cell * 0.1), int(max_cell * 0.5), max_cell]
            size_handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                       markeredgecolor="#333333", markeredgewidth=0.5,
                       markersize=(120 + 1600 * (c / max_cell)) ** 0.5 * 0.52,
                       label=f"n = {c}")
                for c in ref_counts
            ]
            ax5a.legend(handles=size_handles, title="Bubble size (raw count)",
                        title_fontsize=8, fontsize=8, loc="upper right",
                        framealpha=0.9, edgecolor="#cccccc")

            pair_totals_bar = [pair_totals_by_space[p] for p in pair_order]
            bar_colors = [SPACE_COLORS[max(SPACES, key=lambda s: space_counts_by_pair[p].get(s, 0))]
                          for p in pair_order]
            ax5a_bot.bar(range(len(pair_order)), pair_totals_bar, color=bar_colors, alpha=0.9, width=0.6)
            ax5a_bot.set_xticks(range(len(pair_order)))
            ax5a_bot.set_xticklabels(
                [p.replace("_", " ") for p in pair_order], rotation=45, ha="right", fontsize=8,
            )
            ax5a_bot.set_ylabel("Co-occurrence count", labelpad=8)
            ax5a_bot.set_title("Total SetFit-assigned space co-occurrences per helix pair", fontsize=10, pad=6)
            ax5a_bot.grid(axis="y", linestyle="--", alpha=0.25)
            legend_elems = [Patch(facecolor=SPACE_COLORS[s], label=SPACE_LABELS[s]) for s in SPACES]
            ax5a_bot.legend(handles=legend_elems, loc="upper right", ncol=2, fontsize=8, title="TH Space")

            xlim = (-0.5, len(pair_order) - 0.5)
            ax5a.set_xlim(xlim)
            ax5a_bot.set_xlim(xlim)
            fig5a.tight_layout(pad=2.0)
            fig5a.savefig(config.ANALYSIS_SPACES_HELIX_BUBBLE_PATH, dpi=300, bbox_inches="tight")
            plt.close(fig5a)
            print(f"[OK] {config.ANALYSIS_SPACES_HELIX_BUBBLE_PATH}")

            # ── Figure 5b: bubble only, no bottom bar ───────────────────────
            fig5b, (ax5b, ax5b_cbar_ax) = plt.subplots(
                1, 2,
                figsize=(max(14, len(pair_order) * 1.0), 5.5),
                gridspec_kw={"width_ratios": [1, 0.025], "wspace": 0.04},
            )
            xsb, ysb, sizesb, sharesb, countsb = [], [], [], [], []
            for x_idx, pair in enumerate(pair_order):
                pair_total = pair_totals_by_space[pair] or 1
                for y_idx, space in enumerate(SPACES):
                    count = space_counts_by_pair[pair].get(space, 0)
                    if count <= 0:
                        continue
                    xsb.append(x_idx)
                    ysb.append(y_idx)
                    countsb.append(count)
                    sizesb.append(120 + 1600 * (count / max_cell))
                    sharesb.append(count / pair_total)

            scatter5b = ax5b.scatter(
                xsb, ysb, s=sizesb, c=sharesb, cmap="YlGnBu",
                vmin=0.0, vmax=1.0, alpha=0.95, edgecolors="white", linewidths=0.9, zorder=3,
            )
            for xi, yi, cnt in zip(xsb, ysb, countsb):
                if cnt >= max(20, int(max_cell * 0.2)):
                    ax5b.text(xi, yi, str(cnt), ha="center", va="center", fontsize=8, color="#1a1a1a")

            for xline in range(len(pair_order) + 1):
                ax5b.axvline(xline - 0.5, color="#cccccc", linewidth=0.8, zorder=0)
            for yline in range(len(SPACES) + 1):
                ax5b.axhline(yline - 0.5, color="#f5f5f5", linewidth=0.8, zorder=0)

            ax5b.set_yticks(range(len(SPACES)))
            ax5b.set_yticklabels([SPACE_LABELS[s] for s in SPACES])
            ax5b.set_xticks(range(len(pair_order)))
            ax5b.set_xticklabels([p.replace("_", " ") for p in pair_order], rotation=45, ha="right")
            ax5b.set_ylabel("TH Space")
            ax5b.set_title("TH Space Profile per Helix Pair (SetFit)", pad=10)
            ax5b.set_xlim(-0.5, len(pair_order) - 0.5)

            fig5b.colorbar(scatter5b, cax=ax5b_cbar_ax).set_label("Share within helix pair")

            ref_countsb = [int(max_cell * 0.1), int(max_cell * 0.5), max_cell]
            size_handlesb = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                       markeredgecolor="#333333", markeredgewidth=0.5,
                       markersize=(120 + 1600 * (c / max_cell)) ** 0.5 * 0.52,
                       label=f"n = {c}")
                for c in ref_countsb
            ]
            ax5b.legend(handles=size_handlesb, title="Bubble size (raw count)",
                        title_fontsize=8, fontsize=8, loc="upper right",
                        framealpha=0.9, edgecolor="#cccccc")

            fig5b.tight_layout(pad=2.0)
            fig5b.savefig(config.ANALYSIS_SPACES_HELIX_BUBBLE_ONLY_PATH, dpi=300, bbox_inches="tight")
            plt.close(fig5b)
            print(f"[OK] {config.ANALYSIS_SPACES_HELIX_BUBBLE_ONLY_PATH}")

    except Exception as exc:
        print(f"[WARN] Spaces bubble figures skipped: {exc}")

    # ── Figure 7: Confidence boxplot per space (NLI only) ─────────────────
    if any(confidence_by_space.values()):
        fig5, ax5 = plt.subplots(figsize=(8, 4))
        conf_data = [confidence_by_space.get(s, []) for s in SPACES]
        bp = ax5.boxplot(conf_data, patch_artist=True, medianprops={"color": "black", "lw": 2})
        for patch, space in zip(bp["boxes"], SPACES):
            patch.set_facecolor(SPACE_COLORS[space])
        ax5.set_xticks(range(1, len(SPACES) + 1))
        ax5.set_xticklabels([SPACE_LABELS[s] for s in SPACES], rotation=20, ha="right")
        ax5.set_ylabel("NLI Entailment Confidence")
        ax5.set_title("NLI Confidence Distribution per TH Space")
        ax5.set_ylim(0, 1.05)
        fig5.tight_layout()
        fig5.savefig(config.ANALYSIS_SPACES_CONFIDENCE_PNG)
        plt.close(fig5)
        print(f"[OK] {config.ANALYSIS_SPACES_CONFIDENCE_PNG}")
    else:
        print("[SKIP] Confidence boxplot skipped (SetFit has no per-prediction confidence scores)")

    # ── CSV: country × space density table ────────────────────────────────
    with config.ANALYSIS_SPACES_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["country", "paragraphs"] + [SPACE_LABELS[s] for s in SPACES]
                        + [f"{SPACE_LABELS[s]}_density" for s in SPACES] + ["total"])
        for c in sorted_countries:
            n_para = paragraphs_by_country.get(c, 0) or 1
            counts = [space_by_country[c].get(s, 0) for s in SPACES]
            densities = [round(cnt / n_para, 4) for cnt in counts]
            writer.writerow([c, paragraphs_by_country.get(c, 0)] + counts + densities + [sum(counts)])
    print(f"[OK] {config.ANALYSIS_SPACES_CSV}")

    # ── JSON summary ──────────────────────────────────────────────────────
    summary = {
        "generated_at": utc_now_iso(),
        "total_rows": len(rows),
        "global_space_counts": {SPACE_LABELS[s]: global_counts[s] for s in SPACES},
        "confidence_stats": {
            SPACE_LABELS[s]: {
                "mean": round(statistics.mean(confidence_by_space[s]), 4) if confidence_by_space[s] else None,
                "median": round(statistics.median(confidence_by_space[s]), 4) if confidence_by_space[s] else None,
                "n": len(confidence_by_space[s]),
            }
            for s in SPACES
        },
        "by_country": {
            c: {SPACE_LABELS[s]: space_by_country[c].get(s, 0) for s in SPACES}
            for c in sorted_countries
        },
    }
    write_json(config.ANALYSIS_SPACES_JSON, summary)
    print(f"[OK] {config.ANALYSIS_SPACES_JSON}")
    print(">>> ANALYSIS: TH Spaces done.")
