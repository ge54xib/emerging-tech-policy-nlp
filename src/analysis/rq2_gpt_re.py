"""RQ2-GPT-RE: relation type analysis using GPT-RE predictions on the full corpus.

Mirrors rq2.py's relation figures but reads from cooccurrence_gpt_re.jsonl
(produced by src.pipeline.gpt_re) using the relation_type_gpt_re field.

Outputs (in outputs/rq2_gpt_re/):
- rq2_gptrei_relation_by_helix_pair.png   relation × helix pair bubble matrix
- rq2_gptrei_relation_distribution.png    global relation type bar chart
- rq2_gptrei_relation_table.csv           country × pair × relation counts
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict

from src import config
from src.analysis._helpers import thesis_style, write_json, utc_now_iso
from src.analysis.rq2 import (
    HELIXES,
    PAIR_SEP,
    EXPLICIT_RELATION_TYPES,
    RELATION_COLORS,
    TH_SPACE_COLORS,
    _normalize_pair,
    _pair_from_row,
    _th_space,
    _all_pairs,
    _load_jsonl_safe,
    _load_plot_dependencies,
)
from src.utils import normalize_helix, to_str

SOURCE_FILE    = config.STEP3_DIR / "gpt_re" / "cooccurrence_gpt_re.jsonl"
RELATION_FIELD = "relation_type_gpt_re"
OUT_DIR        = config.ANALYSIS_RQ2_GPTREI_DIR


def run() -> None:
    print(">>> ANALYSIS: RQ2 GPT-RE")

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"{SOURCE_FILE} not found. Run: python -m src.pipeline.gpt_re"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = _load_jsonl_safe(SOURCE_FILE)
    print(f"[INFO] Loaded {len(all_rows)} rows from {SOURCE_FILE.name}")

    # Cross-helix pairs only
    rows = [
        r for r in all_rows
        if normalize_helix(to_str(r.get("h1", ""))) != normalize_helix(to_str(r.get("h2", "")))
        and normalize_helix(to_str(r.get("h1", "")))
    ]
    print(f"[INFO] {len(rows)} cross-helix pairs")

    # ── Accumulators ──────────────────────────────────────────────────────────
    relation_counts_global: Counter = Counter()
    relation_counts_by_pair: dict[str, Counter] = defaultdict(Counter)
    relation_counts_by_country: dict[str, Counter] = defaultdict(Counter)
    pair_counts_global: Counter = Counter()

    for row in rows:
        country = to_str(row.get("country", "")).strip().upper() or "UNK"
        pair    = _pair_from_row(row)
        if not pair or PAIR_SEP not in pair:
            continue
        h1, h2 = pair.split(PAIR_SEP, 1)
        if h1 not in HELIXES or h2 not in HELIXES:
            continue

        rel = to_str(row.get(RELATION_FIELD, "no_explicit_relation")).strip() or "no_explicit_relation"
        pair_counts_global[pair] += 1
        if rel != "no_explicit_relation":
            relation_counts_global[rel] += 1
            relation_counts_by_pair[pair][rel] += 1
            relation_counts_by_country[country][rel] += 1

    if not relation_counts_global:
        print("[WARN] No explicit relations found in GPT-RE output.")
        return

    plt, pd, sns = _load_plot_dependencies()

    # ── Figure 1: Relation × helix pair bubble matrix ─────────────────────────
    try:
        all_pairs = _all_pairs()
        explicit_totals = {
            pair: sum(relation_counts_by_pair[pair].get(r, 0) for r in EXPLICIT_RELATION_TYPES)
            for pair in all_pairs
        }
        pair_order = [
            p for p in sorted(all_pairs, key=lambda p: explicit_totals[p], reverse=True)
            if explicit_totals[p] > 0
        ]

        if pair_order:
            max_cell = max(
                relation_counts_by_pair[pair].get(rel, 0)
                for pair in pair_order
                for rel in EXPLICIT_RELATION_TYPES
            ) or 1

            fig = plt.figure(figsize=(max(16, len(pair_order) * 1.1), 12))
            gs  = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.4],
                                   width_ratios=[1, 0.03], hspace=0.08, wspace=0.05)
            ax      = fig.add_subplot(gs[0, 0])
            ax_cbar = fig.add_subplot(gs[0, 1])
            ax_bot  = fig.add_subplot(gs[1, 0], sharex=ax)
            fig.add_subplot(gs[1, 1]).set_visible(False)

            xs, ys, sizes, shares, counts = [], [], [], [], []
            for x_idx, pair in enumerate(pair_order):
                pair_total = explicit_totals[pair] or 1
                for y_idx, rel in enumerate(EXPLICIT_RELATION_TYPES):
                    count = relation_counts_by_pair[pair].get(rel, 0)
                    if count <= 0:
                        continue
                    xs.append(x_idx); ys.append(y_idx); counts.append(count)
                    sizes.append(120 + 1600 * (count / max_cell))
                    shares.append(count / pair_total)

            scatter = ax.scatter(xs, ys, s=sizes, c=shares, cmap="YlGnBu",
                                 vmin=0.0, vmax=1.0, alpha=0.95,
                                 edgecolors="white", linewidths=0.9, zorder=3)
            for xi, yi, cnt in zip(xs, ys, counts):
                if cnt >= max(10, int(max_cell * 0.15)):
                    ax.text(xi, yi, str(cnt), ha="center", va="center",
                            fontsize=8, color="#1a1a1a")

            for xline in range(len(pair_order) + 1):
                ax.axvline(xline - 0.5, color="#cccccc", lw=0.8, zorder=0)
                ax_bot.axvline(xline - 0.5, color="#cccccc", lw=0.8, zorder=0)
            for yline in range(len(EXPLICIT_RELATION_TYPES) + 1):
                ax.axhline(yline - 0.5, color="#f5f5f5", lw=0.8, zorder=0)

            ax.set_yticks(range(len(EXPLICIT_RELATION_TYPES)))
            ax.set_yticklabels([r.replace("_", " ") for r in EXPLICIT_RELATION_TYPES], fontsize=9)
            ax.set_xticks(range(len(pair_order)))
            ax.set_xticklabels([])
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.set_ylabel("Relation type")
            ax.set_title("Relation Type Profile per Helix Pair (GPT-RE)", fontsize=11, pad=8)

            fig.colorbar(scatter, cax=ax_cbar).set_label("Share within helix pair")

            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            ref_counts = [int(max_cell * 0.1), int(max_cell * 0.5), max_cell]
            size_handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                       markeredgecolor="#333333", markeredgewidth=0.5,
                       markersize=(120 + 1600 * (c / max_cell)) ** 0.5 * 0.52,
                       label=f"n = {c}")
                for c in ref_counts
            ]
            ax.legend(handles=size_handles, title="Bubble size", title_fontsize=8,
                      fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#cccccc")

            pair_totals_bar = [explicit_totals[p] for p in pair_order]
            pair_colors     = [TH_SPACE_COLORS[_th_space(p)] for p in pair_order]
            ax_bot.bar(range(len(pair_order)), pair_totals_bar, color=pair_colors, alpha=0.9, width=0.6)
            ax_bot.set_xticks(range(len(pair_order)))
            ax_bot.set_xticklabels([p.replace("_", " ") for p in pair_order],
                                   rotation=45, ha="right", fontsize=8)
            ax_bot.set_ylabel("Number of explicit relations", labelpad=8)
            ax_bot.grid(axis="y", linestyle="--", alpha=0.25)
            legend_elements = [Patch(facecolor=col, label=space) for space, col in TH_SPACE_COLORS.items()]
            ax_bot.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize=8,
                          title="Innovation Space")

            xlim = (-0.5, len(pair_order) - 0.5)
            ax.set_xlim(xlim); ax_bot.set_xlim(xlim)
            fig.tight_layout(pad=2.0)
            out_path = OUT_DIR / "rq2_gptrei_relation_by_helix_pair.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] {out_path}")
    except Exception as exc:
        print(f"[WARN] Bubble matrix skipped: {exc}")

    # ── Figure 2: Global relation type bar chart ──────────────────────────────
    try:
        sorted_rels = sorted(EXPLICIT_RELATION_TYPES, key=lambda r: relation_counts_global[r], reverse=True)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        bars = ax2.bar(
            [r.replace("_", " ") for r in sorted_rels],
            [relation_counts_global[r] for r in sorted_rels],
            color=[RELATION_COLORS[r] for r in sorted_rels],
            edgecolor="white", linewidth=0.8,
        )
        for bar, r in zip(bars, sorted_rels):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(relation_counts_global[r]), ha="center", va="bottom", fontsize=9)
        ax2.set_ylabel("Co-occurrence count")
        ax2.set_title("Global Relation Type Distribution (GPT-RE)")
        ax2.set_xticklabels([r.replace("_", " ") for r in sorted_rels], rotation=20, ha="right")
        fig2.tight_layout()
        out_path2 = OUT_DIR / "rq2_gptrei_relation_distribution.png"
        fig2.savefig(out_path2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"[OK] {out_path2}")
    except Exception as exc:
        print(f"[WARN] Distribution bar chart skipped: {exc}")

    # ── CSV: country × relation type counts ───────────────────────────────────
    csv_path = OUT_DIR / "rq2_gptrei_relation_table.csv"
    countries = sorted(relation_counts_by_country.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["country"] + EXPLICIT_RELATION_TYPES + ["total_explicit"])
        for c in countries:
            counts_row = [relation_counts_by_country[c].get(r, 0) for r in EXPLICIT_RELATION_TYPES]
            writer.writerow([c] + counts_row + [sum(counts_row)])
    print(f"[OK] {csv_path}")

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary = {
        "generated_at": utc_now_iso(),
        "source": str(SOURCE_FILE),
        "total_cross_helix_pairs": len(rows),
        "global_relation_counts": dict(relation_counts_global),
    }
    write_json(OUT_DIR / "rq2_gptrei_summary.json", summary)
    print(">>> ANALYSIS: RQ2 GPT-RE done.")


if __name__ == "__main__":
    run()
