"""Extended thesis analysis: additional figures for RQ1, RQ2, RQ3, and synthesis.

All outputs are NEW files — no existing outputs are modified.
Reads from pre-existing CSV tables produced by rq1/rq2/rq3/descriptives.run().
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from src import config

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Shared visual identity system
# ---------------------------------------------------------------------------
HELIX_COLORS: dict[str, str] = {
    "government":    "#5B7FA6",
    "academia":      "#6B9E78",
    "industry":      "#C47F3C",
    "civil_society": "#B06070",
    "intermediary":  "#8B7BA8",
}
TH_SPACE_COLORS: dict[str, str] = {
    "Knowledge Space":  "#5B7FA6",
    "Innovation Space": "#C47F3C",
    "Consensus Space":  "#9B7D5B",
    "Civil Society":    "#B06070",
    "Internal":         "#A8A8A8",
}
QH_CONFIG_COLORS: dict[str, str] = {
    "Balanced":      "#5B7FA6",
    "Mixed":         "#8B8B8B",
    "Laissez-faire": "#6B9E78",
    "Statist":       "#C4574A",
}
QH_CONFIG_ORDER: list[str] = ["Balanced", "Mixed", "Laissez-faire", "Statist"]
HELIX_ORDER: list[str] = [
    "government", "academia", "industry", "civil_society", "intermediary",
]
PAIR_SEP = "\u2013"  # en-dash used in all pair labels

# Global actor counts from descriptives_actor_summary (used for node sizing)
_HELIX_ACTOR_COUNTS: dict[str, int] = {
    "government":    1037,
    "academia":       557,
    "intermediary":   361,
    "industry":       248,
    "civil_society":   62,
}

# ---------------------------------------------------------------------------
# rcParams base style
# ---------------------------------------------------------------------------
_BASE_RC: dict = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAF8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_cross_helix(pair: str) -> bool:
    parts = pair.split(PAIR_SEP)
    return len(parts) == 2 and parts[0].strip() != parts[1].strip()


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _rq1() -> pd.DataFrame:
    return pd.read_csv(config.ANALYSIS_RQ1_CSV_PATH)


def _rq3() -> pd.DataFrame:
    df = pd.read_csv(config.ANALYSIS_RQ3_CSV_PATH)
    df["Civil_Society_Present"] = (
        df["Civil_Society_Present"].astype(str).str.lower() == "true"
    )
    return df


def _corpus() -> pd.DataFrame:
    return pd.read_csv(config.ANALYSIS_DESCRIPTIVES_CORPUS_CSV)



def _hex_alpha(hex_color: str, pct: int) -> str:
    """Return hex_color with appended 2-digit hex alpha (0–255 → pct 0–100)."""
    a = round(pct / 100 * 255)
    return f"{hex_color}{a:02X}"


# ---------------------------------------------------------------------------
# Figure 8 — rq3_country_trajectories.png  (RQ3, COUNTRY COMPARISON)
# ---------------------------------------------------------------------------
def _fig8_country_trajectories() -> None:
    """Arrow trajectories for multi-document countries on Gov_Share × HBI plane."""
    with plt.rc_context(_BASE_RC):
        corpus = _corpus()
        rq3 = _rq3()[["Country", "QH_Configuration"]]
        final_cfg = rq3.set_index("Country")["QH_Configuration"].to_dict()

        # Compute gov share per document
        corpus["total"] = corpus[["Gov", "Ind", "Acad", "CS", "Int"]].sum(axis=1)
        corpus = corpus[corpus["total"] > 0].copy()
        corpus["GovShr"] = corpus["Gov"] / corpus["total"]

        # Merge DNK (two 2023 parts) into one averaged point
        dk_sub = corpus[corpus["Country"] == "DNK"]
        dk_row = pd.DataFrame([{
            "Country": "DNK",
            "GovShr": float(dk_sub["GovShr"].mean()),
            "HBI": float(dk_sub["HBI"].mean()),
            "Year": int(dk_sub["Year"].iloc[0]),
        }])
        corpus = pd.concat(
            [corpus[corpus["Country"] != "DNK"][["Country", "GovShr", "HBI", "Year"]],
             dk_row],
            ignore_index=True,
        )
        corpus = corpus.sort_values(["Country", "Year"])

        counts = corpus.groupby("Country").size()
        multi_doc = set(counts[counts > 1].index)
        single_doc = set(counts[counts == 1].index)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.spines["left"].set_color("#DDDDDD")
        ax.spines["bottom"].set_color("#DDDDDD")

        # Background quadrant shading
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.60, -0.05), 0.50, 1.15, boxstyle="square,pad=0",
            color="#C4574A", alpha=0.04, zorder=0,
        ))
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.05, 0.80), 0.65, 0.30, boxstyle="square,pad=0",
            color="#5B7FA6", alpha=0.04, zorder=0,
        ))

        # Reference lines
        ax.axvline(0.60, color="#BBBBBB", lw=0.8, ls="--", zorder=1)
        ax.axhline(0.80, color="#BBBBBB", lw=0.8, ls="--", zorder=1)

        # Quadrant corner labels
        ax.text(0.01, 0.01, "Distributed / Low Balance",
                transform=ax.transAxes, fontsize=7, color="#BBBBBB", style="italic")
        ax.text(0.01, 0.96, "Balanced",
                transform=ax.transAxes, fontsize=7, color="#5B7FA6", style="italic",
                va="top")
        ax.text(0.80, 0.01, "Statist",
                transform=ax.transAxes, fontsize=7, color="#C4574A", style="italic")
        ax.text(0.80, 0.96, "Complex / Mixed",
                transform=ax.transAxes, fontsize=7, color="#888888", style="italic",
                va="top")

        # Single-doc countries (background dots)
        for country in single_doc:
            row = corpus[corpus["Country"] == country].iloc[0]
            cfg = final_cfg.get(country, "Mixed")
            col = QH_CONFIG_COLORS.get(cfg, "#888888")
            ax.scatter(row["GovShr"], row["HBI"], s=28, color=col,
                       alpha=0.30, zorder=2, edgecolors="white", lw=0.5)
            ax.text(
                row["GovShr"] + 0.007, row["HBI"] + 0.003,
                country, fontsize=6, color=col, alpha=0.50, zorder=2,
            )

        # Multi-doc trajectories
        for country in sorted(multi_doc):
            sub = corpus[corpus["Country"] == country].sort_values("Year")
            xs = sub["GovShr"].values
            ys = sub["HBI"].values
            years_list = sub["Year"].values
            cfg = final_cfg.get(country, "Mixed")
            col = QH_CONFIG_COLORS.get(cfg, "#888888")

            # Line
            ax.plot(xs, ys, color=col, lw=1.8, alpha=0.85, zorder=5, solid_capstyle="round")

            # Arrows between each step
            for k in range(len(xs) - 1):
                ax.annotate(
                    "",
                    xy=(xs[k + 1], ys[k + 1]),
                    xytext=(xs[k], ys[k]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=col,
                        lw=1.4,
                        mutation_scale=11,
                        alpha=0.85,
                    ),
                    zorder=6,
                )

            # Points with year labels
            for k, (x, y, yr) in enumerate(zip(xs, ys, years_list)):
                marker = "o" if k == 0 else ("s" if k == len(xs) - 1 else "^")
                ax.scatter(x, y, s=55, marker=marker, color=col,
                           zorder=7, edgecolors="white", lw=0.8)
                ax.text(
                    x + 0.012, y + 0.006,
                    f"{country} '{str(yr)[2:]}",
                    fontsize=7.5, color=col, fontweight="bold", zorder=8,
                )

        ax.set_xlabel("Government share of actor mentions (0 → 1)", labelpad=8)
        ax.set_ylabel("Helix Balance Index (0 → 1)", labelpad=8)
        ax.set_xlim(-0.03, 1.08)
        ax.set_ylim(-0.04, 1.06)

        # Legend
        config_handles = [
            mpatches.Patch(color=QH_CONFIG_COLORS[c], label=c, alpha=0.82)
            for c in QH_CONFIG_ORDER
        ]
        marker_handles = [
            ax.scatter([], [], s=45, marker="o", color="#888888", label="First document"),
            ax.scatter([], [], s=45, marker="^", color="#888888", label="Intermediate"),
            ax.scatter([], [], s=45, marker="s", color="#888888", label="Latest document"),
        ]
        l1 = ax.legend(
            handles=config_handles, title="QH Configuration",
            loc="lower right", framealpha=0.93, edgecolor="#DDDDDD", fontsize=8,
        )
        l2 = ax.legend(
            handles=marker_handles, title="Trajectory markers",
            loc="upper right", framealpha=0.93, edgecolor="#DDDDDD", fontsize=8,
        )
        ax.add_artist(l1)

        ax.set_title(
            "Strategy Trajectories of Multi-Document Countries\n"
            "Arrows show direction of change across consecutive strategies  "
            "(single-doc countries shown as faint background dots)",
            fontsize=9.5, fontweight="bold", pad=10,
        )
        fig.tight_layout()
        _save(fig, config.ANALYSIS_RQ3_COUNTRY_TRAJECTORIES_PATH)


# ---------------------------------------------------------------------------
# Figure 12 — rq3_relation_temporal.png  (RQ3, TEMPORAL)
# ---------------------------------------------------------------------------
def _fig12_relation_temporal() -> None:
    """Stacked area: how explicit relation type emphasis shifts across strategy years."""
    with plt.rc_context(_BASE_RC):
        net = _rq2_net()
        explicit = net[net["relation_type"] != "no_explicit_relation"].copy()

        if explicit.empty:
            return  # nothing to plot

        # Count by year × relation_type
        year_rel = (
            explicit.groupby(["year", "relation_type"])
            .size()
            .reset_index(name="Count")
        )

        # Only active relation types
        active_rels = (
            year_rel.groupby("relation_type")["Count"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

        # Pivot and normalise
        pivot = year_rel.pivot_table(
            index="year", columns="relation_type",
            values="Count", aggfunc="sum", fill_value=0,
        )
        for rel in active_rels:
            if rel not in pivot.columns:
                pivot[rel] = 0
        pivot = pivot[active_rels]

        totals = pivot.sum(axis=1).clip(lower=1)
        pct = (pivot.div(totals, axis=0) * 100)

        years = pct.index.tolist()

        # N_docs per year (from rq3 table)
        rq3 = _rq3()
        n_docs = rq3.groupby("Year")["N_Documents"].first().to_dict()

        # Total explicit per year (for annotation)
        year_totals = year_rel.groupby("year")["Count"].sum().to_dict()

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.spines["left"].set_color("#DDDDDD")
        ax.spines["bottom"].set_color("#DDDDDD")

        # Stack from bottom: reversed order so dominant type is at bottom
        stack_order = list(reversed(active_rels))
        cumulative = np.zeros(len(years))

        for rel in stack_order:
            vals = pct[rel].values
            color = RELATION_COLORS.get(rel, "#AAAAAA")
            label = RELATION_LABELS.get(rel, rel)
            ax.fill_between(
                years, cumulative, cumulative + vals,
                alpha=0.82, color=color, label=label, zorder=2,
            )
            # Label at rightmost visible year if large enough
            y_mid = cumulative[-1] + vals[-1] / 2
            if vals[-1] > 8:
                ax.text(
                    years[-1] + 0.1, y_mid,
                    RELATION_LABELS.get(rel, rel).split()[0],
                    va="center", ha="left", fontsize=7, color=color,
                    fontweight="bold",
                )
            cumulative = cumulative + vals

        # X-axis: year with doc count and explicit count
        xtick_labels = []
        for y in years:
            nd = n_docs.get(y, "?")
            ne = year_totals.get(y, 0)
            xtick_labels.append(f"{y}\n(docs={nd}, n={ne})")

        ax.set_xticks(years)
        ax.set_xticklabels(xtick_labels, fontsize=7.5)
        ax.set_ylabel("Share of detected explicit relations (%)", labelpad=8)
        ax.set_xlim(years[0] - 0.4, years[-1] + 2.2)
        ax.set_ylim(0, 108)

        # Annotate years with NO explicit relations detected
        rq3_years = sorted(rq3["Year"].unique())
        silent_years = [y for y in rq3_years if y not in years]
        for y in silent_years:
            ax.axvline(y, color="#EEEEEE", lw=0.6, ls=":", zorder=1)

        ax.legend(
            title="Explicit Relation Type",
            loc="upper left", framealpha=0.93,
            edgecolor="#DDDDDD", fontsize=8,
            bbox_to_anchor=(0.01, 0.99),
        )
        ax.set_title(
            "Temporal Shift in Explicit Relation Type Emphasis\n"
            "Only years with ≥1 detected explicit relation shown  ·  "
            "no_explicit_relation excluded",
            fontsize=9.5, fontweight="bold", pad=10,
        )
        fig.tight_layout()
        _save(fig, config.ANALYSIS_RQ3_RELATION_TEMPORAL_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run() -> None:
    """Generate all extended figures and companion CSVs."""
    _fig8_country_trajectories()


if __name__ == "__main__":
    run()
