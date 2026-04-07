"""RQ3 deliverable: Quadruple Helix country profiles synthesis.

Loads RQ1 JSON output and classifies each country into a QH configuration
(Statist / Balanced / Laissez-faire / Mixed) following Ranga & Etzkowitz (2013).

Outputs:
- rq3_profiles.json         -- classification data
- rq3_country_profiles.png  -- radar grid (one subplot per country)
- rq3_profiles_summary.png  -- scatter (Gov% vs HBI) + summary table
- rq3_table.csv             -- machine-readable profile table
"""

from __future__ import annotations

import csv
import json

from src import config
from src.analysis._helpers import utc_now_iso, write_json, thesis_style


HELIXES = ["government", "industry", "academia", "civil_society", "intermediary"]
HELIX_COLORS = {
    "government":    "#2166ac",
    "industry":      "#d6604d",
    "academia":      "#4dac26",
    "civil_society": "#8073ac",
    "intermediary":  "#f4a582",
}

QH_CONFIG_COLORS = {
    "Statist":       "#d6604d",
    "Balanced":      "#4393c3",
    "Laissez-faire": "#74c476",
    "Mixed":         "#bdbdbd",
}

PAIR_SEP = "\u2013"


def _classify(gov: float, industry: float, academia: float, civil_society: float, hbi: float) -> str:
    """Classify into Ranga & Etzkowitz (2013) Triple Helix configurations (MECE).

    Balanced:      HBI >= 0.80  — all spheres roughly equal
    Statist:       HBI < 0.80 AND gov > 0.50 (government has clear majority)
    Laissez-faire: HBI < 0.80 AND gov <= 0.50 AND industry leads among non-gov spheres
    Mixed:         HBI < 0.80 AND gov <= 0.50 AND academia or civil society leads non-gov
                   (no named configuration in the paper)
    """
    if hbi >= 0.80:
        return "Balanced"
    if gov > 0.50:
        return "Statist"
    non_gov_dominant = max(
        [("industry", industry), ("academia", academia), ("civil_society", civil_society)],
        key=lambda x: x[1],
    )[0]
    if non_gov_dominant == "industry":
        return "Laissez-faire"
    return "Mixed"


def _load_rq1() -> list[dict]:
    if not config.ANALYSIS_RQ1_PATH.exists():
        raise FileNotFoundError(
            f"RQ1 JSON not found: {config.ANALYSIS_RQ1_PATH}. Run rq1.run() first."
        )
    payload = json.loads(config.ANALYSIS_RQ1_PATH.read_text(encoding="utf-8"))
    return payload["rows"]


def _load_rq2_dominant_pair() -> dict[str, str]:
    return {}


def _th_space(pair: str) -> str:
    space_map = {
        f"academia{PAIR_SEP}government":        "Knowledge Space",
        f"academia{PAIR_SEP}industry":           "Innovation Space",
        f"academia{PAIR_SEP}intermediary":       "Innovation Space",
        f"industry{PAIR_SEP}intermediary":       "Innovation Space",
        f"government{PAIR_SEP}industry":         "Consensus Space",
        f"government{PAIR_SEP}intermediary":     "Consensus Space",
        f"academia{PAIR_SEP}civil_society":      "Civil Society",
        f"civil_society{PAIR_SEP}government":    "Civil Society",
        f"civil_society{PAIR_SEP}industry":      "Civil Society",
        f"civil_society{PAIR_SEP}intermediary":  "Civil Society",
    }
    return space_map.get(pair, "Civil Society" if "civil_society" in pair else "Consensus Space")


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
    print(">>> ANALYSIS: RQ3")

    rq1_rows = _load_rq1()
    dominant_pairs = _load_rq2_dominant_pair()

    profiles = []
    for r in rq1_rows:
        country = r["country"]
        shares = r["shares"]
        gov = shares.get("government", 0.0)
        ind = shares.get("industry", 0.0)
        acad = shares.get("academia", 0.0)
        hbi = r["helix_balance_index"]
        cs = shares.get("civil_society", 0.0)

        config_label = _classify(gov, ind, acad, cs, hbi)
        dom_pair = dominant_pairs.get(country, "")
        th_space = _th_space(dom_pair) if dom_pair else ""

        profiles.append({
            "country": country,
            "qh_configuration": config_label,
            "hbi": hbi,
            "shares": shares,
            "civil_society_present": cs > 0.0,
            "dominant_cross_pair": dom_pair,
            "th_space_emphasis": th_space,
        })

    # Sort by configuration cluster, then by HBI descending within each cluster
    _config_order = {"Balanced": 0, "Laissez-faire": 1, "Mixed": 2, "Statist": 3}
    profiles.sort(key=lambda p: (_config_order.get(p["qh_configuration"], 9), -p["hbi"]))

    write_json(
        config.ANALYSIS_RQ3_PATH,
        {
            "id": "rq3_qh_profiles",
            "title": "RQ3 Quadruple Helix Country Profiles",
            "generated_utc": utc_now_iso(),
            "classification_rules": {
                "Balanced": "HBI >= 0.80 (all spheres roughly equal)",
                "Statist": "HBI < 0.80 AND gov > 0.50 (government has clear majority)",
                "Laissez-faire": "HBI < 0.80 AND gov <= 0.50 AND industry leads non-gov spheres",
                "Mixed": "HBI < 0.80 AND gov <= 0.50 AND academia/civil_society leads non-gov spheres",
            },
            "profiles": profiles,
        },
    )

    # ── Figure 1: Radar grid ──────────────────────────────────────────────────
    try:
        plt, np = _load_plot_dependencies()

        n = len(profiles)
        ncols = 5
        nrows = (n + ncols - 1) // ncols
        fig = plt.figure(figsize=(ncols * 3.2, nrows * 3.2))

        angles = np.linspace(0, 2 * np.pi, len(HELIXES), endpoint=False).tolist()
        angles += angles[:1]  # close polygon
        helix_labels = [h.replace("_", " ").title() for h in HELIXES]

        for idx, profile in enumerate(profiles):
            ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)
            values = [profile["shares"].get(h, 0.0) for h in HELIXES]
            values += values[:1]

            color = QH_CONFIG_COLORS[profile["qh_configuration"]]
            ax.plot(angles, values, color=color, linewidth=1.5)
            ax.fill(angles, values, color=color, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(helix_labels, size=6.5)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.25, 0.5, 0.75])
            ax.set_yticklabels(["25%", "50%", "75%"], size=5)
            ax.set_title(
                f"{profile['country']} — {profile['qh_configuration']}\n"
                f"HBI: {profile['hbi']:.2f}",
                size=8, pad=10,
            )

        # Config color legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=col, label=cfg)
            for cfg, col in QH_CONFIG_COLORS.items()
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center", ncol=4, fontsize=9,
            title="QH Configuration (Ranga & Etzkowitz 2013)",
            bbox_to_anchor=(0.5, 0.0),
        )
        fig.suptitle(
            "RQ3: Quadruple Helix Country Profiles\n"
            "(sorted by Helix Balance Index — most balanced first)",
            fontsize=13, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(config.ANALYSIS_RQ3_FIGURE_PATH, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Wrote: {config.ANALYSIS_RQ3_FIGURE_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ3 radar figure skipped: {exc}")

    # ── Figure 2: Scatter (Gov% vs HBI) + summary table ──────────────────────
    try:
        plt, np = _load_plot_dependencies()
        from matplotlib.patches import Patch
        from scipy.spatial import ConvexHull

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1]}
        )

        # Scatter
        for profile in profiles:
            gov = profile["shares"].get("government", 0.0)
            hbi = profile["hbi"]
            color = QH_CONFIG_COLORS[profile["qh_configuration"]]
            ax1.scatter(gov, hbi, color=color, s=90, zorder=5)
            ax1.annotate(
                profile["country"],
                (gov, hbi),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )

        # Reference lines
        ax1.axhline(0.80, color="#4393c3", linestyle="--", linewidth=1.2,
                    label="Balanced threshold (HBI ≥ 0.80)")
        ax1.set_xlabel("Government share of classified actors")
        ax1.set_ylabel("Helix Balance Index (normalized entropy)")
        ax1.set_title(
            "RQ3: QH Configuration Map\n"
            "(Gov% vs. Helix Balance Index)"
        )

        # Try convex hulls per config (need ≥3 points)
        try:
            for cfg, color in QH_CONFIG_COLORS.items():
                pts = np.array([
                    [p["shares"].get("government", 0.0), p["hbi"]]
                    for p in profiles if p["qh_configuration"] == cfg
                ])
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    hull_pts = np.append(pts[hull.vertices], pts[hull.vertices[:1]], axis=0)
                    ax1.fill(hull_pts[:, 0], hull_pts[:, 1],
                             color=color, alpha=0.10, zorder=1)
                    ax1.plot(hull_pts[:, 0], hull_pts[:, 1],
                             color=color, linewidth=0.8, alpha=0.5, zorder=2)
        except Exception:
            pass

        legend_elements = [
            Patch(facecolor=col, label=cfg)
            for cfg, col in QH_CONFIG_COLORS.items()
        ]
        ax1.legend(handles=legend_elements, fontsize=8, loc="upper right")
        ax1.set_xlim(-0.02, 1.02)
        ax1.set_ylim(-0.02, 1.02)

        # Summary table (right panel)
        ax2.axis("off")
        table_data = []
        col_labels = ["Country", "Config", "Gov%", "HBI", "CS?", "Top Pair"]
        for p in profiles:
            gov_pct = f"{p['shares'].get('government', 0)*100:.0f}%"
            cs_present = "✓" if p["civil_society_present"] else "–"
            _abbrev = {"government": "GOV", "industry": "IND", "academia": "ACA",
                       "civil_society": "CIV", "intermediary": "INT"}
            if p["dominant_cross_pair"]:
                parts = p["dominant_cross_pair"].split("–")
                dom = "–".join(_abbrev.get(pt.strip(), pt.strip()[:3].upper()) for pt in parts)
            else:
                dom = "–"
            table_data.append([
                p["country"],
                p["qh_configuration"],
                gov_pct,
                f"{p['hbi']:.2f}",
                cs_present,
                dom,
            ])
        tbl = ax2.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1.0, 1.35)
        # Color rows by configuration
        for row_idx, p in enumerate(profiles):
            color = QH_CONFIG_COLORS[p["qh_configuration"]]
            for col_idx in range(len(col_labels)):
                tbl[row_idx + 1, col_idx].set_facecolor(color + "30")
        ax2.set_title("Country Profiles Summary", fontsize=10)

        fig.suptitle(
            "RQ3: Quadruple Helix System Types in National Quantum Strategies\n"
            "(Ranga & Etzkowitz 2013: Balanced / Statist / Laissez-faire / Mixed)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(config.ANALYSIS_RQ3_SUMMARY_FIGURE_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Wrote: {config.ANALYSIS_RQ3_SUMMARY_FIGURE_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ3 summary figure skipped: {exc}")

    # ── Classification reasoning table ───────────────────────────────────────
    try:
        plt, np = _load_plot_dependencies()

        col_labels = [
            "Country", "Config", "Gov%", "Ind%", "Acad%", "CS%", "HBI",
            "CS\nPresent", "Top Pair", "TH Space", "Triggered rule", "Decisive metric"
        ]

        def _rule_explanation(p: dict) -> tuple[str, str]:
            gov  = p["shares"].get("government",   0.0)
            ind  = p["shares"].get("industry",     0.0)
            acad = p["shares"].get("academia",     0.0)
            hbi  = p["hbi"]
            cfg  = p["qh_configuration"]
            if cfg == "Balanced":
                return "HBI ≥ 0.80", f"HBI = {hbi:.2f}"
            if cfg == "Statist":
                return "HBI < 0.80 & Gov > 50%", f"Gov = {gov*100:.1f}%, HBI = {hbi:.2f}"
            if cfg == "Laissez-faire":
                return "HBI < 0.80 & Gov ≤ 50% & Ind leads", f"Ind = {ind*100:.1f}%, Gov = {gov*100:.1f}%"
            # Mixed — academia or civil society leads non-gov
            cs = p["shares"].get("civil_society", 0.0)
            non_gov_leader = "Acad" if acad >= cs else "CS"
            non_gov_val = max(acad, cs)
            return "HBI < 0.80 & Gov ≤ 50% & non-Ind leads", f"{non_gov_leader} = {non_gov_val*100:.1f}%, Gov = {gov*100:.1f}%"

        table_data = []
        row_colors = []
        _abbrev = {"government": "GOV", "industry": "IND", "academia": "ACA",
                   "civil_society": "CIV", "intermediary": "INT"}
        for p in profiles:
            gov  = p["shares"].get("government",   0.0)
            ind  = p["shares"].get("industry",     0.0)
            acad = p["shares"].get("academia",     0.0)
            cs   = p["shares"].get("civil_society",0.0)
            rule, decisive = _rule_explanation(p)
            dom = p["dominant_cross_pair"]
            if dom:
                parts = dom.split("–")
                dom_abbrev = "–".join(_abbrev.get(pt.strip(), pt.strip()[:3].upper()) for pt in parts)
            else:
                dom_abbrev = "–"
            table_data.append([
                p["country"],
                p["qh_configuration"],
                f"{gov*100:.1f}%",
                f"{ind*100:.1f}%",
                f"{acad*100:.1f}%",
                f"{cs*100:.1f}%",
                f"{p['hbi']:.2f}",
                "✓" if p["civil_society_present"] else "–",
                dom_abbrev,
                p.get("th_space_emphasis", "–") or "–",
                rule,
                decisive,
            ])
            row_colors.append(QH_CONFIG_COLORS[p["qh_configuration"]] + "30")

        n_rows = len(table_data)
        fig_h = max(5, n_rows * 0.42 + 1.5)
        fig, ax = plt.subplots(figsize=(16, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.6)

        # Header styling
        for col_idx in range(len(col_labels)):
            tbl[0, col_idx].set_facecolor("#2c3e50")
            tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

        # Row colors by config
        for row_idx, p in enumerate(profiles):
            color = QH_CONFIG_COLORS[p["qh_configuration"]] + "30"
            for col_idx in range(len(col_labels)):
                tbl[row_idx + 1, col_idx].set_facecolor(color)

        # Widen last two columns
        tbl.auto_set_column_width(col=list(range(len(col_labels))))

        ax.set_title(
            "RQ3: Classification Reasoning per Country\n"
            "Rules: Balanced = HBI≥0.80 · Statist = HBI<0.80 & Gov>50% · "
            "Laissez-faire = HBI<0.80 & Gov≤50% & Ind leads · Mixed = HBI<0.80 & Gov≤50% & non-Ind leads",
            fontsize=10, pad=12,
        )
        fig.tight_layout()
        fig.savefig(config.ANALYSIS_RQ3_REASONING_FIGURE_PATH, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Wrote: {config.ANALYSIS_RQ3_REASONING_FIGURE_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ3 reasoning table skipped: {exc}")

    # ── CSV export ───────────────────────────────────────────────────────────
    try:
        fieldnames = [
            "Country", "QH_Configuration", "Gov_Share", "Ind_Share", "Acad_Share",
            "CS_Share", "Int_Share", "HBI", "Civil_Society_Present",
            "Dominant_Cross_Pair", "TH_Space_Emphasis",
        ]
        with open(config.ANALYSIS_RQ3_CSV_PATH, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for p in profiles:
                writer.writerow({
                    "Country": p["country"],
                    "QH_Configuration": p["qh_configuration"],
                    "Gov_Share": f"{p['shares'].get('government', 0):.4f}",
                    "Ind_Share": f"{p['shares'].get('industry', 0):.4f}",
                    "Acad_Share": f"{p['shares'].get('academia', 0):.4f}",
                    "CS_Share": f"{p['shares'].get('civil_society', 0):.4f}",
                    "Int_Share": f"{p['shares'].get('intermediary', 0):.4f}",
                    "HBI": f"{p['hbi']:.4f}",
                    "Civil_Society_Present": p["civil_society_present"],
                    "Dominant_Cross_Pair": p["dominant_cross_pair"],
                    "TH_Space_Emphasis": p["th_space_emphasis"],
                })
        print(f"[OK] Wrote: {config.ANALYSIS_RQ3_CSV_PATH}")
    except Exception as exc:
        print(f"[WARN] RQ3 CSV skipped: {exc}")

    print(f"[OK] Wrote: {config.ANALYSIS_RQ3_PATH}")


if __name__ == "__main__":
    run()
