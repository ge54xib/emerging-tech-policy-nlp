"""Pipeline architecture figure — Design Science style for quantum policy analysis thesis.

Output: outputs/pipeline_architecture.png + .pdf
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np
from pathlib import Path
from src.analysis._helpers import thesis_style
thesis_style()

# ── Colors ────────────────────────────────────────────────────────────────────
PHASE1_BG  = "#DCDCDC"
PHASE1_BD  = "#777777"
PHASE2_BG  = "#BDD3E8"
PHASE2_BD  = "#2B6CB0"
MOD_A_BG   = "#EBF2FA"
MOD_A_BD   = "#2B6CB0"
MOD_B_BG   = "#BEDAEF"
MOD_B_BD   = "#1A4E8A"
MOD_C_BG   = "#96CDD2"
MOD_C_BD   = "#1A7A82"
PHASE3_BG  = "#F5C842"
PHASE3_BD  = "#B07A00"
ORANGE_BOX = "#E8902A"
ORANGE_BD  = "#994400"
WHITE      = "#FFFFFF"
DARK       = "#111111"
TEAL       = "#16A2A8"
ARROW_C    = "#333333"
GREY       = "#555555"

FIG_W, FIG_H = 22, 12.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")


# ── Helpers ───────────────────────────────────────────────────────────────────
def rbox(cx, cy, w, h, text, sub=None, fc=WHITE, ec=GREY, lw=1.5,
         fs=8.5, bold=False, tc=DARK):
    ax.add_patch(FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    fw = "bold" if bold else "normal"
    if sub:
        ax.text(cx, cy+0.14, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=4, multialignment="center")
        ax.text(cx, cy-0.18, sub, ha="center", va="center",
                fontsize=fs-1.8, color=GREY, zorder=4, style="italic",
                multialignment="center")
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=4, multialignment="center")


def phase_rect(x, y, w, h, title, fc, ec, lw=2.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1))
    tw = len(title) * 0.092 + 0.4
    ax.add_patch(FancyBboxPatch(
        (x+0.12, y+h-0.44), tw, 0.38, boxstyle="round,pad=0.04",
        facecolor=ec, edgecolor="none", linewidth=0, zorder=2))
    ax.text(x+0.3, y+h-0.25, title,
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=WHITE, zorder=3)


def arw(x0, y0, x1, y1, col=ARROW_C, lw=1.5, dashed=False, rad=0.0, label=None):
    ls = (0, (5, 3)) if dashed else "solid"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->, head_width=0.22, head_length=0.14",
                    color=col, lw=lw, linestyle=ls,
                    connectionstyle=f"arc3,rad={rad}"), zorder=5)
    if label:
        ax.text((x0+x1)/2+0.06, (y0+y1)/2+0.12, label,
                ha="center", va="center", fontsize=7, color=col,
                bbox=dict(facecolor=WHITE, edgecolor="none", pad=1.5))


def diamond(cx, cy, w, h, lines, fc=WHITE, ec=GREY, lw=1.5, fs=7.5):
    pts = np.array([[cx, cy+h/2], [cx+w/2, cy], [cx, cy-h/2], [cx-w/2, cy]])
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec,
                         linewidth=lw, zorder=3))
    for i, ln in enumerate(lines):
        ax.text(cx, cy+(len(lines)-1)*0.12-i*0.24, ln,
                ha="center", va="center", fontsize=fs, color=DARK, zorder=4)


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(FIG_W/2, 12.1,
        "INTEGRATED NLP & DESIGN SCIENCE ARCHITECTURE FOR QUANTUM POLICY ANALYSIS",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color=DARK)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Input & Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
P1X, P1Y, P1W, P1H = 0.2, 9.1, 21.6, 2.7
phase_rect(P1X, P1Y, P1W, P1H, "Phase 1: Input & Preprocessing",
           PHASE1_BG, PHASE1_BD)

PY = 10.35   # main flow y
BW, BH = 1.72, 0.72

# Input nodes
rbox(1.1,  PY, BW, BH, "OECD\nTimeline",              fs=8)
rbox(3.0,  PY, BW, BH, "Web Search\n& Validation",    fs=8)
rbox(4.95, PY, BW, BH, "Official\nStrategy PDFs",     fs=8)

arw(1.1+BW/2, PY, 3.0-BW/2, PY)
arw(3.0+BW/2, PY, 4.95-BW/2, PY)

# Decision diamond
DX, DY = 7.1, PY
arw(4.95+BW/2, PY, DX-0.9, PY)
diamond(DX, DY, 1.8, 1.0, ["Is PDF", "English?"])

# Yes → DeepL
DL_X, DL_Y = 9.15, PY + 0.82
rbox(DL_X, DL_Y, 1.92, 0.68, "DeepL Document\nTranslation", fs=8)
arw(DX+0.9, DY+0.12, DL_X-0.96, DL_Y, label="Yes")

# No → Adobe directly
AD_X = 11.1
arw(DX, DY-0.5, AD_X-1.0, PY-0.1, rad=-0.15, label="No")
rbox(AD_X, PY, 1.92, 0.72, "Adobe PDF\nExtract API",        fs=8)
arw(DL_X+0.96, DL_Y, AD_X-0.96, PY+0.18)

ax.text(AD_X, PY-0.55, "Tagged JSON & Layout Tree",
        ha="center", va="center", fontsize=6, color=GREY, style="italic")

# Subsequent steps
ST_X = 13.2
rbox(ST_X, PY, 1.92, 0.72, "Structural\nFiltering",
     sub="Drop TOCs, Footnotes", fs=7.5)
arw(AD_X+0.96, PY, ST_X-0.96, PY)

TB_X = 15.3
rbox(TB_X, PY, 1.92, 0.72, "Text Block\nReconstruction", fs=7.5)
arw(ST_X+0.96, PY, TB_X-0.96, PY)

DS_X = 17.45
rbox(DS_X, PY, 2.1, 0.72, "Clean Plain Text\nDataset (.txt)", fs=7.5)
arw(TB_X+0.96, PY, DS_X-1.05, PY)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Computational NLP Artifact
# ══════════════════════════════════════════════════════════════════════════════
P2X, P2Y, P2W, P2H = 0.2, 0.2, 14.2, 8.65
phase_rect(P2X, P2Y, P2W, P2H, "Phase 2: The Computational NLP Artifact",
           PHASE2_BG, PHASE2_BD)

# Phase 1 → Phase 2 arrows
arw(4.95, PY-BH/2, 2.55, P2Y+P2H-0.08, col=ARROW_C, lw=2.0)  # PDFs → Mod A
arw(DS_X, PY-BH/2, 11.65, P2Y+P2H-0.08, col=ARROW_C, lw=2.0)  # Dataset → Mod C

# ── Module A — Components (Actors) ────────────────────────────────────────────
MA_X, MA_W = 0.4, 4.3
MA_Y, MA_H = 0.4, 8.15
MA_CX = MA_X + MA_W/2
phase_rect(MA_X, MA_Y, MA_W, MA_H, "Module A - Components (Actors)",
           MOD_A_BG, MOD_A_BD, lw=1.5)

rbox(MA_CX, 7.7, 3.6, 0.68,
     "Flair NER\n(ner-english-large)",
     fc=WHITE, ec=MOD_A_BD, fs=8.5, bold=True)
arw(MA_CX, 7.36, MA_CX, 6.7, col=MOD_A_BD)

rbox(MA_CX, 6.3, 3.8, 0.9,
     "Manual Validation &\nHelix Classification",
     sub="Web check, primary mandate,\nhierarchy rules",
     fc=WHITE, ec=MOD_A_BD, fs=8)
arw(MA_CX, 5.85, MA_CX, 5.18, col=MOD_A_BD)

rbox(MA_CX, 4.75, 3.8, 0.86,
     "Classified Actor List",
     sub="Academia · Industry · Gov\nCivil Society · Intermediary",
     fc=WHITE, ec=MOD_A_BD, fs=8)

# Helix colour bands
for i, (hc, hl) in enumerate(zip(
        ["#3060C0", "#E07020", "#60A840", "#C83030"],
        ["Helix", "Helix", "Helix", "Helix"])):
    hy = 4.1 - i * 0.25
    ax.add_patch(FancyBboxPatch(
        (MA_CX+0.35, hy), 1.55, 0.21,
        boxstyle="round,pad=0.02", facecolor=hc, edgecolor="none", zorder=4))
    ax.text(MA_CX+1.12, hy+0.105, hl, ha="center", va="center",
            fontsize=6.5, color=WHITE, fontweight="bold", zorder=5)

# ── Module B — Relationships (NLI) ────────────────────────────────────────────
MB_X, MB_W = 4.9, 4.3
MB_Y, MB_H = 0.4, 8.15
MB_CX = MB_X + MB_W/2
phase_rect(MB_X, MB_Y, MB_W, MB_H, "Module B - Relationships (NLI)",
           MOD_B_BG, MOD_B_BD, lw=1.5)

rbox(MB_CX, 7.7, 3.8, 0.68,
     "Co-occurrence Mapping\n(Sentence-level)",
     fc=WHITE, ec=MOD_B_BD, fs=8.5)
arw(MB_CX, 7.36, MB_CX, 6.7, col=MOD_B_BD)

rbox(MB_CX, 6.3, 3.8, 0.68,
     "Zero-Shot NLI\n(RoBERTa-large-MNLI)",
     fc=WHITE, ec=MOD_B_BD, fs=8.5)
arw(MB_CX, 5.96, MB_CX, 5.38, col=MOD_B_BD)

# Formula box 5 × 4 × 2 = 40 NLI Calls
ax.add_patch(FancyBboxPatch(
    (MB_CX-1.8, 4.68), 3.6, 0.65,
    boxstyle="round,pad=0.04", facecolor=WHITE, edgecolor=MOD_B_BD,
    linewidth=1.2, zorder=3))
for xoff, num, sub in [(-1.12, "5", "Relations"), (-0.32, "4", "Templates"),
                        (0.5,   "2", "Directions")]:
    ax.text(MB_CX+xoff, 5.06, num, ha="center", va="center",
            fontsize=12, fontweight="bold", color=DARK, zorder=4)
    ax.text(MB_CX+xoff, 4.74, sub, ha="center", va="center",
            fontsize=5.5, color=GREY, zorder=4)
for xoff in [-0.7, 0.1]:
    ax.text(MB_CX+xoff, 5.02, "×", ha="center", va="center",
            fontsize=11, color=DARK, zorder=4)
ax.text(MB_CX+0.92, 5.03, "= 40 NLI Calls",
        ha="left", va="center", fontsize=8, fontweight="bold",
        color=DARK, zorder=4)

arw(MB_CX, 4.68, MB_CX, 4.08, col=MOD_B_BD)

rbox(MB_CX, 3.67, 3.8, 0.72,
     "Accepted Inter-Actor\nRelations",
     fc=WHITE, ec=MOD_B_BD, fs=8.5)

for i, txt in enumerate(["5 Relation", "4 Relation type",
                          "5 Relation", "5 Relation type"]):
    ry = 3.12 - i * 0.28
    ax.text(MB_CX-1.7, ry, ">>", ha="left", va="center", fontsize=7, color=MOD_B_BD, zorder=4)
    ax.text(MB_CX-1.25, ry, txt, ha="left", va="center",
            fontsize=7.5, color=DARK, zorder=4)

# Module A → Module B arrow
arw(MA_X+MA_W, 4.75, MB_X, 5.3, col=ARROW_C, lw=1.5)

# ── Module C — Functions / Spaces (NLI) ───────────────────────────────────────
MC_X, MC_W = 9.6, 4.6
MC_Y, MC_H = 0.4, 8.15
MC_CX = MC_X + MC_W/2
phase_rect(MC_X, MC_Y, MC_W, MC_H, "Module C - Functions / Spaces (NLI)",
           MOD_C_BG, MOD_C_BD, lw=1.5)

# NLI UPGRADE badge
bw = 2.0
ax.add_patch(FancyBboxPatch(
    (MC_CX+0.4, 8.18), bw, 0.30,
    boxstyle="round,pad=0.03", facecolor=TEAL, edgecolor="none", zorder=5))
ax.text(MC_CX+0.4+bw/2, 8.33, "NLI UPGRADE",
        ha="center", va="center", fontsize=8, fontweight="bold",
        color=WHITE, zorder=6)

ax.text(MC_CX, 9.0, "*Every Sentence*",
        ha="center", va="center", fontsize=8.5,
        fontstyle="italic", fontweight="bold", color=DARK)

rbox(MC_CX, 7.7, 4.0, 0.68, "Sentence-Level Extraction",
     fc=WHITE, ec=MOD_C_BD, fs=8.5)
arw(MC_CX, 7.36, MC_CX, 6.7, col=MOD_C_BD)

rbox(MC_CX, 6.3, 4.0, 0.68, "Zero-Shot NLI\n(RoBERTa-large-MNLI)",
     fc=WHITE, ec=MOD_C_BD, fs=8.5)
arw(MC_CX, 5.96, MC_CX, 5.28, col=MOD_C_BD)

rbox(MC_CX, 4.85, 4.0, 0.82,
     "Judge against Space Hypotheses:",
     sub="Knowledge · Innovation · Consensus · Public",
     fc=WHITE, ec=MOD_C_BD, fs=8)
arw(MC_CX, 4.44, MC_CX, 3.78, col=MOD_C_BD)

rbox(MC_CX, 3.38, 4.0, 0.72,
     "Sentence-Level Functional\nSpace Probability Map",
     fc=WHITE, ec=MOD_C_BD, fs=8)

# Module B → Module C (implicit — same pipeline)
arw(MB_X+MB_W, 6.3, MC_X, 6.3, col=ARROW_C, lw=1.2, dashed=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Aggregation & Final Output
# ══════════════════════════════════════════════════════════════════════════════
P3X, P3Y, P3W, P3H = 14.6, 0.2, 7.2, 8.65
P3CX = P3X + P3W/2
phase_rect(P3X, P3Y, P3W, P3H, "Phase 3: Aggregation & Final Output",
           PHASE3_BG, PHASE3_BD)

# Phase 2 → Phase 3 arrows
arw(MB_X+MB_W, 3.67, P3X, 5.8, col=ARROW_C, lw=2.0)   # relations
arw(MC_X+MC_W, 3.38, P3X, 4.8, col=ARROW_C, lw=2.0)   # spaces

# Strategy-Level Aggregation
rbox(P3CX, 7.7, 5.8, 0.7, "Strategy-Level Aggregation",
     fc=ORANGE_BOX, ec=ORANGE_BD, lw=1.5, fs=9.5, bold=True, tc=WHITE)
arw(P3CX, 7.35, P3CX, 6.72, col=ORANGE_BD, lw=1.5)

# Metric Calculation
rbox(P3CX, 6.3, 5.8, 0.84,
     "Metric Calculation",
     sub="Helix Balance Index (HBI) · Government Share",
     fc=ORANGE_BOX, ec=ORANGE_BD, lw=1.5, fs=9.5, bold=True, tc=WHITE)
arw(P3CX, 5.88, P3CX, 5.22, col=ORANGE_BD, lw=1.5)

# HBI vs Gov Share Plot
rbox(P3CX, 4.8, 5.8, 0.72, "HBI vs. Gov Share Plot",
     fc=ORANGE_BOX, ec=ORANGE_BD, lw=1.5, fs=9.5, bold=True, tc=WHITE)
arw(P3CX, 4.44, P3CX, 3.78, col=ORANGE_BD, lw=1.5)

# Profile outputs
for px, pl, pc in [
    (P3CX-2.1, "Statist\nProfile",  "#C83020"),
    (P3CX,     "Mixed\nProfile",    "#D07020"),
    (P3CX+2.1, "Balanced\nProfile", "#30A030"),
]:
    arw(P3CX, 3.42, px, 2.72, col=PHASE3_BD, lw=1.4)
    rbox(px, 2.28, 2.1, 0.72, pl,
         fc=WHITE, ec=pc, lw=2.0, fs=8.5, bold=True, tc=pc)

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
out_dir = Path(__file__).parent.parent.parent / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

for fmt in ("png", "pdf"):
    path = out_dir / f"pipeline_architecture.{fmt}"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=WHITE, format=fmt)
    print(f"[OK] Saved: {path}")

plt.close(fig)
