"""Helpers for thesis deliverable JSON outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def thesis_style() -> None:
    """Apply thesis-ready matplotlib style: Times New Roman, readable sizes, clean grid."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.rcParams.update({
        # Font
        "font.family":          "serif",
        "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":            11,
        # Axes
        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "axes.labelsize":       11,
        "axes.labelweight":     "normal",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.color":           "#DDDDDD",
        "grid.linewidth":       0.7,
        # Ticks
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        # Legend
        "legend.fontsize":      10,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#CCCCCC",
        # Figure
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
        # Lines
        "lines.linewidth":      1.8,
        "patch.linewidth":      0.8,
    })


