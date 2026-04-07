"""Run all thesis deliverable analyses (JSON + PNG figures + CSVs)."""

from __future__ import annotations

import importlib

from src.analysis import descriptives, methodology, rq1, rq2, rq2_gpt_re, rq3, rq_extended, spaces, rq_spaces


def _require_plot_dependencies() -> None:
    missing: list[str] = []
    for module in ("matplotlib", "seaborn", "pandas", "numpy"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append(module)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing analysis dependencies: {joined}. "
            "Install with: pip install -r requirements.txt"
        )


def run() -> None:
    _require_plot_dependencies()
    descriptives.run()
    methodology.run()
    rq1.run()
    rq2.run()
    rq2_gpt_re.run()
    rq3.run()
    rq_extended.run()
    spaces.run()
    rq_spaces.run()


if __name__ == "__main__":
    run()
