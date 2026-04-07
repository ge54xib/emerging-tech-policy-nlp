"""Unified entry point for the policy-document NLP pipeline.

Usage
-----
# Run individual pipeline steps
python run.py pipeline --step 0   # PDF extraction (requires Adobe credentials)
python run.py pipeline --step 1   # Named Entity Recognition (Flair)
python run.py pipeline --step 2   # Quadruple Helix classification
python run.py pipeline --step 3   # Paragraph co-occurrence extraction
python run.py pipeline --step all # All steps in sequence

# Generate thesis analysis outputs
python run.py analysis            # All RQs + descriptives + methodology

# Run everything end-to-end
python run.py all
"""

from __future__ import annotations

import argparse
import sys


def _run_pipeline(step: str) -> None:
    from src.pipeline.run_pipeline import main as pipeline_main
    sys.argv = ["run.py", "--step", step]
    pipeline_main()


def _run_analysis() -> None:
    from src.analysis.run_deliverables import run
    run()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="Quantum policy NLP pipeline — Quadruple Helix analysis",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pipeline subcommand
    pipe_parser = subparsers.add_parser(
        "pipeline", help="Run a pipeline step (PDF extraction → NER → classification → co-occurrence)"
    )
    pipe_parser.add_argument(
        "--step",
        required=True,
        choices=["0", "1", "2", "3", "all"],
        help="Step to execute: 0=preprocess, 1=NER, 2=classify, 3=cooccurrence, all=full pipeline",
    )

    # analysis subcommand
    subparsers.add_parser(
        "analysis", help="Generate thesis analysis outputs (RQ1–RQ4, descriptives, methodology)"
    )

    # ui subcommand
    subparsers.add_parser("ui", help="Launch Streamlit UI for manual QH classification")

    # all subcommand
    subparsers.add_parser("all", help="Run full pipeline (steps 0–3) then analysis")

    args = parser.parse_args()

    if args.command == "pipeline":
        _run_pipeline(args.step)
    elif args.command == "analysis":
        _run_analysis()
    elif args.command == "ui":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/app.py"])
    elif args.command == "all":
        _run_pipeline("all")
        _run_analysis()


if __name__ == "__main__":
    main()
