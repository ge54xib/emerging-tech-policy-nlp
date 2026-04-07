"""Simple runner for the policy-document NLP pipeline."""

from __future__ import annotations

import argparse


def _runner(step: str):
    if step == "0":
        from src.pipeline import step0_preprocess

        return step0_preprocess.run
    if step == "1":
        from src.pipeline import step1_ner

        return step1_ner.run
    if step == "2":
        from src.pipeline import step2_classify

        return step2_classify.run
    if step == "3":
        from src.pipeline import step3_cooccurrence

        return step3_cooccurrence.run
    if step == "4":
        from src.pipeline import step4_nli

        return step4_nli.run
    raise ValueError(f"Unsupported step: {step}")


def run_all() -> None:
    for step in ["0", "1", "2", "3", "4"]:
        _runner(step)()


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-document NLP pipeline runner")
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "0", "1", "2", "3", "4"],
        help="Pipeline step to run",
    )
    args = parser.parse_args()

    if args.step == "all":
        run_all()
    else:
        _runner(args.step)()


if __name__ == "__main__":
    main()
