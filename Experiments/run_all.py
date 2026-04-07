"""Run all experiments sequentially.

Usage:
    python Experiments/run_all.py                        # all experiments
    python Experiments/run_all.py --relation             # relation only
    python Experiments/run_all.py --spaces               # spaces only
    python Experiments/run_all.py --skip gpt_re llm_augmenter  # skip specific

Requires API keys for LLM-based experiments:
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

RELATION_EXPERIMENTS = [
    "Relation/nli_pipeline",
    "Relation/nli_sainz_roberta",
    "Relation/nli_sainz_deberta",
    "Relation/claude_cot",
    "Relation/gollie_guidelines",
    "Relation/self_consistency",
    "Relation/gpt_re",
    "Relation/glirel",
]

SPACES_EXPERIMENTS = [
    "Spaces/helix_pair_baseline",
    "Spaces/nli_pipeline",
    "Spaces/instructor_embeddings",
    "Spaces/setfit_current",
    "Spaces/fastfit",
    "Spaces/gpt4_few_shot",
    "Spaces/llm_augmenter",
]


def _run(experiment_path: str, skip: set[str]) -> bool:
    name = experiment_path.split("/")[-1]
    if name in skip:
        print(f"\n{'='*60}\nSKIPPING {experiment_path}\n{'='*60}")
        return True

    run_file = Path(__file__).parent / experiment_path / "run.py"
    print(f"\n{'='*60}\nRUNNING {experiment_path}\n{'='*60}")

    spec = importlib.util.spec_from_file_location("run", run_file)
    mod  = importlib.util.module_from_spec(spec)
    # Isolate sys.path per experiment
    orig_path = sys.path[:]
    sys.path.insert(0, str(run_file.parent))
    try:
        spec.loader.exec_module(mod)
        mod.main()
        print(f"OK: {experiment_path}")
        return True
    except Exception:
        print(f"FAILED: {experiment_path}")
        traceback.print_exc()
        return False
    finally:
        sys.path = orig_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation", action="store_true", help="Run only relation experiments")
    parser.add_argument("--spaces",   action="store_true", help="Run only spaces experiments")
    parser.add_argument("--skip",     nargs="*", default=[], help="Experiment names to skip")
    args = parser.parse_args()

    skip = set(args.skip)

    if args.relation:
        experiments = RELATION_EXPERIMENTS
    elif args.spaces:
        experiments = SPACES_EXPERIMENTS
    else:
        experiments = RELATION_EXPERIMENTS + SPACES_EXPERIMENTS

    results: dict[str, bool] = {}
    for exp in experiments:
        results[exp] = _run(exp, skip)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for exp, ok in results.items():
        status = "OK     " if ok else "FAILED "
        print(f"  {status} {exp}")

    failed = [e for e, ok in results.items() if not ok]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
