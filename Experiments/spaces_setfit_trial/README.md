# SetFit TH-Space Trial (Isolated)

This folder is an isolated trial for Triple Helix space classification using
SetFit (Tunstall et al., 2022 / arXiv:2209.11055), without changing
`src/pipeline/`.

## What this does

The script mirrors the 3-stage workflow:

1. `--sample`  
   Build `spaces_annotation.json` from existing `cooccurrence.jsonl`.
2. `--train`  
   Train a SetFit model on manually annotated examples.
3. `--predict`  
   Re-classify all unique sentences and write `cooccurrence_setfit.jsonl`.

All artifacts stay in this folder by default (`artifacts/`).

## Quick start

```bash
python experiments/spaces_setfit_trial/spaces_setfit_trial.py --sample
```

Then annotate:

- `experiments/spaces_setfit_trial/artifacts/spaces_annotation.json`
- Use `experiments/spaces_setfit_trial/codebook_spaces.md` as guidance.

Train:

```bash
python experiments/spaces_setfit_trial/spaces_setfit_trial.py --train
```

Trial-only bootstrap (no manual annotation yet):

```bash
python experiments/spaces_setfit_trial/spaces_setfit_trial.py --train --bootstrap-from-nli
```

Predict:

```bash
python experiments/spaces_setfit_trial/spaces_setfit_trial.py --predict
```

## Optional custom paths

```bash
python experiments/spaces_setfit_trial/spaces_setfit_trial.py \
  --sample \
  --cooccurrence data/processed/step3/cooccurrence.jsonl \
  --workdir experiments/spaces_setfit_trial/artifacts
```
