"""RQ_Spaces: TH space distribution using SetFit predictions (Tunstall et al. 2022).

Delegates to src.analysis.spaces, which auto-selects cooccurrence_setfit.jsonl
when present (produced by python -m src.pipeline.spaces_setfit --predict).

Outputs: see src/analysis/spaces.py  →  outputs/spaces/
"""
from src.analysis.spaces import run  # noqa: F401
