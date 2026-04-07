"""NLI-based zero-shot relation extraction (Sainz et al. 2021).

For each entity pair (subject, object) found in a paragraph, this module
scores 5 Ranga & Etzkowitz (2013) relation types by treating the paragraph
text as premise and verbalized templates as hypotheses.

Relation types:
  1. technology_transfer
  2. collaboration_conflict_moderation
  3. collaborative_leadership
  4. substitution
  5. networking

Usage:
    scorer = NLIRelationScorer()
    result = scorer.score_pair(paragraph_text, "MIT", "Quantum Corp")
    # result = {"relation_type": "technology_transfer", "confidence": 0.82,
    #           "all_scores": {...}, "direction": "A->B"}
"""

from __future__ import annotations

RELATION_TEMPLATES: dict[str, list[str]] = {
    # Knowledge or IP moves from one sphere to another via explicit transfer mechanisms.
    "technology_transfer": [
        "{subj} licenses patents, know-how, or inventions for use by {obj}.",
        "{subj} transfers research results or intellectual property to {obj} for commercial application.",
        "{subj} operates technology transfer offices, incubators, or science parks that serve {obj}.",
        "{subj} supports spin-off creation, patenting, or licensing activities involving {obj}.",
    ],
    # One actor mediates or resolves active tension or conflict of interest between spheres.
    "collaboration_conflict_moderation": [
        "{subj} helps resolve tensions or conflicts of interest between institutional spheres involving {obj}.",
        "{subj} negotiates a resolution to conflicting interests between {obj} and other institutional spheres.",
        "{subj} develops a partnership structure with {obj} to transform competing interests into collaboration.",
        "{subj} intervenes to mediate institutional tensions between {obj} and other actors.",
    ],
    # One actor takes an asymmetric leadership role, organizing and convening other spheres.
    "collaborative_leadership": [
        "{subj} acts as an innovation organizer, convening {obj} and other institutional spheres into a shared agenda.",
        "{subj} leads and directly coordinates the activities of {obj} within a cross-sector initiative.",
        "{subj} has mobilized or activated {obj} to participate in a jointly driven programme.",
        "{subj} steers {obj} and other actors toward a shared strategic goal.",
    ],
    # One sphere fills a role normally belonging to another that is absent or weak.
    "substitution": [
        "{subj} fills a gap left by the absence or weakness of {obj} in the innovation system.",
        "{subj} takes over a function normally belonging to {obj} because {obj} is absent or underdeveloped.",
        "{subj} provides venture capital, firm formation, or training because {obj} lacks the capacity to do so.",
        "{subj} assumes responsibilities traditionally belonging to {obj} in contexts where {obj} is weak.",
    ],
    # Symmetric connection-building: named/formal alliances, consortia, or bilateral ties already in place.
    # Templates are intentionally specific — require named structures or signed instruments
    # so that DeBERTa does not match generic policy co-mentions.
    "networking": [
        "{subj} and {obj} have signed a memorandum of understanding or a named bilateral cooperation agreement.",
        "{subj} and {obj} are co-listed as member institutions of a named consortium, joint centre, or multilateral network.",
        "{subj} maintains an active formal partnership with {obj} through a named joint programme or standing bilateral initiative.",
        "{subj} and {obj} participate jointly in a named international network or multilateral alliance focused on shared goals.",
    ],
}

RELATION_TYPES = list(RELATION_TEMPLATES.keys())

# ── Triple / Quadruple Helix space classification (sentence-level) ────────────
# Grounded in Ranga & Etzkowitz (2013): spaces are defined by ACTIVITY TYPE
# expressed in a sentence, not by the specific entity pair co-occurring in it.
#
#   Knowledge space  — R&D and knowledge resources, scientific expertise
#   Innovation space — tech transfer, hybrid orgs, IP, commercialisation
#   Consensus space  — policy dialogue, governance, shared agendas
#   Public space     — civil society 4th helix (QH extension)
#
# Zero-shot text classification: sentence → "This text is about {label}."
# No entity pair needed — classifying the sentence alone is more grounded.
SPACE_LABELS: dict[str, str] = {
    "knowledge_space": "research collaboration, scientific knowledge creation, and R&D activities",
    "innovation_space": "technology transfer, commercialisation, start-ups, intellectual property, and firm formation",
    "consensus_space": "governance, policy dialogue, regulation, and strategic consensus building",
    "public_space": "civil society engagement, public benefit, ethics, and equitable access to technology",
}

SPACE_TYPES = list(SPACE_LABELS.keys())

SPACE_HYPOTHESIS_TEMPLATE = "This text is about {label}."

# Fallback entailment index for models with standard MNLI label order
# (0=contradiction, 1=neutral, 2=entailment).
# The actual index is resolved at load time from the model's label2id config.
_ENTAILMENT_IDX_FALLBACK = 2


class NLIRelationScorer:
    """Zero-shot relation classifier using NLI entailment scores.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (must be an NLI / MNLI model).
    threshold:
        Minimum entailment probability to assign a relation type.
        Pairs below this threshold receive ``"no_explicit_relation"``.
    device:
        ``"auto"`` (default) auto-selects cuda > mps > cpu.
        Pass ``"cpu"`` to force CPU inference.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
        threshold: float = 0.5,
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self._device = self._resolve_device(device)
        self._tokenizer = None
        self._model = None
        self._entailment_idx: int = _ENTAILMENT_IDX_FALLBACK
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_pair(self, premise: str, subj: str, obj: str) -> dict:
        """Classify the relation between *subj* and *obj* given *premise*.

        Returns a dict with keys:
          ``relation_type``  – winning relation or ``"no_explicit_relation"``
          ``confidence``     – entailment probability of the winning relation
          ``all_scores``     – dict mapping each relation type to its score
          ``direction``      – ``"A->B"`` or ``"B->A"`` for the best direction
        """
        all_scores: dict[str, float] = {}
        best_directions: dict[str, str] = {}

        for relation_type, templates in RELATION_TEMPLATES.items():
            score_ab, score_ba = self._relation_score(premise, subj, obj, templates)
            best = max(score_ab, score_ba)
            all_scores[relation_type] = round(best, 6)
            best_directions[relation_type] = "A->B" if score_ab >= score_ba else "B->A"

        best_relation = max(all_scores, key=lambda k: all_scores[k])
        best_confidence = all_scores[best_relation]

        if best_confidence < self.threshold:
            return {
                "relation_type": "no_explicit_relation",
                "confidence": round(best_confidence, 6),
                "all_scores": all_scores,
                "direction": best_directions[best_relation],
            }

        return {
            "relation_type": best_relation,
            "confidence": best_confidence,
            "all_scores": all_scores,
            "direction": best_directions[best_relation],
        }

    def score_pairs_batch(
        self, items: list[tuple[str, str, str]]
    ) -> list[dict]:
        """Score multiple (premise, subj, obj) tuples efficiently.

        Builds a flat list of all hypotheses, runs a single batched forward
        pass through the NLI model, then re-assembles the per-pair results.

        Parameters
        ----------
        items:
            List of ``(premise, subj, obj)`` tuples.

        Returns
        -------
        list[dict]
            One result dict per input item (same order).
        """
        if not items:
            return []

        import torch

        n_relations = len(RELATION_TYPES)
        n_templates = len(next(iter(RELATION_TEMPLATES.values())))
        # Each item: n_relations × n_templates × 2 directions
        hypotheses_per_item = n_relations * n_templates * 2

        all_premises: list[str] = []
        all_hypotheses: list[str] = []

        for premise, subj, obj in items:
            for relation_type in RELATION_TYPES:
                templates = RELATION_TEMPLATES[relation_type]
                for tmpl in templates:
                    all_premises.append(premise)
                    all_hypotheses.append(tmpl.format(subj=subj, obj=obj))
                for tmpl in templates:
                    all_premises.append(premise)
                    all_hypotheses.append(tmpl.format(subj=obj, obj=subj))

        entailment_scores = self._batch_entailment(all_premises, all_hypotheses, batch_size=self.batch_size)

        results: list[dict] = []
        offset = 0
        for premise, subj, obj in items:
            all_scores: dict[str, float] = {}
            best_directions: dict[str, str] = {}

            for relation_type in RELATION_TYPES:
                n = n_templates
                ab_scores = entailment_scores[offset: offset + n]
                ba_scores = entailment_scores[offset + n: offset + 2 * n]
                offset += 2 * n

                score_ab = sum(ab_scores) / len(ab_scores)
                score_ba = sum(ba_scores) / len(ba_scores)
                best = max(score_ab, score_ba)
                all_scores[relation_type] = round(best, 6)
                best_directions[relation_type] = "A->B" if score_ab >= score_ba else "B->A"

            best_relation = max(all_scores, key=lambda k: all_scores[k])
            best_confidence = all_scores[best_relation]

            if best_confidence < self.threshold:
                results.append({
                    "relation_type": "no_explicit_relation",
                    "confidence": round(best_confidence, 6),
                    "all_scores": all_scores,
                    "direction": best_directions[best_relation],
                })
            else:
                results.append({
                    "relation_type": best_relation,
                    "confidence": best_confidence,
                    "all_scores": all_scores,
                    "direction": best_directions[best_relation],
                })

        return results

    def classify_space(self, premise: str) -> dict:
        """Classify the TH space of a sentence using zero-shot text classification.

        Spaces are properties of the ACTIVITY TYPE expressed in the sentence,
        independent of which entity pair co-occurs in it (Ranga & Etzkowitz 2013).
        Each candidate label is scored as: "This text is about {label}."

        Returns a dict with keys:
          ``th_space``            – winning space name
          ``th_space_confidence`` – entailment probability of the winner
          ``th_space_scores``     – dict mapping each space to its score
        """
        scores: dict[str, float] = {}
        for space, label in SPACE_LABELS.items():
            hypothesis = SPACE_HYPOTHESIS_TEMPLATE.format(label=label)
            scores[space] = round(self._score_hypothesis(premise, hypothesis), 6)
        best = max(scores, key=lambda k: scores[k])
        return {
            "th_space":            best,
            "th_space_confidence": scores[best],
            "th_space_scores":     scores,
        }

    def classify_spaces_batch(self, sentences: list[str]) -> list[dict]:
        """Classify TH spaces for a list of sentences (no entity pair needed).

        Single batched forward pass: each sentence is scored against all
        ``SPACE_LABELS`` using ``SPACE_HYPOTHESIS_TEMPLATE``.

        Parameters
        ----------
        sentences:
            List of sentence strings.

        Returns
        -------
        list[dict]
            One result dict per input sentence (same order as ``classify_space``).
        """
        if not sentences:
            return []

        all_premises:   list[str] = []
        all_hypotheses: list[str] = []

        for sentence in sentences:
            for space in SPACE_TYPES:
                hypothesis = SPACE_HYPOTHESIS_TEMPLATE.format(label=SPACE_LABELS[space])
                all_premises.append(sentence)
                all_hypotheses.append(hypothesis)

        entailment_scores = self._batch_entailment(
            all_premises, all_hypotheses, batch_size=self.batch_size
        )

        results: list[dict] = []
        offset = 0
        for _ in sentences:
            scores: dict[str, float] = {}
            for space in SPACE_TYPES:
                scores[space] = round(entailment_scores[offset], 6)
                offset += 1
            best = max(scores, key=lambda k: scores[k])
            results.append({
                "th_space":            best,
                "th_space_confidence": scores[best],
                "th_space_scores":     scores,
            })
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for NLI relation extraction. "
                "Install with: pip install transformers"
            ) from exc

        import torch

        print(f"[NLI] Loading model '{self.model_name}' on device '{self._device}' …")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        use_fp16 = "cuda" in self._device
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            dtype=torch.float16 if use_fp16 else torch.float32,
        )
        self._model.to(self._device)
        self._model.eval()
        # Resolve entailment label index from model config (varies across NLI models)
        label2id: dict = getattr(self._model.config, "label2id", {})
        entailment_key = next((k for k in label2id if k.lower() == "entailment"), None)
        if entailment_key is not None:
            self._entailment_idx = int(label2id[entailment_key])
        print(f"[NLI] Model loaded ({sum(p.numel() for p in self._model.parameters()):,} params), entailment_idx={self._entailment_idx}")

    def _score_hypothesis(self, premise: str, hypothesis: str) -> float:
        """Return the entailment probability for a single premise/hypothesis pair."""
        import torch

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0][self._entailment_idx].item())

    def _batch_entailment(
        self, premises: list[str], hypotheses: list[str], batch_size: int = 32
    ) -> list[float]:
        """Return entailment probabilities for a flat list of premise/hypothesis pairs."""
        import torch

        scores: list[float] = []
        for i in range(0, len(premises), batch_size):
            batch_p = premises[i: i + batch_size]
            batch_h = hypotheses[i: i + batch_size]

            inputs = self._tokenizer(
                batch_p,
                batch_h,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            scores.extend(probs[:, self._entailment_idx].tolist())

        return scores

    def _relation_score(
        self,
        premise: str,
        subj: str,
        obj: str,
        templates: list[str],
    ) -> tuple[float, float]:
        """Return (mean_score_A_to_B, mean_score_B_to_A) over all templates."""
        scores_ab = [
            self._score_hypothesis(premise, tmpl.format(subj=subj, obj=obj))
            for tmpl in templates
        ]
        scores_ba = [
            self._score_hypothesis(premise, tmpl.format(subj=obj, obj=subj))
            for tmpl in templates
        ]
        return sum(scores_ab) / len(scores_ab), sum(scores_ba) / len(scores_ba)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
