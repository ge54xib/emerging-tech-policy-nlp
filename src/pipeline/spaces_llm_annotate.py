"""LLM-assisted TH space annotation using Claude API.

Workflow
--------
Stage 1 — Annotate with LLM:
    python -m src.pipeline.spaces_llm_annotate

    Classifies all unique cross-helix sentences in cooccurrence.jsonl using
    Claude. Writes data/processed/step3/spaces_llm_review.json with:
      - ``sentence``      central sentence text
      - ``entities``      cross-helix entities present
      - ``pair_space``    theoretically mapped space (from helix pair)
      - ``llm_space``     Claude's predicted space
      - ``llm_reasoning`` Claude's brief reasoning
      - ``verified``      set to "true" after human review to include in training

    Sentences already in spaces_labels.json (manual training set) are excluded.
    Supports resume: already-processed sentences are skipped on re-run.

Stage 2 — Review:
    Open spaces_llm_review.json, set ``verified``: ``"true"`` for correct entries.

Stage 3 — Train:
    python -m src.pipeline.spaces_setfit --train
    (automatically uses both spaces_labels.json and spaces_llm_review.json)

Valid labels: knowledge_space, innovation_space, consensus_space, public_space
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

SPACE_LABELS = ["knowledge_space", "innovation_space", "consensus_space", "public_space"]
MODEL_ID = "claude-haiku-4-5-20251001"
REQUESTS_PER_MINUTE = 40   # stay under rate limits (50 rpm hard limit)
RANDOM_SEED = 42

PAIR_TO_SPACE = {
    frozenset({"academia", "government"}):     "knowledge_space",
    frozenset({"academia", "industry"}):       "innovation_space",
    frozenset({"academia", "intermediary"}):   "innovation_space",
    frozenset({"industry", "intermediary"}):   "innovation_space",
    frozenset({"government", "industry"}):     "consensus_space",
    frozenset({"government", "intermediary"}): "consensus_space",
}

SYSTEM_PROMPT = """You are an expert annotator for science and innovation policy research.
You classify sentences from national quantum strategy documents into one of four
Triple/Quadruple Helix innovation spaces, based on Ranga & Etzkowitz (2013).

Space definitions:

knowledge_space — Activities that generate, share, or build scientific/technical knowledge.
  Assign when: conducting or funding research, scientific collaboration, joint labs,
  education/training in S&T, publishing results, building research capabilities.
  Signal words: research, R&D, science, knowledge, laboratory, experiment, training,
  skills, capability, infrastructure.
  Rule: "we will invest in quantum research" → consensus_space (governance act), NOT knowledge_space.

innovation_space — Activities that turn knowledge into commercial or societal value.
  Assign when: technology transfer, commercialisation, spin-offs, patents/IP, incubators,
  science parks, venture capital, procurement as demand-pull.
  Signal words: commercialise, transfer, spin-off, start-up, incubator, patent, license,
  IP, venture capital, market, product, science park, procurement.

consensus_space — Governance, coordination, and strategic agenda-setting.
  Assign when: policy development, national strategies/roadmaps, governance bodies,
  regulatory frameworks, standards, stakeholder dialogue, coordination across sectors,
  funding programmes as governance instruments.
  Signal words: strategy, policy, governance, coordinate, regulate, standard, framework,
  roadmap, advisory, council, ministry, agenda, dialogue, consultation, fund (as instrument).
  Distinction from knowledge_space: if describing WHAT R&D is done → knowledge_space;
  if describing HOW it is coordinated/governed → consensus_space.

public_space — Civil society engagement, societal concerns, democratic oversight.
  Assign when: public engagement, science communication, ethical considerations,
  responsible innovation, equity/inclusion, democratic oversight, public trust.
  Signal words: public, society, citizen, ethics, responsible, trust, equity, inclusion,
  diversity, awareness, engagement, societal, democratic.

Decision rules:
1. Focus on the ACTIVITY (verb), not the actors.
2. public_space > consensus_space when civil society is the explicit subject.
3. innovation_space > knowledge_space when commercialisation is the explicit activity.
4. If the sentence lists activities across spaces, pick the most prominent one.

Respond with JSON only: {"space": "<label>", "reasoning": "<one sentence>"}
Valid labels: knowledge_space, innovation_space, consensus_space, public_space"""


def _config():
    from src import config
    return config


def _llm_review_path() -> Path:
    return _config().STEP3_DIR / "spaces_llm_review.json"


def _pair_space(h1: str, h2: str) -> str:
    if "civil_society" in (h1, h2):
        return "public_space"
    return PAIR_TO_SPACE.get(frozenset({h1, h2}), "")


def _load_training_sentences() -> set[str]:
    """Return sentences already in the manual training set."""
    path = Path(__file__).parent.parent.parent / "Experiments" / "Spaces" / "spaces_labels.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(e.get("sentence", "")).strip() for e in data}


def _load_existing_results() -> dict[str, dict]:
    """Return successfully-processed sentences from a previous run (skip failed/empty)."""
    path = _llm_review_path()
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    # Only skip entries with a valid llm_space — retry failed ones on re-run
    return {str(e["sentence"]).strip(): e for e in data
            if str(e.get("llm_space", "")).strip() in SPACE_LABELS}


def _collect_candidates() -> list[dict]:
    """Collect unique cross-helix sentences from cooccurrence.jsonl."""
    cfg = _config()
    training_sents = _load_training_sentences()

    seen: set[tuple] = set()
    entities_by_key: dict[tuple, list[dict]] = defaultdict(list)
    rows_all: list[dict] = []

    with cfg.FILE_COOCCURRENCE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_all.append(row)

    for row in rows_all:
        key = (str(row["doc_id"]), int(row["paragraph_id"]), int(row["sentence_id"]))
        if row.get("h1") != row.get("h2"):
            for ef, hf in [("entity_1", "h1"), ("entity_2", "h2")]:
                ent, helix = row.get(ef, ""), row.get(hf, "")
                if ent:
                    entry = {"entity": ent, "helix": helix}
                    if entry not in entities_by_key[key]:
                        entities_by_key[key].append(entry)

    candidates: list[dict] = []
    for row in rows_all:
        if row.get("h1") == row.get("h2"):
            continue
        key = (str(row["doc_id"]), int(row["paragraph_id"]), int(row["sentence_id"]))
        if key in seen:
            continue
        seen.add(key)

        central = row.get("central_sent_text") or row.get("sent_text", "")
        if len(central) < 20:
            continue
        if central.strip() in training_sents:
            continue

        pair_space = _pair_space(row.get("h1", ""), row.get("h2", ""))
        candidates.append({
            "sentence":   central,
            "entities":   entities_by_key[key],
            "pair_space": pair_space,
            "doc_id":     str(row["doc_id"]),
            "country":    row.get("country", ""),
        })

    # Exclude evaluation set sentences (prevent train/eval leakage)
    eval_path = Path(__file__).parent.parent.parent / "evaluation" / "annotation_spaces.json"
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        eval_sents = {str(e.get("sentence", "")).strip() for e in eval_data}
        before = len(candidates)
        candidates = [c for c in candidates if c["sentence"].strip() not in eval_sents]
        print(f"[LLM] Excluded {before - len(candidates)} eval-set sentences from annotation pool")

    return candidates


def _classify_sentence(client, sentence: str) -> tuple[str, str]:
    """Call Claude and return (space, reasoning). Returns ('', '') on error."""
    try:
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=128,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": sentence}],
        )
        text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        space = parsed.get("space", "").strip()
        reasoning = parsed.get("reasoning", "").strip()
        if space not in SPACE_LABELS:
            return "", reasoning
        return space, reasoning
    except Exception as exc:
        print(f"  [warn] API error: {exc}")
        return "", ""


def main() -> None:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("anthropic SDK required. Install with: pip install anthropic") from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic(api_key=api_key)

    candidates = _collect_candidates()
    existing = _load_existing_results()

    to_process = [c for c in candidates if c["sentence"].strip() not in existing]
    already_done = len(candidates) - len(to_process)

    print(f"[LLM] {len(candidates)} unique cross-helix sentences")
    print(f"[LLM] {already_done} already processed (resuming), {len(to_process)} remaining")

    results: dict[str, dict] = dict(existing)
    interval = 60.0 / REQUESTS_PER_MINUTE

    for i, cand in enumerate(to_process):
        sent = cand["sentence"].strip()
        llm_space, llm_reasoning = _classify_sentence(client, sent)

        entry = {
            "doc_id":        cand["doc_id"],
            "country":       cand["country"],
            "sentence":      sent,
            "entities":      cand["entities"],
            "pair_space":    cand["pair_space"],
            "llm_space":     llm_space,
            "llm_reasoning": llm_reasoning,
        }
        results[sent] = entry

        if (i + 1) % 50 == 0 or (i + 1) == len(to_process):
            out = sorted(results.values(), key=lambda e: (e.get("llm_space") or ""))
            _llm_review_path().write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  [{i + 1}/{len(to_process)}] saved ({len(out)} total)")

        time.sleep(interval)

    # Final save and summary — sort by llm_space so user can browse class by class
    out = sorted(results.values(), key=lambda e: (e.get("llm_space") or ""))
    _llm_review_path().write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    from collections import Counter
    space_counts = Counter(e["llm_space"] for e in out)
    agree = sum(1 for e in out if e["llm_space"] and e["pair_space"] and
                e["llm_space"] == e["pair_space"])
    total_with_pair = sum(1 for e in out if e["pair_space"])

    print(f"\n[LLM] Done. Wrote {len(out)} entries to {_llm_review_path()}")
    print("[LLM] Space distribution (LLM predictions):")
    for space in SPACE_LABELS:
        print(f"  {space}: {space_counts[space]}")
    if total_with_pair:
        print(f"[LLM] Agreement with helix-pair mapping: {agree}/{total_with_pair} "
              f"({agree / total_with_pair * 100:.1f}%)")
    print('\nSet "verified": "true" for correct entries, then run --train.')


if __name__ == "__main__":
    main()
