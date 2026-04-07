"""R5 — GPT-RE: entity-aware demo retrieval + gold label-induced reasoning.

Paper: Wan, Z., et al. (2023). GPT-RE: In-context Learning for Relation Extraction
       using Large Language Models. EMNLP 2023.

Method (from paper, exactly):

Stage 1 — Entity-aware demonstration retrieval:
  - Embed each demo candidate using entity-prompted SimCSE:
      "The relation between [E1] {e1} [/E1] and [E2] {e2} [/E2] in context: {sentence}"
  - Model: princeton-nlp/sup-simcse-roberta-large
  - For each test example, retrieve k=4 nearest demos by cosine similarity
  - Entity-type constraint (key paper contribution): prefer demos whose (h1, h2) helix
    pair matches the target; fill remaining slots with globally nearest

Stage 2 — Gold label-induced reasoning (key paper contribution):
  - For each retrieved demo, generate a reasoning sentence:
      "What are the clues that lead to the relation between {e1} and {e2} to be
       {true_label} in the sentence '{sentence}'? It is because: ..."
  - Cache reasonings in outputs/reasoning_cache.json to avoid recomputation
  - Each demo in the final ICL prompt includes: sentence, typed entities, reasoning, label

Final classification:
  - Assemble ICL prompt with k=4 demos + test sentence
  - GPT-4 outputs relation label directly (no reasoning step needed at test time)
  - Parse output → valid label; fallback to no_explicit_relation

Run:
    export OPENAI_API_KEY=...
    python Experiments/Relation/gpt_re/run.py

Requires:
    pip install openai scikit-learn transformers torch
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from eval_utils import (
    RELATION_LABELS,
    load_relation_eval,
    mark_entities,
    mark_entities_typed,
    save_outputs,
)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
ANNOTATION_DEMOS = Path(__file__).parent.parent / "relation_labels.json"

SIMCSE_MODEL   = "princeton-nlp/sup-simcse-roberta-large"
GPT_MODEL      = "gpt-4o"
K_DEMOS        = 4
REASONING_CACHE_FILE = Path(__file__).parent / "outputs" / "reasoning_cache.json"

RELATION_DEFS = {
    "technology_transfer":             "market or non-market transfer of technology, knowledge, or IP — the core activity of innovation systems",
    "collaborative_leadership":        "an innovation organizer exercises convening power to bring spheres together, coordinating top-down and bottom-up processes",
    "substitution":                    "one sphere fills gaps when another is weak; one actor takes the role of the other",
    "networking":                      "formal or informal networks — a manifestation of the collective nature of science, technology and innovation",
    "collaboration_conflict_moderation":"triadic moderation of tensions and conflicts, turning diverging interests into convergence and win-win outcomes",
    "no_explicit_relation":            "no Triple Helix relationship is described",
}


# ── SimCSE embedding ──────────────────────────────────────────────────────────

def _load_simcse():
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SIMCSE_MODEL)
    model = AutoModel.from_pretrained(SIMCSE_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def _embed(texts: list[str], tokenizer, model, device: str) -> torch.Tensor:
    """Encode a list of texts to pooled embeddings (SimCSE-style: CLS token)."""
    batch = tokenizer(texts, padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True, return_dict=True)
        embeddings = out.last_hidden_state[:, 0]  # CLS token
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings.cpu()


def _entity_query(e1: str, e2: str, sentence: str) -> str:
    """Entity-prompted SimCSE query format (Wan et al. 2023)."""
    return f"The relation between [E1] {e1} [/E1] and [E2] {e2} [/E2] in context: {sentence}"


# ── Reasoning cache ───────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if REASONING_CACHE_FILE.exists():
        return json.loads(REASONING_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    REASONING_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    REASONING_CACHE_FILE.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _generate_reasoning(client, demo: dict, cache: dict) -> str:
    """Gold label-induced reasoning (Stage 2, Wan et al. 2023)."""
    cache_key = f"{demo.get('doc_id')}|{demo['entity_1']}|{demo['entity_2']}"
    if cache_key in cache:
        return cache[cache_key]

    e1, e2 = demo["entity_1"], demo["entity_2"]
    label  = demo["true_relation"]
    sent   = demo.get("sentence") or demo.get("central_sent_text", "")
    prompt = (
        f"What are the clues that lead to the relation between {e1} and {e2} "
        f"to be {label} in the sentence '{sent}'? It is because: "
    )
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.0,
    )
    reasoning = "It is because: " + (response.choices[0].message.content or "").strip()
    cache[cache_key] = reasoning
    return reasoning


# ── Demo selection ────────────────────────────────────────────────────────────

def _select_demos(
    target: dict,
    demos: list[dict],
    demo_embeddings: torch.Tensor,
    tokenizer,
    model,
    device: str,
    k: int = K_DEMOS,
) -> list[int]:
    """Entity-aware retrieval: prefer demos with matching (h1,h2) helix pair."""
    target_query = _entity_query(
        target["entity_1"], target["entity_2"],
        target.get("sentence") or target.get("central_sent_text", "")
    )
    target_emb = _embed([target_query], tokenizer, model, device)  # (1, D)
    sims = (demo_embeddings @ target_emb.T).squeeze(-1)            # (N,)

    target_pair = frozenset({target["h1"], target["h2"]})

    # Sort by similarity; prefer matching helix pair
    ranked = sorted(range(len(demos)), key=lambda i: -sims[i].item())
    matched   = [i for i in ranked if frozenset({demos[i]["h1"], demos[i]["h2"]}) == target_pair]
    unmatched = [i for i in ranked if i not in set(matched)]

    selected = (matched + unmatched)[:k]
    return selected


# ── Final prompt assembly ─────────────────────────────────────────────────────

def _build_prompt(target: dict, selected_demos: list[dict], reasonings: list[str]) -> str:
    rel_defs = "\n".join(f"- {lbl}: {defn}" for lbl, defn in RELATION_DEFS.items())
    system = (
        "You are a relation extraction expert for science and technology policy analysis "
        "using the Triple Helix framework (Ranga & Etzkowitz 2013).\n\n"
        f"Relation types:\n{rel_defs}"
    )

    demo_blocks = []
    for demo, reasoning in zip(selected_demos, reasonings):
        sent = mark_entities_typed(
            demo.get("sentence") or demo.get("central_sent_text", ""),
            demo["entity_1"], demo["h1"], demo["entity_2"], demo["h2"]
        )
        demo_blocks.append(
            f"Sentence: {sent}\n"
            f"Entities: <{demo['h1']}>{demo['entity_1']}</{demo['h1']}> and "
            f"<{demo['h2']}>{demo['entity_2']}</{demo['h2']}>\n"
            f"Reasoning: {reasoning}\n"
            f"Relation: {demo['true_relation']}"
        )

    test_sent = mark_entities_typed(
        target.get("sentence") or target.get("central_sent_text", ""),
        target["entity_1"], target["h1"], target["entity_2"], target["h2"]
    )
    test_block = (
        f"Sentence: {test_sent}\n"
        f"Entities: <{target['h1']}>{target['entity_1']}</{target['h1']}> and "
        f"<{target['h2']}>{target['entity_2']}</{target['h2']}>\n"
        f"Relation:"
    )

    demos_text = "\n\n---\n\n".join(demo_blocks)
    user = f"{demos_text}\n\n---\n\n{test_block}"
    return system, user


def _parse_label(text: str) -> str:
    text = text.strip().strip("*").lower()
    for lbl in RELATION_LABELS:
        if text == lbl or text.startswith(lbl):
            return lbl
    for lbl in RELATION_LABELS:
        if lbl in text:
            return lbl
    return "no_explicit_relation"


# ── Main ──────────────────────────────────────────────────────────────────────

def predict(entries: list[dict], demos: list[dict]) -> tuple[list[str], list[str]]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    cache = _load_cache()

    # Build SimCSE embeddings for all demo candidates
    print("Building SimCSE embeddings for demo pool...")
    tokenizer, sim_model, device = _load_simcse()
    demo_queries = [
        _entity_query(d["entity_1"], d["entity_2"],
                      d.get("sentence") or d.get("central_sent_text", ""))
        for d in demos
    ]
    # Embed in batches of 64
    all_embs = []
    for i in range(0, len(demo_queries), 64):
        batch_embs = _embed(demo_queries[i:i+64], tokenizer, sim_model, device)
        all_embs.append(batch_embs)
    demo_embeddings = torch.cat(all_embs, dim=0)
    print(f"Embedded {len(demos)} demo candidates")

    # Pre-generate reasonings for all demo candidates (cached)
    print("Generating gold label-induced reasonings (Stage 2)...")
    for demo in demos:
        _generate_reasoning(client, demo, cache)
    _save_cache(cache)
    print("Reasonings cached.")

    true_labels, pred_labels = [], []

    for i, entry in enumerate(entries):
        # Select k=4 entity-aware demos from separate demo pool (relation_labels.json)
        selected_idx = _select_demos(entry, demos, demo_embeddings, tokenizer, sim_model, device)
        selected = [demos[j] for j in selected_idx]
        reasonings = [
            cache.get(f"{d.get('doc_id')}|{d['entity_1']}|{d['entity_2']}", "")
            for d in selected
        ]

        system, user = _build_prompt(entry, selected, reasonings)

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=32,
            temperature=0.0,
        )
        pred = _parse_label(response.choices[0].message.content or "")
        true_labels.append(entry["true_relation"])
        pred_labels.append(pred)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(entries)} done")

    return true_labels, pred_labels


def main() -> None:
    entries = load_relation_eval()
    print(f"Loaded {len(entries)} labeled relation examples (eval set)")

    if not ANNOTATION_DEMOS.exists():
        raise FileNotFoundError(
            f"Demo pool not found: {ANNOTATION_DEMOS}\n"
            "Fill in true_relation for each entry in Experiments/Relation/gpt_re/relation_labels.json"
        )
    demos = [
        e for e in json.loads(ANNOTATION_DEMOS.read_text(encoding="utf-8"))
        if e.get("true_relation", "").strip()
    ]
    print(f"Loaded {len(demos)} demo candidates from relation_labels.json")

    true_labels, pred_labels = predict(entries, demos)

    predictions = [
        {
            "id":   i,
            "true": t,
            "pred": p,
            "text": entries[i].get("sentence") or entries[i].get("central_sent_text", ""),
        }
        for i, (t, p) in enumerate(zip(true_labels, pred_labels))
    ]

    approach_dir = Path(__file__).parent
    save_outputs(approach_dir, predictions, true_labels, pred_labels, RELATION_LABELS)


if __name__ == "__main__":
    main()
