"""GPT-RE pipeline step: apply GPT-RE relation extraction to the full corpus.

Paper: Wan, Z., et al. (2023). GPT-RE: In-context Learning for Relation Extraction
       using Large Language Models. EMNLP 2023.

Method:
  Stage 1 — Entity-aware SimCSE demo retrieval (princeton-nlp/sup-simcse-roberta-large)
  Stage 2 — Gold label-induced reasoning (GPT-4o, cached)
  Final    — ICL prompt with k=4 demos → GPT-4o → relation label

Input:  data/processed/step3/cooccurrence.jsonl  (all co-occurrence pairs)
Output: data/processed/step3/cooccurrence_gpt_re.jsonl
        — same rows as input + field: relation_type_gpt_re

Demo pool: Experiments/Relation/relation_labels.json  (92 labeled pairs)
Cache:     data/processed/step3/gpt_re_reasoning_cache.json

Resume: already-processed pairs (matched by sentence+e1+e2) are skipped.
        Output file is written incrementally every SAVE_EVERY rows.

Run:
    export OPENAI_API_KEY=...
    python -m src.pipeline.gpt_re

Requires:
    pip install openai transformers torch
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from src import config
from src.utils import normalize_helix, to_str

SIMCSE_MODEL   = "princeton-nlp/sup-simcse-roberta-large"
GPT_MODEL      = "gpt-4o-mini"
K_DEMOS        = 4
SAVE_EVERY     = 50

_REPO_ROOT     = Path(__file__).resolve().parents[2]
DEMO_POOL_FILE = _REPO_ROOT / "Experiments" / "Relation" / "relation_labels.json"
GPT_RE_DIR     = config.STEP3_DIR / "gpt_re"
OUTPUT_FILE    = GPT_RE_DIR / "cooccurrence_gpt_re.jsonl"
CACHE_FILE     = GPT_RE_DIR / "reasoning_cache.json"

RELATION_LABELS = [
    "technology_transfer",
    "collaboration_conflict_moderation",
    "collaborative_leadership",
    "substitution",
    "networking",
    "no_explicit_relation",
]

RELATION_DEFS = {
    "technology_transfer":              "market or non-market transfer of technology, knowledge, or IP — the core activity of innovation systems",
    "collaborative_leadership":         "an innovation organizer exercises convening power to bring spheres together, coordinating top-down and bottom-up processes",
    "substitution":                     "one sphere fills gaps when another is weak; one actor takes the role of the other",
    "networking":                       "formal or informal networks — a manifestation of the collective nature of science, technology and innovation",
    "collaboration_conflict_moderation": "triadic moderation of tensions and conflicts, turning diverging interests into convergence and win-win outcomes",
    "no_explicit_relation":             "no Triple Helix relationship is described",
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
    batch = tokenizer(texts, padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True, return_dict=True)
        embeddings = out.last_hidden_state[:, 0]
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings.cpu()


def _entity_query(e1: str, e2: str, sentence: str) -> str:
    return f"The relation between [E1] {e1} [/E1] and [E2] {e2} [/E2] in context: {sentence}"


# ── Reasoning cache ───────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    GPT_RE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _generate_reasoning(client, demo: dict, cache: dict) -> str:
    cache_key = f"{demo.get('doc_id', '')}|{demo['entity_1']}|{demo['entity_2']}"
    if cache_key in cache:
        return cache[cache_key]

    e1, e2   = demo["entity_1"], demo["entity_2"]
    label    = demo["true_relation"]
    sent     = demo.get("sentence") or demo.get("central_sent_text", "")
    prompt   = (
        f"What are the clues that lead to the relation between {e1} and {e2} "
        f"to be {label} in the sentence '{sent}'? It is because: "
    )
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128,
                temperature=0.0,
            )
            break
        except Exception as exc:
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                wait = 10 * (attempt + 1)
                print(f"  [rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
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
) -> list[int]:
    target_query = _entity_query(
        target["entity_1"], target["entity_2"],
        target.get("sentence") or target.get("central_sent_text", "")
    )
    target_emb = _embed([target_query], tokenizer, model, device)
    sims = (demo_embeddings @ target_emb.T).squeeze(-1)

    target_pair = frozenset({
        normalize_helix(to_str(target.get("h1", ""))),
        normalize_helix(to_str(target.get("h2", ""))),
    })
    ranked    = sorted(range(len(demos)), key=lambda i: -sims[i].item())
    matched   = [i for i in ranked if frozenset({demos[i].get("h1", ""), demos[i].get("h2", "")}) == target_pair]
    unmatched = [i for i in ranked if i not in set(matched)]
    return (matched + unmatched)[:K_DEMOS]


# ── Prompt assembly ───────────────────────────────────────────────────────────

def _mark_typed(text: str, e1: str, h1: str, e2: str, h2: str) -> str:
    import re
    def _ins(t, entity, open_tag, close_tag):
        m = re.search(re.escape(entity), t, re.IGNORECASE)
        if m:
            return t[:m.start()] + open_tag + t[m.start():m.end()] + close_tag + t[m.end():]
        return t
    marked = _ins(text, e1, f"<{h1}>", f"</{h1}>")
    marked = _ins(marked, e2, f"<{h2}>", f"</{h2}>")
    return marked


def _build_prompt(target: dict, selected_demos: list[dict], reasonings: list[str]) -> tuple[str, str]:
    rel_defs = "\n".join(f"- {lbl}: {defn}" for lbl, defn in RELATION_DEFS.items())
    system = (
        "You are a relation extraction expert for science and technology policy analysis "
        "using the Triple Helix framework (Ranga & Etzkowitz 2013).\n\n"
        f"Relation types:\n{rel_defs}"
    )
    demo_blocks = []
    for demo, reasoning in zip(selected_demos, reasonings):
        sent = _mark_typed(
            demo.get("sentence") or demo.get("central_sent_text", ""),
            demo["entity_1"], demo.get("h1", ""), demo["entity_2"], demo.get("h2", "")
        )
        demo_blocks.append(
            f"Sentence: {sent}\n"
            f"Entities: <{demo.get('h1', '')}>{demo['entity_1']}</{demo.get('h1', '')}> and "
            f"<{demo.get('h2', '')}>{demo['entity_2']}</{demo.get('h2', '')}>\n"
            f"Reasoning: {reasoning}\n"
            f"Relation: {demo['true_relation']}"
        )
    h1 = to_str(target.get("h1", ""))
    h2 = to_str(target.get("h2", ""))
    test_sent = _mark_typed(
        target.get("sentence") or target.get("central_sent_text", ""),
        target["entity_1"], h1, target["entity_2"], h2
    )
    test_block = (
        f"Sentence: {test_sent}\n"
        f"Entities: <{h1}>{target['entity_1']}</{h1}> and <{h2}>{target['entity_2']}</{h2}>\n"
        f"Relation:"
    )
    user = "\n\n---\n\n".join(demo_blocks) + "\n\n---\n\n" + test_block
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

def run() -> None:
    from openai import OpenAI

    if not DEMO_POOL_FILE.exists():
        raise FileNotFoundError(f"Demo pool not found: {DEMO_POOL_FILE}")

    demos = [
        e for e in json.loads(DEMO_POOL_FILE.read_text(encoding="utf-8"))
        if e.get("true_relation", "").strip()
    ]
    print(f"Demo pool: {len(demos)} labeled entries")

    # Load full corpus
    all_rows: list[dict] = []
    with config.FILE_COOCCURRENCE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    all_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"Loaded {len(all_rows)} rows from cooccurrence.jsonl")

    # Filter to cross-helix pairs
    cross_helix = [
        r for r in all_rows
        if normalize_helix(to_str(r.get("h1", ""))) != normalize_helix(to_str(r.get("h2", "")))
        and normalize_helix(to_str(r.get("h1", "")))
    ]
    print(f"Cross-helix pairs: {len(cross_helix)}")

    # Resume: load already-processed output
    done_keys: set[tuple] = set()
    output_rows: dict[int, dict] = {}  # original index → output row
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "relation_type_gpt_re" in row:
                        key = (
                            row.get("sentence") or row.get("central_sent_text", ""),
                            row.get("entity_1", ""),
                            row.get("entity_2", ""),
                        )
                        done_keys.add(key)
                except json.JSONDecodeError:
                    pass
        print(f"Resume: {len(done_keys)} pairs already processed")

    # Build SimCSE embeddings for demo pool
    print("Building SimCSE embeddings for demo pool...")
    tokenizer, sim_model, device = _load_simcse()
    demo_queries = [
        _entity_query(d["entity_1"], d["entity_2"],
                      d.get("sentence") or d.get("central_sent_text", ""))
        for d in demos
    ]
    all_embs = []
    for i in range(0, len(demo_queries), 64):
        all_embs.append(_embed(demo_queries[i:i+64], tokenizer, sim_model, device))
    demo_embeddings = torch.cat(all_embs, dim=0)
    print(f"Embedded {len(demos)} demos")

    # Pre-generate reasonings (cached)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    cache = _load_cache()
    print(f"Pre-generating reasonings ({len(cache)} cached)...")
    for demo in demos:
        _generate_reasoning(client, demo, cache)
    _save_cache(cache)
    print("Reasonings ready.")

    # Process all cross-helix pairs
    to_process = [
        r for r in cross_helix
        if (r.get("sentence") or r.get("central_sent_text", ""),
            r.get("entity_1", ""), r.get("entity_2", "")) not in done_keys
    ]
    print(f"To process: {len(to_process)} pairs")

    results: list[dict] = []
    for i, row in enumerate(to_process):
        e1 = row.get("entity_1", "")
        e2 = row.get("entity_2", "")

        selected_idx = _select_demos(row, demos, demo_embeddings, tokenizer, sim_model, device)
        selected = [demos[j] for j in selected_idx]
        reasonings = [
            cache.get(f"{d.get('doc_id', '')}|{d['entity_1']}|{d['entity_2']}", "")
            for d in selected
        ]

        system, user = _build_prompt(row, selected, reasonings)

        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=32,
                    temperature=0.0,
                )
                break
            except Exception as exc:
                if "429" in str(exc) or "rate_limit" in str(exc).lower():
                    wait = 10 * (attempt + 1)
                    print(f"  [rate limit] waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        pred = _parse_label(response.choices[0].message.content or "")
        out_row = {**row, "relation_type_gpt_re": pred}
        results.append(out_row)
        time.sleep(1.5)

        if (i + 1) % 50 == 0:
            _flush(results, done_keys)
            results = []
            print(f"  {i+1}/{len(to_process)} done (saved)")
        elif (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(to_process)} done")

    if results:
        _flush(results, done_keys)

    print(f"[OK] GPT-RE complete → {OUTPUT_FILE}")


def _flush(rows: list[dict], done_keys: set) -> None:
    """Append new rows to output file and update done_keys."""
    GPT_RE_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            key = (
                row.get("sentence") or row.get("central_sent_text", ""),
                row.get("entity_1", ""),
                row.get("entity_2", ""),
            )
            done_keys.add(key)


if __name__ == "__main__":
    run()
