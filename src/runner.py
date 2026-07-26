"""Shared execution machinery for 05_run (sweeps), 04b (baselines) and
04c (position ablation): assemble -> generate -> score -> judge, with
block-level parquet checkpoints so an interrupted run resumes for free.

Generation prompt is pinned (recorded in the manifest): the budget governs the
CONTEXT tokens; the instruction/question overhead is identical across arms, so
arm comparisons are apples-to-apples. Both context tokens and the API-reported
prompt tokens are logged.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DATA,
    cache_get_json,
    cache_put_json,
    llm_generate,
    n_tokens,
    openrouter_client,
)
from metrics import exact_match, f1_score, judge_scores  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arms import assemble  # noqa: E402

GEN_SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "Reply with the shortest exact answer (a few words). No explanation, no punctuation "
    "beyond what the answer itself requires. If the context does not contain the answer, "
    "give your single best guess."
)
GEN_SYSTEM_NO_CONTEXT = (
    "Answer the question from your own knowledge. Reply with the shortest exact answer "
    "(a few words). No explanation."
)
GEN_MAX_TOKENS = 128

PARTIAL_DIR = DATA / "eval_partial"
PARTIAL_DIR.mkdir(exist_ok=True)

def build_prompt(question: str, context: str | None) -> tuple[str, str]:
    if context is None:
        return GEN_SYSTEM_NO_CONTEXT, f"Question: {question}\nAnswer:"
    return GEN_SYSTEM, f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def assemble_cached(q: dict, cands, budget: int, arm: str, extra: dict) -> dict:
    """Assembly results cached (SQLite) — llmlingua compute and RECOMP calls are
    reused by 04c and by re-runs."""
    key = f"{arm}_{budget}_{q['question_id']}"
    hit = cache_get_json("assembled", key)
    if hit is not None:
        return hit
    res = assemble(q["question"], cands, budget, arm, **extra)
    d = dict(text=res.text, chunk_ids=res.chunk_ids,
             gen_context_tokens=res.gen_context_tokens,
             assembly_input_tokens=res.assembly_input_tokens,
             assembly_output_tokens=res.assembly_output_tokens,
             assembly_latency_s=res.assembly_latency_s,
             assembly_cost_usd=res.assembly_cost_usd, meta=res.meta)
    cache_put_json("assembled", key, d)
    return d


def run_record(q: dict, context_text: str | None, *, sweep: str, arm: str, budget,
               assembly: dict | None, gold_in_pool: bool, client, with_judge=True) -> dict:
    system, prompt = build_prompt(q["question"], context_text)
    t0 = time.time()
    gen = llm_generate(prompt, system=system, max_tokens=GEN_MAX_TOKENS,
                       temperature=0.0, client=client)
    golds = [str(g) for g in q["gold_answers"]]
    rec = dict(
        question_id=q["question_id"], dataset=q["dataset"], hop_type=q["hop_type"],
        content_type=q["content_type"], sweep=sweep, arm=arm, budget=budget,
        gen_context_tokens=n_tokens(context_text) if context_text else 0,
        gen_input_tokens=gen.input_tokens, output_tokens=gen.output_tokens,
        assembly_input_tokens=assembly["assembly_input_tokens"] if assembly else 0,
        assembly_output_tokens=assembly["assembly_output_tokens"] if assembly else 0,
        latency_assembly_s=assembly["assembly_latency_s"] if assembly else 0.0,
        latency_gen_s=gen.latency_s,
        cost_gen_usd=gen.cost_usd,
        cost_assembly_usd=assembly["assembly_cost_usd"] if assembly else 0.0,
        predicted_answer=gen.text.strip(),
        gold_answer=json.dumps(golds),
        em=exact_match(gen.text, golds),
        f1=f1_score(gen.text, golds),
        gen_cached=gen.cached,
        retrieval_gold_in_pool=gold_in_pool,
        arm_meta=json.dumps((assembly or {}).get("meta", {})),
        wall_s=round(time.time() - t0, 3),
    )
    if with_judge:
        rec.update(judge_scores(q["question"], context_text or "", rec["predicted_answer"],
                                client=client))
    return rec


def run_block(questions: list[dict], *, sweep: str, arm: str, budget, ctx,
              workers: int = 8, with_judge=True, force=False) -> pd.DataFrame:
    """One (sweep, arm, budget) block with its own checkpoint parquet.
    Assembly runs serially (LLMLingua/MPS is not thread-safe); generation and
    judging run in parallel threads against cached HTTP clients."""
    ckpt = PARTIAL_DIR / f"{sweep}__{arm}__{budget}.parquet"
    if ckpt.exists() and not force:
        df = pd.read_parquet(ckpt)
        if len(df) >= len(questions):
            return df
    client = openrouter_client()

    assembled = []
    for q in questions:
        cands = ctx.candidates(q["question_id"])
        gold_ids = set(q.get("gold_passage_ids", []))
        gold_in_pool = any(ctx.chunk(c.chunk_id) and
                           c.chunk_id.rsplit("_c", 1)[0] in gold_ids for c in cands)
        extra = ctx.arm_ctx(q["question_id"]) if arm == "graph_select" else {}
        a = assemble_cached(q, cands, budget, arm, extra)
        assembled.append((q, a, gold_in_pool))

    def work(item):
        q, a, gip = item
        return run_record(q, a["text"], sweep=sweep, arm=arm, budget=budget,
                          assembly=a, gold_in_pool=gip, client=client,
                          with_judge=with_judge)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(work, assembled))

    df = pd.DataFrame(records)
    df.to_parquet(ckpt, index=False)
    return df
