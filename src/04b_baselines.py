"""Stage 04b — reference conditions that make the arm scores interpretable.

  no_context    closed-book floor: how much does retrieval add at all?
  gold_context  perfect-selector ceiling: gold supporting passages only
  random_chunks seeded random chunks at the 1000-token budget (chance control)
  full_context  all 100 candidates, no budget cap: what does the budget cost?

One generation per primary question each (~2,400 calls). Judged like the arms.
Writes data/baseline_records.parquet. Requires user-approved API spend.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, SEED, append_log, n_tokens, openrouter_client, update_manifest  # noqa: E402
from runner import run_record  # noqa: E402

BASELINES = DATA / "baseline_records.parquet"
RANDOM_BUDGET = 1000
SEP = "\n\n"


def hash_stable(s: str) -> int:
    """Process-independent string hash (builtin hash() is randomized)."""
    import hashlib

    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def build_context(cond: str, q: dict, ctx, rng=None) -> str | None:
    if cond == "no_context":
        return None
    if cond == "gold_context":
        return SEP.join(str(p) for p in q["gold_passages"])
    if cond == "full_context":
        return SEP.join(c.text for c in ctx.candidates(q["question_id"]))
    if cond == "random_chunks":
        # per-question stream: the random context is a pure function of
        # (SEED, question_id) — identical under --limit, reordering, or resume,
        # so LLM-cache reuse works across runs
        rng = np.random.default_rng(
            [SEED, abs(hash_stable(q["question_id"])) % 2**31])
        ids = rng.permutation(len(ctx.chunk_ids))
        parts, total = [], 0
        for j in ids:
            c = ctx.chunk(ctx.chunk_ids[j])
            if total + c.n_tokens + 2 > RANDOM_BUDGET:
                break
            parts.append(c.text)
            total += c.n_tokens + 2
        return SEP.join(parts)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    args, _ = ap.parse_known_args()

    from retrieval_ctx import RetrievalContext

    ctx = RetrievalContext()
    qp = pd.read_parquet(DATA / "questions_primary_clean.parquet").to_dict("records")
    if args.limit:
        qp = qp[: args.limit]

    # skip only if the existing file COVERS the requested questions for every
    # condition — a --limit smoke run must never freeze the full deliverable
    if BASELINES.exists() and not args.force:
        prev = pd.read_parquet(BASELINES)
        want = {q["question_id"] for q in qp}
        covered = all(
            want <= set(prev[prev.arm == cond]["question_id"])
            for cond in ["no_context", "gold_context", "random_chunks", "full_context"]
        )
        if covered:
            print("[04b] existing records cover all requested questions, skipping")
            return
        print(f"[04b] existing file covers fewer questions than requested "
              f"({prev.question_id.nunique()} < {len(want)}) — re-running")

    # projected spend gate (mirrors 05): full_context reads ~12.8k tokens/question
    n_calls = 4 * len(qp)
    est = (len(qp) * (90 + 500 + 1000 + 13000) * 0.10 / 1e6  # generation input
           + n_calls * (1500 * 0.15 + 60 * 0.60) / 1e6)      # judge, rough
    print(f"[04b] {n_calls} generations + judging, projected <= ${est * 3:.2f}")
    if not args.yes:
        resp = input("Proceed? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted before spending")
            return

    client = openrouter_client()

    frames = []
    for cond in ["no_context", "gold_context", "random_chunks", "full_context"]:
        t0 = time.time()
        items = [(q, build_context(cond, q, ctx)) for q in qp]

        def gold_pool_flag(q, _cond):
            # honest per-condition semantics: only conditions that draw from the
            # retrieval pool have a meaningful gold-in-pool flag
            if _cond != "full_context":
                return _cond == "gold_context"
            gold_ids = set(q.get("gold_passage_ids", []))
            return any(c.chunk_id.rsplit("_c", 1)[0] in gold_ids
                       for c in ctx.candidates(q["question_id"]))

        def work(item, _cond=cond):
            q, text = item
            budget = {"no_context": 0, "gold_context": -1,
                      "random_chunks": RANDOM_BUDGET, "full_context": -1}[_cond]
            try:
                rec = run_record(q, text, sweep="baseline", arm=_cond, budget=budget,
                                 assembly=None, gold_in_pool=gold_pool_flag(q, _cond),
                                 client=client)
                rec["failed"] = False
                return rec
            except Exception as e:  # containment, like run_block: disclose, don't die
                print(f"  RECORD FAILED baseline/{_cond} {q['question_id']}: {e!r}")
                import json as _json

                return dict(question_id=q["question_id"], dataset=q["dataset"],
                            hop_type=q["hop_type"], content_type=q["content_type"],
                            sweep="baseline", arm=_cond, budget=budget,
                            gen_context_tokens=0, gen_input_tokens=0, output_tokens=0,
                            assembly_input_tokens=0, assembly_output_tokens=0,
                            latency_assembly_s=0.0, latency_gen_s=0.0,
                            cost_gen_usd=0.0, cost_assembly_usd=0.0,
                            predicted_answer="",
                            gold_answer=_json.dumps([str(g) for g in q["gold_answers"]]),
                            em=float("nan"), f1=float("nan"), gen_cached=False,
                            retrieval_gold_in_pool=False, empty_context=False,
                            arm_meta=_json.dumps({"error": repr(e)[:300]}), wall_s=0.0,
                            failed=True, faithfulness=float("nan"),
                            answer_relevance=float("nan"), judge_cost_usd=0.0,
                            judge_cached=False, judge_parse_ok=False)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            recs = list(ex.map(work, items))
        df = pd.DataFrame(recs)
        frames.append(df)
        print(f"[04b] {cond:14s} n={len(df)} EM={df['em'].mean():.3f} "
              f"F1={df['f1'].mean():.3f} mean_ctx_tokens={df['gen_context_tokens'].mean():.0f} "
              f"({time.time()-t0:.0f}s)")

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(BASELINES, index=False)
    summary = out.groupby("arm")[["em", "f1", "gen_context_tokens"]].mean().round(3).to_string()
    update_manifest(stage04b=dict(n_records=len(out)))
    append_log("Stage 04b baselines", f"```\n{summary}\n```")


if __name__ == "__main__":
    main()
