"""Stage 05 — the budget-matched sweeps.

Primary:    6 arms (5 Synopsis arms + naive_topk_dedup confound control)
            x 4 budgets {500,1000,2000,4000} x 600 questions = 14,400 generations
Structured: same shape on the ~600-item structured set (RQ4)          = 14,400
Sensitivity: graph_select with hops=1 at the 1000 budget (primary)    =    600

Every generation is judged (fixed judge) and scored EM/F1. Everything is
cached; blocks checkpoint to data/eval_partial/. Cost is projected and
CONFIRMED before spending.

Usage:
  05_run.py --dry-run                 projection only
  05_run.py --limit 20 --yes          smoke test
  05_run.py --sweep primary --yes     one sweep
  05_run.py --yes                     everything
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    ARMS,
    BUDGETS,
    DATA,
    GENERATOR_MODEL,
    JUDGE_MODEL,
    OUTPUTS,
    SEED,
    append_log,
    update_manifest,
)
from runner import GEN_MAX_TOKENS, PARTIAL_DIR, run_block  # noqa: E402

EVAL_RECORDS = DATA / "eval_records.parquet"

SWEEP_ARMS = ARMS + ["naive_topk_dedup"]

# $/M from the OpenRouter catalog on 2026-07-26 (recorded; actual costs are
# taken from API usage responses, this is only for the upfront projection)
PRICE_GEN_IN, PRICE_GEN_OUT = 0.10, 0.30
PRICE_JUDGE_IN, PRICE_JUDGE_OUT = 0.15, 0.60
PRICE_RERANK_SEARCH = 0.002


def load_questions(which: str) -> list[dict]:
    f = DATA / ("questions_primary_clean.parquet" if which == "primary"
                else "questions_structured_clean.parquet")
    df = pd.read_parquet(f)
    return df.to_dict("records")


def project_cost(n_primary: int, n_structured: int, sweeps: list[str]) -> float:
    total = 0.0
    lines = []
    for sweep, n_q in (("primary", n_primary), ("structured", n_structured)):
        if sweep not in sweeps or n_q == 0:
            continue
        n_gen = len(SWEEP_ARMS) * len(BUDGETS) * n_q
        avg_ctx = float(np.mean(BUDGETS))
        gen_cost = n_gen * ((avg_ctx + 90) * PRICE_GEN_IN + 40 * PRICE_GEN_OUT) / 1e6
        judge_cost = n_gen * ((avg_ctx + 260) * PRICE_JUDGE_IN + 60 * PRICE_JUDGE_OUT) / 1e6
        recomp_cost = len(BUDGETS) * n_q * (2 * avg_ctx * PRICE_GEN_IN + avg_ctx * PRICE_GEN_OUT) / 1e6
        rerank_cost = n_q * PRICE_RERANK_SEARCH
        sweep_total = gen_cost + judge_cost + recomp_cost + rerank_cost
        total += sweep_total
        lines.append(f"  {sweep:11s} {n_gen:6d} generations  "
                     f"gen ${gen_cost:5.2f} + judge ${judge_cost:5.2f} + "
                     f"recomp ${recomp_cost:5.2f} + rerank ${rerank_cost:5.2f} "
                     f"= ${sweep_total:6.2f}")
    print("\nProjected cost (upper bound, ignores cache hits):")
    print("\n".join(lines))
    print(f"  TOTAL ~ ${total:.2f}   (generator {GENERATOR_MODEL}, judge {JUDGE_MODEL})\n")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N questions per sweep")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--force", action="store_true", help="re-run completed blocks")
    ap.add_argument("--sweep", choices=["primary", "structured", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--tier1-only", action="store_true",
                    help="run only Tier 1 datasets (hotpotqa, multihop_rag, squad_v2) "
                         "as a complete first pass")
    args, _ = ap.parse_known_args()

    sweeps = ["primary", "structured"] if args.sweep == "both" else [args.sweep]
    q_primary = load_questions("primary")
    q_structured = load_questions("structured")
    if args.tier1_only:
        tier1 = {"hotpotqa", "multihop_rag", "squad_v2"}
        q_primary = [q for q in q_primary if q["dataset"] in tier1]
        q_structured = []
        print(f"[05] --tier1-only: {len(q_primary)} primary questions, structured deferred")
    if args.limit:
        q_primary, q_structured = q_primary[: args.limit], q_structured[: args.limit]

    total = project_cost(len(q_primary) if "primary" in sweeps else 0,
                         len(q_structured) if "structured" in sweeps else 0, sweeps)
    if args.dry_run:
        return
    if not args.yes:
        resp = input(f"Proceed with ~${total:.2f} of API spend? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted")
            return

    from retrieval_ctx import RetrievalContext

    ctx = RetrievalContext()
    t_start = time.time()
    spent = 0.0
    frames = []

    # Tier-1-first ordering inside each sweep: sort questions so HotpotQA/
    # MultiHop-RAG/SQuAD run before Tier 2/3 — an interruption costs breadth,
    # never the core comparison.
    tier = {"hotpotqa": 0, "multihop_rag": 0, "squad_v2": 0,
            "nq_open_gold": 1, "ms_marco": 1, "liverag": 2, "wikitablequestions": 0}
    q_primary.sort(key=lambda q: tier.get(q["dataset"], 3))

    def uncached_cost(df: pd.DataFrame) -> float:
        """This-run spend: cached generations/judgements replay their catalog
        cost_usd but cost nothing now — count only uncached rows."""
        gen = df.loc[~df["gen_cached"].astype(bool), "cost_gen_usd"].sum()
        judge = 0.0
        if "judge_cached" in df:
            judge = df.loc[~df["judge_cached"].astype(bool), "judge_cost_usd"].sum()
        asm = df["cost_assembly_usd"].sum()  # arms already zero cached reuses
        return float(gen + judge + asm)

    for sweep, questions in (("primary", q_primary), ("structured", q_structured)):
        if sweep not in sweeps or not questions:
            continue
        for arm in SWEEP_ARMS:
            for budget in BUDGETS:
                t0 = time.time()
                df = run_block(questions, sweep=sweep, arm=arm, budget=budget,
                               ctx=ctx, workers=args.workers, force=args.force)
                block_cost = uncached_cost(df)
                spent += block_cost
                frames.append(df)
                print(f"[05] {sweep:10s} {arm:22s} b={budget:5d}  n={len(df)}  "
                      f"EM={df['em'].mean():.3f} F1={df['f1'].mean():.3f}  "
                      f"${block_cost:.3f}  {time.time()-t0:.0f}s  (total ${spent:.2f})")

    # graph sensitivity: hops=1 at the 1000 budget (the brief's hyperparameter
    # run). Distinct arm_label + arm_kwargs give it its own assembly-cache keys
    # — it can NEVER silently reuse the hops=2 contexts.
    if not args.skip_sensitivity and "primary" in sweeps and q_primary:
        df = run_block(q_primary, sweep="sensitivity_h1", arm="graph_select",
                       budget=1000, ctx=ctx, workers=args.workers, force=args.force,
                       arm_label="graph_select_h1", arm_kwargs={"hops": 1})
        spent += uncached_cost(df)
        frames.append(df)
        print(f"[05] sensitivity graph hops=1 b=1000 n={len(df)} EM={df['em'].mean():.3f}")

    # Rebuild eval_records from ALL checkpoints on disk, not just this
    # invocation's frames — a later `--sweep structured` run must never clobber
    # the primary sweep out of the file.
    ckpts = sorted(PARTIAL_DIR.glob("*.parquet"))
    all_df = (pd.concat([pd.read_parquet(p) for p in ckpts], ignore_index=True)
              if ckpts else pd.DataFrame())
    if len(all_df):
        all_df.to_parquet(EVAL_RECORDS, index=False)
        print(f"[05] eval_records rebuilt from {len(ckpts)} checkpoints: {len(all_df)} records")

    # judge spot-check sample (Synopsis promises ~50 hand-verified judgements
    # with a reported agreement rate). Includes the judged context so a human
    # CAN verify, plus empty columns for the human verdict; 08_summary computes
    # the agreement rate once they are filled in.
    if len(all_df):
        from common import cache_get_json

        rng = np.random.default_rng(SEED)
        judged = all_df[(all_df["judge_parse_ok"] == True)  # noqa: E712
                        & all_df["sweep"].isin(["primary", "structured"])]
        sample = judged.iloc[rng.permutation(len(judged))[:50]].copy()

        def ctx_snippet(row):
            a = cache_get_json("assembled", f"{row['arm']}_{row['budget']}_{row['question_id']}")
            return (a or {}).get("text", "")[:2000]

        sample["context_snippet"] = sample.apply(ctx_snippet, axis=1)
        sample["human_faithfulness"] = ""  # to be filled by hand
        sample["human_agrees"] = ""        # yes/no, to be filled by hand
        sample[["question_id", "sweep", "arm", "budget", "context_snippet",
                "predicted_answer", "gold_answer", "em", "f1", "faithfulness",
                "answer_relevance", "human_faithfulness", "human_agrees"]].to_csv(
            OUTPUTS / "judge_spot_check_sample.csv", index=False)

    elapsed = (time.time() - t_start) / 60
    summary = (f"records: {len(all_df)}  |  actual spend this run: ${spent:.2f}  |  "
               f"{elapsed:.0f} min\n")
    if len(all_df):
        pivot = all_df.pivot_table(index="arm", columns="budget", values="em", aggfunc="mean").round(3)
        summary += f"\nEM by arm x budget:\n{pivot.to_string()}"
    print(summary)
    from runner import GEN_SYSTEM, GEN_SYSTEM_NO_CONTEXT
    from metrics import JUDGE_PROMPT, JUDGE_SYSTEM

    update_manifest(stage05=dict(n_records=int(len(all_df)),
                                 uncached_spend_this_run_usd=round(spent, 2),
                                 limit=args.limit, sweeps=sweeps,
                                 gen_max_tokens=GEN_MAX_TOKENS),
                    # the pinned prompts are control variables — record them
                    generation_system_prompt=GEN_SYSTEM,
                    generation_system_prompt_no_context=GEN_SYSTEM_NO_CONTEXT,
                    judge_system_prompt=JUDGE_SYSTEM,
                    judge_prompt_template=JUDGE_PROMPT)
    append_log(f"Stage 05 sweep ({'+'.join(sweeps)}, limit={args.limit})",
               f"```\n{summary}\n```")


if __name__ == "__main__":
    main()
