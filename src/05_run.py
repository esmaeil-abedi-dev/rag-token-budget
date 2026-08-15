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
        if sweep == "primary":  # hops=1 sensitivity block rides on the primary sweep
            sens = n_q * ((1000 + 90) * PRICE_GEN_IN + 40 * PRICE_GEN_OUT
                          + (1000 + 260) * PRICE_JUDGE_IN + 60 * PRICE_JUDGE_OUT) / 1e6
            sweep_total += sens
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
    ap.add_argument("--arms", type=str, default=None,
                    help="comma-separated subset of arms to run this invocation "
                         "(e.g. defer compress_llmlingua to a GPU session); "
                         "omitted arms keep their existing checkpoints")
    args, _ = ap.parse_known_args()

    run_arms = SWEEP_ARMS
    if args.arms:
        requested = [a.strip() for a in args.arms.split(",") if a.strip()]
        bad = set(requested) - set(SWEEP_ARMS)
        if bad:
            raise SystemExit(f"unknown arms: {sorted(bad)}; valid: {SWEEP_ARMS}")
        run_arms = [a for a in SWEEP_ARMS if a in requested]
        print(f"[05] arm subset this invocation: {run_arms}")

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

    # Tier-1-first ordering inside each sweep: sort questions so HotpotQA/
    # MultiHop-RAG/SQuAD run before Tier 2/3 — an interruption costs breadth,
    # never the core comparison.
    tier = {"hotpotqa": 0, "multihop_rag": 0, "squad_v2": 0,
            "nq_open_gold": 1, "ms_marco": 1, "liverag": 2, "wikitablequestions": 0}
    q_primary.sort(key=lambda q: tier.get(q["dataset"], 3))

    def uncached_cost(df: pd.DataFrame) -> float:
        """This-run spend only: checkpoint-resumed blocks and cached calls
        replay their historical cost_usd but cost nothing now."""
        if "resumed_from_checkpoint" in df and df["resumed_from_checkpoint"].all():
            return 0.0
        live = df[~df.get("resumed_from_checkpoint", pd.Series(False, index=df.index))]
        gen = live.loc[~live["gen_cached"].astype(bool), "cost_gen_usd"].sum()
        judge = 0.0
        if "judge_cached" in live:
            judge = live.loc[~live["judge_cached"].astype(bool), "judge_cost_usd"].sum()
        # assembly replayed from the assembled cache costs nothing this run
        # (assembly_from_cache flag); fresh assemblies already zero their own
        # cached sub-calls (rerank_cached / summary_cached)
        fresh = live[~live.get("assembly_from_cache",
                               pd.Series(False, index=live.index)).astype(bool)]
        asm = float(fresh["cost_assembly_usd"].sum())
        return float(gen + judge + asm)

    for sweep, questions in (("primary", q_primary), ("structured", q_structured)):
        if sweep not in sweeps or not questions:
            continue
        for arm in run_arms:
            for budget in BUDGETS:
                t0 = time.time()
                df = run_block(questions, sweep=sweep, arm=arm, budget=budget,
                               ctx=ctx, workers=args.workers, force=args.force)
                block_cost = uncached_cost(df)
                spent += block_cost
                print(f"[05] {sweep:10s} {arm:22s} b={budget:5d}  n={len(df)}  "
                      f"EM={df['em'].mean():.3f} F1={df['f1'].mean():.3f}  "
                      f"${block_cost:.3f}  {time.time()-t0:.0f}s  (total ${spent:.2f})")

    # graph sensitivity: hyperparameter variants at the 1000 budget (expansion
    # hops AND the relevance/centrality weight alpha — instructor feedback asked
    # for wider exploration). Distinct arm_labels + arm_kwargs give each variant
    # its own assembly-cache keys — never reusing the hops=2/alpha=0.7 contexts.
    SENSITIVITY_VARIANTS = [
        ("sensitivity_h1", "graph_select_h1", {"hops": 1}),
        ("sensitivity_a05", "graph_select_a05", {"alpha": 0.5}),
    ]
    if (not args.skip_sensitivity and "primary" in sweeps and q_primary
        and "graph_select" in run_arms):
        for sweep_name, label, kwargs in SENSITIVITY_VARIANTS:
            df = run_block(q_primary, sweep=sweep_name, arm="graph_select",
                           budget=1000, ctx=ctx, workers=args.workers, force=args.force,
                           arm_label=label, arm_kwargs=kwargs)
            spent += uncached_cost(df)
            print(f"[05] sensitivity {label} b=1000 n={len(df)} EM={df['em'].mean():.3f}")

    # Rebuild eval_records from checkpoints on disk (so a later restricted run
    # can never clobber earlier sweeps out of the file) — but with a coverage
    # guard: a stale --limit smoke checkpoint for a block NOT touched this run
    # must be QUARANTINED, not silently reported as a real sweep.
    full_ids = {
        "primary": set(pd.read_parquet(DATA / "questions_primary_clean.parquet")["question_id"]),
        "structured": set(pd.read_parquet(DATA / "questions_structured_clean.parquet")["question_id"]),
    }
    full_ids["sensitivity_h1"] = full_ids["primary"]
    full_ids["sensitivity_a05"] = full_ids["primary"]
    # under --limit / --tier1-only, THIS run's reduced sets are the expectation
    # — but ONLY for sweeps this run actually executed. A sweep not executed
    # here keeps the FULL expectation: an empty this-run set would satisfy
    # `set() <= anything` and wave stale smoke checkpoints straight through.
    sens_ids = ({q["question_id"] for q in q_primary}
                if not args.skip_sensitivity else set())
    this_run_ids = {"primary": {q["question_id"] for q in q_primary},
                    "structured": {q["question_id"] for q in q_structured},
                    "sensitivity_h1": sens_ids, "sensitivity_a05": sens_ids}
    executed = {s for s in ("primary", "structured") if s in sweeps and this_run_ids[s]}
    if not args.skip_sensitivity and "primary" in executed and "graph_select" in run_arms:
        executed.update({"sensitivity_h1", "sensitivity_a05"})
    reduced = bool(args.limit or args.tier1_only)

    from runner import corpus_fingerprint

    fp_now = corpus_fingerprint()
    frames_ok, quarantined = [], []
    for p in sorted(PARTIAL_DIR.glob("*.parquet")):
        if p.name.startswith("._"):  # macOS AppleDouble junk from zip transfers
            continue
        parts = p.stem.split("__")
        if len(parts) >= 4 and parts[3] != fp_now:
            quarantined.append(f"{p.name} (stale corpus fingerprint)")
            continue
        cdf_ = pd.read_parquet(p)
        sweep_name = parts[0]
        if reduced and sweep_name in executed:
            expected = this_run_ids.get(sweep_name)
        else:
            expected = full_ids.get(sweep_name)
        if expected is None or expected <= set(cdf_["question_id"]):
            frames_ok.append(cdf_)
        else:
            quarantined.append(p.name)
    if quarantined:
        print(f"[05] WARNING: {len(quarantined)} under-covered checkpoints EXCLUDED "
              f"from eval_records (smoke residue): {quarantined}")
    all_df = pd.concat(frames_ok, ignore_index=True) if frames_ok else pd.DataFrame()
    if len(all_df):
        all_df.to_parquet(EVAL_RECORDS, index=False)
        print(f"[05] eval_records rebuilt from {len(frames_ok)} checkpoints: "
              f"{len(all_df)} records ({int(all_df.get('failed', pd.Series(dtype=bool)).sum())} failed, disclosed)")

    # judge spot-check sample (Synopsis promises ~50 hand-verified judgements
    # with a reported agreement rate). Includes the judged context so a human
    # CAN verify, plus empty columns for the human verdict; 08_summary computes
    # the agreement rate once they are filled in.
    spot_path = OUTPUTS / "judge_spot_check_sample.csv"
    existing_spot = pd.read_csv(spot_path) if spot_path.exists() else None
    has_human_work = (existing_spot is not None and "human_agrees" in existing_spot
                      and existing_spot["human_agrees"].astype(str).str.strip()
                      .isin(["yes", "no"]).any())
    if has_human_work:
        print("[05] judge_spot_check_sample.csv contains hand-entered verdicts — NOT overwriting")
    elif len(all_df):
        from common import cache_get_json
        from runner import corpus_fingerprint

        rng = np.random.default_rng(SEED)
        judged = all_df[(all_df["judge_parse_ok"] == True)  # noqa: E712
                        & all_df["sweep"].isin(["primary", "structured"])
                        & ~all_df.get("failed", False)]
        sample = judged.iloc[rng.permutation(len(judged))[:50]].copy()

        def ctx_full(row):
            # FULL context — the human verifies faithfulness against what the
            # judge actually saw, not a 2k-char stub
            a = cache_get_json(
                "assembled",
                f"{corpus_fingerprint()}_{row['arm']}_{row['budget']}_{row['question_id']}")
            return (a or {}).get("text", "")

        sample["context"] = sample.apply(ctx_full, axis=1)
        n_empty = int((sample["context"].str.strip() == "").sum())
        if n_empty:
            # a context-less spot check is unverifiable — fail LOUDLY, never
            # write a silently useless file (this happened once: a WAL-truncated
            # cache restore emptied the assembled table)
            print(f"[05] WARNING: {n_empty}/{len(sample)} spot-check contexts missing "
                  f"from the assembled cache — NOT writing the sample; rebuild the "
                  f"cache or use the deterministic reconstruction path")
            sample = sample.iloc[0:0]
        if len(sample):
            sample["human_faithfulness"] = ""  # to be filled by hand
            sample["human_agrees"] = ""        # yes/no, to be filled by hand
            sample[["question_id", "sweep", "arm", "budget", "context",
                    "predicted_answer", "gold_answer", "em", "f1", "faithfulness",
                    "answer_relevance", "human_faithfulness", "human_agrees"]].to_csv(
                spot_path, index=False)

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
