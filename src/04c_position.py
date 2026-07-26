"""Stage 04c — lost-in-the-middle ablation (Liu et al. 2024, the study's own
motivating effect, here measured rather than cited).

Design: take the best-EM arm at the 1000-token budget (from eval_records),
hold each question's selected evidence set constant, and vary ONLY where the
gold chunk sits: start, middle, end. Questions whose selected set contains no
gold chunk are skipped and counted (an arm cannot position evidence it never
selected).

Writes outputs/position_ablation.csv + fig_position_effect.png.
Requires the primary sweep to exist. ~<=1,800 calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, OUTPUTS, append_log, cache_get_json, openrouter_client, update_manifest  # noqa: E402
from runner import run_record  # noqa: E402

POSITIONS = ["start", "middle", "end"]
ABLATION_BUDGET = 1000
OUT_CSV = OUTPUTS / "position_ablation.csv"
OUT_FIG = OUTPUTS / "fig_position_effect.png"
SEP = "\n\n"


def reorder(texts_gold: list[str], texts_other: list[str], where: str) -> str:
    if where == "start":
        parts = texts_gold + texts_other
    elif where == "end":
        parts = texts_other + texts_gold
    else:
        half = len(texts_other) // 2
        parts = texts_other[:half] + texts_gold + texts_other[half:]
    return SEP.join(parts)


# Arms whose cached chunk_ids ARE the selected evidence set. Compression arms
# store the full candidate pool and emit rewritten text, so "hold the evidence
# set constant and move the gold chunk" is undefined for them.
SELECTION_ARMS = ["naive_topk", "naive_topk_dedup", "rerank_topk", "graph_select"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    args, _ = ap.parse_known_args()

    # skip only when a previous run processed AT LEAST this many input
    # questions (recorded in the manifest) — a --limit smoke run must never
    # freeze the full deliverable
    import json as _json

    from common import MANIFEST_PATH, n_tokens
    from retrieval_ctx import RetrievalContext

    n_want = len(pd.read_parquet(DATA / "questions_primary_clean.parquet"))
    if args.limit:
        n_want = min(n_want, args.limit)
    if OUT_CSV.exists() and not args.force:
        prev_manifest = _json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
        n_prev = prev_manifest.get("stage04c", {}).get("n_input_questions", 0)
        if n_prev >= n_want:
            print(f"[04c] previous run covered {n_prev} input questions >= {n_want}, skipping")
            return
        print(f"[04c] previous run covered only {n_prev} < {n_want} input questions — re-running")

    ctx = RetrievalContext()
    ev = pd.read_parquet(DATA / "eval_records.parquet")
    prim = ev[(ev.sweep == "primary") & (ev.budget == ABLATION_BUDGET)
              & (ev.arm.isin(SELECTION_ARMS))]
    if not len(prim):
        raise SystemExit("[04c] no primary records for selection arms — run 05 first")
    best_arm = prim.groupby("arm")["em"].mean().idxmax()
    overall_best = (ev[(ev.sweep == "primary") & (ev.budget == ABLATION_BUDGET)]
                    .groupby("arm")["em"].mean().idxmax())
    note = ("" if overall_best == best_arm else
            f" (overall best {overall_best} is a compression arm — position is "
            f"undefined for rewritten text, so the best SELECTION arm is ablated; disclosed)")
    print(f"[04c] ablating {best_arm} at {ABLATION_BUDGET}{note}")

    qp = pd.read_parquet(DATA / "questions_primary_clean.parquet").to_dict("records")
    if args.limit:
        qp = qp[: args.limit]

    from runner import corpus_fingerprint

    fp = corpus_fingerprint()  # keys are fingerprint-prefixed by the writer (runner.py)
    jobs, skipped_no_assembly, skipped_no_gold, skipped_over_budget = [], 0, 0, 0
    for q in qp:
        a = cache_get_json("assembled", f"{fp}_{best_arm}_{ABLATION_BUDGET}_{q['question_id']}")
        if a is None:
            skipped_no_assembly += 1
            continue
        gold_pids = set(q["gold_passage_ids"])
        gold_chunks = [cid for cid in a["chunk_ids"]
                       if cid.rsplit("_c", 1)[0] in gold_pids]
        other_chunks = [cid for cid in a["chunk_ids"] if cid not in gold_chunks]
        if not gold_chunks or not other_chunks:
            skipped_no_gold += 1
            continue
        tg = [ctx.chunk(c).text for c in gold_chunks]
        to = [ctx.chunk(c).text for c in other_chunks]
        variants = [(pos, reorder(tg, to, pos)) for pos in POSITIONS]
        # same chunks, same separators — only BPE-boundary jitter is possible;
        # verify rather than assume, and never truncate (that would change the
        # held-constant evidence set)
        if any(n_tokens(text) > ABLATION_BUDGET for _, text in variants):
            skipped_over_budget += 1
            continue
        for pos, text in variants:
            jobs.append((q, pos, text))

    print(f"[04c] {len(jobs)} generations "
          f"(skipped: {skipped_no_assembly} no cached assembly, "
          f"{skipped_no_gold} no gold in selected set, "
          f"{skipped_over_budget} BPE-jitter over budget)")
    if not jobs:
        raise SystemExit("[04c] nothing to ablate — run the primary sweep first "
                         "(all questions skipped; see counts above)")
    if not args.yes:
        resp = input(f"[04c] ~{len(jobs)} paid generations + judging. Proceed? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted before spending")
            return
    client = openrouter_client()
    skipped = skipped_no_assembly + skipped_no_gold + skipped_over_budget

    def work(job):
        q, pos, text = job
        r = run_record(q, text, sweep="position", arm=f"{best_arm}@{pos}",
                       budget=ABLATION_BUDGET, assembly=None, gold_in_pool=True,
                       client=client)
        r["position"] = pos
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(work, jobs))
    df = pd.DataFrame(recs)
    agg = df.groupby("position")[["em", "f1", "faithfulness"]].agg(["mean", "count"]).round(4)
    agg.to_csv(OUT_CSV)
    print(agg.to_string(), f"\n({time.time()-t0:.0f}s)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    means = df.groupby("position")["em"].mean().reindex(POSITIONS)
    sem = (df.groupby("position")["em"].std() / (df.groupby("position")["em"].count() ** 0.5)).reindex(POSITIONS)
    ax.bar(POSITIONS, means.values, yerr=1.96 * sem.values, capsize=6, color="tab:blue", alpha=0.8)
    ax.set_ylabel("Exact match")
    ax.set_title(f"Gold-evidence position vs accuracy — {best_arm}, {ABLATION_BUDGET}-token budget\n"
                 f"(evidence set held constant; n={len(df)//3} questions)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)

    df.to_parquet(DATA / "position_records.parquet", index=False)
    update_manifest(stage04c=dict(
        best_arm=best_arm, overall_best_arm=overall_best,
        disclosure=note.strip() or "best selection arm was also the overall best",
        n_input_questions=len(qp), n_questions=len(df) // 3,
        skipped_no_assembly=skipped_no_assembly,
        skipped_no_gold_in_selection=skipped_no_gold,
        skipped_over_budget=skipped_over_budget))
    append_log("Stage 04c position ablation",
               f"arm={best_arm} @ {ABLATION_BUDGET} tokens\n```\n{agg.to_string()}\n```\n"
               f"skipped: {skipped_no_assembly} no assembly, {skipped_no_gold} no gold "
               f"in selection, {skipped_over_budget} BPE jitter.")


if __name__ == "__main__":
    main()
