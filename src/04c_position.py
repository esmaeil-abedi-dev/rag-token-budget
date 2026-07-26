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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if OUT_CSV.exists() and not args.force:
        print("[04c] exists, skipping")
        return

    from retrieval_ctx import RetrievalContext

    ctx = RetrievalContext()
    ev = pd.read_parquet(DATA / "eval_records.parquet")
    prim = ev[(ev.sweep == "primary") & (ev.budget == ABLATION_BUDGET)]
    best_arm = prim.groupby("arm")["em"].mean().idxmax()
    print(f"[04c] best arm at {ABLATION_BUDGET}: {best_arm}")

    qp = pd.read_parquet(DATA / "questions_primary_clean.parquet").to_dict("records")
    if args.limit:
        qp = qp[: args.limit]
    client = openrouter_client()

    jobs, skipped = [], 0
    for q in qp:
        a = cache_get_json("assembled", f"{best_arm}_{ABLATION_BUDGET}_{q['question_id']}")
        if a is None:
            skipped += 1
            continue
        gold_pids = set(q["gold_passage_ids"])
        gold_chunks = [cid for cid in a["chunk_ids"]
                       if cid.rsplit("_c", 1)[0] in gold_pids]
        other_chunks = [cid for cid in a["chunk_ids"] if cid not in gold_chunks]
        if not gold_chunks or not other_chunks:
            skipped += 1
            continue
        tg = [ctx.chunk(c).text for c in gold_chunks]
        to = [ctx.chunk(c).text for c in other_chunks]
        for pos in POSITIONS:
            jobs.append((q, pos, reorder(tg, to, pos)))

    print(f"[04c] {len(jobs)} generations ({skipped} questions skipped: no gold in selected set)")

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

    update_manifest(stage04c=dict(best_arm=best_arm, n_questions=len(df) // 3,
                                  skipped_no_gold=skipped))
    append_log("Stage 04c position ablation",
               f"arm={best_arm} @ {ABLATION_BUDGET} tokens\n```\n{agg.to_string()}\n```\n"
               f"{skipped} questions skipped (no gold chunk in the arm's selected set).")


if __name__ == "__main__":
    main()
