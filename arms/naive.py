"""Arm 1: naive top-k — fill the budget by descending retrieval similarity.
Also: naive_topk_dedup, the confound control for the graph arm (same duplicate
filter the graph expansion implies, but zero graph structure)."""

from __future__ import annotations

from . import finalize, greedy_fill


def naive_topk(question, candidates, budget, **ctx):
    ordered = sorted(candidates, key=lambda c: -c.score)
    text, ids = greedy_fill(ordered, budget)
    return finalize("naive_topk", text, ids, budget)


def _word_jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def naive_topk_dedup(question, candidates, budget, *, jaccard_threshold: float = 0.8, **ctx):
    ordered = sorted(candidates, key=lambda c: -c.score)
    kept = []
    for c in ordered:
        if any(_word_jaccard(c.text, k.text) >= jaccard_threshold for k in kept):
            continue
        kept.append(c)
    text, ids = greedy_fill(kept, budget)
    return finalize("naive_topk_dedup", text, ids, budget,
                    meta={"jaccard_threshold": jaccard_threshold,
                          "n_deduped": len(candidates) - len(kept)})
