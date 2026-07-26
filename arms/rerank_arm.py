"""Arm 2: cross-encoder rerank of the top ~100 candidates via OpenRouter
(cohere/rerank-v3.5), then fill the budget by rerank score.

Honest accounting: the reranker READS every candidate — that is assembly input
cost the generator-only APT never sees. Cohere rerank is priced per search
(~$2.00 / 1k searches), recorded per call; token cost is logged for APT_total.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import n_tokens, rerank  # noqa: E402

from . import finalize, greedy_fill

RERANK_COST_PER_SEARCH_USD = 0.002  # cohere rerank list price, recorded in the log


def rerank_topk(question, candidates, budget, **ctx):
    t0 = time.time()
    ranked, cached = rerank(question, [c.text for c in candidates])
    latency = time.time() - t0
    ordered = [candidates[i] for i, _score in ranked]
    text, ids = greedy_fill(ordered, budget)
    assembly_in = sum(c.n_tokens for c in candidates) + n_tokens(question)
    return finalize(
        "rerank_topk", text, ids, budget,
        assembly_input_tokens=assembly_in,
        assembly_latency_s=round(latency, 3),
        # per-search price is paid once per question, not once per budget:
        # cached reuses cost $0
        assembly_cost_usd=0.0 if cached else RERANK_COST_PER_SEARCH_USD,
        meta={"n_candidates": len(candidates), "rerank_cached": cached},
    )
