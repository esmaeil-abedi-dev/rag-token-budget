"""The context-assembly strategies under comparison.

Every arm implements:  assemble(question, candidates, budget, **ctx) -> AssembledContext
and MUST return a context whose real token count (generator tokenizer) is <= budget.
`assemble()` asserts this — a budget overrun is an experiment-invalidating bug,
never something to hide.

Token accounting per arm (feeds APT_total):
  assembly_input_tokens   tokens the arm itself read (reranker/compressor/summarizer)
  assembly_output_tokens  tokens the arm itself generated (RECOMP summaries)
Arms with no model of their own (naive, graph) have both = 0.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import Chunk, n_tokens  # noqa: E402

SEP = "\n\n"


@dataclass
class AssembledContext:
    arm: str
    text: str
    chunk_ids: list[str]
    gen_context_tokens: int
    assembly_input_tokens: int = 0
    assembly_output_tokens: int = 0
    assembly_latency_s: float = 0.0
    assembly_cost_usd: float = 0.0
    meta: dict = field(default_factory=dict)


def greedy_fill(ordered: list[Chunk], budget: int) -> tuple[str, list[str]]:
    """Concatenate chunks in the given order while the running total stays
    within budget; chunks that do not fit are skipped, later smaller ones may
    still fit. Separator tokens are counted for real, not estimated."""
    sep_tokens = n_tokens(SEP)
    parts: list[str] = []
    ids: list[str] = []
    total = 0
    for c in ordered:
        cost = c.n_tokens + (sep_tokens if parts else 0)
        if total + cost > budget:
            continue
        parts.append(c.text)
        ids.append(c.chunk_id)
        total += cost
    return SEP.join(parts), ids


def finalize(arm: str, text: str, chunk_ids: list[str], budget: int, **kw) -> AssembledContext:
    """Build the result and enforce the budget invariant with a real token count."""
    real = n_tokens(text)
    assert real <= budget, (
        f"BUDGET VIOLATION in {arm}: {real} tokens > budget {budget} — refusing to continue"
    )
    result = AssembledContext(arm=arm, text=text, chunk_ids=chunk_ids,
                              gen_context_tokens=real, **kw)
    if not text.strip():  # pathological but silent otherwise — flag it
        result.meta["empty_context"] = True
        print(f"  WARNING: {arm} produced an EMPTY context at budget {budget}")
    return result


def get_arm(name: str):
    from . import compress, graph, naive, recomp, rerank_arm

    registry = {
        "naive_topk": naive.naive_topk,
        "naive_topk_dedup": naive.naive_topk_dedup,
        "rerank_topk": rerank_arm.rerank_topk,
        "compress_llmlingua": compress.compress_llmlingua,
        "summarize_recomp": recomp.summarize_recomp,
        "graph_select": graph.graph_select,
    }
    return registry[name]


def assemble(question: str, candidates: list[Chunk], budget: int, arm: str, **ctx) -> AssembledContext:
    result = get_arm(arm)(question, candidates, budget, **ctx)
    assert result.gen_context_tokens <= budget  # double enforcement at the boundary
    return result
