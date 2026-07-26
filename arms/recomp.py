"""Arm 4: RECOMP-style summarization (Xu et al. 2024): extractive sentence
selection followed by an abstractive summary that compresses the retrieved
passages to fit the budget.

Pipeline (recorded): (1) extractive — sentences scored by embedding-free lexical
relevance to the question, kept up to 2x budget; (2) abstractive — the fixed
generator model summarizes the extract down to <= budget tokens (its call is
assembly cost, logged separately and cached like everything else).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import llm_generate, n_tokens, truncate_to_tokens  # noqa: E402

from . import SEP, finalize

_SENT_RX = re.compile(r"(?<=[.!?])\s+")

SUMMARY_SYSTEM = (
    "You compress retrieved evidence for a question-answering system. "
    "Preserve every fact, name, number and date that could help answer the question. "
    "Do not add information. Output only the compressed evidence."
)


def _extractive(question: str, candidates, token_cap: int) -> str:
    qwords = set(question.lower().split())
    sents: list[tuple[int, str]] = []
    i = 0
    for c in sorted(candidates, key=lambda x: -x.score):
        for s in _SENT_RX.split(c.text):
            if s.strip():
                sents.append((i, s.strip()))
                i += 1
    scored = sorted(
        sents,
        key=lambda t: (-len(qwords & set(t[1].lower().split())) / (len(t[1].split()) + 1e-9), t[0]),
    )
    picked, total = [], 0
    for pos, s in scored:
        st = n_tokens(s) + 1
        if total + st > token_cap:
            if total >= token_cap * 0.9:
                break
            continue
        picked.append((pos, s))
        total += st
    picked.sort()
    return " ".join(s for _, s in picked)


def summarize_recomp(question, candidates, budget, **ctx):
    t0 = time.time()
    extract = _extractive(question, candidates, token_cap=2 * budget)
    prompt = (
        f"Question: {question}\n\nEvidence:\n{extract}\n\n"
        f"Compress the evidence to at most {budget} tokens while keeping everything "
        f"needed to answer the question."
    )
    res = llm_generate(prompt, system=SUMMARY_SYSTEM, max_tokens=budget, temperature=0.0)
    text = res.text.strip()
    truncated = False
    if n_tokens(text) > budget:
        text = truncate_to_tokens(text, budget)
        truncated = True
    latency = time.time() - t0
    return finalize(
        "summarize_recomp", text, [c.chunk_id for c in candidates], budget,
        assembly_input_tokens=res.input_tokens,
        assembly_output_tokens=res.output_tokens,
        assembly_latency_s=round(latency, 3),
        assembly_cost_usd=res.cost_usd,
        meta={"summary_truncated": truncated, "extract_tokens": n_tokens(extract),
              "summary_cached": res.cached},
    )
