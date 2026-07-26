"""Arm 3: LLMLingua-2 token-level compression of the retrieved set to fit the budget.

Uses the published `llmlingua` package (LLMLingua-2 config). The compressor is a
local token-classification model — that IS the method, not a substitution.
Model: llmlingua-2-bert-base-multilingual (the smaller published config) on
MPS if available, else CPU; choice recorded in the run metadata.

If llmlingua is unusable at runtime, `FALLBACK_ACTIVE` flips on and a clearly
labelled sentence-level informativeness pruner runs instead (a reimplementation,
NOT LLMLingua proper) — the label propagates into every record and the manifest.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import n_tokens, truncate_to_tokens  # noqa: E402

from . import SEP, finalize

LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

_compressor = None
FALLBACK_ACTIVE = False
_fallback_recorded = False


def _record_fallback(reason: str) -> None:
    """The fallback label must reach the manifest, not just per-record meta."""
    global FALLBACK_ACTIVE, _fallback_recorded
    FALLBACK_ACTIVE = True
    if not _fallback_recorded:
        from common import update_manifest

        update_manifest(
            deviation=f"compress_llmlingua fallback (sentence pruning, NOT LLMLingua): {reason}"
        )
        _fallback_recorded = True


def _get_compressor():
    global _compressor
    if _compressor is not None or FALLBACK_ACTIVE:
        return _compressor
    try:
        import torch
        from llmlingua import PromptCompressor

        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        _compressor = PromptCompressor(
            model_name=LLMLINGUA_MODEL, use_llmlingua2=True, device_map=device
        )
        print(f"  [compress] LLMLingua-2 loaded on {device}")
    except Exception as e:
        print(f"  [compress] LLMLingua unavailable ({e!r}) — FALLBACK sentence pruner active")
        _record_fallback(f"install/load failure: {type(e).__name__}")
    return _compressor


_SENT_RX = re.compile(r"(?<=[.!?])\s+")


def _fallback_sentence_prune(question: str, joined: str, budget: int) -> str:
    """Documented fallback: sentence-level informativeness pruning (word-overlap
    with the question, tie-broken by position). NOT LLMLingua proper."""
    qwords = set(question.lower().split())
    sents = _SENT_RX.split(joined)
    scored = sorted(
        enumerate(sents),
        key=lambda t: (-len(qwords & set(t[1].lower().split())) / (len(t[1].split()) + 1e-9), t[0]),
    )
    picked, total = [], 0
    for pos, s in scored:
        st = n_tokens(s) + 1
        if total + st > budget:
            continue
        picked.append((pos, s))
        total += st
    picked.sort()  # restore document order
    return " ".join(s for _, s in picked)


def compress_llmlingua(question, candidates, budget, **ctx):
    ordered = sorted(candidates, key=lambda c: -c.score)
    joined = SEP.join(c.text for c in ordered)
    # pool read only: LLMLingua-2 is question-agnostic, so the question is
    # never part of what the compressor consumes
    assembly_in = sum(c.n_tokens for c in ordered)

    t0 = time.time()
    comp = _get_compressor()
    text = None
    if comp is not None:
        try:
            # NOTE: LLMLingua-2's compressor is question-AGNOSTIC — verified in
            # llmlingua 0.2.2, compress_prompt_llmlingua2 accepts no question.
            # target_token is measured in the compressor's own tokenizer and is
            # advisory; the real (Qwen) budget is enforced below.
            res = comp.compress_prompt(
                [c.text for c in ordered],
                target_token=budget,
                use_sentence_level_filter=False,
            )
            text = res["compressed_prompt"]
            method = f"llmlingua2:{LLMLINGUA_MODEL.split('/')[-1]} (question-agnostic)"
        except Exception as e:  # runtime failure (e.g. MPS OOM), not just load
            print(f"  [compress] LLMLingua runtime failure ({e!r}) — fallback for this call")
            _record_fallback(f"llmlingua runtime failure: {type(e).__name__}")
            text = None
    if text is None:
        text = _fallback_sentence_prune(question, joined, budget)
        method = "FALLBACK_sentence_prune (reimplementation, NOT LLMLingua)"
    latency = time.time() - t0

    if n_tokens(text) > budget:  # hard enforcement in real generator tokens
        text = truncate_to_tokens(text, budget)

    return finalize(
        "compress_llmlingua", text, [c.chunk_id for c in ordered], budget,
        assembly_input_tokens=assembly_in,
        assembly_latency_s=round(latency, 3),
        # per-call truth, not the sticky global: one runtime failure must not
        # mislabel later successful LLMLingua calls as fallbacks
        meta={"method": method, "fallback": method.startswith("FALLBACK")},
    )
