"""Scoring: official SQuAD normalization for EM and token-level F1,
plus the fixed-judge RAGAS-style faithfulness/relevance scorer."""

from __future__ import annotations

import json
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import JUDGE_MODEL, llm_generate  # noqa: E402

_ARTICLES_RX = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """Official SQuAD normalization: lowercase, strip punctuation/articles/extra ws."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES_RX.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, golds: list[str]) -> int:
    p = normalize_answer(prediction)
    return int(any(p == normalize_answer(g) for g in golds))


def f1_score(prediction: str, golds: list[str]) -> float:
    """Token-level F1 = 2PR/(P+R), max over gold answers (official SQuAD)."""
    best = 0.0
    p_toks = normalize_answer(prediction).split()
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not p_toks or not g_toks:
            best = max(best, float(p_toks == g_toks))
            continue
        g_counts: dict[str, int] = {}
        for t in g_toks:
            g_counts[t] = g_counts.get(t, 0) + 1
        p_counts: dict[str, int] = {}
        for t in p_toks:
            p_counts[t] = p_counts.get(t, 0) + 1
        overlap = sum(min(c, g_counts.get(t, 0)) for t, c in p_counts.items())
        if overlap == 0:
            continue
        prec = overlap / len(p_toks)
        rec = overlap / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return round(best, 4)


# ---------------------------------------------------------------- fixed judge

JUDGE_SYSTEM = (
    "You are a strict evaluation judge for retrieval-augmented QA. "
    "Return ONLY a JSON object, no prose."
)

JUDGE_PROMPT = """Question: {question}

Retrieved context (may be truncated):
{context}

Model answer: {answer}

Rate on [0.0, 1.0]:
- "faithfulness": fraction of the model answer's factual claims that are directly \
supported by the retrieved context (1.0 = fully grounded, 0.0 = unsupported).
- "answer_relevance": how directly the model answer addresses the question.

JSON only: {{"faithfulness": <float>, "answer_relevance": <float>}}"""

_JSON_RX = re.compile(r"\{[^{}]*\}")


def judge_scores(question: str, context: str, answer: str, *, client=None) -> dict:
    """One fixed-judge call (cached like every other paid call). The judge model
    and this exact prompt are pinned in the manifest — never varied mid-run."""
    ctx = context if len(context) < 24000 else context[:24000]  # char guard only
    res = llm_generate(
        JUDGE_PROMPT.format(question=question, context=ctx or "(no context)", answer=answer),
        model=JUDGE_MODEL, system=JUDGE_SYSTEM, max_tokens=60, temperature=0.0,
        client=client,
    )
    out = {"faithfulness": float("nan"), "answer_relevance": float("nan"),
           "judge_cost_usd": res.cost_usd, "judge_cached": res.cached,
           "judge_parse_ok": False}
    m = _JSON_RX.search(res.text)
    if m:
        try:
            d = json.loads(m.group(0))
            out["faithfulness"] = max(0.0, min(1.0, float(d.get("faithfulness"))))
            out["answer_relevance"] = max(0.0, min(1.0, float(d.get("answer_relevance"))))
            out["judge_parse_ok"] = True
        except (ValueError, TypeError, KeyError):
            pass
    return out
