# Pre-registration — Answer Quality per Token

Committed **before** `05_run.py` executes any evaluation sweep. The commit hash and
timestamp of this file are the evidence that the analysis below was fixed before any
result was seen. (Stage 01–04 artifacts — sampling, cleaning, indexing, budget
compliance — involve no answer generation and no outcome data.)

**Amendment 1 (2026-07-26, still pre-data):** an independent statistical review of the
analysis code, conducted before any evaluation ran, found two unit-of-analysis errors in
the original wording's implementation: RQ2 slopes computed per question×arm (5×
pseudo-replication) and RQ4 pooling four correlated budget records per question into an
independence-assuming test. Both are corrected below: **RQ2 uses one slope per question**
(F1 averaged across arms per budget) and **RQ4 compares at the 1,000-token budget only**
(one record per question). No outcome data existed at amendment time — verifiable from
the git history (no eval_records, no sweep checkpoints).

## Design (fixed)

- 5 Synopsis arms (`naive_topk`, `rerank_topk`, `compress_llmlingua`,
  `summarize_recomp`, `graph_select`) + `naive_topk_dedup` confound control.
- Budgets: 500 / 1,000 / 2,000 / 4,000 context tokens (generator tokenizer:
  Qwen/Qwen3-30B-A3B-Instruct-2507; `add_special_tokens=False`).
- Primary sample: 600 questions (seed 42), stratified HotpotQA 150 / MultiHop-RAG 150 /
  SQuAD v2 75 / NQ-open-gold 75 / MS MARCO 75 / LiveRAG 75; hop-balanced 300/300.
- Structured sample (RQ4): 600 WikiTableQuestions items, same arms and budgets.
- Generator `qwen/qwen3-30b-a3b-instruct-2507` (CoreWeave bf16, pinned), temperature 0.
- Judge `openai/gpt-4o-mini`, fixed prompt (see `src/metrics.py`), temperature 0.
- Graph arm defaults: hops = 2, alpha = 0.7 (relevance weight); sensitivity run at
  hops = 1, budget 1,000. No other hyperparameter search will be performed.

## Metrics (exact definitions)

- **EM**: official SQuAD normalization (lowercase; strip punctuation, articles,
  extra whitespace); match against any gold answer.
- **F1**: token-level 2PR/(P+R), max over gold answers, same normalization.
- **Faithfulness / answer relevance**: fixed-judge scores in [0,1] (prompt pinned).
- **APT_generator** = EM / mean generator prompt tokens (Synopsis metric).
- **APT_total** = EM / mean (generator prompt + assembly input + assembly output tokens).

## Primary comparison per RQ (one each; everything else is secondary)

| RQ | Primary comparison | Test |
|----|--------------------|------|
| RQ1 | best Synopsis arm vs `naive_topk`, pooled primary sample, **budget 1,000** | two-proportion z (Cohen's h) + McNemar exact; the winner is selected on the same data (max-selection) — reported with that caveat attached |
| RQ2 | F1-per-log2(budget) slope, multi-hop vs single-hop; **one slope per question** (F1 averaged over the 5 arms at each budget, then least-squares slope) | Welch t on per-question slopes (paired t on question-level F1\@4000 vs F1\@500 within hop as secondary) |
| RQ3 | `graph_select` vs `naive_topk`, pooled primary sample, **budget 1,000** | two-proportion z (Cohen's h) + McNemar exact |
| RQ4 | `graph_select` structured vs prose, **at budget 1,000 only** (one record per question; pooling budgets would violate independence) | two-proportion z (unpaired: different question sets; content type is confounded with dataset — disclosed) |

- alpha = 0.05; **Benjamini–Hochberg FDR** within each (RQ × test) family;
  raw p, adjusted p, and family size all reported.
- Effect sizes with 95% CIs (bootstrap, 2,000 resamples for h; 10,000 paired
  resamples over questions for accuracy CIs) are reported alongside every p.

## Exclusion rules (fixed)

1. Questions dropped only by the pre-outcome cleaning rules already executed
   (empty gold answer; gold evidence absent from the cleaned corpus). No
   post-hoc exclusions based on model output.
2. Generation API failures after 6 retries: recorded as missing, question
   excluded pairwise from affected comparisons, count disclosed.
3. Judge parse failures: faithfulness recorded as NaN, row retained for EM/F1.

## What counts as a negative result (stated in advance)

- RQ1: no arm beats `naive_topk` with BH-adjusted p < 0.05 at the primary budget.
- RQ3: `graph_select` does not beat `naive_topk` (or beats it but
  `naive_topk_dedup` closes >= half the EM gap — then the honest finding is
  "deduplication, not graph structure").
- RQ4: no arm shows a structured-vs-prose drop with |h| >= 0.2.
- A negative result on any RQ is reported as a finding, not reframed.

## Disclosure

Every deviation from the graded Synopsis lives in `data/manifest.json` and
`outputs/RESULTS_SUMMARY.md`. Known at pre-registration time: NQ served via
`florin-hf/nq_open_gold` (full NQ is ~40 GB); LiveRAG via `LiveRAG/Benchmark`
(long-form answers — EM unreliable there, F1/faithfulness used for the
contamination check); MultiHop-RAG evaluation questions come from its train
split (the only split published).
