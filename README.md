# RAG Token-Budget: Comparing Context-Assembly Strategies Under Fixed Token Budgets

A reproducible benchmark that answers one question: **at a fixed token budget, which way of assembling retrieved context gives the best answer quality per token?**

Retrieval-augmented generation (RAG) improves answer accuracy by putting retrieved
text into the prompt, but every token costs money and latency — and past a point,
longer context *hurts* accuracy (the "lost in the middle" effect). This project holds
the generator and retriever fixed and compares five context-assembly strategies at
**matched token budgets**, producing an accuracy-per-token Pareto frontier and a
practitioner decision guide.

> Capstone project — QM640 Data Analytics Capstone, Walsh College.
> This is an experimental benchmarking study (comparative, not predictive): it applies
> ML/AI systems and evaluates them with formal statistical inference.

---

## Research questions

| RQ  | Question | Test |
|-----|----------|------|
| RQ1 | At a matched token budget, which assembly strategy is most accurate? | McNemar's paired test (α = 0.05) |
| RQ2 | Does the accuracy-per-token trade-off differ on single-hop vs multi-hop questions? | Two-proportion test, Cohen's h |
| RQ3 | Does graph-based selection beat flat similarity retrieval on the frontier? | McNemar's paired test (multi-hop) |
| RQ4 | How robust is each strategy on structured / code-like passages? | McNemar's paired test (structured subset) |

## Context-assembly strategies compared (the independent variable)

1. **Naive top-k** — concatenate the k most similar chunks until the budget is filled (baseline).
2. **Reranked top-k** — cross-encoder rerank, then fill the budget (strong baseline).
3. **Prompt compression** — LLMLingua-2 and LongLLMLingua.
4. **Summarization compression** — RECOMP (extractive + abstractive).
5. **Graph-based selection** — seed then expand/prune along graph edges (the novel arm).

Everything else is held constant: one fixed generator, one fixed retriever, token budget
swept over **{500, 1k, 2k, 4k, 8k}** as a controlled variable.

---

## Stack

Built from scratch on open components — no private data, no pre-existing system.

- **Gateway:** [OpenRouter](https://openrouter.ai) — a single API for embeddings, reranking, and generation.
- **Embeddings:** an open model (e.g. BGE-M3, 1024-dim) served via OpenRouter.
- **Vector store:** [pgvector](https://github.com/pgvector/pgvector) (dense) + **BM25** (sparse baseline).
- **Generator:** one fixed mid-size open-weight model served via OpenRouter (a second model as a robustness check).
- **Evaluation:** Exact Match, token-level F1, RAGAS (faithfulness / answer relevance / context precision), recall@k, MRR, tokens, cost, latency.

## Datasets (all public; no Kaggle data)

| Dataset | Role | Link |
|---------|------|------|
| HotpotQA | Primary multi-hop QA | https://hotpotqa.github.io/ |
| MultiHop-RAG | Multi-hop RAG over a news corpus | https://github.com/yixuantt/MultiHop-RAG |
| SQuAD | Single-hop reading comprehension | https://rajpurkar.github.io/SQuAD-explorer/ |
| Natural Questions | Single-hop contrast | https://ai.google.com/research/NaturalQuestions |
| MS MARCO | Single-hop passage QA | https://microsoft.github.io/msmarco/ |
| LiveRAG (SIGIR 2025) | Contamination-robust 2025 benchmark | https://arxiv.org/abs/2507.04942 |

---

## Metrics

- **Answer quality:** Exact Match, token-level F1; RAGAS faithfulness / answer relevance / context precision for free-form answers (single fixed judge).
- **Retriever quality:** recall@k, MRR (measured separately so it is not confused with generation quality).
- **Efficiency:** input tokens (the budget axis, provider-neutral), estimated cost (tokens × price), latency.
- **Headline composite:** Accuracy-per-Token, `APT(strategy, budget) = Quality / input_tokens`. A strategy Pareto-dominates another at a budget if it reaches at least the quality at no more tokens.

## Sample size / statistical power

- α = 0.05 (two-sided), power = 0.80.
- Paired McNemar tests (RQ1/RQ3/RQ4): `n = (z_{α/2}·√p_disc + z_β·√(p_disc − d²))² / d²`.
- To detect a 5-point Exact-Match gap (d = 0.05, p_disc = 0.30): **n = 940** questions.
- RQ2 (two-proportion, Cohen's h = 0.167): **n ≈ 564** per hop-type.
- RQ4 (structured subset, p_disc = 0.35): **n = 1,097**.
- **Final target: N = max(RQ1–RQ4) = 1,097** evaluation questions per dataset per arm.

---

## Repository layout

```
rag-token-budget/
├── README.md
├── requirements.txt
├── .env.example              # OPENROUTER_API_KEY, DATABASE_URL (pgvector)
├── config/
│   └── budgets.yaml           # token budgets, model IDs, retriever settings
├── data/
│   ├── loaders/               # dataset download + normalization
│   └── build_index.py         # chunk, embed (OpenRouter), load into pgvector + BM25
├── arms/
│   ├── naive_topk.py
│   ├── rerank_topk.py
│   ├── llmlingua.py           # LLMLingua-2 / LongLLMLingua
│   ├── recomp.py
│   └── graph_select.py        # novel graph-based selection arm
├── eval/
│   ├── run_experiment.py      # score every arm × budget × dataset
│   ├── metrics.py             # EM, F1, RAGAS, recall@k, MRR, APT
│   └── stats.py               # McNemar, two-proportion, bootstrap CIs
├── analysis/
│   ├── pareto_frontier.py     # accuracy-per-token curves
│   └── decision_guide.py      # per-budget dominance table
└── results/                   # per-question result tables (CSV/Parquet)
```

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure (copy and fill in your keys — never commit .env)
cp .env.example .env
#   OPENROUTER_API_KEY=...
#   DATABASE_URL=postgresql://.../ragbudget   # a pgvector-enabled Postgres

# 3. Build the corpus + index (chunk, embed via OpenRouter, load pgvector + BM25)
python data/build_index.py --dataset hotpotqa

# 4. Run the experiment across all arms and budgets
python eval/run_experiment.py --dataset hotpotqa --budgets 500 1000 2000 4000 8000

# 5. Build the accuracy-per-token frontier and decision guide
python analysis/pareto_frontier.py
python analysis/decision_guide.py
```

## Reproducibility

- All datasets are public; none are drawn from Kaggle.
- The retrieval and generation stack is built entirely on open components through OpenRouter,
  so the pipeline runs end to end with a single API key and a pgvector-enabled Postgres.
- Random seeds, model IDs, and the graph edge/expansion policy are fixed in `config/` before evaluation.
- Cost is reported first as input tokens (provider-neutral), then as dollars under a stated price.

## Status

Capstone in progress. The four flat arms (naive, reranked, compression, summarization) are the
guaranteed deliverable; the **graph-based selection arm** is the novel contribution, with a
mid-project go/no-go checkpoint on whether to give it first-class treatment or hold it as a stretch goal.

## License

Code released under the MIT License. Each dataset remains under its own license and terms of use.

## Author

Esmaeil Abedi — QM640 Data Analytics Capstone, Walsh College.
