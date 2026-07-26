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
> The full analysis plan was **pre-registered before any evaluation ran** — see
> [PREREGISTRATION.md](PREREGISTRATION.md); the lab journal is
> [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

---

## Research questions

| RQ  | Question | Primary test (see PREREGISTRATION.md) |
|-----|----------|----------|
| RQ1 | At a matched token budget, which assembly strategy is most accurate? | Two-proportion z (Cohen's h) + McNemar exact, best arm vs naive at 1,000 tokens |
| RQ2 | Does the accuracy-per-token trade-off differ on single-hop vs multi-hop questions? | Welch t on per-question F1-vs-log2(budget) slopes |
| RQ3 | Does graph-based selection beat flat similarity retrieval? | Two-proportion z (Cohen's h) + McNemar exact, graph vs naive at 1,000 tokens |
| RQ4 | How robust is each strategy on structured/table content? | Two-proportion z at 1,000 tokens, structured sweep vs prose (unpaired, disclosed) |

All test families are Benjamini–Hochberg FDR-corrected; effect sizes are reported with
bootstrap CIs; both the pre-registered tests and the paired-appropriate McNemar tests
are reported side by side.

## Context-assembly strategies (the independent variable)

1. **`naive_topk`** — concatenate the most similar chunks until the budget is full (baseline).
2. **`rerank_topk`** — cross-encoder rerank (Cohere rerank-v3.5 via OpenRouter), then fill.
3. **`compress_llmlingua`** — LLMLingua-2 token-level compression (question-agnostic, per the published method; labelled fallback if unavailable).
4. **`summarize_recomp`** — RECOMP-style extractive selection + abstractive summarization to budget.
5. **`graph_select`** — **the novel arm**: entity/similarity graph over chunks, seed-and-expand (1–2 hops), relevance × centrality scoring, connected-set fill.

Plus controls: `naive_topk_dedup` (isolates deduplication from graph structure), four
reference conditions (no-context floor, gold-context ceiling, random chunks,
uncapped full context), and a hops-1 sensitivity run of the graph arm.

Everything else is held constant: one generator, one retriever, one judge; token budget
swept over **{500, 1,000, 2,000, 4,000}** (fixed by the graded Synopsis).

## Stack (all models through one OpenRouter key)

- **Generator:** `qwen/qwen3-30b-a3b-instruct-2507` (open weights; pinned to one bf16 provider; temperature 0). Budgets enforced with **its own tokenizer** — token accounting is the object of study, so no proxy tokenizers.
- **Judge:** `openai/gpt-4o-mini` (fixed model + pinned prompt; cross-vendor from the generator).
- **Embeddings:** `baai/bge-m3` (1024-dim) via OpenRouter.
- **Vector store:** pgvector (Postgres 17 in podman) persists all vectors; query-time retrieval uses exact cosine over the same vectors (no ANN variance). BM25 (`rank_bm25`) as the sparse baseline.
- **Caching:** every paid call (generation, judging, embedding, rerank) is cached in SQLite keyed by content hash — interrupted runs resume without re-spending.

## Datasets (all public; no Kaggle data)

| Dataset | Role | Sampled |
|---------|------|---------|
| HotpotQA (distractor dev) | primary multi-hop | 150 |
| MultiHop-RAG | second multi-hop (news) | 150 |
| SQuAD v2 (dev) | single-hop | 75 |
| Natural Questions (nq_open_gold variant) | single-hop | 75 |
| MS MARCO v2.1 (dev) | single-hop | 75 |
| LiveRAG Benchmark | contamination control (post-cutoff) | 75 |
| WikiTableQuestions | structured/table content for RQ4 | 600 (separate sweep) |

Primary sample: **600 questions** (300 multi-hop / 300 single-hop, seed 42), the scale
the graded Synopsis commits to; the structured RQ4 sweep is a second powered 600.

## Metrics

- **Answer quality:** SQuAD-normalized Exact Match; token-level F1; judge-scored faithfulness and answer relevance.
- **Retrieval quality (validated before use):** doc-level recall@{1,5,10,20,50} + MRR, with a hard gate (HotpotQA dense recall@50 ≥ 0.7) before any arm runs.
- **Efficiency:** three token counters per record — generator input, assembly input, assembly output — giving **APT under two accountings**: `APT_generator` (Synopsis metric) and `APT_total` (honest end-to-end cost including what rerankers/compressors read). The Pareto frontier is plotted both ways; a ranking flip is a headline finding.

## Sample size / statistical power

From the Synopsis (α = 0.05, power = 0.80): N = max(170, 34, 356, 564) = **564**, met by
the 600-question pooled comparisons. Per-dataset slices (~75–150) are **exploratory
only**. `src/07_power.py` recomputes every requirement from the *observed* effect sizes
(two-proportion with the factor of 2; McNemar discordant-pair formula) and reports
assumed vs observed side by side.

## Repository layout

```
├── PREREGISTRATION.md        # analysis plan, committed before any evaluation
├── EXPERIMENT_LOG.md         # append-only lab journal (decisions + runs + reasons)
├── requirements.txt          # ranges; requirements.lock.txt = exact recorded versions
├── run_all.sh                # 01 -> 08 end to end (paid stages gated)
├── src/
│   ├── common.py             # seed, tokenizer, OpenRouter clients, SQLite caches, manifest
│   ├── 01_acquire.py         # datasets + seeded sampling + profile
│   ├── 02_clean.py           # cleaning log (graded), chunking 128/32
│   ├── 03_index.py           # embeddings -> pgvector + BM25; retrieval validation gate
│   ├── 04_arms.py            # chunk graph build + budget-compliance proof
│   ├── 04b_baselines.py      # floor / ceiling / random / full-context references
│   ├── 04c_position.py       # lost-in-the-middle ablation
│   ├── 05_run.py             # the sweeps (cost projection + confirmation)
│   ├── 06_analyze.py         # figures + pre-registered & McNemar tests + BH-FDR
│   ├── 07_power.py           # observed-effect sample sizes
│   └── 08_summary.py         # outputs/RESULTS_SUMMARY.md (the deliverable)
├── arms/                     # the five strategies + dedup control
├── data/                     # gitignored except data/sample/
└── outputs/                  # committed figures + CSVs the report cites
```

## Quickstart

```bash
uv venv --python 3.11 .venv && uv pip install -r requirements.txt --python .venv/bin/python
cp .env.example .env          # add OPENROUTER_API_KEY; never commit .env
podman run -d --name ragtb-pgvector -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=ragtb -p 127.0.0.1:5434:5432 -v ragtb_pgdata:/var/lib/postgresql/data \
  docker.io/pgvector/pgvector:pg17

./run_all.sh                  # full pipeline; 05 prints projected cost and asks first
.venv/bin/python src/05_run.py --limit 20 --yes    # smoke test
.venv/bin/python src/05_run.py --dry-run           # cost projection only
```

## Reproducibility

- Seed 42 with an independent RNG stream per dataset (the sample is a pure function of the seed).
- `PYTHONHASHSEED=0` in `run_all.sh`; all order-sensitive set iterations are sorted.
- Model IDs, provider pin, prompts, chunking parameters, and every deviation live in `data/manifest.json`.
- All datasets public; no Kaggle. Cost reported first in tokens, then dollars from API-returned usage.

## License

Code released under the MIT License. Each dataset remains under its own license and terms of use.

## Author

Esmaeil Abedi — QM640 Data Analytics Capstone, Walsh College.
