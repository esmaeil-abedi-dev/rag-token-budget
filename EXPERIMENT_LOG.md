# Experiment Log — Answer Quality per Token

Append-only lab journal for the QM640 capstone experiments. Every decision, its reason,
every stage run, and every result or failure gets an entry here. The machine-readable
counterpart is `data/manifest.json`; the report-facing summary is `outputs/RESULTS_SUMMARY.md`.

Conventions: newest entries at the bottom. Nothing is ever edited away — corrections get a
new entry referencing the old one. A deviation from the graded Synopsis is always marked
**DEVIATION** and mirrored in the manifest.

---

## 2026-07-26 — Design locked

### Study design (bound by the submitted Synopsis)

- 5 arms: `naive_topk`, `rerank_topk`, `compress_llmlingua`, `summarize_recomp`,
  `graph_select` (novel arm). Plus reference conditions (`no_context`, `gold_context`,
  `random_chunks`, `full_context`) and a `naive_topk_dedup` confound control.
- Token budgets: {500, 1000, 2000, 4000} input tokens — fixed by the Synopsis.
- Primary sample: **600 questions total** (not per dataset), stratified across datasets,
  balanced on hop type, seed 42 → 5 × 4 × 600 = 12,000 generator calls.
- Second powered sweep: ~600 structured/code questions for RQ4 (~12,000 more calls).
- Position ablation (lost-in-the-middle): best arm, one mid budget, gold evidence at
  start/middle/end (~1,800 calls).
- Pre-registration: `PREREGISTRATION.md` committed before `05_run.py` executes.
- Statistics: pre-registered two-proportion (Cohen's h) for RQ1/RQ3/RQ4 and paired t-test
  for RQ2, reported **alongside** McNemar exact (the paired-appropriate test) with
  discordant pairs; Benjamini–Hochberg FDR per RQ family; paired bootstrap 95% CIs
  (10k resamples, resampling questions); effect sizes with CIs lead the reporting.
- Metrics: SQuAD-normalized EM, token-level F1, RAGAS-style faithfulness (fixed judge),
  APT_generator = acc/gen_input_tokens (Synopsis metric) **and** APT_total =
  acc/(gen + assembly_input + assembly_output tokens) — dual accounting because rerank/
  compress/summarize arms read the ~12.8k-token candidate pool before the generator runs.

### Models (all via OpenRouter, one API key) — and why

| Role | Model | Reason |
|---|---|---|
| Generator (fixed, all runs) | `qwen/qwen3-30b-a3b-instruct-2507` | User's choice 2026-07-26: open weights → strongest reproducibility story for an academic report; cheapest viable ($0.048–0.10/M in); non-reasoning → stable latency/cost; exact budget tokenization via its public HF tokenizer. |
| Generator provider pin | CoreWeave (bf16, `allow_fallbacks: false`) | OpenRouter routes across providers with different quantizations (fp8/bf16); pinning one bf16 deployment removes provider variance from the experiment. |
| Judge (fixed, all runs) | `openai/gpt-4o-mini` | Synopsis requires one fixed judge as a control variable. Chosen cross-vendor from the generator so the judge never grades its own model family (self-preference bias). |
| Embeddings | `baai/bge-m3` (1024-dim) | Exactly the model the Synopsis names. Verified available on OpenRouter's embeddings endpoint at $0.01/M — so NO local substitution needed. |
| Reranker | `cohere/rerank-v3.5` | True cross-encoder reranking served via OpenRouter (fallback: `nvidia/llama-nemotron-rerank-vl-1b-v2:free`). |

Rejected generator alternatives, for the record: `gpt-4o-mini` (dated mid-2024; closed),
`gpt-4.1-mini` (3–8× cost, closed), `gemini-2.5-flash*` (tokenizer not public → budgets
would be approximate — disqualifying for a token-budget study), GPT-5.x family (reasoning
models: hidden reasoning tokens add cost/latency noise with no benefit for short-form QA).

### Tokenization

Budgets are enforced with the generator's own tokenizer
(`Qwen/Qwen3-30B-A3B-Instruct-2507` via `transformers.AutoTokenizer`,
`add_special_tokens=False` for context counting). tiktoken was dropped when the generator
moved off OpenAI models — a mismatched tokenizer would make every budget slightly wrong,
and budget exactness is the object of study.

### Infrastructure

- Machine: macOS (Darwin 25.0.0), 16 GiB RAM, no GPU. Python 3.11.13 in `.venv` (uv).
- Vector store: **pgvector 0.8.5** on Postgres 17 in a podman 5.8.2 container
  (`ragtb-pgvector`, port 127.0.0.1:5434, volume `ragtb_pgdata`) — satisfies the
  Synopsis's pgvector commitment; numpy fallback only if the container becomes unusable
  (would be a recorded DEVIATION).
- Sparse baseline: `rank_bm25`.
- All LLM/judge/embedding/rerank calls are disk-cached in `llm_cache/` keyed by
  sha256(prompt, model, params) → interrupted runs resume without paying twice.
- Seed 42 everywhere sampling occurs.

### Pinned package versions (Python 3.11.13)

```
datasets==5.0.0        pandas==3.0.5         pyarrow==25.0.0      numpy==2.4.6
rank-bm25==0.2.2       datasketch==2.0.0     psycopg==3.3.4       pgvector==0.5.0
openai==2.48.0         transformers==5.14.1  huggingface-hub==1.24.0
llmlingua==0.2.2       spacy==3.8.14         matplotlib==3.11.1   scipy==1.17.1
statsmodels==0.14.6    tokenizers==0.22.2    torch==2.13.0
```

(torch/llmlingua are for the LLMLingua-2 arm's local compressor — the published method
itself, not a served-model substitution.)

### Datasets (tiered; Tier 1 required)

- Tier 1: HotpotQA distractor dev (`hotpotqa/hotpot_qa`), MultiHop-RAG
  (`yixuantt/MultiHop-RAG`), SQuAD v2 dev (`rajpurkar/squad_v2`).
- Tier 2: Natural Questions (NQ-open variant — full NQ is ~40 GB; variant choice will be
  recorded), MS MARCO v2.1 dev (`microsoft/ms_marco`).
- Tier 3: LiveRAG (contamination control; availability risk — may require challenge
  registration; failure to load will be recorded, not hidden) + structured/code QA set
  for RQ4 (planned: WikiTableQuestions, plus `content_type=structured` tagging of
  table/code/list passages).
- No Kaggle data (course rule).

### Projected scale and cost (to be confirmed by 05_run's projection gate)

~61,000 API calls total ≈ $15–20 at Qwen pricing (generation ~$4, RECOMP assembly ~$3,
judge ~$8–12, embeddings+rerank <$1). Hard confirmation prompt before each sweep.

---

## 2026-07-26 — Repo scaffolded

- Git repo initialized earlier by user; `.gitignore` commit `ecb909d` (local briefs,
  `.env`, `data/`, caches ignored; `outputs/` committed because the report cites it).
- Created: `.env.example`, `requirements.txt`, `src/common.py` (seed, tokenizer, caches,
  OpenRouter clients, manifest), `run_all.sh`, dirs `src/ arms/ data/ outputs/ notebooks/`.
- pgvector container started and `CREATE EXTENSION vector` verified (0.8.5).
- Integration check passed (all live, ~$0.00001 spent):
  Qwen tokenizer loads and counts; generator via pinned CoreWeave answered correctly
  (20 in / 2 out tokens, cost reported by API); `baai/bge-m3` returned 1024-dim vectors;
  `cohere/rerank-v3.5` top-ranked the relevant document (score 0.903); pgvector 0.8.5
  reachable; judge `openai/gpt-4o-mini` responds. All four OpenRouter integrations are
  therefore real, not assumed.
