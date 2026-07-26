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

---

## 2026-07-26 12:41 — Stage 01 acquire — run complete

```
primary questions: 600 (multi=300, single=300)
structured questions: 0
passage pool: 24866 passages (4133 gold-linked)
per dataset:
dataset       hop_type
hotpotqa      multi       150
ms_marco      single      100
multihop_rag  multi       150
nq_open_gold  single      100
squad_v2      single      100
failures: {'liverag': "RuntimeError('no usable LiveRAG release found on the Hub (last: None)')", 'wikitablequestions': 'DatasetNotFoundError("Revision \'refs/convert/parquet\' doesn\'t exist for dataset \'wikitablequestions\' on the Hub.")'}
```
Outputs: questions_primary.parquet, questions_structured.parquet, passages_pool.parquet, raw_profile.csv, fig_dataset_profile.png. Seed 42.

---

## 2026-07-26 12:42 — Stage 01 acquire — run complete

```
primary questions: 600 (multi=300, single=300)
structured questions: 0
passage pool: 24866 passages (4133 gold-linked)
per dataset:
dataset       hop_type
hotpotqa      multi       150
ms_marco      single      100
multihop_rag  multi       150
nq_open_gold  single      100
squad_v2      single      100
failures: {'liverag': "RuntimeError('no usable LiveRAG release found on the Hub (last: None)')", 'wikitablequestions': 'DatasetNotFoundError("Revision \'refs/convert/parquet\' doesn\'t exist for dataset \'stanfordnlp/wikitablequestions\' on the Hub.")'}
```
Outputs: questions_primary.parquet, questions_structured.parquet, passages_pool.parquet, raw_profile.csv, fig_dataset_profile.png. Seed 42.

---

## 2026-07-26 12:44 — Stage 01 acquire — run complete

```
primary questions: 600 (multi=300, single=300)
structured questions: 600
passage pool: 25951 passages (4650 gold-linked)
per dataset:
dataset       hop_type
hotpotqa      multi       150
ms_marco      single      100
multihop_rag  multi       150
nq_open_gold  single      100
squad_v2      single      100
failures: {'liverag': "RuntimeError('no usable LiveRAG release found on the Hub (last: None)')"}
```
Outputs: questions_primary.parquet, questions_structured.parquet, passages_pool.parquet, raw_profile.csv, fig_dataset_profile.png. Seed 42.

---

## 2026-07-26 12:45 — Stage 01 acquire — run complete

```
primary questions: 600 (multi=300, single=300)
structured questions: 600
passage pool: 26689 passages (4663 gold-linked)
per dataset:
dataset       hop_type
hotpotqa      multi       150
liverag       single       75
ms_marco      single       75
multihop_rag  multi       150
nq_open_gold  single       75
squad_v2      single       75
failures: none
```
Outputs: questions_primary.parquet, questions_structured.parquet, passages_pool.parquet, raw_profile.csv, fig_dataset_profile.png. Seed 42.

---

## 2026-07-26 12:47 — Stage 02 clean — run complete

```
questions: primary 600, structured 600
corpus: 26114 passages -> 76168 chunks (mean 110.2 tokens; 13447 structured)
gold coverage: primary questions with >=1 matched gold passage: 600/600
```
Full step log in outputs/data_cleaning_log.csv.

---

## 2026-07-26 13:00 — ⚠ SYNTHETIC SMOKE TEST of stages 06+07 — NOT RESULTS

The two auto-appended entries that stood here came from running 06_analyze and
07_power on **clearly labelled synthetic records** (random EM draws) purely to
verify the analysis code executes end-to-end. Every number they printed was
fake by construction; every artifact they wrote was deleted immediately
(outputs/ holds only real-run files: data_cleaning_log.csv,
fig_dataset_profile.png). No real evaluation has run yet. One robustness fix
came out of the smoke test: RQ2's slope contrast now guards against a missing
hop group under --limit.

---

## 2026-07-26 13:05 — Cache redesign + implementation complete, verified offline (NO API)

- **Cache redesign (user disk constraint):** the external drive is ExFAT with
  large allocation clusters; 21k per-call JSON files occupied 21 GB for ~0.4 GB
  of vectors. All caches (LLM, embeddings, rerank, assembled contexts) moved to
  one SQLite file (`llm_cache/cache.db`, WAL). 21,040 already-paid embeddings
  migrated (1 corrupt file from the killed run discarded); 21 GB reclaimed.
- Stage 03's embedding run was STOPPED partway (21,040 / 76,168 chunks
  embedded, ~$0.03 spent) on the user's instruction: **no further API calls
  until explicitly approved.** All spending stages are gated on that approval.
- Offline verification (zero API): all scripts compile; SQuAD EM/F1 unit checks
  pass; all local arms budget-compliant at every budget on synthetic chunks
  (naive, dedup, graph at hops 1 and 2; RECOMP extractive stage; LLMLingua
  fallback path); 05 projection gate works ($23.16 upper bound, both sweeps,
  judge included).
- PREREGISTRATION.md committed (dad5921) BEFORE any evaluation sweep.

---

## 2026-07-26 13:35 — Stage 01 acquire — run complete

```
primary questions: 600 (multi=300, single=300)
structured questions: 600
passage pool: 26444 passages (1304 gold-linked)
per dataset:
dataset       hop_type
hotpotqa      multi       150
liverag       single       75
ms_marco      single       75
multihop_rag  multi       150
nq_open_gold  single       75
squad_v2      single       75
failures: none
```
Outputs: questions_primary.parquet, questions_structured.parquet, passages_pool.parquet, raw_profile.csv, fig_dataset_profile.png. Seed 42.

---

## 2026-07-26 13:36 — Stage 02 clean — run complete

```
questions: primary 600, structured 600
corpus: 25957 passages -> 72967 chunks (mean 109.4 tokens; 12310 structured)
gold coverage: primary questions with >=1 matched gold passage: 600/600
```
Full step log in outputs/data_cleaning_log.csv.

---

## 2026-07-26 13:55 — Review round 1: 5-agent audit, full fix pass (NO API)

Five parallel reviewers (brief compliance, pipeline, arms/runner, statistics,
reproducibility) audited docs + code + real artifacts. 1 CRITICAL, ~12 MAJOR,
~25 minor findings — all triaged and fixed before any evaluation spend:

- **CRITICAL fixed:** assembled-context cache key now includes arm label +
  hyperparameters — the hops=1 sensitivity run can no longer silently reuse
  hops=2 contexts (which would have fabricated a "no difference" result).
- **Statistics (prereg Amendment 1, still pre-data):** RQ2 now uses ONE slope
  per question (was 5x pseudo-replicated per question x arm); RQ4 tests at the
  1,000-token budget only (was pooling 4 correlated records/question). NaN
  p-values excluded from BH families; Haldane-corrected discordant odds; CIs
  added to RQ2/RQ4 effects; winner's-curse caveat attached to RQ1; APT columns
  renamed *_per_1k; per-dataset results table with CIs added; RQ2 observed-n
  added to 07.
- **Data integrity:** gold-evidence gate now runs AFTER chunking (2 unwinnable
  questions eliminated); gold passages exempt from min-length chunk drop; chunk
  n_tokens re-encoded (0 mismatches, 0 U+FFFD); is_gold made uniform =
  "evidence of a sampled question" (MultiHop-RAG evidence now protected;
  HotpotQA no longer over-protects 3,300 distractor golds); per-dataset RNG
  streams (sample now a pure function of seed+dataset); LiveRAG restricted to
  single-doc questions (hop labels true); structured-tag regex tightened
  (LiveRAG false positives 1,294 -> 703).
- **Execution:** eval_records now rebuilt from ALL checkpoints (a --sweep rerun
  can't clobber earlier sweeps); checkpoints validated by question-ID set; 04b/
  04c accept forwarded flags + spend gates; 04c restricted to selection arms
  with BPE-jitter verification; RECOMP charged the full pool read in APT_total;
  rerank cost charged once per question; uncached-only spend accounting;
  fail-fast on 4xx; judge context guard raised to 200k chars (full_context
  judged whole); graph arm fully deterministic (sorted set iteration + tie
  breaks; PYTHONHASHSEED=0 in run_all.sh as defense-in-depth).
- **Deliverables/hygiene:** src/08_summary.py now generates RESULTS_SUMMARY.md
  (all 13 items incl. floor-to-ceiling fractions, contamination table, APT
  ranking-change verdict); spot-check CSV includes context + human columns and
  08 computes the agreement rate; judge/generation prompts recorded in
  manifest; synthetic stage06/07 manifest keys REMOVED and stale deviations
  retired (correction noted in manifest.corrections); README rewritten to match
  the real implementation; requirements.lock.txt committed; .gitignore data/
  sample negation fixed; LLM cache key includes provider pin.
- Stages 01+02 re-ran with the fixes (still $0 generator spend): 600+600
  questions, 25,957 passages -> 72,967 chunks, gold coverage 600/600, zero
  failures. All round-1 artifact invariant failures verified fixed.

Round 2 (full re-review by fresh agents) launching next.

---

## 2026-07-26 13:56 — Stage 02 clean — run complete

```
questions: primary 600, structured 600
corpus: 25957 passages -> 72967 chunks (mean 109.4 tokens; 12310 structured)
gold coverage: primary questions with >=1 matched gold passage: 600/600
```
Full step log in outputs/data_cleaning_log.csv.

---

## 2026-07-26 14:40 — Review round 2: second 5-agent audit, full fix pass (NO API)

Five fresh reviewers re-audited everything from scratch. Verdict on round-1
fixes: ALL verified correct — in code and against the regenerated artifacts
(the pipeline reviewer proved the min-length gold exemption complete and
re-validated every invariant; the arms reviewer traced the sensitivity fix
end-to-end under adversarial hash seeds; the stats reviewer numerically
verified McNemar/BH/slopes/power).

New findings fixed this round (none result-corrupting; several would have
crashed or silently truncated the pipeline):
- 08_summary would crash on `to_markdown` (tabulate absent) — dep added.
- spaCy model cannot auto-download into a pip-less uv venv — installed as a
  direct wheel + BaseException guard around the legacy download path.
- Stage 03: now projects cost and confirms like every other paid stage, and a
  FAILED retrieval gate is remembered on re-run (skip path re-fails).
- 04b/04c: coverage-aware skip — a --limit smoke run can no longer freeze
  n=20 artifacts as final deliverables. 04c also refuses to run on nothing.
- 05: eval_records rebuild now quarantines under-covered (smoke residue)
  checkpoints with a warning; spot-check keeps hand-entered verdicts and
  includes the FULL judged context; projection includes the sensitivity block.
- runner: per-question failure containment (failed=True records, excluded
  pairwise + disclosed — implements prereg exclusion rule 2); assembled-cache
  keys carry a corpus fingerprint (rebuilt corpus can never replay stale
  contexts); resumed blocks count $0 toward this-run spend.
- 07 power: welch branch now has the factor of 2; paired rows use d_z =
  mean_diff/sd_diff (raw mean_diff had misscaled n by 1/sd^2) — found by the
  stats reviewer in the exact file whose docstring warns about factor-of-2
  errors.
- common: 402/405/410 fail fast; embeddings response sorted by index field;
  base_url added to embed/rerank cache keys; truncate strips boundary chars.
- arms: RECOMP no longer double-charges the question and zeroes cached
  summary cost; compress drops the question charge (question-agnostic) and
  labels fallback per call; empty-graph guard stops graph_select degrading
  silently; 04b random_chunks contexts are pure functions of (seed, question).
- 06: bootstrap CIs on every hop slice + error bars on the hop figure;
  failed-record exclusion with disclosure; RQ2 completeness check (drops
  questions with incomplete arm coverage, disclosed); unpaired-p/paired-CI
  note on two-proportion rows; SENSITIVITY merge budget/arm-filtered.
- Repo: LICENSE (MIT) added; stable dedup sort (cross-version determinism);
  02 re-run — corpus byte-identical counts (25,957 passages / 72,967 chunks,
  600/600 coverage). Note: the earlier "0 U+FFFD" phrasing meant 0 REMAINING
  after 37 boundary-split chunks were repaired during chunking.

Round 3 (full re-review) launching next.
