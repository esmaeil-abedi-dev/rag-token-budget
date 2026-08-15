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

---

## 2026-07-26 14:09 — Review round 3 (in progress): fixes from first two verdicts (NO API)

Correction to the round-2 entry: its header was hand-stamped "14:40" but the
commit (05f2bcc) landed 13:58 — entries from here on use the actual clock.

Round-3 reviewers 1+2 (brief compliance; pipeline+repro) both returned
NOT CLEAN on the SAME single MAJOR: 04c read the assembled cache without the
round-2 corpus-fingerprint prefix (a propagation gap in my own round-2 fix) —
the position ablation would have deterministically found zero cached
assemblies after a successful sweep. Fixed; key format now verified identical
across the one writer and both readers.

Also fixed from their minors: corpus fingerprint now hashes chunk text, not
just positional IDs; 04b gained per-record failure containment (schema
key-identical to runner's, verified by reviewer 2); 06 prints a loud
SMOKE-SCALE warning (+ manifest note) when eval_records under-covers the
design, and guards RQ2 against missing budget columns; 04 prints its cost
projection even under --yes; 402/4xx fail-fast added to embed/rerank retry
loops (was llm-only); 03's skip path now also refuses to skip when outputs
exist without a manifest record; .gitignore un-ignores future
data/sample/*.parquet.

Embedding-cache bookkeeping, for the audit trail (reviewer 2 verified all of
this bitwise): round 2 added base_url to the embed cache key, which made the
21,040 pre-paid old-format rows unreachable. A LOCAL re-key pass copied the
5,695 vectors whose texts survive in the current corpus to new-format keys
(byte-identical copies — zero API calls, zero spend), and the 21,040
unreachable rows (15,345 stale-corpus + 5,695 superseded originals) have now
been pruned + VACUUMed (96 MB -> 26 MB). Stage 03 will embed the remaining
~68.5k genuinely new texts for ~$0.08 when approved.

Reviewer 3 (arms + stats) still running; round 4 follows regardless, since
round 3 was NOT CLEAN.

---

## 2026-07-26 14:11 — Review round 3 complete: all three verdicts in, all findings fixed (NO API)

Reviewer 3 (arms + stats) numerically re-verified the whole statistical stack
(two-prop/h vs statsmodels, McNemar vs exact binomial, BH, RQ2 slopes, all
four 07 power branches within <2% of statsmodels power solvers, effect_dz CSV
round-trip, 08's column references) and confirmed graph-arm determinism
byte-identical across hash seeds. Verdicts: 3x NOT CLEAN, converging on two
MAJORs — both now fixed and regression-tested:

1. 04c assembled-cache read lacked the round-2 corpus-fingerprint prefix
   (found independently by all three reviewers): fixed; key format verified
   identical across the writer and both readers.
2. Checkpoint-quarantine hole: sweeps NOT executed in a reduced run got empty
   expectations, so set() <= anything admitted stale smoke checkpoints (e.g.
   --limit 20 smoke then --tier1-only would report RQ4 on n=20). Fixed: only
   sweeps actually executed this run use reduced expectations; regression
   test passes.

Minors fixed: fingerprint hashes chunk TEXT (not just positional ids);
assembly_from_cache flag -> spend accounting ignores replays (incl. rerank);
failure-record judge fields gated on with_judge (schema parity per mode);
04c per-record failure containment + skip gate now also re-runs when the best
arm changed; fig_structured_vs_prose gained bootstrap CIs; RQ1 guards missing
arms under partial sweeps; 402/4xx fail-fast in embed/rerank; 03 refuses to
skip when outputs exist without a manifest record; unreachable emb cache rows
pruned (21,040 removed, 96 MB -> 26 MB) with the re-key documented above.

ROUND 4 is the verification round: same three lenses, full re-review.

---

## 2026-07-26 14:24 — Review round 4: split verdict, second fix batch (NO API)

Auditor 1 (brief + repo): VERDICT CLEAN — verified both round-3 MAJORs fixed via
byte-level key simulation and exhaustive workflow simulation of the quarantine
logic (5 workflows + adversarial variants); full brief-compliance and hygiene
sweeps passed. Its three minors were fixed anyway: checkpoint filenames now
carry the corpus fingerprint (stale-corpus checkpoints quarantined by name),
and 06 guards an empty stats frame.

Auditor 2 (execution + stats): VERDICT NOT CLEAN — confirmed all round-3 fixes
work (incl. running 05's main() against a stubbed run_block through five
workflows, and unit-testing all four 07 power branches), but found two new
MAJORs, both now fixed:
1. 04c's skip gate counted ATTEMPTED questions, not ablated ones — a
   tier1-only pass would freeze a partial ablation as final. Skip now also
   requires skipped_no_assembly == 0 and persists the failure count.
2. fig_structured_vs_prose CIs bootstrapped correlated rows (4 budgets per
   question) — now bootstraps per-question mean EM, per the brief's
   "resample questions, not runs".
Minors fixed: 04b failure rows carry assembly_from_cache and no longer count
as coverage (transient failures retry without --force); 07 guards zero
computable tests; position_ablation.csv header flattened.

ROUND 5 verifies this small surface; loop continues until a fully clean round.

---

## 2026-07-26 14:31 — Review round 5: VERDICT CLEAN — loop closed (NO API)

Final-gate auditor verified all six round-4 fixes by offline simulation
(04c skip decision table 6/6 correct; per-question bootstrap CI demonstrably
wider than the anti-conservative row bootstrap, 0.173 vs 0.098 on correlated
synthetic data; checkpoint fingerprint parse unbroken for every sweep/label
combination; failure-row schema parity; empty-frame guards; flat CSV header)
and swept the whole repo once more: NOTHING MAJOR OR CRITICAL REMAINS.

Residuals accepted as-is or fixed: 05 block resume now also retries failed
rows (same treatment 04b got); the two remaining notes are unreachable-in-
practice degenerate paths, documented here.

Loop totals: 5 rounds, 13 reviewer reports, ~85 findings triaged — every
CRITICAL/MAJOR fixed and re-verified by later rounds; total generator spend
during the entire loop: $0 (the only paid activity remains the ~$0.03 of
partial embeddings from before the freeze). The pipeline is ready for the
user's run decision: 03 (embeddings ~$0.08) -> 04 (graph + compliance
~$0.25) -> 05 smoke (--limit 20, ~$1) -> full sweeps + 04b/04c
(~$23-26 ceiling) -> 06/07/08 (free).

---

## 2026-07-26 14:44 — Stage 03 index — retrieval validated

```
           dataset       method  recall@1  recall@5  recall@10  recall@20  recall@50    mrr
          hotpotqa         bm25    0.2933    0.5567     0.6467     0.7267     0.7967 0.6954
          hotpotqa dense_bge_m3    0.4433    0.8000     0.9033     0.9333     0.9467 0.9291
           liverag         bm25    0.7200    0.8400     0.8933     0.9067     0.9067 0.7803
           liverag dense_bge_m3    0.9467    1.0000     1.0000     1.0000     1.0000 0.9711
          ms_marco         bm25    0.1667    0.5467     0.7200     0.7933     0.8267 0.3392
          ms_marco dense_bge_m3    0.4000    0.8200     0.9200     0.9733     0.9867 0.5924
      multihop_rag         bm25    0.2139    0.5394     0.6378     0.7200     0.8122 0.6319
      multihop_rag dense_bge_m3    0.2844    0.6656     0.8417     0.9533     0.9900 0.7717
      nq_open_gold         bm25    0.3733    0.5600     0.6800     0.7067     0.7333 0.4662
      nq_open_gold dense_bge_m3    0.7067    0.8667     0.8800     0.9067     0.9467 0.7805
          squad_v2         bm25    0.4667    0.7067     0.7600     0.8000     0.8667 0.5768
          squad_v2 dense_bge_m3    0.8400    0.9067     0.9600     0.9600     1.0000 0.8739
wikitablequestions         bm25    0.1133    0.1517     0.1800     0.2017     0.2500 0.1349
wikitablequestions dense_bge_m3    0.2650    0.4217     0.5017     0.6100     0.7333 0.3456
            POOLED         bm25    0.2280    0.3787     0.4414     0.4821     0.5344 0.3685
            POOLED dense_bge_m3    0.4043    0.6186     0.7040     0.7808     0.8546 0.5865
```
HotpotQA dense recall@50 = 0.947 (gate 0.7) — PASS

---

## 2026-07-26 14:51 — Stage 03 interpretation (first paid stage after the go decision)

Spend: ~$0.08 for 74,156 embeddings (68,461 new + 5,695 cache hits), exactly
as projected. Cumulative project spend ≈ $0.11.

What the validation MEANS for the study (recorded for the report's EDA section):

1. **Gate passed decisively** (HotpotQA dense recall@50 = 0.947 vs 0.7): the
   arm comparison is not capped by retrieval quality — downstream differences
   are attributable to assembly strategy. This was a precondition, not a
   finding about the RQs.
2. **Dense (BGE-M3) > BM25 on every dataset** (pooled recall@50 0.855 vs
   0.534; MRR 0.587 vs 0.369). The fixed-retriever choice is now an
   evidence-backed decision, not an assumption.
3. **HotpotQA MRR = 0.929**: the FIRST gold passage ranks near the top almost
   always; the multi-hop challenge is fitting the SECOND bridging passage
   into the budget — precisely the mechanism the graph arm hypothesizes it
   exploits. Good news for the sensitivity of RQ3.
4. **Known ceilings, disclosed up front**: HotpotQA full-evidence ceiling is
   ~95% (recall@50 = 0.947) — shared equally by all arms, comparisons stay
   fair. WikiTableQuestions is the retrieval outlier (dense 0.733, BM25
   collapses to 0.250): serialized tables embed/match poorly, so ~27% of
   structured questions start with the gold table outside the top-50 pool.
   **RQ4's structured-content penalty is therefore part retrieval, part
   assembly** — the per-record retrieval_gold_in_pool flag exists to separate
   the two in 06's stratification; the report must state this confound.

Stage 04 (graph build + budget compliance) launched next.

---

## 2026-07-26 17:00 — Compute moved to Google Colab (Mac froze on stage 04)

The Mac (16 GiB) froze during stage 04's spaCy NER pass over 72,967 chunks;
the run was killed with no data loss (stage 04 had written nothing yet; all
paid work lives in llm_cache/cache.db, which is intact).

Changes for the Colab environment (both recorded, neither affects results):
- SKIP_PGVECTOR=1 supported in 03: no Postgres server on Colab, vectors
  persist as embeddings.npy only — a recorded DEVIATION; query-time retrieval
  was ALWAYS exact in-memory cosine, so the science is unchanged. The local
  pgvector store (already loaded from the Mac run) remains the repo's
  persistent store.
- LLMLingua compressor now uses CUDA when available (Colab T4), else MPS/CPU.

New: notebooks/colab_run.ipynb — mounts Drive, restores prior progress
(cache.db, checkpoints, outputs), installs deps, runs 01→08 with a sync to
MyDrive/ragtb/sync/ after every stage plus a 10-minute background sync during
the long sweep, pauses for smoke-test inspection before the full spend, and
renders all figures + RESULTS_SUMMARY.md inline. colab_bundle.zip (307 MB:
code + data parquets + manifest + outputs + the 26 MB embedding cache) is the
transfer vehicle — gitignored, regenerable. Stage 03 re-derives its index on
Colab from the embedding cache at ~$0 (all 74,156 vectors cached).

State at handoff: stages 01-03 complete and validated (gate passed 0.947);
stage 04 onward runs on Colab. Cumulative spend ≈ $0.11.

---

## 2026-07-26 15:16 — Stage 01 acquire — run complete

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

## 2026-07-26 15:19 — Stage 02 clean — run complete

```
questions: primary 600, structured 600
corpus: 25957 passages -> 72967 chunks (mean 109.4 tokens; 12310 structured)
gold coverage: primary questions with >=1 matched gold passage: 600/600
```
Full step log in outputs/data_cleaning_log.csv.

---

## 2026-07-26 15:38 — Stage 03 index — retrieval validated

```
           dataset       method  recall@1  recall@5  recall@10  recall@20  recall@50    mrr
          hotpotqa         bm25    0.2933    0.5567     0.6467     0.7267     0.7967 0.6954
          hotpotqa dense_bge_m3    0.4433    0.8000     0.9033     0.9333     0.9467 0.9291
           liverag         bm25    0.7200    0.8400     0.8933     0.9067     0.9067 0.7803
           liverag dense_bge_m3    0.9467    1.0000     1.0000     1.0000     1.0000 0.9711
          ms_marco         bm25    0.1667    0.5467     0.7200     0.7933     0.8267 0.3388
          ms_marco dense_bge_m3    0.4000    0.8200     0.9200     0.9733     0.9867 0.5924
      multihop_rag         bm25    0.2139    0.5394     0.6378     0.7200     0.8122 0.6319
      multihop_rag dense_bge_m3    0.2844    0.6656     0.8417     0.9533     0.9900 0.7717
      nq_open_gold         bm25    0.3733    0.5600     0.6800     0.7067     0.7333 0.4661
      nq_open_gold dense_bge_m3    0.7067    0.8667     0.8800     0.9067     0.9467 0.7805
          squad_v2         bm25    0.4667    0.7067     0.7600     0.8000     0.8667 0.5768
          squad_v2 dense_bge_m3    0.8400    0.9067     0.9600     0.9600     1.0000 0.8739
wikitablequestions         bm25    0.1133    0.1517     0.1800     0.2017     0.2500 0.1349
wikitablequestions dense_bge_m3    0.2650    0.4217     0.5017     0.6100     0.7333 0.3455
            POOLED         bm25    0.2280    0.3787     0.4414     0.4821     0.5344 0.3685
            POOLED dense_bge_m3    0.4043    0.6186     0.7040     0.7808     0.8546 0.5865
```
HotpotQA dense recall@50 = 0.947 (gate 0.7) — PASS

---

## 2026-07-26 16:10 — Stage 04 — budget compliance proven

25 questions x 4 budgets x 6 arms, zero violations (assemble() asserts <= budget in real generator tokens).

```
               arm  budget  n  mean_realized  max_realized  mean_utilization  mean_assembly_in  mean_assembly_out  mean_assembly_s  violations
compress_llmlingua     500 25         494.64           500             0.989          11652.80               0.00            4.145           0
compress_llmlingua    1000 25         994.12          1000             0.994          11652.80               0.00            3.528           0
compress_llmlingua    2000 25        1981.72          2000             0.991          11652.80               0.00            4.079           0
compress_llmlingua    4000 25        3927.76          4000             0.982          11652.80               0.00            5.153           0
      graph_select     500 25         478.68           493             0.957              0.00               0.00            0.014           0
      graph_select    1000 25         974.52           989             0.975              0.00               0.00            0.016           0
      graph_select    2000 25        1968.92          1978             0.984              0.00               0.00            0.022           0
      graph_select    4000 25        3944.32          3962             0.986              0.00               0.00            0.051           0
        naive_topk     500 25         478.28           497             0.957              0.00               0.00            0.327           0
        naive_topk    1000 25         975.76           997             0.976              0.00               0.00            0.005           0
        naive_topk    2000 25        1974.40          1996             0.987              0.00               0.00            0.009           0
        naive_topk    4000 25        3969.32          3996             0.992              0.00               0.00            0.016           0
  naive_topk_dedup     500 25         478.28           497             0.957              0.00               0.00            0.194           0
  naive_topk_dedup    1000 25         975.36           997             0.975              0.00               0.00            0.248           0
  naive_topk_dedup    2000 25        1974.40          1996             0.987              0.00               0.00            0.199           0
  naive_topk_dedup    4000 25        3969.28          3996             0.992              0.00               0.00            0.236           0
       rerank_topk     500 25         481.04           498             0.962          11676.80               0.00            0.576           0
       rerank_topk    1000 25         976.60           999             0.977          11676.80               0.00            0.006           0
       rerank_topk    2000 25        1976.40          1998             0.988          11676.80               0.00            0.011           0
       rerank_topk    4000 25        3964.28          3996             0.991          11676.80               0.00            0.018           0
  summarize_recomp     500 25         188.00           500             0.376          12679.20             188.96            2.419           0
  summarize_recomp    1000 25         276.28          1000             0.276          13639.92             277.24            3.226           0
  summarize_recomp    2000 25         358.16          2000             0.179          15571.28             359.12            3.956           0
  summarize_recomp    4000 25         359.16          1386             0.090          19444.24             360.16            6.085           0
```

---

## 2026-07-26 16:45 — Stage 05 sweep (primary+structured, limit=20)

```
records: 980  |  actual spend this run: $0.48  |  31 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.100  0.150  0.175  0.275
graph_select        0.150  0.125  0.175  0.200
graph_select_h1       NaN  0.250    NaN    NaN
naive_topk          0.275  0.225  0.225  0.275
naive_topk_dedup    0.275  0.225  0.225  0.275
rerank_topk         0.250  0.250  0.275  0.250
summarize_recomp    0.225  0.250  0.250  0.275
```

---

## 2026-07-26 20:02 — Stage 05 sweep (primary+structured, limit=20)

```
records: 5620  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.100  0.150  0.175  0.275
graph_select        0.150  0.125  0.175  0.200
graph_select_h1       NaN  0.250    NaN    NaN
naive_topk          0.474  0.484  0.492  0.500
naive_topk_dedup    0.275  0.225  0.225  0.275
rerank_topk         0.500  0.505  0.516  0.511
summarize_recomp    0.225  0.250  0.250  0.275
```

---

## 2026-07-26 20:13 — Stage 05 sweep (primary+structured, limit=20)

```
records: 5620  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.100  0.150  0.175  0.275
graph_select        0.150  0.125  0.175  0.200
graph_select_h1       NaN  0.250    NaN    NaN
naive_topk          0.474  0.484  0.492  0.500
naive_topk_dedup    0.275  0.225  0.225  0.275
rerank_topk         0.500  0.505  0.516  0.511
summarize_recomp    0.225  0.250  0.250  0.275
```

---

## 2026-07-26 20:19 — Stage 05 sweep (primary+structured, limit=20)

```
records: 5620  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.100  0.150  0.175  0.275
graph_select        0.150  0.125  0.175  0.200
graph_select_h1       NaN  0.250    NaN    NaN
naive_topk          0.474  0.484  0.492  0.500
naive_topk_dedup    0.275  0.225  0.225  0.275
rerank_topk         0.500  0.505  0.516  0.511
summarize_recomp    0.225  0.250  0.250  0.275
```

---

## 2026-07-26 20:24 — Stage 05 sweep (primary+structured, limit=20)

```
records: 5620  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.100  0.150  0.175  0.275
graph_select        0.150  0.125  0.175  0.200
graph_select_h1       NaN  0.250    NaN    NaN
naive_topk          0.474  0.484  0.492  0.500
naive_topk_dedup    0.275  0.225  0.225  0.275
rerank_topk         0.500  0.505  0.516  0.511
summarize_recomp    0.225  0.250  0.250  0.275
```

---

## 2026-07-27 05:02 — Stage 05 sweep (primary+structured, limit=None)

```
records: 24600  |  actual spend this run: $8.78  |  518 min

EM by arm x budget:
budget             500    1000   2000   4000
arm                                         
graph_select      0.177  0.203  0.219  0.237
graph_select_h1     NaN  0.330    NaN    NaN
naive_topk        0.329  0.336  0.341  0.343
naive_topk_dedup  0.327  0.336  0.341  0.343
rerank_topk       0.360  0.360  0.364  0.364
summarize_recomp  0.240  0.285  0.312  0.342
```

---

## 2026-08-14 18:17 — Stage 05 sweep (primary+structured, limit=20)

```
records: 24760  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.100  0.150  0.175  0.275
graph_select        0.177  0.203  0.219  0.237
graph_select_h1       NaN  0.330    NaN    NaN
naive_topk          0.329  0.336  0.341  0.343
naive_topk_dedup    0.327  0.336  0.341  0.343
rerank_topk         0.360  0.360  0.364  0.364
summarize_recomp    0.240  0.285  0.312  0.342
```

---

## 2026-08-14 18:18 — Stage 05 sweep (primary+structured, limit=None)

```
records: 24600  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget             500    1000   2000   4000
arm                                         
graph_select      0.177  0.203  0.219  0.237
graph_select_h1     NaN  0.330    NaN    NaN
naive_topk        0.329  0.336  0.341  0.343
naive_topk_dedup  0.327  0.336  0.341  0.343
rerank_topk       0.360  0.360  0.364  0.364
summarize_recomp  0.240  0.285  0.312  0.342
```

---

## 2026-08-15 05:51 — Stage 05 sweep (primary+structured, limit=20)

```
records: 28240  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.154  0.178  0.308  0.392
graph_select        0.177  0.203  0.219  0.237
graph_select_h1       NaN  0.330    NaN    NaN
naive_topk          0.329  0.336  0.341  0.343
naive_topk_dedup    0.327  0.336  0.341  0.343
rerank_topk         0.360  0.360  0.364  0.364
summarize_recomp    0.240  0.285  0.312  0.342
```

---

## 2026-08-15 05:51 — Stage 05 sweep (primary+structured, limit=None)

```
records: 28200  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.154  0.178  0.315  0.400
graph_select        0.177  0.203  0.219  0.237
graph_select_h1       NaN  0.330    NaN    NaN
naive_topk          0.329  0.336  0.341  0.343
naive_topk_dedup    0.327  0.336  0.341  0.343
rerank_topk         0.360  0.360  0.364  0.364
summarize_recomp    0.240  0.285  0.312  0.342
```

---

## 2026-08-15 07:03 — Stage 05 sweep (primary+structured, limit=None)

```
records: 29400  |  actual spend this run: $0.83  |  71 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.154  0.178  0.202  0.258
graph_select        0.177  0.203  0.219  0.237
graph_select_h1       NaN  0.330    NaN    NaN
naive_topk          0.329  0.336  0.341  0.343
naive_topk_dedup    0.327  0.336  0.341  0.343
rerank_topk         0.360  0.360  0.364  0.364
summarize_recomp    0.240  0.285  0.312  0.342
```

---

## 2026-08-15 07:08 — Stage 04b baselines

```
                  em     f1  gen_context_tokens
arm                                            
full_context   0.507  0.654           11431.118
gold_context   0.562  0.718             335.223
no_context     0.300  0.399               0.000
random_chunks  0.233  0.336             927.810
```

---

## 2026-08-15 07:11 — Stage 04c position ablation

arm=rerank_topk @ 1000 tokens
```
  position  em_mean  em_count  f1_mean  f1_count  faithfulness_mean  faithfulness_count
0      end   0.5295       576   0.6774       576             0.9151                 576
1   middle   0.5156       576   0.6634       576             0.8998                 576
2    start   0.5365       576   0.6842       576             0.9259                 576
```
skipped: 0 no assembly, 24 no gold in selection, 0 BPE jitter.

---

## 2026-08-15 07:11 — Stage 06 analysis

```
results rows: 135; tests: 41 (34 significant after BH-FDR); figures written (300 dpi)
```

---

## 2026-08-15 07:11 — Stage 07 power refresh

```
 rq                                              comparison budget             test  observed_effect  n_assumed_synopsis  n_observed_effect                                                       formula  alpha  power  primary_comparison
RQ1                               rerank_topk vs naive_topk   1000 two_proportion_z           0.0467               170.0               7204                                        n = 2(z_a/2+z_b)^2/h^2   0.05    0.8                True
RQ1                               rerank_topk vs naive_topk   1000    mcnemar_exact           1.7568               170.0               1199                                       discordant-pair formula   0.05    0.8                True
RQ3                              graph_select vs naive_topk   1000 two_proportion_z          -0.3578               356.0                123                                        n = 2(z_a/2+z_b)^2/h^2   0.05    0.8                True
RQ3                              graph_select vs naive_topk   1000    mcnemar_exact           0.0909               356.0                 51                                       discordant-pair formula   0.05    0.8                True
RQ2 per-question F1-per-log2(budget) slope: multi vs single  slope          welch_t           0.2060                34.0                370 n per group = 2(z_a/2+z_b)^2/d^2 (two-sample t approximation)   0.05    0.8                True
RQ4                graph_select: structured vs prose @ 1000   1000 two_proportion_z          -0.6240               564.0                 41                                        n = 2(z_a/2+z_b)^2/h^2   0.05    0.8                True
```

---

## 2026-08-15 07:11 — Stage 08 summary

RESULTS_SUMMARY.md regenerated (with sweep records).

---

## 2026-08-15 07:21 — Stage 05 sweep (primary+structured, limit=20)

```
records: 29400  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.154  0.178  0.202  0.258
graph_select        0.177  0.203  0.219  0.237
graph_select_h1       NaN  0.330    NaN    NaN
naive_topk          0.329  0.336  0.341  0.343
naive_topk_dedup    0.327  0.336  0.341  0.343
rerank_topk         0.360  0.360  0.364  0.364
summarize_recomp    0.240  0.285  0.312  0.342
```

---

## 2026-08-15 07:21 — Stage 05 sweep (primary+structured, limit=None)

```
records: 29400  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.154  0.178  0.202  0.258
graph_select        0.177  0.203  0.219  0.237
graph_select_h1       NaN  0.330    NaN    NaN
naive_topk          0.329  0.336  0.341  0.343
naive_topk_dedup    0.327  0.336  0.341  0.343
rerank_topk         0.360  0.360  0.364  0.364
summarize_recomp    0.240  0.285  0.312  0.342
```

---

## 2026-08-15 07:21 — Stage 05 sweep (primary+structured, limit=None)

```
records: 29400  |  actual spend this run: $0.00  |  0 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.154  0.178  0.202  0.258
graph_select        0.177  0.203  0.219  0.237
graph_select_h1       NaN  0.330    NaN    NaN
naive_topk          0.329  0.336  0.341  0.343
naive_topk_dedup    0.327  0.336  0.341  0.343
rerank_topk         0.360  0.360  0.364  0.364
summarize_recomp    0.240  0.285  0.312  0.342
```

---

## 2026-08-15 07:22 — Stage 04b baselines

```
                  em     f1  gen_context_tokens
arm                                            
full_context   0.507  0.654           11431.118
gold_context   0.565  0.721             340.153
no_context     0.300  0.399               0.000
random_chunks  0.233  0.336             927.810
```

---

## 2026-08-15 07:22 — Stage 06 analysis

```
results rows: 135; tests: 41 (34 significant after BH-FDR); figures written (300 dpi)
```

---

## 2026-08-15 07:22 — Stage 07 power refresh

```
 rq                                              comparison budget             test  observed_effect  n_assumed_synopsis  n_observed_effect                                                       formula  alpha  power  primary_comparison
RQ1                               rerank_topk vs naive_topk   1000 two_proportion_z           0.0467               170.0               7204                                        n = 2(z_a/2+z_b)^2/h^2   0.05    0.8                True
RQ1                               rerank_topk vs naive_topk   1000    mcnemar_exact           1.7568               170.0               1199                                       discordant-pair formula   0.05    0.8                True
RQ3                              graph_select vs naive_topk   1000 two_proportion_z          -0.3578               356.0                123                                        n = 2(z_a/2+z_b)^2/h^2   0.05    0.8                True
RQ3                              graph_select vs naive_topk   1000    mcnemar_exact           0.0909               356.0                 51                                       discordant-pair formula   0.05    0.8                True
RQ2 per-question F1-per-log2(budget) slope: multi vs single  slope          welch_t           0.2060                34.0                370 n per group = 2(z_a/2+z_b)^2/d^2 (two-sample t approximation)   0.05    0.8                True
RQ4                graph_select: structured vs prose @ 1000   1000 two_proportion_z          -0.6240               564.0                 41                                        n = 2(z_a/2+z_b)^2/h^2   0.05    0.8                True
```

---

## 2026-08-15 07:22 — Stage 08 summary

RESULTS_SUMMARY.md regenerated (with sweep records).

---

## 2026-08-15 — Repair session verified; full data merged to the local repo

Read all 21 executed notebook cells of the repair run: cells 1-14 replayed at
$0; 04b's coverage gate refused the skip and re-ran cleanly (warm-up fix
worked — zero failed records). **gold_context ceiling is now n=600: EM 0.565,
F1 0.721** (was 0.562 on n=589). 04c correctly skipped; 06/07/08 regenerated;
pushed. Cosmetic nit noted: 04b's re-run message prints unique-question count
("600 < 600") while the check is per-condition coverage.

Merged from the Drive sync into the local repo: eval_records (29,400 records,
0 failed), baseline_records (2,400, 0 failed), position_records (1,728), all
49 full checkpoints, manifest. The synced cache.db arrived MALFORMED (SQLite
WAL synced mid-write without its -wal sidecar — a known hazard of file-copy
backups of live SQLite). `.recover` salvaged the full emb table (74,156
vectors); the generation/judge/assembled caches from the Colab sessions are
lost in this copy. No result is affected — results live in checkpoints — but
future re-generation would re-pay. Lesson recorded: sync SQLite only after
`PRAGMA wal_checkpoint(TRUNCATE)` or via the sqlite backup API.

STUDY DATA NOW FULLY LOCAL AND VERIFIED. Outstanding: (1) 50 hand verdicts in
outputs/judge_spot_check_sample.csv -> rerun 08 for the agreement rate;
(2) update the Interim Report docx with final numbers.

---

## 2026-08-15 10:05 — Stage 05 sweep (primary, limit=None)

```
records: 30000  |  actual spend this run: $0.17  |  2 min

EM by arm x budget:
budget               500    1000   2000   4000
arm                                           
compress_llmlingua  0.154  0.178  0.202  0.258
graph_select        0.177  0.203  0.219  0.237
graph_select_a05      NaN  0.293    NaN    NaN
graph_select_h1       NaN  0.330    NaN    NaN
naive_topk          0.329  0.336  0.341  0.343
naive_topk_dedup    0.327  0.336  0.341  0.343
rerank_topk         0.360  0.360  0.364  0.364
summarize_recomp    0.240  0.285  0.312  0.342
```

---

## 2026-08-15 10:06 — Stage 06 analysis

```
results rows: 138; tests: 42 (35 significant after BH-FDR); figures written (300 dpi)
```

---

## 2026-08-15 10:06 — Stage 07 power refresh

```
 rq                                              comparison budget             test  observed_effect  n_assumed_synopsis  n_observed_effect                                                       formula  alpha  power  primary_comparison
RQ1                               rerank_topk vs naive_topk   1000 two_proportion_z           0.0467               170.0               7204                                        n = 2(z_a/2+z_b)^2/h^2   0.05    0.8                True
RQ1                               rerank_topk vs naive_topk   1000    mcnemar_exact           1.7568               170.0               1199                                       discordant-pair formula   0.05    0.8                True
RQ3                              graph_select vs naive_topk   1000 two_proportion_z          -0.3578               356.0                123                                        n = 2(z_a/2+z_b)^2/h^2   0.05    0.8                True
RQ3                              graph_select vs naive_topk   1000    mcnemar_exact           0.0909               356.0                 51                                       discordant-pair formula   0.05    0.8                True
RQ2 per-question F1-per-log2(budget) slope: multi vs single  slope          welch_t           0.2060                34.0                370 n per group = 2(z_a/2+z_b)^2/d^2 (two-sample t approximation)   0.05    0.8                True
RQ4                graph_select: structured vs prose @ 1000   1000 two_proportion_z          -0.6240               564.0                 41                                        n = 2(z_a/2+z_b)^2/h^2   0.05    0.8                True
```

---

## 2026-08-15 10:06 — Stage 08 summary

RESULTS_SUMMARY.md regenerated (with sweep records).

---

## 2026-08-15 — Instructor-feedback coding items implemented (alpha run ~$0.30)

1. **RQ4 retrieval-vs-assembly decomposition** (outputs/rq4_decomposition.csv):
   at matched retrieval the structured penalty barely shrinks — retrieval
   share is only 0.02-0.05 EM of a ~0.32 gap. The structured-content penalty
   is predominantly an ASSEMBLY/GENERATION effect, not the WTQ retrieval
   weakness previously flagged as a possible confound.
2. **LiveRAG contamination investigated** (outputs/contamination_check.csv):
   at matched retrieval (gold_in_pool both sides), older benchmarks EM 0.384 /
   F1 0.533 vs LiveRAG 0.008 / 0.247 — but LiveRAG faithfulness is HIGHER
   (0.807 vs 0.750). Post-cutoff the generator still grounds answers the judge
   accepts; the EM/F1 collapse is largely the long-form-gold metric artifact.
   Residual memorization inflating older-benchmark EM cannot be excluded, but
   "capability collapse on unseen data" is ruled out.
3. **Graph alpha sensitivity** (new sensitivity_a05 sweep, 600 records):
   alpha=0.5 EM 0.293 vs primary alpha=0.7 0.323 (McNemar odds 2.16, p=.013)
   vs hops=1 0.330 (null). The arm is sensitive to the relevance weight,
   insensitive to expansion depth, and dominated by naive top-k under every
   configuration probed — the negative result is not a tuning artifact.
4. **Practitioner decision guide** added to RESULTS_SUMMARY (winner by
   segment x budget with margins, carrying the RQ1 small-margin caveat).

eval_records now 30,000 records (50 blocks), 0 failed.
