"""Stage 01 — acquire datasets, sample with the fixed seed, profile.

Outputs:
  data/questions_primary.parquet     600 questions, stratified, hop-balanced
  data/questions_structured.parquet  ~600 structured/code questions (RQ4)
  data/passages_pool.parquet         all candidate passages (gold + distractors)
  data/raw_profile.csv               per-dataset raw profile (graded deliverable)
  outputs/fig_dataset_profile.png    distribution panel

Tiers (Synopsis contract): T1 HotpotQA + MultiHop-RAG + SQuAD v2 (required),
T2 NQ + MS MARCO, T3 LiveRAG + structured set. A tier that fails to load is
recorded in the manifest and the experiment log — never silently dropped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DATA,
    OUTPUTS,
    SEED,
    append_log,
    skip_if_exists,
    update_manifest,
)

# One independent, fixed-index RNG stream per dataset: the sample each dataset
# draws is then a pure function of (SEED, dataset), regardless of load order,
# network state, or another dataset failing. A single shared stream consumed
# sequentially would re-deal every downstream sample whenever anything upstream
# changed (observed across the first stage-01 attempts).
_DS_STREAM = {"hotpotqa": 1, "multihop_rag": 2, "squad_v2": 3, "nq_open_gold": 4,
              "ms_marco": 5, "liverag": 6, "wikitablequestions": 7}


def rng_for(dataset: str) -> np.random.Generator:
    return np.random.default_rng([SEED, _DS_STREAM[dataset]])

Q_PRIMARY = DATA / "questions_primary.parquet"
Q_STRUCTURED = DATA / "questions_structured.parquet"
PASSAGES = DATA / "passages_pool.parquet"
RAW_PROFILE = DATA / "raw_profile.csv"
FIG_PROFILE = OUTPUTS / "fig_dataset_profile.png"

# Sample allocation (Synopsis: 600 total, balanced hop_type, stratified by dataset)
N_MULTI = {"hotpotqa": 150, "multihop_rag": 150}
N_SINGLE_TOTAL = 300  # split evenly across whichever single-hop sets load
N_STRUCTURED = 600

# Distractor pool sizes (extra questions whose passages pad the retrieval corpus
# so retrieval is non-trivial; recorded here as design constants)
N_DISTRACTOR_Q = {"hotpotqa": 1500, "squad_v2": 1000, "nq_open_gold": 1000, "ms_marco": 500}


def _load(repo: str, *args, **kwargs):
    """load_dataset with a fallback to the Hub's auto-converted parquet branch
    (datasets>=5 no longer runs script-based builders)."""
    from datasets import load_dataset

    try:
        return load_dataset(repo, *args, **kwargs)
    except Exception as e:
        print(f"  primary load failed ({e!r}); trying refs/convert/parquet")
        return load_dataset(repo, *args, revision="refs/convert/parquet", **kwargs)


def profile_rows(name, rows, fields) -> dict:
    """Raw-profile one dataset: counts, null rates, length distributions (words —
    labelled as such; exact-token accounting applies to budgets, not profiling)."""
    n = len(rows)
    prof = {"dataset": name, "n_records": n, "fields": ", ".join(fields)}
    for f_ in fields:
        vals = [r.get(f_) for r in rows]
        empty = sum(1 for v in vals if v is None or (isinstance(v, (str, list)) and len(v) == 0))
        prof[f"null_rate.{f_}"] = round(empty / max(n, 1), 4)
        first = next((v for v in vals if v is not None), None)
        prof[f"type.{f_}"] = type(first).__name__ if first is not None else "NoneType"
    qlens = [len(str(r.get("question", "")).split()) for r in rows]
    alens = [
        len(str(r["gold_answers"][0]).split()) if r.get("gold_answers") else 0 for r in rows
    ]
    clens = [
        sum(len(p.split()) for p in r.get("gold_passages", [])) for r in rows
    ]
    sfs = [r.get("n_supporting_facts", 0) for r in rows]
    for label, arr in [("question_words", qlens), ("answer_words", alens),
                       ("gold_context_words", clens), ("supporting_facts", sfs)]:
        a = np.array(arr, dtype=float)
        prof[f"{label}.mean"] = round(float(a.mean()), 2) if len(a) else 0
        prof[f"{label}.median"] = round(float(np.median(a)), 2) if len(a) else 0
        prof[f"{label}.p90"] = round(float(np.percentile(a, 90)), 2) if len(a) else 0
    return prof


# ---------------------------------------------------------------- per-dataset acquire
# Each returns (questions, passages, profile_dict). Question dict fields:
# question_id, dataset, question, gold_answers, hop_type, content_type,
# gold_titles, gold_passages, n_supporting_facts


def acquire_hotpotqa():
    ds = _load("hotpotqa/hotpot_qa", "distractor", split="validation")
    rows = []
    for r in ds:
        ctx_titles = list(r["context"]["title"])
        ctx_sents = list(r["context"]["sentences"])
        sf_titles = sorted(set(r["supporting_facts"]["title"]))
        gold_passages = [
            "".join(ctx_sents[ctx_titles.index(t)]) for t in sf_titles if t in ctx_titles
        ]
        rows.append(
            dict(
                question_id=f"hotpot_{r['id']}",
                dataset="hotpotqa",
                question=r["question"],
                gold_answers=[r["answer"]],
                hop_type="multi",
                content_type="prose",
                gold_titles=sf_titles,
                gold_passages=gold_passages,
                n_supporting_facts=len(r["supporting_facts"]["title"]),
                _all_titles=ctx_titles,
                _all_passages=["".join(s) for s in ctx_sents],
            )
        )
    prof = profile_rows("hotpotqa (distractor dev)", rows,
                        ["question", "gold_answers", "gold_passages"])

    idx = rng_for("hotpotqa").permutation(len(rows))
    sampled = [rows[i] for i in idx[: N_MULTI["hotpotqa"]]]
    distractor_q = [rows[i] for i in idx[N_MULTI["hotpotqa"]:
                                         N_MULTI["hotpotqa"] + N_DISTRACTOR_Q["hotpotqa"]]]
    # is_gold means "supporting evidence of a SAMPLED question" — uniformly
    # across datasets. Distractor questions' own golds are just distractors.
    passages = []
    for r, is_sampled in [(x, True) for x in sampled] + [(x, False) for x in distractor_q]:
        for t, p in zip(r.pop("_all_titles"), r.pop("_all_passages")):
            passages.append(dict(dataset="hotpotqa", title=t, text=p, content_type="prose",
                                 is_gold=is_sampled and t in r["gold_titles"]))
    return sampled, passages, prof


def acquire_multihop_rag():
    qs = _load("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    corpus = _load("yixuantt/MultiHopRAG", "corpus", split="train")
    rows = []
    for i, r in enumerate(qs):
        ev = r.get("evidence_list") or []
        # null/insufficient questions have answer "Insufficient information."
        rows.append(
            dict(
                question_id=f"mhrag_{i}",
                dataset="multihop_rag",
                question=r["query"],
                gold_answers=[r["answer"]],
                hop_type="multi",
                content_type="prose",
                gold_titles=sorted({e["title"] for e in ev}),
                gold_passages=[e["fact"] for e in ev],
                n_supporting_facts=len(ev),
                _qtype=r.get("question_type", ""),
            )
        )
    prof = profile_rows("multihop_rag", rows, ["question", "gold_answers", "gold_passages"])
    usable = [r for r in rows if r["gold_answers"][0].strip().lower()
              not in ("insufficient information.", "insufficient information")]
    idx = rng_for("multihop_rag").permutation(len(usable))
    sampled = []
    for i in idx:
        r = dict(usable[i]); r.pop("_qtype", None)
        sampled.append(r)
        if len(sampled) == N_MULTI["multihop_rag"]:
            break
    # corpus: the full 609-article news corpus. An article is gold iff its
    # title is a SAMPLED question's evidence source (uniform is_gold semantics
    # — this protects real evidence from near-dup removal in 02).
    sampled_gold_titles = {t for r in sampled for t in r["gold_titles"]}
    passages = [
        dict(dataset="multihop_rag", title=c["title"], text=c["body"],
             content_type="prose", is_gold=c["title"] in sampled_gold_titles)
        for c in corpus
    ]
    return sampled, passages, prof


def acquire_squad(n_target: int):
    ds = _load("rajpurkar/squad_v2", split="validation")
    rows = []
    for r in ds:
        answers = [a for a in r["answers"]["text"]]
        rows.append(
            dict(
                question_id=f"squad_{r['id']}",
                dataset="squad_v2",
                question=r["question"],
                gold_answers=answers,  # empty => unanswerable (v2)
                hop_type="single",
                content_type="prose",
                gold_titles=[r["title"]],
                gold_passages=[r["context"]],
                n_supporting_facts=1,
            )
        )
    prof = profile_rows("squad_v2 (dev)", rows, ["question", "gold_answers", "gold_passages"])
    answerable = [r for r in rows if r["gold_answers"]]
    prof["note"] = (f"sampled from answerable subset ({len(answerable)}/{len(rows)}); "
                    "unanswerable items lack a gold string for EM/F1")
    idx = rng_for("squad_v2").permutation(len(answerable))
    sampled = [answerable[i] for i in idx[:n_target]]
    distractors = [answerable[i] for i in idx[n_target: n_target + N_DISTRACTOR_Q["squad_v2"]]]
    sampled_ids = {r["question_id"] for r in sampled}
    seen, passages = {}, []
    for r in sampled + distractors:
        c = r["gold_passages"][0]
        gold = r["question_id"] in sampled_ids
        if c not in seen:
            seen[c] = len(passages)
            passages.append(dict(dataset="squad_v2", title=r["gold_titles"][0], text=c,
                                 content_type="prose", is_gold=gold))
        elif gold:  # same context serves a sampled question too — keep it gold
            passages[seen[c]]["is_gold"] = True
    return sampled, passages, prof


def acquire_nq(n_target: int):
    """NQ-open with gold passages (florin-hf/nq_open_gold, used by the 'Power of
    Noise' paper). Recorded as the NQ variant since full NQ is ~40 GB HTML."""
    ds = _load("florin-hf/nq_open_gold", split="test")
    rows = []
    for i, r in enumerate(ds):
        rows.append(
            dict(
                question_id=f"nq_{r.get('example_id', i)}",
                dataset="nq_open_gold",
                question=r["question"],
                gold_answers=list(r["answers"]),
                hop_type="single",
                content_type="prose",
                gold_titles=[r.get("title", "")],
                gold_passages=[r["text"]],
                n_supporting_facts=1,
            )
        )
    prof = profile_rows("nq_open_gold (test)", rows, ["question", "gold_answers", "gold_passages"])
    idx = rng_for("nq_open_gold").permutation(len(rows))
    sampled = [rows[i] for i in idx[:n_target]]
    distractors = [rows[i] for i in idx[n_target: n_target + N_DISTRACTOR_Q["nq_open_gold"]]]
    sampled_ids = {r["question_id"] for r in sampled}
    passages = [
        dict(dataset="nq_open_gold", title=r["gold_titles"][0], text=r["gold_passages"][0],
             content_type="prose", is_gold=r["question_id"] in sampled_ids)
        for r in sampled + distractors
    ]
    return sampled, passages, prof


def acquire_msmarco(n_target: int):
    ds = _load("microsoft/ms_marco", "v2.1", split="validation")
    rows = []
    for r in ds:
        answers = [a for a in r["answers"] if a and a != "No Answer Present."]
        ptexts = list(r["passages"]["passage_text"])
        sel = list(r["passages"]["is_selected"])
        gold = [t for t, s in zip(ptexts, sel) if s == 1]
        rows.append(
            dict(
                question_id=f"msmarco_{r['query_id']}",
                dataset="ms_marco",
                question=r["query"],
                gold_answers=answers,
                hop_type="single",
                content_type="prose",
                gold_titles=[],
                gold_passages=gold,
                n_supporting_facts=len(gold),
                _all_passages=ptexts,
            )
        )
    prof = profile_rows("ms_marco v2.1 (dev)", rows, ["question", "gold_answers", "gold_passages"])
    usable = [r for r in rows if r["gold_answers"] and r["gold_passages"]]
    prof["note"] = f"sampled from answered subset with a selected passage ({len(usable)}/{len(rows)})"
    idx = rng_for("ms_marco").permutation(len(usable))
    sampled = [usable[i] for i in idx[:n_target]]
    distractor_q = [usable[i] for i in idx[n_target: n_target + N_DISTRACTOR_Q["ms_marco"]]]
    passages = []
    for r, is_sampled in [(x, True) for x in sampled] + [(x, False) for x in distractor_q]:
        golds = set(r["gold_passages"]) if is_sampled else set()
        for t in r.pop("_all_passages"):
            passages.append(dict(dataset="ms_marco", title="", text=t,
                                 content_type="prose", is_gold=t in golds))
    return sampled, passages, prof


def acquire_liverag(n_target: int):
    """LiveRAG (SIGIR 2025) — contamination control. Official release may be gated;
    failure is recorded, not hidden."""
    from huggingface_hub import list_datasets

    # LiveRAG/Benchmark (DataMorgana-generated, post-cutoff): explicit schema.
    # NOTE for analysis: answers are long-form -> EM unreliable; contamination
    # comparison uses F1 + judged faithfulness.
    try:
        ds = _load("LiveRAG/Benchmark", split="train")
        rows = []
        for i, r in enumerate(ds):
            docs = [d.get("content", "") for d in (r.get("Supporting_Documents") or [])]
            docs = [d for d in docs if d.strip()]
            rows.append(
                dict(question_id=f"liverag_{r.get('Index', i)}", dataset="liverag",
                     question=str(r["Question"]),
                     gold_answers=[str(r["Answer"]).strip()],
                     hop_type="single", content_type="prose", gold_titles=[],
                     gold_passages=docs, n_supporting_facts=len(docs))
            )
        # single-doc questions only: liverag sits in the single-hop stratum, and
        # multi-doc DataMorgana items would mislabel the 300/300 hop balance
        usable = [r for r in rows
                  if r["gold_answers"][0] and len(r["gold_passages"]) == 1]
        if len(usable) >= n_target:
            prof = profile_rows("liverag (LiveRAG/Benchmark)", rows,
                                ["question", "gold_answers", "gold_passages"])
            prof["note"] = ("long-form answers: EM unreliable, use F1/faithfulness; "
                            "sampled from single-document questions only "
                            f"({len(usable)}/{len(rows)}) to keep hop labels true")
            idx = rng_for("liverag").permutation(len(usable))
            sampled = [usable[i] for i in idx[:n_target]]
            distractors = [usable[i] for i in idx[n_target: n_target + 800]]
            sampled_ids = {r["question_id"] for r in sampled}
            passages = []
            for r in sampled + distractors:
                for p in r["gold_passages"]:
                    passages.append(dict(dataset="liverag", title="", text=p,
                                         content_type="prose",
                                         is_gold=r["question_id"] in sampled_ids))
            return sampled, passages, prof, "LiveRAG/Benchmark"
    except Exception as e:
        print(f"  LiveRAG/Benchmark direct load failed: {e!r}")

    cands = [d.id for d in list_datasets(search="LiveRAG", limit=20)]
    print(f"  LiveRAG hub candidates: {cands}")
    last_err = None
    for repo in cands:
        try:
            ds = _load(repo, split="train")
            cols = set(ds.column_names)
            qcol = next((c for c in ["question", "query", "Question"] if c in cols), None)
            acol = next((c for c in ["answer", "answers", "Answer", "gold_answer"] if c in cols), None)
            pcol = next((c for c in ["context", "passages", "documents", "gold_passage", "passage"]
                         if c in cols), None)
            if not (qcol and acol):
                continue
            rows = []
            for i, r in enumerate(ds):
                ans = r[acol] if isinstance(r[acol], list) else [str(r[acol])]
                gp = []
                if pcol:
                    gp = r[pcol] if isinstance(r[pcol], list) else [str(r[pcol])]
                    gp = [str(x) for x in gp]
                rows.append(
                    dict(question_id=f"liverag_{i}", dataset="liverag", question=str(r[qcol]),
                         gold_answers=[str(a) for a in ans if str(a).strip()],
                         hop_type="single", content_type="prose", gold_titles=[],
                         gold_passages=gp, n_supporting_facts=len(gp))
                )
            usable = [r for r in rows if r["gold_answers"] and len(r["gold_passages"]) == 1]
            if len(usable) < n_target:
                continue
            prof = profile_rows(f"liverag ({repo})", rows,
                                ["question", "gold_answers", "gold_passages"])
            prof["note"] = f"loaded from {repo}; single-document questions only"
            idx = rng_for("liverag").permutation(len(usable))
            sampled = [usable[i] for i in idx[:n_target]]
            distractors = [usable[i] for i in idx[n_target: n_target + 800]]
            sampled_ids = {r["question_id"] for r in sampled}
            passages = []
            for r in sampled + distractors:
                for p in r["gold_passages"]:
                    passages.append(dict(dataset="liverag", title="", text=p,
                                         content_type="prose",
                                         is_gold=r["question_id"] in sampled_ids))
            return sampled, passages, prof, repo
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"no usable LiveRAG release found on the Hub (last: {last_err!r})")


def serialize_table(table: dict, max_rows: int = 25) -> str:
    header = table["header"]
    lines = [" | ".join(header), " | ".join(["---"] * len(header))]
    for row in table["rows"][:max_rows]:
        lines.append(" | ".join(str(c) for c in row))
    if len(table["rows"]) > max_rows:
        lines.append(f"... ({len(table['rows']) - max_rows} more rows)")
    return "\n".join(lines)


def acquire_wtq(n_target: int):
    """WikiTableQuestions — the structured/code set RQ4 requires."""
    # official stanfordnlp repo is script-based (unsupported by datasets>=5);
    # lighteval mirror is parquet-native with identical fields + a table_md render
    ds = _load("lighteval/wikitablequestions", split="test")
    train = _load("lighteval/wikitablequestions", split="train")
    rows = []
    for split_name, split in [("test", ds), ("train", train)]:
        for r in split:
            tid = r["table"].get("name", "") or f"tbl_{split_name}_{r['id']}"
            rows.append(
                dict(
                    question_id=f"wtq_{split_name}_{r['id']}",
                    dataset="wikitablequestions",
                    question=r["question"],
                    gold_answers=list(r["answers"]),
                    hop_type="single",
                    content_type="structured",
                    gold_titles=[tid],
                    gold_passages=[r.get("table_md") or serialize_table(r["table"])],
                    n_supporting_facts=1,
                )
            )
    prof = profile_rows("wikitablequestions (dev+train pool)", rows,
                        ["question", "gold_answers", "gold_passages"])
    usable = [r for r in rows if r["gold_answers"]]
    idx = rng_for("wikitablequestions").permutation(len(usable))
    sampled = [usable[i] for i in idx[:n_target]]
    distractors = [usable[i] for i in idx[n_target: n_target + 1000]]
    sampled_ids = {r["question_id"] for r in sampled}
    seen, passages = {}, []
    for r in sampled + distractors:
        t = r["gold_passages"][0]
        gold = r["question_id"] in sampled_ids
        if t not in seen:
            seen[t] = len(passages)
            passages.append(dict(dataset="wikitablequestions", title=r["gold_titles"][0],
                                 text=t, content_type="structured", is_gold=gold))
        elif gold:  # table shared with a sampled question stays gold
            passages[seen[t]]["is_gold"] = True
    return sampled, passages, prof


# ---------------------------------------------------------------- figure


def make_profile_figure(qdf: pd.DataFrame, sdf: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_q = pd.concat([qdf, sdf])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0][0]
    counts = all_q.groupby(["dataset", "hop_type"]).size().unstack(fill_value=0)
    counts.plot.barh(stacked=True, ax=ax)
    ax.set_title("Sampled questions by dataset and hop type")
    ax.set_xlabel("questions")

    ax = axes[0][1]
    for d, g in all_q.groupby("dataset"):
        ax.hist(g["question"].str.split().str.len(), bins=30, alpha=0.5, label=d)
    ax.set_title("Question length (words)")
    ax.set_xlabel("words"); ax.set_ylabel("count"); ax.legend(fontsize=7)

    ax = axes[0][2]
    for d, g in all_q.groupby("dataset"):
        lens = g["gold_answers"].apply(lambda a: len(str(a[0]).split()) if len(a) else 0)
        ax.hist(lens.clip(upper=40), bins=40, alpha=0.5, label=d)
    ax.set_title("Gold answer length (words, clipped at 40)")
    ax.set_xlabel("words"); ax.legend(fontsize=7)

    ax = axes[1][0]
    for d, g in all_q.groupby("dataset"):
        lens = g["gold_passages"].apply(lambda ps: sum(len(str(p).split()) for p in ps))
        ax.hist(lens.clip(upper=1500), bins=40, alpha=0.5, label=d)
    ax.set_title("Gold context length (words, clipped at 1500)")
    ax.set_xlabel("words"); ax.legend(fontsize=7)

    ax = axes[1][1]
    all_q.boxplot(column="n_supporting_facts", by="dataset", ax=ax, rot=30)
    ax.set_title("Supporting facts / gold passages per question")
    ax.set_xlabel("")

    ax = axes[1][2]
    ct = all_q.groupby(["dataset", "content_type"]).size().unstack(fill_value=0)
    ct.plot.barh(stacked=True, ax=ax, color=["tab:blue", "tab:orange"])
    ax.set_title("Content type by dataset")
    ax.set_xlabel("questions")

    fig.suptitle("Dataset profile — sampled evaluation sets (seed 42)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_PROFILE, dpi=300)
    print(f"  wrote {FIG_PROFILE}")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args, _ = ap.parse_known_args()

    outputs = [Q_PRIMARY, Q_STRUCTURED, PASSAGES, RAW_PROFILE, FIG_PROFILE]
    # a previous run that recorded failures (or an empty mandatory output) is
    # NOT complete — outputs existing is necessary but not sufficient to skip
    prev_failures = {}
    if MANIFEST := (DATA / "manifest.json"):
        if MANIFEST.exists():
            import json as _json

            prev_failures = _json.loads(MANIFEST.read_text()).get("stage01", {}).get("failures", {})
    complete = not prev_failures and all(p.exists() for p in outputs)
    if complete and len(pd.read_parquet(Q_STRUCTURED)) == 0:
        complete = False
    if complete and skip_if_exists(outputs, args.force, "01_acquire"):
        return
    if not complete and not args.force and all(p.exists() for p in outputs):
        print(f"[01] outputs exist but previous run was incomplete "
              f"(failures={list(prev_failures)}) — re-running")

    profiles, failures = [], {}
    all_questions, all_passages = [], []

    # --- Tier 1 (required) + Tier 2 + Tier 3, each isolated so one failure
    # never takes down the others
    print("[01] HotpotQA ...")
    qs, ps, prof = acquire_hotpotqa()
    all_questions += qs; all_passages += ps; profiles.append(prof)

    print("[01] MultiHop-RAG ...")
    qs, ps, prof = acquire_multihop_rag()
    all_questions += qs; all_passages += ps; profiles.append(prof)

    # Single-hop sets: LiveRAG probed first (it changes the even split of 300).
    loaded_single = []
    try:
        print("[01] LiveRAG (contamination control) ...")
        qs, ps, prof, repo = acquire_liverag(N_SINGLE_TOTAL // 4)
        loaded_single.append(("liverag", qs, ps, prof))
        update_manifest(liverag_source=repo)
    except Exception as e:
        failures["liverag"] = repr(e)
        print(f"  LiveRAG unavailable: {e!r}")

    single_sets = {"squad_v2": acquire_squad, "nq_open_gold": acquire_nq,
                   "ms_marco": acquire_msmarco}
    n_remaining = N_SINGLE_TOTAL - sum(len(x[1]) for x in loaded_single)
    names = list(single_sets)
    for i, name in enumerate(names):
        # split what's left evenly across the sets not yet loaded (incl. failures)
        sets_left = len(names) - i
        n = n_remaining // sets_left + (1 if n_remaining % sets_left else 0)
        try:
            print(f"[01] {name} (n={n}) ...")
            qs, ps, prof = single_sets[name](n)
            loaded_single.append((name, qs, ps, prof))
            n_remaining -= len(qs)
        except Exception as e:
            failures[name] = repr(e)
            print(f"  {name} FAILED: {e!r}")

    if n_remaining > 0:
        print(f"  [01] single-hop shortfall: {n_remaining} unfilled — recorded as deviation")
        failures["_single_hop_shortfall"] = f"{n_remaining} questions unfilled"

    for name, qs, ps, prof in loaded_single:
        all_questions += qs; all_passages += ps; profiles.append(prof)

    print("[01] WikiTableQuestions (structured, RQ4) ...")
    try:
        sqs, sps, sprof = acquire_wtq(N_STRUCTURED)
        profiles.append(sprof)
        all_passages += sps
    except Exception as e:
        failures["wikitablequestions"] = repr(e)
        sqs = []
        print(f"  wikitablequestions FAILED: {e!r}")

    # ---------------- write outputs
    qdf = pd.DataFrame(all_questions)
    for col in ("gold_answers", "gold_titles", "gold_passages"):
        qdf[col] = qdf[col].apply(list)
    sdf = pd.DataFrame(sqs)
    if len(sdf):
        for col in ("gold_answers", "gold_titles", "gold_passages"):
            sdf[col] = sdf[col].apply(list)

    pdf = pd.DataFrame(all_passages)
    pdf.insert(0, "passage_id", [f"p{i:06d}" for i in range(len(pdf))])

    qdf.to_parquet(Q_PRIMARY, index=False)
    sdf.to_parquet(Q_STRUCTURED, index=False)
    pdf.to_parquet(PASSAGES, index=False)
    pd.DataFrame(profiles).to_csv(RAW_PROFILE, index=False)
    make_profile_figure(qdf, sdf)

    counts = qdf.groupby(["dataset", "hop_type"]).size()
    summary = (
        f"primary questions: {len(qdf)} "
        f"(multi={sum(qdf.hop_type == 'multi')}, single={sum(qdf.hop_type == 'single')})\n"
        f"structured questions: {len(sdf)}\n"
        f"passage pool: {len(pdf)} passages "
        f"({pdf.is_gold.sum()} gold-linked)\n"
        f"per dataset:\n{counts.to_string()}\n"
        f"failures: {failures or 'none'}"
    )
    print(summary)

    manifest_kwargs = dict(
        stage01=dict(
            primary_n=len(qdf),
            structured_n=len(sdf),
            passages_n=len(pdf),
            allocation={f"{k[0]}/{k[1]}": int(v) for k, v in counts.items()},
            failures=failures,
        )
    )
    if failures:
        manifest_kwargs["deviation"] = f"stage01 dataset failures: {sorted(failures)}"
    else:
        # clean run supersedes any stale failure deviations from earlier attempts
        manifest_kwargs["resolve_deviation_prefix"] = "stage01 dataset failures"
    update_manifest(**manifest_kwargs)
    append_log(
        "Stage 01 acquire — run complete",
        f"```\n{summary}\n```\n"
        f"Outputs: questions_primary.parquet, questions_structured.parquet, "
        f"passages_pool.parquet, raw_profile.csv, fig_dataset_profile.png. Seed {SEED}.",
    )


if __name__ == "__main__":
    main()
