"""Stage 02 — cleaning pipeline. Every step logs before/after counts because
the cleaning log is a graded deliverable (rubric table), not a debug aid.

Steps (in order):
  Q1  drop questions with empty/missing gold answers
  P1  strip markup/boilerplate from passages
  P2  normalize unicode (NFKC) + whitespace
  P3  deduplicate passages — exact (normalized hash)
  P4  deduplicate passages — near-dup (MinHash LSH, 5-word shingles, thr 0.9)
  Q2  drop questions whose gold evidence is absent from the cleaned corpus
      (also records gold_passage_ids for retrieval validation in 03)
  C1  chunk to 128 tokens / 32 overlap (generator tokenizer), drop <20-token chunks
  C2  tag structured-looking chunks in prose corpora (tables/code/lists)

Outputs:
  data/corpus_chunks.parquet
  data/questions_primary_clean.parquet, data/questions_structured_clean.parquet
  outputs/data_cleaning_log.csv   (exact rubric columns)
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    DATA,
    MIN_CHUNK_TOKENS,
    OUTPUTS,
    append_log,
    get_tokenizer,
    skip_if_exists,
    update_manifest,
)

Q_PRIMARY = DATA / "questions_primary.parquet"
Q_STRUCTURED = DATA / "questions_structured.parquet"
PASSAGES = DATA / "passages_pool.parquet"

CORPUS = DATA / "corpus_chunks.parquet"
QP_CLEAN = DATA / "questions_primary_clean.parquet"
QS_CLEAN = DATA / "questions_structured_clean.parquet"
CLEAN_LOG = OUTPUTS / "data_cleaning_log.csv"

LOG_ROWS: list[dict] = []


def log_step(issue, variables, detection, treatment, rationale, n_before, n_after):
    pct = 0.0 if n_before == 0 else round(100 * (n_before - n_after) / n_before, 2)
    LOG_ROWS.append(
        {
            "Issue": issue,
            "Variables Affected": variables,
            "Detection Method": detection,
            "Treatment Applied": treatment,
            "Rationale": rationale,
            "N Before": n_before,
            "N After": n_after,
            "Pct Removed": pct,
        }
    )
    print(f"  [{issue}] {n_before} -> {n_after} ({pct}%)")


_MARKUP_RES = [
    (re.compile(r"<[^>]{1,200}>"), " "),          # HTML tags
    (re.compile(r"\[\d+\]|\[citation needed\]", re.I), " "),  # wiki refs
    (re.compile(r"&(amp|nbsp|quot|lt|gt|#\d+);"), " "),       # HTML entities
]


def strip_markup(text: str) -> str:
    for rx, repl in _MARKUP_RES:
        text = rx.sub(repl, text)
    return text


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def norm_key(text: str) -> str:
    """Aggressive normalization for duplicate detection only."""
    return re.sub(r"\W+", " ", text.lower()).strip()


# Markdown tables, code fences, or a genuine bullet LIST (>=3 bullet lines —
# two lines starting with a dash are routinely just prose/transcripts and
# produced an ~8% false-positive rate on LiveRAG in review round 1)
STRUCT_RX = re.compile(
    r"(\|.+\|.+\n.*\|)|(```)|(^\s*[-*•]\s+.+\n\s*[-*•]\s+.+\n\s*[-*•]\s+)", re.M
)


def looks_structured(text: str) -> bool:
    return bool(STRUCT_RX.search(text)) or text.count(" | ") >= 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args, _ = ap.parse_known_args()

    if skip_if_exists([CORPUS, QP_CLEAN, QS_CLEAN, CLEAN_LOG,
                       DATA / "passages_clean.parquet"], args.force, "02_clean"):
        return

    qp = pd.read_parquet(Q_PRIMARY)
    qs = pd.read_parquet(Q_STRUCTURED)
    pf = pd.read_parquet(PASSAGES)

    # ---- Q1: empty/missing gold answers
    for name, df_ in (("primary", qp), ("structured", qs)):
        n0 = len(df_)
        keep = df_["gold_answers"].apply(
            lambda a: len(a) > 0 and any(str(x).strip() for x in a)
        )
        if name == "primary":
            qp = df_[keep].reset_index(drop=True)
            n1 = len(qp)
        else:
            qs = df_[keep].reset_index(drop=True)
            n1 = len(qs)
        log_step(
            f"Empty/missing gold answers ({name})",
            "gold_answers",
            "len==0 or all-whitespace strings",
            "Drop question",
            "EM/F1 need a gold string; unanswerable items are unscoreable",
            n0, n1,
        )

    # ---- P1: markup/boilerplate
    n_affected = int(pf["text"].apply(lambda t: any(rx.search(t) for rx, _ in _MARKUP_RES)).sum())
    pf["text"] = pf["text"].apply(strip_markup)
    log_step(
        "Markup/boilerplate in passages",
        "text",
        f"regex (HTML tags, wiki refs, entities); {n_affected} passages matched",
        "Strip matched spans",
        "Markup wastes budget tokens and pollutes embeddings",
        len(pf), len(pf),
    )

    # ---- P2: unicode + whitespace
    before_texts = pf["text"].copy()
    pf["text"] = pf["text"].apply(normalize)
    n_changed = int((before_texts != pf["text"]).sum())
    log_step(
        "Unicode/whitespace irregularities",
        "text",
        f"NFKC + whitespace collapse changed {n_changed} passages",
        "Normalize NFKC, collapse runs of spaces/newlines",
        "Stable tokenization and duplicate detection",
        len(pf), len(pf),
    )

    # drop passages emptied by cleaning
    n0 = len(pf)
    pf = pf[pf["text"].str.len() > 0].reset_index(drop=True)
    if len(pf) != n0:
        log_step("Passages emptied by cleaning", "text", "len==0 after strip",
                 "Drop passage", "No content left", n0, len(pf))

    # ---- P3: exact dedup (keep first, prefer gold copies)
    n0 = len(pf)
    pf["_key"] = pf["text"].apply(norm_key)
    pf = (
        # stable sort: which duplicate copy survives must not depend on the
        # sort implementation's tie order across pandas versions
        pf.sort_values("is_gold", ascending=False, kind="stable")
        .drop_duplicates("_key", keep="first")
        .sort_index()
        .reset_index(drop=True)
    )
    log_step(
        "Exact duplicate passages",
        "text",
        "normalized-text hash collision",
        "Keep one copy (gold-linked copy preferred)",
        "Duplicates waste budget and double-count evidence",
        n0, len(pf),
    )

    # ---- P4: near-dup via MinHash LSH
    from datasketch import MinHash, MinHashLSH

    n0 = len(pf)
    lsh = MinHashLSH(threshold=0.9, num_perm=64)
    keep_mask = [True] * len(pf)
    for i, (text, gold) in enumerate(zip(pf["text"], pf["is_gold"])):
        words = text.lower().split()
        m = MinHash(num_perm=64)
        for j in range(max(len(words) - 4, 1)):
            m.update(" ".join(words[j : j + 5]).encode())
        dup = lsh.query(m)
        if dup and not gold:  # never drop a gold passage as a near-dup
            keep_mask[i] = False
        else:
            lsh.insert(str(i), m)
    pf = pf[keep_mask].reset_index(drop=True)
    log_step(
        "Near-duplicate passages",
        "text",
        "MinHash LSH, 5-word shingles, Jaccard >= 0.9",
        "Drop later copy (gold passages exempt)",
        "Near-dups from overlapping sources inflate retrieval scores",
        n0, len(pf),
    )

    # ---- C1: chunk to fixed token length with the generator tokenizer.
    # Runs BEFORE the gold-evidence gate (Q2) so the gate sees what actually
    # survives into the retrievable corpus — a sub-minimum gold passage that
    # yields no chunk must fail Q2, not slip past it (review round 1 found two
    # such unwinnable questions shipped in the primary set).
    tok = get_tokenizer()
    chunks = []
    texts = pf["text"].tolist()
    encs = tok(texts, add_special_tokens=False)["input_ids"]
    n_pass = len(pf)
    n_fffd = 0
    for (_, row), ids in zip(pf.iterrows(), encs):
        step = CHUNK_TOKENS - CHUNK_OVERLAP
        i = 0
        c = 0
        while i < len(ids):
            piece = ids[i : i + CHUNK_TOKENS]
            # gold passages are exempt from the min-length drop: dropping a
            # question's only evidence for being short makes it unwinnable
            if len(piece) >= MIN_CHUNK_TOKENS or (bool(row["is_gold"]) and c == 0):
                text = tok.decode(piece)
                if "�" in text:  # token slice split a multi-byte char
                    text = text.replace("�", "")
                    n_fffd += 1
                # store the REAL re-encoded count: decode of a token slice is
                # not round-trip stable, and budget math must never overrun
                real_n = len(tok.encode(text, add_special_tokens=False))
                chunks.append(
                    dict(
                        chunk_id=f"{row['passage_id']}_c{c}",
                        passage_id=row["passage_id"],
                        dataset=row["dataset"],
                        title=row["title"],
                        text=text,
                        n_tokens=real_n,
                        content_type=row["content_type"],
                        is_gold=bool(row["is_gold"]),
                    )
                )
                c += 1
            if i + CHUNK_TOKENS >= len(ids):
                break
            i += step
    cdf = pd.DataFrame(chunks)
    log_step(
        f"Chunking ({CHUNK_TOKENS}-token target, {CHUNK_OVERLAP} overlap)",
        "text -> chunks",
        f"generator tokenizer; sub-{MIN_CHUNK_TOKENS}-token tails dropped "
        f"(gold passages exempt); {n_fffd} chunks with boundary-split chars repaired; "
        f"n_tokens re-encoded for exactness",
        f"{n_pass} passages -> {len(cdf)} chunks",
        "Fixed-size chunks make budget arithmetic exact across arms "
        "(negative Pct Removed = unit change, passages become chunks)",
        n_pass, len(cdf),
    )

    # ---- Q2: gold evidence must exist in the CHUNKED corpus.
    # Exact/near dedup is global across datasets, so a gold paragraph's
    # surviving copy may live in another dataset's partition — match same-
    # dataset first, then fall back to the whole corpus.
    chunked_pids = set(cdf["passage_id"])
    pf_chunked = pf[pf["passage_id"].isin(chunked_pids)]
    keys_by_ds: dict[str, dict] = {
        d: dict(zip(g["_key"], g["passage_id"])) for d, g in pf_chunked.groupby("dataset")
    }
    keys_global = dict(zip(pf_chunked["_key"], pf_chunked["passage_id"]))

    def match_gold(row) -> list[str]:
        """passage_ids (with >=1 retrievable chunk) carrying this question's gold
        evidence: exact normalized match, else containment for sentence facts."""
        local = keys_by_ds.get(row["dataset"], {})
        pids = []
        for gp in row["gold_passages"]:
            k = norm_key(strip_markup(normalize(str(gp))))
            if not k:
                continue
            if k in local:
                pids.append(local[k])
                continue
            if k in keys_global:
                pids.append(keys_global[k])
                continue
            hit = next((pid for key_, pid in local.items() if k in key_), None)
            if hit is None:
                hit = next((pid for key_, pid in keys_global.items() if k in key_), None)
            if hit:
                pids.append(hit)
        return sorted(set(pids))

    cleaned = {}
    for name, df_ in (("primary", qp), ("structured", qs)):
        n0 = len(df_)
        df_ = df_.copy()
        df_["gold_passage_ids"] = df_.apply(match_gold, axis=1)
        n_gold = df_["gold_passages"].apply(len)
        n_found = df_["gold_passage_ids"].apply(len)
        keep = (n_found > 0) | (n_gold == 0)
        cleaned[name] = df_[keep].reset_index(drop=True)
        log_step(
            f"Gold evidence absent from chunked corpus ({name})",
            "gold_passages, gold_passage_ids",
            "normalized exact/containment match vs passages with >=1 chunk",
            "Drop question",
            "An arm cannot retrieve evidence the corpus no longer contains",
            n0, len(cleaned[name]),
        )
    qp, qs = cleaned["primary"], cleaned["structured"]

    # ---- C2: tag structured-looking chunks in prose corpora (feeds RQ4)
    prose_mask = cdf["content_type"] == "prose"
    struct_hits = cdf.loc[prose_mask, "text"].apply(looks_structured)
    cdf.loc[prose_mask & cdf.index.isin(struct_hits[struct_hits].index), "content_type"] = "structured"
    log_step(
        "Structured content inside prose corpora",
        "content_type",
        "regex: markdown tables, code fences, bullet runs, pipe-delimited rows",
        f"Tagged {int(struct_hits.sum())} chunks content_type=structured",
        "RQ4 compares structured vs prose; the label must exist on every chunk",
        len(cdf), len(cdf),
    )

    pf.drop(columns=["_key"]).to_parquet(DATA / "passages_clean.parquet", index=False)
    cdf.to_parquet(CORPUS, index=False)
    qp.to_parquet(QP_CLEAN, index=False)
    qs.to_parquet(QS_CLEAN, index=False)
    pd.DataFrame(LOG_ROWS).to_csv(CLEAN_LOG, index=False)

    summary = (
        f"questions: primary {len(qp)}, structured {len(qs)}\n"
        f"corpus: {len(pf)} passages -> {len(cdf)} chunks "
        f"(mean {cdf['n_tokens'].mean():.1f} tokens; "
        f"{int((cdf['content_type'] == 'structured').sum())} structured)\n"
        f"gold coverage: primary questions with >=1 matched gold passage: "
        f"{int((qp['gold_passage_ids'].apply(len) > 0).sum())}/{len(qp)}"
    )
    print(summary)
    update_manifest(
        stage02=dict(
            chunk_tokens=CHUNK_TOKENS,
            chunk_overlap=CHUNK_OVERLAP,
            min_chunk_tokens=MIN_CHUNK_TOKENS,
            n_chunks=len(cdf),
            n_passages_clean=len(pf),
            questions_primary_clean=len(qp),
            questions_structured_clean=len(qs),
        )
    )
    append_log("Stage 02 clean — run complete",
               f"```\n{summary}\n```\nFull step log in outputs/data_cleaning_log.csv.")


if __name__ == "__main__":
    main()
