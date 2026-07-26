"""Stage 03 — embed all chunks, build dense (pgvector) + sparse (BM25) indexes,
then VALIDATE retrieval before any arm uses it.

Validation: recall@k (k = 1,5,10,20,50) and MRR against gold supporting facts,
per dataset (HotpotQA is the gate: dense recall@50 < 0.7 => stop and report).
Retrieval is over the full pooled corpus (all datasets together) — one realistic
corpus, recorded as a design decision.

Outputs:
  pgvector table `chunks` (+ HNSW index)      the retrieval store (Synopsis: pgvector)
  data/embeddings.npy, data/chunk_ids.json    convenience copy for the graph arm
  data/bm25.pkl                               sparse baseline
  outputs/retrieval_quality.csv
  outputs/fig_retrieval_recall.png
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DATA,
    OUTPUTS,
    PG_DSN,
    append_log,
    embed_texts,
    skip_if_exists,
    update_manifest,
)

CORPUS = DATA / "corpus_chunks.parquet"
QP_CLEAN = DATA / "questions_primary_clean.parquet"
QS_CLEAN = DATA / "questions_structured_clean.parquet"

EMB_NPY = DATA / "embeddings.npy"
CHUNK_IDS = DATA / "chunk_ids.json"
BM25_PKL = DATA / "bm25.pkl"
RQ_CSV = OUTPUTS / "retrieval_quality.csv"
FIG_RECALL = OUTPUTS / "fig_retrieval_recall.png"

KS = [1, 5, 10, 20, 50]
RECALL_GATE = 0.7  # on HotpotQA dense recall@50, per the brief


def embed_parallel(texts: list[str], workers: int = 6, slice_size: int = 96) -> np.ndarray:
    """Thread-parallel wrapper over the cached embed_texts client."""
    slices = [(i, texts[i : i + slice_size]) for i in range(0, len(texts), slice_size)]
    out: list = [None] * len(texts)
    done = 0

    def run(arg):
        start, batch = arg
        return start, embed_texts(batch, batch_size=slice_size)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start, vecs in ex.map(run, slices):
            for j, v in enumerate(vecs):
                out[start + j] = v
            done += len(vecs)
            if done % (slice_size * workers * 4) < slice_size:
                print(f"  embedded {done}/{len(texts)}")
    return np.asarray(out, dtype=np.float32)


def load_into_pgvector(cdf: pd.DataFrame, emb: np.ndarray):
    import psycopg
    from pgvector.psycopg import register_vector

    dim = emb.shape[1]
    with psycopg.connect(PG_DSN) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(
            f"""CREATE TABLE chunks (
                chunk_id text PRIMARY KEY,
                passage_id text,
                dataset text,
                title text,
                text text,
                n_tokens int,
                content_type text,
                is_gold boolean,
                embedding vector({dim})
            )"""
        )
        with conn.cursor().copy(
            "COPY chunks (chunk_id, passage_id, dataset, title, text, n_tokens, "
            "content_type, is_gold, embedding) FROM STDIN"
        ) as copy:
            copy.set_types(["text", "text", "text", "text", "text", "int4", "text", "bool", "vector"])
            for row, vec in zip(cdf.itertuples(index=False), emb):
                copy.write_row(
                    (row.chunk_id, row.passage_id, row.dataset, row.title, row.text,
                     int(row.n_tokens), row.content_type, bool(row.is_gold), vec)
                )
        print("  building HNSW index (cosine) ...")
        conn.execute(
            "CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
        conn.commit()


def dense_topk(qvecs: np.ndarray, emb: np.ndarray, k: int) -> np.ndarray:
    """Exact cosine top-k via the in-memory matrix. NOTE: exact search is used
    for validation here AND for query-time retrieval in 04/05 (retrieval_ctx) —
    pgvector holds the same vectors as the persistent store, so ANN recall
    variance never contaminates the arm comparison. Recorded in the manifest."""
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    q_n = qvecs / (np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9)
    sims = q_n @ emb_n.T
    return np.argsort(-sims, axis=1)[:, :k]


def _unique_pids(idx_row, pid_of, cap: int) -> list:
    """Chunk ranks -> UNIQUE passage ids in rank order: recall@k must count
    passage slots, not let one long passage's 5 chunks occupy 5 of the top k."""
    seen, out = set(), []
    for j in idx_row:
        p = pid_of[j]
        if p not in seen:
            seen.add(p)
            out.append(p)
            if len(out) == cap:
                break
    return out


def evaluate_retrieval(questions: pd.DataFrame, cdf: pd.DataFrame, emb: np.ndarray,
                       bm25, qvecs: np.ndarray) -> pd.DataFrame:
    """Doc-level recall@k and MRR: a gold passage counts as retrieved when any of
    its chunks appears among the top-k DISTINCT passages."""
    pid_of = cdf["passage_id"].to_numpy()
    rows = []
    # retrieve deep enough that k DISTINCT passages survive chunk-dedup even in
    # corpora where one long article contributes dozens of top chunks
    # (MultiHop-RAG p95 = 56 chunks/passage)
    depth = max(KS) * 20
    top_dense = dense_topk(qvecs, emb, min(depth, emb.shape[0]))

    for qi, (_, q) in enumerate(questions.iterrows()):
        gold = set(q["gold_passage_ids"])
        if not gold:
            continue
        ranked_pids = _unique_pids(top_dense[qi], pid_of, max(KS))
        scores = bm25.get_scores(str(q["question"]).lower().split())
        bm_idx = np.argsort(-scores)[:depth]
        bm_pids = _unique_pids(bm_idx, pid_of, max(KS))

        for method, plist in (("dense_bge_m3", ranked_pids), ("bm25", bm_pids)):
            first_hit = next((r + 1 for r, p in enumerate(plist) if p in gold), None)
            rec = {}
            for k in KS:
                found = {p for p in plist[:k] if p in gold}
                rec[k] = len(found) / len(gold)
            rows.append(
                dict(dataset=q["dataset"], question_id=q["question_id"], method=method,
                     mrr=1.0 / first_hit if first_hit else 0.0,
                     **{f"recall@{k}": rec[k] for k in KS})
            )
    per_q = pd.DataFrame(rows)
    agg = (
        per_q.groupby(["dataset", "method"])[[f"recall@{k}" for k in KS] + ["mrr"]]
        .mean()
        .round(4)
        .reset_index()
    )
    pooled = (
        per_q.groupby("method")[[f"recall@{k}" for k in KS] + ["mrr"]]
        .mean().round(4).reset_index()
    )
    pooled.insert(0, "dataset", "POOLED")
    return pd.concat([agg, pooled], ignore_index=True)


def make_figure(rq: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, method in zip(axes, ["dense_bge_m3", "bm25"]):
        sub = rq[rq["method"] == method]
        for _, r in sub.iterrows():
            style = dict(lw=2.5, marker="o") if r["dataset"] == "POOLED" else dict(lw=1, marker=".", alpha=0.7)
            ax.plot(KS, [r[f"recall@{k}"] for k in KS], label=r["dataset"], **style)
        ax.axhline(RECALL_GATE, color="red", ls="--", lw=1, label=f"gate {RECALL_GATE} @50")
        ax.set_title(f"{method} — doc-level recall@k")
        ax.set_xlabel("k"); ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("recall (gold passages found / gold passages)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Retrieval validation before use — fixed retriever, pooled corpus")
    fig.tight_layout()
    fig.savefig(FIG_RECALL, dpi=300)
    print(f"  wrote {FIG_RECALL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    args, _ = ap.parse_known_args()

    outputs_03 = [EMB_NPY, CHUNK_IDS, BM25_PKL, RQ_CSV, FIG_RECALL,
                  DATA / "question_embeddings.npy", DATA / "question_ids.json"]
    if skip_if_exists(outputs_03, args.force, "03_index"):
        # a completed-but-FAILED gate must keep failing on re-run, and outputs
        # WITHOUT a manifest record (crash before update_manifest) don't count
        # as a completed stage either
        manifest = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else {}
        gate = manifest.get("stage03", {}).get("gate_passed")
        if gate is False:
            print("[03] previous run FAILED the retrieval gate — refusing to pass silently")
            sys.exit(2)
        if gate is True:
            return
        print("[03] outputs exist but no stage03 manifest record — re-validating")

    cdf = pd.read_parquet(CORPUS)
    qp = pd.read_parquet(QP_CLEAN)
    qs = pd.read_parquet(QS_CLEAN)

    # spend gate (every paid stage projects and confirms)
    total_tokens = int(cdf["n_tokens"].sum()) + 40 * (len(qp) + len(qs))
    est = total_tokens * 0.01 / 1e6  # bge-m3 $0.01/M; cache hits make this an upper bound
    print(f"[03] embedding projection: ~{total_tokens/1e6:.1f}M tokens ≈ ${est:.2f} "
          f"(upper bound; already-cached vectors are free)")
    if not args.yes:
        resp = input("Proceed with embedding spend? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted before spending")
            return
    from common import EMBEDDING_MODEL

    print(f"[03] embedding {len(cdf)} chunks via OpenRouter ({EMBEDDING_MODEL}) ...")
    emb = embed_parallel(cdf["text"].tolist())
    np.save(EMB_NPY, emb)
    CHUNK_IDS.write_text(json.dumps(cdf["chunk_id"].tolist()))

    print("[03] loading pgvector ...")
    load_into_pgvector(cdf, emb)

    print("[03] building BM25 ...")
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([t.lower().split() for t in cdf["text"]])
    with BM25_PKL.open("wb") as f:
        pickle.dump(bm25, f)

    print("[03] embedding questions ...")
    allq = pd.concat([qp, qs], ignore_index=True)
    qvecs = embed_parallel(allq["question"].tolist())
    np.save(DATA / "question_embeddings.npy", qvecs)
    (DATA / "question_ids.json").write_text(json.dumps(allq["question_id"].tolist()))

    print("[03] validating retrieval ...")
    rq = evaluate_retrieval(allq, cdf, emb, bm25, qvecs)
    rq.to_csv(RQ_CSV, index=False)
    print(rq.to_string(index=False))
    make_figure(rq)

    hotpot_dense = rq[(rq.dataset.str.startswith("hotpot")) & (rq.method == "dense_bge_m3")]
    gate_val = float(hotpot_dense["recall@50"].iloc[0]) if len(hotpot_dense) else float("nan")
    gate_ok = gate_val >= RECALL_GATE

    update_manifest(
        stage03=dict(
            n_chunks=len(cdf),
            embedding_dim=int(emb.shape[1]),
            vector_store=("pgvector (podman) persists all vectors + HNSW index; "
                          "query-time retrieval uses exact in-memory cosine over the "
                          "same vectors to eliminate ANN recall variance"),
            hotpot_dense_recall_at_50=gate_val,
            gate_passed=bool(gate_ok),
        )
    )
    append_log(
        "Stage 03 index — retrieval validated" if gate_ok else
        "Stage 03 index — RETRIEVAL GATE FAILED",
        f"```\n{rq.to_string(index=False)}\n```\n"
        f"HotpotQA dense recall@50 = {gate_val:.3f} (gate {RECALL_GATE}) — "
        f"{'PASS' if gate_ok else 'FAIL: downstream arms are capped by retrieval; stopping.'}",
    )
    if not gate_ok:
        print(f"\nRETRIEVAL GATE FAILED: HotpotQA dense recall@50 = {gate_val:.3f} < {RECALL_GATE}.")
        print("Stopping per the brief — the report must attribute weakness to retrieval, not the arms.")
        sys.exit(2)
    print(f"[03] gate passed: HotpotQA dense recall@50 = {gate_val:.3f}")


if __name__ == "__main__":
    main()
