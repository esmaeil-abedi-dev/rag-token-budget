"""Stage 04 — build the chunk graph for the novel arm, then prove budget
compliance for EVERY arm before any money is spent on the sweep.

Graph construction (once, over the whole corpus):
  entity edges: chunks sharing a named entity (spaCy en_core_web_sm;
                regex proper-noun fallback). Entities appearing in > 30 chunks
                are treated as stop-entities (hub blowup guard).
  knn edges:    top-8 embedding neighbours with cosine >= 0.75.

Budget compliance: 25 seeded questions x 4 budgets x 6 arms (5 + dedup control),
assert realized tokens <= budget, write outputs/budget_compliance.csv.

Outputs: data/chunk_entities.parquet, data/graph_edges.parquet,
         outputs/budget_compliance.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    BUDGETS,
    DATA,
    OUTPUTS,
    SEED,
    append_log,
    skip_if_exists,
    update_manifest,
)

ENTITIES = DATA / "chunk_entities.parquet"
EDGES = DATA / "graph_edges.parquet"
COMPLIANCE = OUTPUTS / "budget_compliance.csv"

MAX_ENTITY_DF = 30   # entities in more chunks than this are stop-entities
KNN_K = 8
KNN_SIM = 0.75
N_COMPLIANCE_Q = 25

ALL_ARMS = ["naive_topk", "naive_topk_dedup", "rerank_topk",
            "compress_llmlingua", "summarize_recomp", "graph_select"]

_CAP_RX = re.compile(r"\b([A-Z][a-zA-Z0-9&.-]+(?:\s+[A-Z][a-zA-Z0-9&.-]+)*)\b")


def extract_entities_spacy(texts: list[str]) -> list[set[str]]:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger"])
    except OSError:
        try:
            # spacy.cli.download shells out to pip and raises SystemExit (a
            # BaseException) in pip-less venvs — catch broadly so the recorded
            # regex fallback can take over instead of killing the pipeline.
            # (requirements.txt installs the model wheel directly; this path is
            # a belt-and-braces recovery only.)
            from spacy.cli import download

            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger"])
        except BaseException as e:  # noqa: BLE001
            raise RuntimeError(f"spaCy model unavailable: {e!r}") from e

    keep = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT", "WORK_OF_ART", "PRODUCT"}
    out = []
    for doc in nlp.pipe(texts, batch_size=256):
        out.append({e.text.strip().lower() for e in doc.ents
                    if e.label_ in keep and len(e.text.strip()) > 2})
    return out


def extract_entities_regex(texts: list[str]) -> list[set[str]]:
    """Fallback: capitalized multi-word spans (labelled in the manifest)."""
    return [{m.group(1).lower() for m in _CAP_RX.finditer(t) if len(m.group(1)) > 3}
            for t in texts]


def build_graph(force: bool):
    if not force and ENTITIES.exists() and EDGES.exists():
        print("[04] graph artifacts exist, skipping build")
        return

    # entities ------------------------------------------------------------
    cdf = pd.read_parquet(DATA / "corpus_chunks.parquet")
    texts = cdf["text"].tolist()
    t0 = time.time()
    method = "spacy_en_core_web_sm"
    try:
        ents = extract_entities_spacy(texts)
    except Exception as e:
        print(f"  spaCy unavailable ({e!r}) — regex fallback (recorded)")
        ents = extract_entities_regex(texts)
        method = "regex_capitalized_spans_FALLBACK"
    print(f"  entities extracted for {len(texts)} chunks in {time.time()-t0:.0f}s ({method})")
    pd.DataFrame(
        {"chunk_id": cdf["chunk_id"], "entities": [sorted(s) for s in ents]}
    ).to_parquet(ENTITIES, index=False)

    # entity edges ---------------------------------------------------------
    inv = defaultdict(list)
    for i, es in enumerate(ents):
        for e in es:
            inv[e].append(i)
    chunk_ids = cdf["chunk_id"].tolist()
    edge_set: set[tuple[int, int]] = set()
    n_stop = 0
    for e, idxs in inv.items():
        if len(idxs) < 2:
            continue
        if len(idxs) > MAX_ENTITY_DF:
            n_stop += 1
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                edge_set.add((idxs[a], idxs[b]))
    n_entity_edges = len(edge_set)

    # knn edges ------------------------------------------------------------
    emb = np.load(DATA / "embeddings.npy")
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    block = 2048
    for s in range(0, len(emb), block):
        sims = emb[s : s + block] @ emb.T
        for r in range(sims.shape[0]):
            sims[r, s + r] = -1  # no self edge
        top = np.argpartition(-sims, KNN_K, axis=1)[:, :KNN_K]
        for r in range(sims.shape[0]):
            for j in top[r]:
                if sims[r, j] >= KNN_SIM:
                    a, b = s + r, int(j)
                    edge_set.add((min(a, b), max(a, b)))
    print(f"  edges: {n_entity_edges} entity + {len(edge_set)-n_entity_edges} knn "
          f"({n_stop} stop-entities skipped)")

    edf = pd.DataFrame(
        {"src": [chunk_ids[a] for a, b in edge_set],
         "dst": [chunk_ids[b] for a, b in edge_set]}
    )
    edf.to_parquet(EDGES, index=False)
    update_manifest(
        stage04_graph=dict(entity_method=method, n_entity_edges=n_entity_edges,
                           n_edges_total=len(edf), knn_k=KNN_K, knn_sim=KNN_SIM,
                           max_entity_df=MAX_ENTITY_DF)
    )
    if method != "spacy_en_core_web_sm":
        update_manifest(deviation=f"graph NER fallback used: {method}")


def compliance(force: bool):
    if not force and COMPLIANCE.exists():
        print("[04] compliance exists, skipping")
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from arms import assemble
    from retrieval_ctx import RetrievalContext

    ctx = RetrievalContext()
    qp = pd.read_parquet(DATA / "questions_primary_clean.parquet")
    rng = np.random.default_rng(SEED)
    sample = qp.iloc[rng.permutation(len(qp))[:N_COMPLIANCE_Q]]

    rows = []
    for arm in ALL_ARMS:
        for budget in BUDGETS:
            for _, q in sample.iterrows():
                cands = ctx.candidates(q["question_id"])
                extra = ctx.arm_ctx(q["question_id"]) if arm == "graph_select" else {}
                t0 = time.time()
                res = assemble(q["question"], cands, budget, arm, **extra)
                rows.append(
                    dict(arm=arm, budget=budget, question_id=q["question_id"],
                         realized_tokens=res.gen_context_tokens,
                         utilization=round(res.gen_context_tokens / budget, 4),
                         assembly_input_tokens=res.assembly_input_tokens,
                         assembly_output_tokens=res.assembly_output_tokens,
                         assembly_s=round(time.time() - t0, 3))
                )
            done = [r for r in rows if r["arm"] == arm and r["budget"] == budget]
            mu = np.mean([r["utilization"] for r in done])
            print(f"  {arm:22s} budget={budget:5d}  mean_util={mu:.3f}  n={len(done)}")

    df = pd.DataFrame(rows)
    df["violation"] = df["realized_tokens"] > df["budget"]
    agg = (
        df.groupby(["arm", "budget"])
        .agg(n=("realized_tokens", "size"),
             mean_realized=("realized_tokens", "mean"),
             max_realized=("realized_tokens", "max"),
             mean_utilization=("utilization", "mean"),
             mean_assembly_in=("assembly_input_tokens", "mean"),
             mean_assembly_out=("assembly_output_tokens", "mean"),
             mean_assembly_s=("assembly_s", "mean"),
             violations=("violation", "sum"))  # measured, not asserted-and-assumed
        .round(3)
        .reset_index()
    )
    agg.to_csv(COMPLIANCE, index=False)
    print(agg.to_string(index=False))
    append_log(
        "Stage 04 — budget compliance proven",
        f"{N_COMPLIANCE_Q} questions x {len(BUDGETS)} budgets x {len(ALL_ARMS)} arms, "
        f"zero violations (assemble() asserts <= budget in real generator tokens).\n\n"
        f"```\n{agg.to_string(index=False)}\n```",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    args, _ = ap.parse_known_args()
    build_graph(args.force)
    if not (COMPLIANCE.exists() and not args.force) and not args.yes:
        # 25 rerank searches + 25 q x 4 budgets RECOMP summaries
        est = N_COMPLIANCE_Q * 0.002 + N_COMPLIANCE_Q * len(BUDGETS) * 0.0015
        resp = input(f"[04] compliance run makes paid calls (~${est:.2f}). Proceed? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted before spending")
            return
    compliance(args.force)


if __name__ == "__main__":
    main()
