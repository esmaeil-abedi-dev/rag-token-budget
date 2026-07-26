"""Query-time retrieval context shared by stages 04, 04b, 04c and 05.

The fixed retriever every arm shares: exact cosine top-k over the BGE-M3
matrix (pgvector holds the same vectors as the persistent store; exact search
is used at query time so ANN recall variance cannot contaminate the arm
comparison — recorded in the manifest).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, Chunk  # noqa: E402

N_CANDIDATES = 100  # the candidate pool the arms see (rerank reads all of it)


class RetrievalContext:
    def __init__(self):
        self.cdf = pd.read_parquet(DATA / "corpus_chunks.parquet")
        emb = np.load(DATA / "embeddings.npy")
        self.emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        self.chunk_ids = json.loads((DATA / "chunk_ids.json").read_text())
        self.id_to_idx = {c: i for i, c in enumerate(self.chunk_ids)}

        qv = np.load(DATA / "question_embeddings.npy")
        self.qvecs = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
        self.question_ids = json.loads((DATA / "question_ids.json").read_text())
        self.q_to_idx = {q: i for i, q in enumerate(self.question_ids)}

        self._chunks: dict[str, Chunk] = {}
        for row in self.cdf.itertuples(index=False):
            self._chunks[row.chunk_id] = Chunk(
                chunk_id=row.chunk_id, doc_title=row.title, text=row.text,
                n_tokens=int(row.n_tokens), dataset=row.dataset,
                content_type=row.content_type,
            )

        gp = DATA / "graph_edges.parquet"
        self.graph: dict[str, list[str]] = {}
        if gp.exists():
            edf = pd.read_parquet(gp)
            for s, d in zip(edf["src"], edf["dst"]):
                self.graph.setdefault(s, []).append(d)
                self.graph.setdefault(d, []).append(s)

    def chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def query_vec(self, question_id: str) -> np.ndarray:
        return self.qvecs[self.q_to_idx[question_id]]

    def candidates(self, question_id: str, k: int = N_CANDIDATES) -> list[Chunk]:
        """Top-k chunks by exact cosine; each Chunk carries its similarity score."""
        qv = self.query_vec(question_id)
        sims = self.emb @ qv
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        top = top[np.argsort(-sims[top], kind="stable")]
        out = []
        for j in top:
            c = self._chunks[self.chunk_ids[j]]
            fields = {**c.__dict__, "score": float(sims[j]), "extra": dict(c.extra)}
            out.append(Chunk(**fields))
        return out

    def arm_ctx(self, question_id: str) -> dict:
        """Extra kwargs the graph arm needs."""
        if not self.graph:
            raise RuntimeError(
                "graph_edges.parquet missing/empty — run src/04_arms.py first; "
                "running graph_select without the graph would silently degrade "
                "the novel arm to flat similarity and invalidate RQ3"
            )
        return dict(
            graph=self.graph,
            chunk_lookup=self._chunks,
            query_vec=self.query_vec(question_id),
            emb=self.emb,
            id_to_idx=self.id_to_idx,
        )
