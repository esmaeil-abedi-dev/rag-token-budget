"""Arm 5 (novel): graph-based selection.

Graph built once over the whole corpus in 04_arms.py:
  nodes = chunks
  edges = shared named entities (spaCy NER, noun-phrase fallback) OR
          top-k embedding-similarity neighbours
Query time: seed nodes by dense similarity, expand 1-2 hops along edges,
score nodes by  alpha * query_similarity + (1 - alpha) * subgraph degree
centrality,  then fill the budget with the highest-scoring connected set.

Hypothesis: multi-hop questions gain from bridging evidence that flat
similarity misses. Hyperparameters (hops, alpha) are exposed for the
sensitivity run the brief requires.
"""

from __future__ import annotations

import numpy as np

from . import finalize, greedy_fill

N_SEEDS = 20
MAX_SUBGRAPH = 400


def graph_select(question, candidates, budget, *,
                 graph=None, chunk_lookup=None, query_vec=None,
                 emb=None, id_to_idx=None,
                 hops: int = 2, alpha: float = 0.7, **ctx):
    if graph is None or chunk_lookup is None or query_vec is None:
        raise ValueError("graph_select needs graph/chunk_lookup/query_vec/emb/id_to_idx in ctx")

    seeds = [c.chunk_id for c in sorted(candidates, key=lambda c: -c.score)[:N_SEEDS]]

    # --- expand
    nodes = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        nxt = set()
        for cid in frontier:
            nxt.update(graph.get(cid, ()))
        nxt -= nodes
        # cap growth, preferring neighbours similar to the query
        if len(nodes) + len(nxt) > MAX_SUBGRAPH:
            idxs = [id_to_idx[c] for c in nxt if c in id_to_idx]
            if idxs:
                sims = emb[idxs] @ query_vec
                order = np.argsort(-sims)
                keep = [list(nxt)[i] for i in order[: MAX_SUBGRAPH - len(nodes)]]
                nxt = set(keep)
        nodes |= nxt
        frontier = nxt
        if not frontier:
            break

    node_list = [c for c in nodes if c in id_to_idx]
    idxs = np.array([id_to_idx[c] for c in node_list])
    sims = emb[idxs] @ query_vec  # unit-normalized upstream => cosine

    # --- degree centrality inside the induced subgraph
    node_set = set(node_list)
    deg = np.array([sum(1 for nb in graph.get(c, ()) if nb in node_set) for c in node_list],
                   dtype=float)
    if deg.max() > 0:
        deg = deg / deg.max()
    score = alpha * sims + (1 - alpha) * deg
    by_id = dict(zip(node_list, score))

    # --- fill the budget with the highest-scoring *connected* set
    selected: list[str] = []
    sel_set: set[str] = set()
    used = 0
    disconnected_picks = 0

    def fits(cid):
        return used + chunk_lookup[cid].n_tokens + 2 <= budget

    pool = sorted(node_list, key=lambda c: -by_id[c])
    while pool:
        if not selected:
            cand_pool = [c for c in pool if fits(c)]
        else:
            reachable = {nb for c in sel_set for nb in graph.get(c, ()) if nb in node_set}
            cand_pool = [c for c in pool if c in reachable and fits(c)]
            if not cand_pool:  # nothing connected fits — fall back to global best
                cand_pool = [c for c in pool if fits(c)]
                if cand_pool:
                    disconnected_picks += 1
        if not cand_pool:
            break
        best = max(cand_pool, key=lambda c: by_id[c])
        pool.remove(best)
        selected.append(best)
        sel_set.add(best)
        used += chunk_lookup[best].n_tokens + 2

    ordered = sorted((chunk_lookup[c] for c in selected), key=lambda ch: -by_id[ch.chunk_id])
    text, ids = greedy_fill(ordered, budget)
    return finalize(
        "graph_select", text, ids, budget,
        meta={"hops": hops, "alpha": alpha, "subgraph_nodes": len(node_list),
              "disconnected_picks": disconnected_picks, "n_seeds": len(seeds)},
    )
