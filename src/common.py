"""Shared utilities: paths, seed, tokenizer, OpenRouter clients, caches, manifest.

Token accounting is the object of study: all budget math uses the real
tokenizer of the fixed generator model (Qwen3, via transformers), never
word counts and never a mismatched tokenizer.

Everything model-shaped runs through OpenRouter with one API key:
generator, judge, embeddings, reranker.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests as _requests
from dotenv import load_dotenv

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
LLM_CACHE = ROOT / "llm_cache"
EMB_CACHE = ROOT / "llm_cache" / "embeddings"
MANIFEST_PATH = DATA / "manifest.json"

for _d in (DATA, OUTPUTS, LLM_CACHE, EMB_CACHE):
    _d.mkdir(exist_ok=True, parents=True)

load_dotenv(ROOT / ".env")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")
GENERATOR_PROVIDER = os.getenv("GENERATOR_PROVIDER", "coreweave")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "baai/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cohere/rerank-v3.5")
PG_DSN = os.getenv("PG_DSN", "postgresql://rag:rag@127.0.0.1:5434/ragtb")

# HF repo whose tokenizer matches GENERATOR_MODEL exactly
GENERATOR_HF_TOKENIZER = os.getenv(
    "GENERATOR_HF_TOKENIZER", "Qwen/Qwen3-30B-A3B-Instruct-2507"
)

# Chunking parameters (recorded in the manifest; cited in the report)
CHUNK_TOKENS = 128
CHUNK_OVERLAP = 32
MIN_CHUNK_TOKENS = 20

BUDGETS = [500, 1000, 2000, 4000]
ARMS = ["naive_topk", "rerank_topk", "compress_llmlingua", "summarize_recomp", "graph_select"]

_tokenizer = None


def get_tokenizer():
    """The generator's own tokenizer — budgets are enforced in its real tokens."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(GENERATOR_HF_TOKENIZER)
    return _tokenizer


def n_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tok = get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens])


# ---------------------------------------------------------------- manifest


def update_manifest(**kwargs) -> dict:
    """Merge key/value pairs into data/manifest.json (the audit trail the
    Interim Report discloses deviations from)."""
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    manifest.setdefault("seed", SEED)
    manifest.setdefault("generator_model", GENERATOR_MODEL)
    manifest.setdefault("generator_provider_pinned", GENERATOR_PROVIDER)
    manifest.setdefault("judge_model", JUDGE_MODEL)
    manifest.setdefault("embedding_model", f"{EMBEDDING_MODEL} via OpenRouter")
    manifest.setdefault("rerank_model", f"{RERANK_MODEL} via OpenRouter")
    manifest.setdefault("tokenizer", GENERATOR_HF_TOKENIZER)
    manifest.setdefault("deviations", [])
    for k, v in kwargs.items():
        if k == "deviation":  # append, never overwrite
            if v not in manifest["deviations"]:
                manifest["deviations"].append(v)
        else:
            manifest[k] = v
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------- OpenRouter


# ---------------------------------------------------------------- cache DB
# One SQLite file instead of per-call JSON files: the external drive is ExFAT
# with large allocation clusters, so tens of thousands of tiny files waste
# ~1000x their true size (21 GB observed for 0.4 GB of vectors).

import sqlite3  # noqa: E402
import threading  # noqa: E402

_db_local = threading.local()
_DB_PATH = LLM_CACHE / "cache.db"


def _cache_db() -> sqlite3.Connection:
    conn = getattr(_db_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_DB_PATH, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, payload TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        conn.execute("CREATE TABLE IF NOT EXISTS assembled (key TEXT PRIMARY KEY, payload TEXT)")
        conn.commit()
        _db_local.conn = conn
    return conn


def cache_get_json(table: str, key: str):
    row = _cache_db().execute(f"SELECT payload FROM {table} WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def cache_put_json(table: str, key: str, obj) -> None:
    conn = _cache_db()
    conn.execute(f"INSERT OR REPLACE INTO {table} (key, payload) VALUES (?, ?)",
                 (key, json.dumps(obj)))
    conn.commit()


def cache_get_vec(key: str):
    import numpy as _np

    row = _cache_db().execute("SELECT vec FROM emb WHERE key=?", (key,)).fetchone()
    return _np.frombuffer(row[0], dtype=_np.float32).tolist() if row else None


def cache_put_vecs(items: list[tuple[str, list[float]]]) -> None:
    import numpy as _np

    conn = _cache_db()
    conn.executemany(
        "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
        [(k, _np.asarray(v, dtype=_np.float32).tobytes()) for k, v in items],
    )
    conn.commit()


def _api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing. Copy .env.example to .env and set it.")
    return key


def openrouter_client():
    from openai import OpenAI

    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=_api_key())


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    cached: bool = False


def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def llm_generate(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    system: str | None = None,
    client=None,
    max_retries: int = 6,
) -> LLMResult:
    """Cached, retried chat completion. Cache key = hash(prompt, model, params);
    a crashed run resumes from llm_cache/ without paying twice.

    The generator model is pinned to one provider (bf16) so every run hits
    the same deployment; other models (judge) route normally.
    """
    model = model or GENERATOR_MODEL
    params = {"temperature": temperature, "max_tokens": max_tokens, "system": system}
    key = _cache_key({"prompt": prompt, "model": model, "params": params})
    hit = cache_get_json("llm", key)
    if hit is not None:
        return LLMResult(**{**hit, "cached": True})

    if client is None:
        client = openrouter_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    extra_body: dict = {"usage": {"include": True}}
    if model == GENERATOR_MODEL and GENERATOR_PROVIDER:
        extra_body["provider"] = {"only": [GENERATOR_PROVIDER], "allow_fallbacks": False}

    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            latency = time.time() - t0
            usage = resp.usage
            cost = 0.0
            if usage is not None and getattr(usage, "model_extra", None):
                cost = float(usage.model_extra.get("cost") or 0.0)
            result = LLMResult(
                text=resp.choices[0].message.content or "",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_s=round(latency, 3),
                cost_usd=cost,
            )
            cache_put_json(
                "llm", key,
                {
                    "text": result.text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "latency_s": result.latency_s,
                    "cost_usd": result.cost_usd,
                },
            )
            return result
        except Exception as e:  # rate limits, transient network
            last_err = e
            wait = min(2**attempt, 30)
            print(f"  LLM call failed ({e!r}); retry {attempt + 1}/{max_retries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err!r}")


def embed_texts(texts: list[str], *, batch_size: int = 64) -> "list[list[float]]":
    """Embed via OpenRouter (BGE-M3, 1024-dim), disk-cached per text hash."""
    out: list = [None] * len(texts)
    missing_idx = []
    for i, t in enumerate(texts):
        k = _cache_key({"embed": t, "model": EMBEDDING_MODEL})
        hit = cache_get_vec(k)
        if hit is not None:
            out[i] = hit
        else:
            missing_idx.append(i)

    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    for start in range(0, len(missing_idx), batch_size):
        batch = missing_idx[start : start + batch_size]
        payload = {"model": EMBEDDING_MODEL, "input": [texts[i] for i in batch]}
        for attempt in range(6):
            try:
                r = _requests.post(
                    f"{OPENROUTER_BASE_URL}/embeddings", headers=headers, json=payload, timeout=120
                )
                r.raise_for_status()
                data = r.json()["data"]
                break
            except Exception as e:
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 30))
                print(f"  embed batch retry {attempt + 1}: {e!r}")
        puts = []
        for j, item in zip(batch, data):
            vec = item["embedding"]
            puts.append((_cache_key({"embed": texts[j], "model": EMBEDDING_MODEL}), vec))
            out[j] = vec
        cache_put_vecs(puts)
    return out


def rerank(query: str, documents: list[str], *, top_n: int | None = None) -> list[tuple[int, float]]:
    """Rerank via OpenRouter's Cohere-compatible endpoint, disk-cached.

    Returns [(index_into_documents, relevance_score)] sorted descending.
    """
    k = _cache_key({"rerank": query, "docs": documents, "model": RERANK_MODEL, "top_n": top_n})
    hit = cache_get_json("llm", f"rerank_{k}")
    if hit is not None:
        return [tuple(x) for x in hit]

    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    payload = {"model": RERANK_MODEL, "query": query, "documents": documents}
    if top_n:
        payload["top_n"] = top_n
    for attempt in range(6):
        try:
            r = _requests.post(
                f"{OPENROUTER_BASE_URL}/rerank", headers=headers, json=payload, timeout=120
            )
            r.raise_for_status()
            results = r.json()["results"]
            break
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))
            print(f"  rerank retry {attempt + 1}: {e!r}")
    ranked = sorted(
        [(item["index"], float(item["relevance_score"])) for item in results],
        key=lambda x: -x[1],
    )
    cache_put_json("llm", f"rerank_{k}", ranked)
    return ranked


# ---------------------------------------------------------------- chunks


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str
    n_tokens: int
    dataset: str
    content_type: str = "prose"  # prose | structured
    score: float = 0.0  # retrieval similarity, filled at query time
    extra: dict = field(default_factory=dict)


def append_log(title: str, body: str) -> None:
    """Append a timestamped entry to EXPERIMENT_LOG.md (the lab journal)."""
    log = ROOT / "EXPERIMENT_LOG.md"
    stamp = time.strftime("%Y-%m-%d %H:%M")
    with log.open("a") as f:
        f.write(f"\n---\n\n## {stamp} — {title}\n\n{body.strip()}\n")


def skip_if_exists(paths: list[Path], force: bool, stage: str) -> bool:
    """True if the stage should be skipped (all outputs exist, no --force)."""
    if force:
        return False
    if all(p.exists() for p in paths):
        print(f"[{stage}] outputs exist, skipping (use --force to rebuild):")
        for p in paths:
            print(f"  {p.relative_to(ROOT)}")
        return True
    return False
