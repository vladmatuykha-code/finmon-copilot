"""Індекс нормативки: векторний пошук (Qdrant + fastembed), BM25 і гібрид через Reciprocal Rank Fusion.

    python -m finmon.rag.index            # побудувати індекси для всіх стратегій нарізки
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from ..config import INDEX_DIR
from .chunking import Chunk, build_all

RRF_K = 60
STOP = {"та", "або", "що", "які", "яка", "який", "яке", "для", "від", "при", "щодо", "його", "її", "їх", "цього", "цієї", "також", "якщо", "коли", "чи", "не", "як", "на", "до", "за", "по", "із", "з", "у", "в", "і", "й", "а"}


def tokenize(text: str) -> list[str]:
    """Проста нормалізація для української: нижній регістр і обрізання слова до 6 літер замість стемера."""
    words = re.findall(r"[a-zа-яіїєґ0-9'’]+", text.lower())
    return [w[:6] for w in words if w not in STOP and len(w) > 1]


EMBED_MODELS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 128 токенів, швидка
    "e5": "intfloat/multilingual-e5-large",  # 512 токенів, повільніша і точніша
}


@lru_cache(maxsize=2)
def embedder(model_key: str):
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODELS[model_key], cache_dir=str(INDEX_DIR / "models"))


def embed(texts: list[str], model_key: str = "minilm", kind: str = "passage") -> np.ndarray:
    if model_key == "e5":  # e5 очікує префікси query:/passage:
        texts = [f"{kind}: {t}" for t in texts]
    vecs = np.array(list(embedder(model_key).embed(texts, batch_size=32)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _qdrant(model_key: str = "minilm") -> QdrantClient:
    """Окреме локальне сховище на кожну модель ембедингів (локальний Qdrant блокує теку на час роботи)."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(INDEX_DIR / ("qdrant" if model_key == "minilm" else f"qdrant_{model_key}")))


def collection(strategy: str, model_key: str) -> str:
    return strategy if model_key == "minilm" else f"{strategy}__{model_key}"


def build(strategies: list[str] | None = None, model_key: str = "minilm") -> None:
    all_chunks = build_all()
    client = _qdrant(model_key)
    for name, chunks in all_chunks.items():
        if strategies and name not in strategies:
            continue
        (INDEX_DIR / f"{name}.chunks.json").write_text(json.dumps([c.__dict__ for c in chunks], ensure_ascii=False), encoding="utf-8")
        vecs = embed([c.embed_text for c in chunks], model_key)
        coll = collection(name, model_key)
        if client.collection_exists(coll):
            client.delete_collection(coll)
        client.create_collection(coll, vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.COSINE))
        client.upsert(coll, [PointStruct(id=i, vector=v.tolist(), payload={**c.__dict__}) for i, (c, v) in enumerate(zip(chunks, vecs))])
        np.save(INDEX_DIR / f"{coll}.npy", vecs)  # копія для пошуку в пам'яті без Qdrant
        print(f"{coll}: {len(chunks)} chunks indexed")
    client.close()


def export_vectors(model_key: str = "minilm") -> None:
    """Зберегти вектори з Qdrant у .npy (для індексів, побудованих до появи цієї функції)."""
    client = _qdrant(model_key)
    for coll in [c.name for c in client.get_collections().collections]:
        n = client.count(coll).count
        pts, _ = client.scroll(coll, limit=n, with_vectors=True, with_payload=False)
        vecs = np.array([p.vector for p in sorted(pts, key=lambda p: p.id)], dtype=np.float32)
        np.save(INDEX_DIR / f"{coll}.npy", vecs)
        print(coll, vecs.shape)
    client.close()


class Retriever:
    """Пошук по одній стратегії нарізки. mode: bm25 | dense | hybrid."""

    def __init__(self, strategy: str = "structural_short", model_key: str = "minilm", client: QdrantClient | None = None, backend: str = "auto"):
        self.strategy = strategy
        self.model_key = model_key
        self.coll = collection(strategy, model_key)
        self.client, self.vecs = client, None
        npy = INDEX_DIR / f"{self.coll}.npy"
        if backend == "memory" or (backend == "auto" and client is None and npy.exists()):
            self.vecs = np.load(npy)  # 1–2 тис. векторів: точний пошук у пам'яті швидший за будь-яку БД
        elif self.client is None:
            self.client = _qdrant(model_key)
        data = json.loads((INDEX_DIR / f"{strategy}.chunks.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk(**d) for d in data]
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in self.chunks])

    def _dense(self, query: str, n: int) -> list[int]:
        qv = embed([query], self.model_key, kind="query")[0]
        if self.vecs is not None:
            return list(np.argsort(-(self.vecs @ qv))[:n])
        res = self.client.query_points(self.coll, query=qv.tolist(), limit=n).points
        return [p.id for p in res]

    def _bm25(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:n])

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
        if mode == "dense":
            ids = self._dense(query, k)
        elif mode == "bm25":
            ids = self._bm25(query, k)
        else:
            fused: dict[int, float] = {}
            for ranking in (self._dense(query, 40), self._bm25(query, 40)):
                for rank, idx in enumerate(ranking):
                    fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
            ids = [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]
        out = []
        for i in ids:
            c = self.chunks[int(i)]
            out.append({"chunk_id": c.chunk_id, "source": c.header or c.refs[0] if c.refs else c.doc, "refs": c.refs, "text": c.text})
        return out


@lru_cache(maxsize=4)
def get_retriever(strategy: str = "structural_short", model_key: str = "minilm") -> Retriever:
    return Retriever(strategy, model_key)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "export":
        export_vectors(args[1] if len(args) > 1 else "minilm")
    else:
        model = args[0] if args else "minilm"
        build(args[1:] or None, model)
