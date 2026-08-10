"""RAG：新闻/公告 索引与检索。"""
from core.logging import get_logger
from core.store import Store

logger = get_logger("core.rag")


def build_index(provider, news_items: list[dict], store: Store, symbol: str) -> None:
    if not news_items:
        return
    store.clear_chunks()
    texts, metas = [], []
    for i, item in enumerate(news_items):
        text = f"{item['title']}。{item['content']}"[:500]
        texts.append(text)
        metas.append({
            "title": item["title"], "date": item.get("date", ""),
            "source": item.get("source", ""), "url": item.get("url", ""),
            "symbol": item.get("symbol", symbol),
        })
    vectors = provider.embed(texts)
    for i, vec in enumerate(vectors):
        store.save_chunk({
            "chunk_id": f"{symbol}:{i}",
            "source_id": str(i),
            "text": texts[i],
            "meta": _to_json(metas[i]),
            "embedding": _float32_bytes(vec),
        })


def retrieve(provider, query: str, store: Store, top_k: int = 5) -> list[dict]:
    chunks = store.get_chunks()
    if not chunks:
        return []
    qvec = provider.embed([query])[0]
    scored = []
    for c in chunks:
        sim = _cosine(qvec, _bytes_to_float32(c["embedding"]))
        meta = _from_json(c["meta"])
        scored.append({
            "text": c["text"],
            "title": meta.get("title", ""),
            "date": meta.get("date", ""),
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
            "similarity": round(sim, 4),
        })
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    x, y = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) or 1.0
    return float(np.dot(x, y) / denom)


def _float32_bytes(vec: list[float]) -> bytes:
    import numpy as np

    return np.asarray(vec, dtype=np.float32).tobytes()


def _bytes_to_float32(raw: bytes) -> list[float]:
    import numpy as np

    return np.frombuffer(raw, dtype=np.float32).tolist()


def _to_json(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def _from_json(raw: str) -> dict:
    import json

    return json.loads(raw)
