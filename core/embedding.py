"""文本向量化 Provider。默认用确定性哈希，可切换 sentence-transformers。"""
import hashlib

import numpy as np

from core.logging import get_logger

logger = get_logger("core.embedding")


class EmbeddingProvider:
    def embed(self, texts: list[str], dim: int = 64) -> list[list[float]]:
        raise NotImplementedError


class HashEmbedding(EmbeddingProvider):
    """离线确定性哈希向量，供测试与默认降级。"""

    def embed(self, texts: list[str], dim: int = 512) -> list[list[float]]:
        # 注：dim 取 512 而非 64——64 维下不同 n-gram 哈希到同一桶的概率过高，
        # 会导致与查询无关的文本因哈希碰撞而获得虚高相似度（检索结果错误）。
        # 512 维在离线确定性哈希下显著降低碰撞，同时保持接口与存储格式不变。
        vectors = []
        for text in texts:
            vec = np.zeros(dim, dtype=float)
            grams = _ngrams(text, n=3)
            for gram in grams:
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                vec[h % dim] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors


def _ngrams(text: str, n: int = 3):
    text = text.replace(" ", "")
    return [text[i : i + n] for i in range(max(0, len(text) - n + 1))]


def get_embedding_provider() -> EmbeddingProvider:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

        class STEmbedding(EmbeddingProvider):
            def embed(self, texts: list[str], dim: int = 64) -> list[list[float]]:
                return [v.tolist() for v in model.encode(texts)]

        logger.info("已加载 sentence-transformers 模型")
        return STEmbedding()
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法加载 sentence-transformers（%s），使用 HashEmbedding 降级", exc)
        return HashEmbedding()
