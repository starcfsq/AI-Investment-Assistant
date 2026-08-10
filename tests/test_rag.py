import tempfile

from core.embedding import HashEmbedding
from core.rag import build_index, retrieve
from core.store import Store


def test_retrieve_finds_most_similar():
    store = Store(tempfile.mkdtemp() + "/t.db")
    emb = HashEmbedding()
    news = [
        {"title": "半导体板块大涨", "content": "芯片需求旺盛", "date": "2026-08-09",
         "source": "s", "url": "http://a", "symbol": "600001"},
        {"title": "白酒板块回调", "content": "白酒库存偏高", "date": "2026-08-08",
         "source": "s", "url": "http://b", "symbol": "600002"},
    ]
    build_index(emb, news, store, symbol="all")
    hits = retrieve(emb, "半导体 芯片", store, top_k=1)
    assert hits[0]["title"] == "半导体板块大涨"


def test_retrieve_empty_index_returns_empty():
    store = Store(tempfile.mkdtemp() + "/t.db")
    emb = HashEmbedding()
    assert retrieve(emb, "测试", store) == []
