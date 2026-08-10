import tempfile
from pathlib import Path

from core.store import Store


def make_store():
    return Store(tempfile.mkdtemp() + "/t.db")


def test_cache_set_get_roundtrip():
    s = make_store()
    assert s.cache_get("k1") is None
    s.cache_set("k1", "v1", 3600)
    assert s.cache_get("k1") == "v1"


def test_cache_expired():
    s = make_store()
    s.cache_set("k1", "v1", -1)
    assert s.cache_get("k1") is None


def test_account_default():
    s = make_store()
    acc = s.get_account()
    assert acc["cash"] == 0.0


def test_save_and_reload_account():
    s = make_store()
    s.save_account({"cash": 999.0, "initial_capital": 1000000.0,
                    "period_start": "2026-08-01", "updated_at": "2026-08-10"})
    acc = s.get_account()
    assert acc["cash"] == 999.0


def test_trade_and_period_roundtrip():
    s = make_store()
    s.insert_trade({"time": "2026-08-10 10:00:00", "symbol": "510300",
                    "name": "沪深300ETF", "side": "buy", "price": 3.9,
                    "qty": 10000, "fee": 39.0, "pnl": None, "status": "open"})
    trades = s.list_trades()
    assert len(trades) == 1 and trades[0]["side"] == "buy"
    s.insert_period({"period_id": "2026-08", "start": "2026-08-01", "end": "2026-08-31",
                     "initial_capital": 1000000.0, "final_nav": 1010000.0,
                     "win_rate": 0.6, "return_pct": 1.0, "benchmark_return": 0.5})
    assert s.list_periods()[0]["win_rate"] == 0.6


def test_news_and_chunk():
    s = make_store()
    s.insert_news({"title": "标题", "content": "内容", "date": "2026-08-09",
                   "source": "来源", "url": "http://x", "symbol": "600519"})
    assert len(s.list_news(symbol="600519")) == 1
    s.save_chunk({"chunk_id": "c1", "source_id": "1", "text": "片段",
                  "meta": '{"title":"标题"}', "embedding": b"1"})
    assert len(s.get_chunks()) == 1
