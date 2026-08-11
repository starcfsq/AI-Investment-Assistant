"""build_history_summary: 把虚拟账户历史聚合为对话可用的结构化摘要。"""
import tempfile

from ai.chat import answer_question
from core.account import SimAccount
from core.history import build_history_summary
from core.store import Store

PORTFOLIO = {
    "core": {"name": "沪深300ETF(510300)", "weight": 0.7},
    "satellite": [{"name": "半导体ETF(512480)", "weight": 0.2},
                  {"name": "医药ETF(512010)", "weight": 0.1}],
}
PRICES = {"510300": 3.9, "512480": 1.2, "512010": 0.8}


def _new(initial_capital: float = 100000.0):
    store = Store(tempfile.mkdtemp() + "/t.db")
    acc = SimAccount(store, initial_capital=initial_capital)
    acc.ensure_initialized()
    return acc, store


def test_empty_store_returns_empty_summary():
    acc, store = _new()
    h = build_history_summary(store, acc)
    assert h["account"]["position_count"] == 0
    assert h["positions"] == []
    assert h["recent_trades"] == []
    assert h["periods"] == []
    assert h["iters"] == []


def test_contains_account_and_positions():
    acc, store = _new()
    acc.execute(PORTFOLIO, PRICES)
    h = build_history_summary(store, acc)
    assert h["account"]["period_id"] == acc.current_period_id()
    assert h["account"]["position_count"] == 3
    assert len(h["positions"]) == 3
    by_sym = {p["symbol"]: p for p in h["positions"]}
    assert by_sym["510300"]["qty"] > 0
    assert abs(by_sym["510300"]["cost_price"] - 3.9) < 1e-6


def test_recent_trades_reverse_chronological():
    acc, store = _new()
    for i, t in enumerate(["2026-08-01 10:00:00",
                           "2026-08-05 10:00:00",
                           "2026-08-10 10:00:00"]):
        store.insert_trade({
            "time": t, "symbol": "510300", "name": "沪深300ETF(510300)",
            "side": "buy", "price": 3.9, "qty": 1.0, "fee": 0.0,
            "pnl": None, "status": "open",
        })
    h = build_history_summary(store, acc)
    assert [t["time"] for t in h["recent_trades"]] == [
        "2026-08-10 10:00:00", "2026-08-05 10:00:00", "2026-08-01 10:00:00",
    ]


def test_positions_truncated_to_10():
    acc, store = _new()
    for i in range(12):
        store.save_position({
            "symbol": f"{600000 + i}", "name": f"标的{i}",
            "qty": 100.0, "cost_price": 10.0, "updated_at": "2026-08-11 00:00:00",
        })
    h = build_history_summary(store, acc)
    assert len(h["positions"]) == 10


def test_trades_truncated_to_20():
    acc, store = _new()
    for i in range(25):
        store.insert_trade({
            "time": f"2026-08-01 00:00:0{i % 10}", "symbol": "510300",
            "name": "沪深300ETF(510300)", "side": "buy", "price": 3.9,
            "qty": 1.0, "fee": 0.0, "pnl": None, "status": "open",
        })
    h = build_history_summary(store, acc)
    assert len(h["recent_trades"]) == 20


def test_periods_and_iters_truncated_to_5():
    acc, store = _new()
    for i in range(7):
        store.insert_period({
            "period_id": f"2026-{i:02d}", "start": f"2026-{i:02d}-01",
            "end": "2026-08-10", "initial_capital": 100000.0, "final_nav": 100000.0,
            "win_rate": 0.5, "return_pct": 1.0, "benchmark_return": 0.5,
        })
    for i in range(7):
        store.insert_iter({
            "version": f"v{i}", "run_at": "2026-08-01 00:00:00",
            "weights_json": "{}", "backtest_window": "w", "win_rate": 0.6,
            "excess_return": 0.02, "data_until": "2026-08-10",
        })
    h = build_history_summary(store, acc)
    assert len(h["periods"]) == 5
    assert len(h["iters"]) == 5


def test_curve_points_and_period_stats():
    acc, store = _new()
    acc.execute(PORTFOLIO, PRICES)
    acc.snapshot(PRICES)
    acc.snapshot({k: v * 1.05 for k, v in PRICES.items()})
    acc.snapshot({k: v * 0.95 for k, v in PRICES.items()})
    h = build_history_summary(store, acc)
    curve = h["curve"]
    navs = [s["nav"] for s in store.list_snapshots(acc.current_period_id())]
    assert curve["latest"]["nav"] == navs[-1]
    assert curve["first"]["nav"] == navs[0]
    assert curve["max"]["nav"] == max(navs)
    assert curve["min"]["nav"] == min(navs)
    assert "win_rate" in h["period"]
    assert "return_pct" in h["period"]


class _RecordingClient:
    """捕获发给 LLM 的消息，返回结构完整的 chat 输出。"""

    def __init__(self):
        self.messages = None

    def chat_json(self, messages, schema):
        self.messages = messages
        return {"answer": "ok", "references": [], "confidence": 0.7,
                "disclaimer": "d"}


def test_answer_question_embeds_history_in_prompt():
    acc, store = _new()
    acc.execute(PORTFOLIO, PRICES)
    history = build_history_summary(store, acc)
    client = _RecordingClient()
    out = answer_question(client, "结合我的持仓，现在该买什么？",
                          {"trend": {"state": "低估"}}, [], history)
    # 返回结构完整
    assert out["answer"] == "ok"
    assert "confidence" in out and "disclaimer" in out
    # 历史摘要真正进入 user prompt（LLM 可见持仓数字）
    user = client.messages[-1]["content"]
    assert "虚拟账户历史" in user
    assert "510300" in user
