import tempfile

from core.account import SimAccount
from core.store import Store

PORTFOLIO = {
    "core": {"name": "沪深300ETF(510300)", "weight": 0.7},
    "satellite": [{"name": "半导体ETF(512480)", "weight": 0.2},
                  {"name": "医药ETF(512010)", "weight": 0.1}],
}
PRICES = {"510300": 3.9, "512480": 1.2, "512010": 0.8}


def _new_account():
    store = Store(tempfile.mkdtemp() + "/t.db")
    acc = SimAccount(store, initial_capital=100000.0)
    acc.ensure_initialized()
    return acc, store


def test_execute_buys_positions():
    acc, store = _new_account()
    trades = acc.execute(PORTFOLIO, PRICES)
    assert len(trades) == 3
    assert all(t["side"] == "buy" for t in trades)
    pos = {p["symbol"]: p["qty"] for p in store.list_positions()}
    assert abs(pos["510300"] - 0.7 * 100000 / 3.9) < 1


def test_snapshot_calc_nav():
    acc, store = _new_account()
    acc.execute(PORTFOLIO, PRICES)
    acc.snapshot(PRICES)
    stats = acc.period_stats()
    assert abs(stats["nav"] - 100000.0) < 1000.0  # 含费用，接近初始资金


def test_win_rate_after_profit_sell():
    acc, store = _new_account()
    acc.execute(PORTFOLIO, PRICES)
    # 全部卖出，价格涨 10%
    up = {k: v * 1.1 for k, v in PRICES.items()}
    acc.execute({"core": None, "satellite": []}, up)
    stats = acc.period_stats()
    assert stats["win_rate"] == 1.0


def test_reset_period_archives():
    acc, store = _new_account()
    acc.execute(PORTFOLIO, PRICES)
    acc.snapshot(PRICES, benchmark_return=0.5)
    # 模拟跨月：把 period_start 改到上个月，触发阶段归档
    state = store.get_account()
    state["period_start"] = "2026-07-01"
    store.save_account(state)
    acc.maybe_reset_period(benchmark_return=0.5)
    assert len(store.list_periods()) == 1
    acc.ensure_initialized()
    assert acc.period_stats()["cash"] == 100000.0
