import pandas as pd
import pytest


def _fake_provider():
    """内存 fake，snapshot_at 只做日期切片，不访问网络。"""
    class P:
        pass
    p = P
    p.index_df = pd.DataFrame({"date": pd.to_datetime(
        ["2026-01-05", "2026-02-02", "2026-03-02"]), "close": [1.0, 2.0, 3.0]})
    return p


def test_history_snapshot_slices_by_date():
    from core.simulation import _slice_by_date
    df = pd.DataFrame({"date": pd.to_datetime(
        ["2026-01-05", "2026-02-02", "2026-03-02"]), "close": [1.0, 2.0, 3.0]})
    out = _slice_by_date(df, "2026-02-01")
    assert list(out["close"]) == [1.0]


def test_snapshot_at_keeps_bench_parallel():
    from core.simulation import _snapshot_at
    index = pd.DataFrame({"date": pd.to_datetime(
        ["2026-01-05", "2026-02-02", "2026-03-02"]), "close": [1.0, 2.0, 3.0]})
    data = {"index_df": index, "val_df": pd.DataFrame(),
            "bond_df": pd.DataFrame(),
            "quotes": pd.DataFrame(), "flow": pd.DataFrame(),
            "hist": {}, "bench": index}
    snap = _snapshot_at(data, "2026-02-01")
    assert list(snap["index_df"]["close"]) == [1.0]
    assert list(snap["bench"]["close"]) == [1.0]


def test_monthly_rebalance_dates():
    from core.simulation import _monthly_rebalance_dates
    dates = pd.to_datetime([
        "2026-01-05", "2026-01-28", "2026-01-29", "2026-01-30",
        "2026-02-02", "2026-02-27", "2026-03-02", "2026-03-31"])
    out = _monthly_rebalance_dates(dates)
    # 每月最后交易日
    assert out == ["2026-01-30", "2026-02-27", "2026-03-31"]


def test_run_year_simulation_returns_structure():
    from core.simulation import run_year_simulation
    from core.store import Store
    import tempfile
    store = Store(tempfile.mkdtemp() + "/t.db")
    provider = _FakeSimProvider()
    out = run_year_simulation(provider, store, lookback_days=90)
    assert {"stats", "curve", "trades", "rebalances"} <= set(out)
    assert out["stats"]["total_return"] is not None
    assert len(out["curve"]) >= 1


def test_curve_first_point_is_initial_cash_and_trades_dated(monkeypatch):
    """净值曲线首点在首次调仓之前 = 初始资金（无持仓）；调仓交易时间为调仓日。"""
    from core.simulation import run_year_simulation
    from core.store import Store
    import tempfile
    monkeypatch.setenv("ACCOUNT_INITIAL_CAPITAL", "100000")
    store = Store(tempfile.mkdtemp() + "/t.db")
    provider = _FakeSimProvider()
    out = run_year_simulation(provider, store, lookback_days=90)
    # 首个调仓日为每月最后交易日（2026-01-30），晚于首条曲线日 2026-01-05
    assert out["rebalances"][0]["date"] == "2026-01-30"
    # 首曲线点在首次调仓之前：净值 = 初始资金，且没有任何持仓
    assert out["curve"][0]["date"] == "2026-01-05"
    assert out["curve"][0]["nav"] == 100000.0
    # 模拟调仓买入交易的时间戳 = 调仓日字符串（而非墙钟时间）
    buys = [t for t in out["trades"] if t["side"] == "buy"]
    assert buys and buys[0]["time"] == "2026-01-30"


class _FakeSimProvider:
    """小型历史数据的 fake provider，供模拟测试离线使用。

    每月含多个交易日，使首个调仓日（每月最后交易日）落在首条曲线日期之后，
    从而可以断言"调仓前"的净值点。
    """

    def __init__(self):
        import pandas as pd
        dates = pd.to_datetime(["2026-01-05", "2026-01-30",
                                "2026-02-27", "2026-03-31"])
        self.index = pd.DataFrame({"date": dates,
                                   "close": [4000.0, 4100.0, 4200.0, 4300.0]})
        self.val = pd.DataFrame({"date": dates,
                                 "pe": [13.0, 13.5, 14.0, 14.5],
                                 "pb": [1.4, 1.4, 1.5, 1.5]})
        self.bond = pd.DataFrame({"date": dates,
                                  "cn_10y": [2.5, 2.5, 2.6, 2.6]})
        self.quotes = pd.DataFrame({"name": ["医疗服务"], "pct_change": [2.0]})
        self.hist = {"医疗服务": pd.DataFrame(
            {"date": dates, "close": [100.0, 105.0, 110.0, 115.0]})}

    def index_daily(self, symbol="沪深300"):
        return self.index

    def index_valuation(self, name):
        return self.val

    def bond_yield(self):
        return self.bond

    def sector_quote(self):
        return self.quotes

    def sector_flow(self):
        return pd.DataFrame()

    def sector_hist(self, name):
        return self.hist.get(name, pd.DataFrame())

    def benchmark_index_code(self):
        return "sh000300"

    def etf_close(self, code6):
        import pandas as pd
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-01-30",
                                    "2026-02-27", "2026-03-31"]),
            "close": [3.9, 4.0, 4.1, 4.2],
        })
