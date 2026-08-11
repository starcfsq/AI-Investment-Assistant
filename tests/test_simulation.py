import pandas as pd
import pytest
from datetime import datetime


class _FixedNow:
    """固定 'now'，让 run_year_simulation 的 lookback 窗口可预测，测试结果确定。"""

    NOW = datetime(2026, 6, 15, 10, 0, 0)

    @classmethod
    def now(cls):
        return cls.NOW


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


def test_run_year_simulation_returns_structure(monkeypatch):
    from core.simulation import run_year_simulation
    from core.store import Store
    import tempfile
    monkeypatch.setattr("core.simulation.datetime", _FixedNow)
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
    monkeypatch.setattr("core.simulation.datetime", _FixedNow)
    monkeypatch.setenv("ACCOUNT_INITIAL_CAPITAL", "100000")
    store = Store(tempfile.mkdtemp() + "/t.db")
    provider = _FakeSimProvider()
    out = run_year_simulation(provider, store, lookback_days=90)
    assert out["rebalances"], "窗口内应有调仓"
    first_rebal = out["rebalances"][0]["date"]
    # 窗口内首条曲线日在首次调仓之前：净值 = 初始资金，且没有任何持仓
    assert out["curve"][0]["date"] < first_rebal
    assert out["curve"][0]["nav"] == 100000.0
    # 模拟调仓买入交易的时间戳 = 调仓日字符串（而非墙钟时间）
    buys = [t for t in out["trades"] if t["side"] == "buy"]
    assert buys and buys[0]["time"] == first_rebal


def test_curve_bounded_to_lookback_window(monkeypatch):
    """C1：曲线被裁剪到 lookback 窗口（非全量历史），首点为初始资金。"""
    from core.simulation import run_year_simulation
    from core.store import Store
    import tempfile
    monkeypatch.setattr("core.simulation.datetime", _FixedNow)
    monkeypatch.setenv("ACCOUNT_INITIAL_CAPITAL", "100000")
    store = Store(tempfile.mkdtemp() + "/t.db")
    provider = _FakeSimProvider(n_years=3)
    out = run_year_simulation(provider, store, lookback_days=180)
    assert "error" not in out, out.get("error")
    assert out["curve"], "曲线不应为空"
    span = (pd.Timestamp(out["curve"][-1]["date"])
            - pd.Timestamp(out["curve"][0]["date"])).days
    assert span <= 200, f"曲线日期跨度应 <= ~200 天，实际 {span}"
    assert span >= 60, f"曲线日期跨度不应过短，实际 {span}"
    assert out["curve"][0]["nav"] == 100000.0, "首点应为初始资金"


class _FakeSimProvider:
    """离线 fake：历史数据覆盖 n_years 年，lookback 窗口落在其中。

    每个工作日一条数据，窗口起点本身即为曲线首点（早于当月最后交易日），
    因此可断言"调仓前"的净值点。默认 3 年保证既有测试（lookback_days=90）
    与新测试（lookback_days=180）都能命中窗口。
    """

    def __init__(self, n_years=3):
        import pandas as pd
        end = pd.Timestamp(_FixedNow.NOW)
        start = end - pd.DateOffset(years=n_years)
        dates = pd.bdate_range(start, end)
        n = len(dates)
        self._dates = dates
        self.index = pd.DataFrame({"date": dates,
                                   "close": [4000.0 + i * 0.5 for i in range(n)]})
        self.val = pd.DataFrame({"date": dates,
                                 "pe": [13.0 + (i % 20) * 0.05 for i in range(n)],
                                 "pb": [1.4 + (i % 20) * 0.01 for i in range(n)]})
        self.bond = pd.DataFrame({"date": dates,
                                  "cn_10y": [2.5 + (i % 10) * 0.02 for i in range(n)]})
        self.quotes = pd.DataFrame({"name": ["医疗服务"], "pct_change": [2.0]})
        self.hist = {"医疗服务": pd.DataFrame(
            {"date": dates, "close": [100.0 + i * 0.1 for i in range(n)]})}

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
        n = len(self._dates)
        return pd.DataFrame({"date": self._dates,
                             "close": [3.9 + i * 0.001 for i in range(n)]})
