from datetime import datetime
import pytest


def test_is_trading_day_weekday():
    from core.auto_invest import is_trading_day
    assert is_trading_day(datetime(2026, 8, 10))  # 周一
    assert is_trading_day(datetime(2026, 8, 14))  # 周五
    assert not is_trading_day(datetime(2026, 8, 15))  # 周六
    assert not is_trading_day(datetime(2026, 8, 16))  # 周日


def test_run_auto_invest_returns_result(monkeypatch):
    from core.auto_invest import run_auto_invest
    called = {}
    def fake_run(provider, account, store):
        called["ok"] = True
        return {"trend": {"state": "低估"}}
    import core.auto_invest as ai
    monkeypatch.setattr(ai, "_do_invest", fake_run)
    out = run_auto_invest(None, None, None)
    assert called.get("ok")
    assert out["trend"]["state"] == "低估"


def test_run_auto_invest_returns_error_on_failure(monkeypatch):
    from core.auto_invest import run_auto_invest
    def boom(*a, **k):
        raise RuntimeError("boom")
    import core.auto_invest as ai
    monkeypatch.setattr(ai, "_do_invest", boom)
    out = run_auto_invest(None, None, None)
    assert "error" in out
    assert "boom" in out["error"]


def test_do_invest_without_prices_does_not_crash(monkeypatch):
    from core.auto_invest import _do_invest
    from core.account import SimAccount
    from core.store import Store
    import tempfile, pandas as pd
    store = Store(tempfile.mkdtemp() + "/t.db")
    account = SimAccount(store, initial_capital=100000.0)
    account.ensure_initialized()

    class FakeProvider:
        def run_analysis_result(self):
            return {"trend": {"state": "低估"}, "portfolio": {
                "core": {"name": "沪深300ETF(510300)", "weight": 0.7},
                "satellite": []}}
        def stock_spot(self):
            return pd.DataFrame()
        def etf_spot(self):
            return pd.DataFrame()
    provider = FakeProvider()
    # _do_invest 内部 `from core.analyze import run_analysis` 在调用时解析，
    # 因此必须 patch core.analyze.run_analysis 而不是 core.auto_invest.run_analysis。
    import core.analyze as ca
    monkeypatch.setattr(ca, "run_analysis", lambda p: p.run_analysis_result())
    result = _do_invest(provider, account, store)
    assert result["portfolio"]["core"]["name"] == "沪深300ETF(510300)"
    # 无实时价格时应跳过买入，账户现金保持不变。
    assert store.get_account()["cash"] == 100000.0
