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
