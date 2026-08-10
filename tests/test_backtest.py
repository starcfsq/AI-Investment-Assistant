import numpy as np
import pandas as pd
import pytest

from core.backtest import backtest_sectors

WEIGHTS = {"sector": {"rs": 0.4, "flow": 0.3, "momentum": 0.3}}


def _series(n, trend=1.0, start=100.0):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    closes = start * np.linspace(1.0, trend, n)
    return pd.DataFrame({"date": dates, "close": closes})


def test_backtest_returns_metrics():
    hist = {
        "A": _series(300, 1.5),
        "B": _series(300, 1.2),
        "C": _series(300, 1.0),
    }
    bench = _series(300, 1.1)
    result = backtest_sectors(hist, bench, WEIGHTS, lookahead_days=40)
    assert "win_rate" in result
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["n_samples"] >= 1
