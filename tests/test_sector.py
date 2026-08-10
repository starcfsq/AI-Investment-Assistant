import pandas as pd
from core.sector import score_sectors

WEIGHTS = {"sector": {"rs": 0.4, "flow": 0.3, "momentum": 0.3}}


def _dummy_hist(n=65, start_val=100.0):
    import numpy as np
    dates = pd.date_range("2026-01-01", periods=n)
    return pd.DataFrame({"date": dates, "close": np.linspace(start_val, start_val * 1.2, n)})


def test_score_orders_by_score():
    quotes = pd.DataFrame({"name": ["A", "B"], "pct_change": [2.0, 1.0]})
    flow = pd.DataFrame({"name": ["A", "B"], "net_inflow": [10.0, 5.0]})
    hist = {"A": _dummy_hist(), "B": _dummy_hist(start_val=100.0)}
    bench = _dummy_hist(start_val=100.0)
    result = score_sectors(quotes, flow, hist, bench, WEIGHTS)
    assert result[0]["name"] == "A"
    assert 0 <= result[0]["score"] <= 100


def test_score_fields_present():
    quotes = pd.DataFrame({"name": ["X"], "pct_change": [1.0]})
    flow = pd.DataFrame({"name": ["X"], "net_inflow": [8.0]})
    hist = {"X": _dummy_hist()}
    bench = _dummy_hist()
    result = score_sectors(quotes, flow, hist, bench, WEIGHTS)
    item = result[0]
    for k in ("name", "rs", "flow", "momentum", "score"):
        assert k in item
