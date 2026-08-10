import pandas as pd
import pytest

from core.trend import analyze_trend, ma_deviation, pct_rank_historical

WEIGHTS = {"trend": {"ma": 0.3, "valuation": 0.4, "bond": 0.3}}


def test_pct_rank_historical():
    history = pd.Series([10, 20, 30, 40, 50])
    assert pct_rank_historical(45, history) == pytest.approx(80.0)
    assert pct_rank_historical(5, history) == pytest.approx(0.0)


def test_ma_deviation():
    assert ma_deviation(110, 100) == pytest.approx(0.1)


def test_trend_high_pe_is_overvalued():
    # PE 百分位 95 → 便宜分低；MA 站上、债股中性 → 综合高估风险
    index = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=250),
                          "close": [100.0] * 250})
    val = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=100),
                        "pe": range(1, 101), "pb": range(1, 101)})
    bond = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=100),
                         "cn_10y": [3.0] * 100})
    result = analyze_trend(index, val, bond, WEIGHTS)
    assert result["detail"]["pe_pct"] > 90
    assert result["state"] in ("高估风险", "中性合理")


def test_trend_low_pe_is_opportunity():
    index = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=250),
                          "close": [100.0] * 250})
    val = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=100),
                        "pe": [95.0] * 100, "pb": [5.0] * 100})
    bond = pd.DataFrame({"date": pd.date_range("2015-01-01", periods=100),
                         "cn_10y": [3.0] * 100})
    result = analyze_trend(index, val, bond, WEIGHTS)
    assert result["state"] in ("低估机会", "中性合理")
