"""大盘长期趋势分析。"""
import numpy as np
import pandas as pd

from core.logging import get_logger

logger = get_logger("core.trend")

TREND_STATES = {"opportunity": "低估机会", "neutral": "中性合理", "risk": "高估风险"}


def pct_rank_historical(value: float, history: pd.Series) -> float:
    """value 在 history 中的百分位（0-100）。"""
    clean = history.dropna()
    if clean.empty:
        return 50.0
    below = float((clean < value).sum())
    return round(below / len(clean) * 100.0, 1)


def ma_deviation(close: float, ma_value: float) -> float:
    return close / ma_value - 1.0


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(min(hi, max(lo, x)))


def _ma(series: pd.Series, window: int = 250) -> float:
    if len(series) < window:
        return float(series.mean())
    return float(series.iloc[-window:].mean())


def analyze_trend(index_df: pd.DataFrame, val_df: pd.DataFrame,
                  bond_df: pd.DataFrame, weights: dict) -> dict:
    """综合判断大盘长期趋势状态。

    - index_df: date/close（用于 MA250 偏离）
    - val_df: date/pe/pb（用于估值百分位）
    - bond_df: date/cn_10y（用于股债性价比）
    """
    w = weights["trend"]
    if index_df is None or index_df.empty or "close" not in index_df.columns:
        return {
            "signals": {"ma": 50.0, "valuation": 50.0, "bond": 50.0},
            "state": "中性合理", "composite": 50.0,
            "detail": {"ma_dev": 0.0, "pe_pct": 50.0, "pb_pct": 50.0,
                       "bond_equity_pct": 50.0},
            "data_until": "",
        }
    last_close = float(index_df["close"].iloc[-1])
    last_date = str(index_df["date"].iloc[-1])[:10]
    ma_value = _ma(index_df["close"], 250)
    dev = ma_deviation(last_close, ma_value)
    ma_signal = _clamp(dev * 500.0 + 50.0)

    pe_pct = pct_rank_historical(_last_pe(val_df), val_df["pe"])
    pb_pct = pct_rank_historical(_last_pb(val_df), val_df["pb"])
    valuation_signal = 100.0 - pe_pct

    bond_signal = 50.0
    bond_pct = 50.0
    if not bond_df.empty and not val_df.empty:
        eq_earn = 1.0 / _last_pe(val_df) if _last_pe(val_df) > 0 else 0.0
        last_bond = float(bond_df["cn_10y"].iloc[-1]) / 100.0
        if last_bond > 0:
            ratio_series = (1.0 / val_df["pe"].replace(0, np.nan)
                            - bond_df["cn_10y"].iloc[-1] / 100.0)
            bond_pct = pct_rank_historical(eq_earn - last_bond, ratio_series)
            bond_signal = bond_pct

    composite = (w["ma"] * ma_signal + w["valuation"] * valuation_signal
                 + w["bond"] * bond_signal)

    if composite >= 60:
        state = TREND_STATES["opportunity"]
    elif composite <= 40:
        state = TREND_STATES["risk"]
    else:
        state = TREND_STATES["neutral"]

    return {
        "signals": {"ma": round(ma_signal, 1),
                    "valuation": round(valuation_signal, 1),
                    "bond": round(bond_signal, 1)},
        "state": state,
        "composite": round(composite, 1),
        "detail": {"ma_dev": round(dev, 4), "pe_pct": pe_pct,
                   "pb_pct": pb_pct, "bond_equity_pct": round(bond_pct, 1)},
        "data_until": last_date,
    }


def _last_pe(val_df: pd.DataFrame) -> float:
    col = "pe"
    clean = val_df[col].dropna()
    return float(clean.iloc[-1]) if not clean.empty else 0.0


def _last_pb(val_df: pd.DataFrame) -> float:
    clean = val_df["pb"].dropna()
    return float(clean.iloc[-1]) if not clean.empty else 0.0
