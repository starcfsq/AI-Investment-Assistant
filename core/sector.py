"""板块分析：RS 相对强度 + 资金流 + 动量。"""
import numpy as np
import pandas as pd

from core.logging import get_logger

logger = get_logger("core.sector")


def score_sectors(quotes: pd.DataFrame, flow: pd.DataFrame,
                  hist: dict[str, pd.DataFrame], bench_df: pd.DataFrame,
                  weights: dict) -> list[dict]:
    w = weights["sector"]
    bench_hist = bench_df
    bench_3m = _pct_return(bench_hist, 63)
    rows = []
    for name in quotes["name"]:
        h = hist.get(name)
        if h is None or h.empty:
            continue
        ret_3m = _pct_return(h, 63)
        ret_20d = _pct_return(h, 20)
        rs_raw = ret_3m - bench_3m
        flow_raw = _flow_for(flow, name)
        momentum_raw = ret_20d
        rows.append({
            "name": name,
            "rs_raw": rs_raw,
            "flow_raw": flow_raw,
            "momentum_raw": momentum_raw,
        })
    if not rows:
        return []
    rs = _minmax([r["rs_raw"] for r in rows])
    fl = _minmax([r["flow_raw"] for r in rows])
    mo = _minmax([r["momentum_raw"] for r in rows])
    for i, r in enumerate(rows):
        r["rs"] = round(rs[i], 1)
        r["flow"] = round(fl[i], 1)
        r["momentum"] = round(mo[i], 1)
        r["score"] = round(w["rs"] * r["rs"] + w["flow"] * r["flow"]
                           + w["momentum"] * r["momentum"], 1)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def _pct_return(df: pd.DataFrame, window: int) -> float:
    if df is None or len(df) < 2:
        return 0.0
    close = df["close"].dropna()
    if len(close) == 0:
        return 0.0
    start = close.iloc[max(0, len(close) - 1 - window)]
    end = close.iloc[-1]
    return float(end / start - 1.0) if start else 0.0


def _flow_for(flow: pd.DataFrame, name: str) -> float:
    if flow.empty:
        return 0.0
    m = flow[flow["name"] == name]
    if m.empty:
        return 0.0
    try:
        return float(m["net_inflow"].iloc[0])
    except (TypeError, ValueError):
        return 0.0


def _minmax(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return [50.0] * len(arr)
    return [round(float((v - lo) / (hi - lo) * 100.0), 1) for v in arr]
