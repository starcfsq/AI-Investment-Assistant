"""回测引擎：用历史板块数据评估打分权重的推荐胜率。"""
import numpy as np
import pandas as pd

from core.logging import get_logger
from core.sector import _pct_return, _minmax

logger = get_logger("core.backtest")


def backtest_sectors(sector_hist: dict[str, pd.DataFrame], bench: pd.DataFrame,
                     weights: dict, lookahead_days: int = 63) -> dict:
    if not sector_hist or bench.empty:
        return {"win_rate": 0.0, "excess_return": 0.0, "n_samples": 0,
                "wins": 0, "data_until": ""}
    bench_dates = pd.to_datetime(bench["date"])
    start = bench_dates.iloc[0]
    end = bench_dates.iloc[-1]
    t = start + np.timedelta64(lookahead_days, "D")
    wins, samples = 0, 0
    excess_returns = []
    while t + np.timedelta64(lookahead_days, "D") <= end:
        rows = []
        for name, h in sector_hist.items():
            h = h.copy()
            h["date"] = pd.to_datetime(h["date"])
            past = h[h["date"] <= t]
            if len(past) < 30:
                continue
            ret_3m = _pct_return(past, 63)
            ret_20d = _pct_return(past, 20)
            rows.append({"name": name, "rs_raw": ret_3m,
                         "flow_raw": 0.0, "momentum_raw": ret_20d})
        if len(rows) < 3:
            t += np.timedelta64(lookahead_days, "D")
            continue
        rs = _minmax([r["rs_raw"] for r in rows])
        mo = _minmax([r["momentum_raw"] for r in rows])
        for i, r in enumerate(rows):
            r["score"] = (weights["sector"]["rs"] * rs[i]
                          + weights["sector"]["momentum"] * mo[i])
        top3 = sorted(rows, key=lambda r: r["score"], reverse=True)[:3]
        # 组合区间收益
        port_rets = []
        for r in top3:
            h = sector_hist[r["name"]]
            h = h.copy()
            h["date"] = pd.to_datetime(h["date"])
            start_p = h[h["date"] <= t]
            end_p = h[(h["date"] > t) & (h["date"] <= t + np.timedelta64(lookahead_days, "D"))]
            if start_p.empty or end_p.empty:
                continue
            port_rets.append(float(end_p["close"].iloc[-1] / start_p["close"].iloc[-1] - 1.0))
        if not port_rets:
            t += np.timedelta64(lookahead_days, "D")
            continue
        port_ret = float(np.mean(port_rets))
        b_start = bench[bench_dates <= t]
        b_end = bench[(bench_dates > t) & (bench_dates <= t + np.timedelta64(lookahead_days, "D"))]
        if b_start.empty or b_end.empty:
            t += np.timedelta64(lookahead_days, "D")
            continue
        bench_ret = float(b_end["close"].iloc[-1] / b_start["close"].iloc[-1] - 1.0)
        samples += 1
        if port_ret > bench_ret:
            wins += 1
        excess_returns.append(port_ret - bench_ret)
        t += np.timedelta64(lookahead_days, "D")
    return {
        "win_rate": round(wins / samples, 3) if samples else 0.0,
        "excess_return": round(float(np.mean(excess_returns)), 4) if excess_returns else 0.0,
        "n_samples": samples,
        "wins": wins,
        "data_until": str(end)[:10],
    }
