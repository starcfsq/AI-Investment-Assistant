"""个股筛选：多因子打分。"""
import numpy as np

from core.logging import get_logger

logger = get_logger("core.stock")


def rank_stocks(candidates: list[dict], weights: dict) -> list[dict]:
    if not candidates:
        return []
    w = weights["stock"]
    roes = _minmax([c.get("roe", 0.0) for c in candidates])
    grows = _minmax([c.get("growth", 0.0) for c in candidates])
    divs = _minmax([c.get("dividend", 0.0) for c in candidates])
    vals = [_valuation_score(c) for c in candidates]
    vals = _minmax(vals)
    for i, c in enumerate(candidates):
        c["score"] = round(w["roe"] * roes[i] + w["growth"] * grows[i]
                           + w["valuation"] * vals[i] + w["dividend"] * divs[i], 1)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def score_stock(candidate: dict, weights: dict) -> float:
    ranked = rank_stocks([candidate], weights)
    return ranked[0]["score"] if ranked else 0.0


def _valuation_score(c: dict) -> float:
    pe_pct = c.get("pe_pct")
    if pe_pct is not None:
        return 100.0 - float(pe_pct)
    pe = c.get("pe")
    if pe:
        return float(1.0 / pe) if pe > 0 else 0.0
    return 50.0


def _minmax(values: list[float]) -> list[float]:
    arr = np.asarray([float(v) for v in values])
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return [50.0] * len(arr)
    return [round(float((v - lo) / (hi - lo) * 100.0), 1) for v in arr]
