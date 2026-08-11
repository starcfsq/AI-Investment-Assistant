"""把数据与各分析模块串成一次完整分析。"""
from datetime import datetime

import pandas as pd

from core.config import load_weights
from core.logging import get_logger
from core.portfolio import build_portfolio
from core.sector import score_sectors
from core.stock import rank_stocks
from core.trend import analyze_trend

logger = get_logger("core.analyze")


def _df(value) -> pd.DataFrame:
    """None/缺失 → 空 DataFrame。不能用 `value or df`，会触发 bool(DataFrame) 歧义。"""
    return pd.DataFrame() if value is None else value


def analyze_at(data: dict, weights: dict) -> dict:
    """给定数据切片 + 权重，计算趋势/板块/组合。纯函数，不访问网络。"""
    index_df = _df(data.get("index_df"))
    val_df = _df(data.get("val_df"))
    bond_df = _df(data.get("bond_df"))
    quotes = _df(data.get("quotes"))
    flow = _df(data.get("flow"))
    hist = data.get("hist") or {}
    bench = _df(data.get("bench"))
    trend = {}
    if not index_df.empty:
        try:
            trend = analyze_trend(index_df, val_df, bond_df, weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning("趋势分析失败: %s", exc)
    sectors = []
    if not quotes.empty and "name" in quotes.columns and not bench.empty:
        try:
            sectors = score_sectors(quotes, flow, hist, bench, weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning("板块分析失败: %s", exc)
    portfolio = build_portfolio(sectors[:4]) if sectors else {}
    warnings = _sufficiency_warnings(index_df, val_df, quotes, sectors, [])
    return {"trend": trend, "sectors": sectors[:10], "portfolio": portfolio,
            "warnings": warnings}


def run_analysis(provider) -> dict:
    weights = load_weights()
    index_df = val_df = bond_df = quotes = flow = bench = spot = None
    try:
        index_df = provider.index_daily("沪深300")
        val_df = provider.index_valuation("沪深300")
        bond_df = provider.bond_yield()
    except Exception as exc:  # noqa: BLE001
        logger.warning("趋势数据拉取失败: %s", exc)
    hist = {}
    try:
        quotes = provider.sector_quote()
        flow = provider.sector_flow()
        if not quotes.empty and "name" in quotes.columns:
            for name in list(quotes["name"])[:30]:
                h = provider.sector_hist(name)
                if not h.empty:
                    hist[name] = h
        bench = provider.index_daily(provider.benchmark_index_code())
    except Exception as exc:  # noqa: BLE001
        logger.warning("板块数据拉取失败: %s", exc)
    try:
        spot = provider.stock_spot()
    except Exception as exc:  # noqa: BLE001
        logger.warning("行情拉取失败: %s", exc)
    base = analyze_at({
        "index_df": index_df, "val_df": val_df, "bond_df": bond_df,
        "quotes": quotes, "flow": flow, "hist": hist, "bench": bench,
    }, weights)
    stocks = _rank_stocks(spot, base["sectors"], provider, weights)
    warnings = _sufficiency_warnings(index_df, val_df, quotes,
                                     base["sectors"], stocks)
    data_until = base["trend"].get("data_until",
                                   datetime.now().strftime("%Y-%m-%d"))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trend": base["trend"], "sectors": base["sectors"], "stocks": stocks,
        "portfolio": base["portfolio"], "data_until": data_until,
        "data_quality": provider.quality_report(), "warnings": warnings,
    }


def _rank_stocks(spot, sectors, provider, weights):
    """实时个股选股（依赖网络财务数据；无 spot/sectors 时返回空）。"""
    if spot is None or spot.empty or not sectors:
        return []
    try:
        top_names = [s["name"] for s in sectors[:5]]
        candidates = []
        pool = _candidate_pool(spot, top_names)
        for _, row in pool.head(20).iterrows():
            fin = provider.stock_financial(row["code"])
            if fin.get("pe"):
                candidates.append({
                    "code": row["code"], "name": row["name"],
                    "roe": fin.get("roe", 0.0), "growth": fin.get("growth", 0.0),
                    "pe": fin.get("pe", 0.0), "pe_pct": fin.get("pe_pct"),
                    "dividend": fin.get("dividend", 0.0),
                })
        return rank_stocks(candidates, weights)[:10]
    except Exception as exc:  # noqa: BLE001
        logger.warning("选股失败: %s", exc)
        return []


def _candidate_pool(spot: pd.DataFrame, top_names: list[str]) -> pd.DataFrame:
    """从行情池生成选股候选。

    优先按 top 板块名模糊匹配股票名；板块名与股票名通常不直接对应，
    匹配失败时回退为全市场活跃候选（按涨跌幅降序取前 30），保证选股功能可用。
    """
    pool = spot[spot["name"].str.contains("|".join(top_names), regex=True, na=False)]
    if pool.empty and "pct_change" in spot.columns:
        pool = spot.sort_values("pct_change", ascending=False).head(30)
    return pool


def _sufficiency_warnings(index_df, val_df, quotes, sectors, stocks) -> list[str]:
    """数据充分性检查：不足时明确告警，而非静默出错。"""
    warnings = []
    if index_df is not None and len(index_df) < 250:
        warnings.append("趋势数据不足：指数 MA250 样本不足")
    if val_df is not None and len(val_df) < 120:
        warnings.append("趋势数据不足：估值历史过短")
    if quotes is not None and len(quotes) < 10:
        warnings.append("板块覆盖不足")
    if len(sectors) < 10:
        warnings.append("板块打分样本不足")
    if len(stocks) < 5:
        warnings.append("选股候选不足")
    return warnings
