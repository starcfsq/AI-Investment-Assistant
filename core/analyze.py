"""把数据与各分析模块串成一次完整分析。"""
from datetime import datetime

from core.config import load_weights
from core.logging import get_logger
from core.portfolio import build_portfolio
from core.sector import score_sectors
from core.stock import rank_stocks
from core.trend import analyze_trend

logger = get_logger("core.analyze")


def run_analysis(provider) -> dict:
    weights = load_weights()
    trend = {}
    index_df, val_df, quotes = None, None, None
    try:
        index_df = provider.index_daily("沪深300")
        val_df = provider.index_valuation("沪深300")
        bond_df = provider.bond_yield()
        if not index_df.empty:
            trend = analyze_trend(index_df, val_df, bond_df, weights)
    except Exception as exc:  # noqa: BLE001
        logger.warning("趋势分析失败: %s", exc)

    sectors = []
    try:
        quotes = provider.sector_quote()
        flow = provider.sector_flow()
        hist = {}
        for name in list(quotes["name"])[:30]:
            h = provider.sector_hist(name)
            if not h.empty:
                hist[name] = h
        bench = provider.index_daily(provider.benchmark_index_code())
        if not quotes.empty and not bench.empty:
            sectors = score_sectors(quotes, flow, hist, bench, weights)
    except Exception as exc:  # noqa: BLE001
        logger.warning("板块分析失败: %s", exc)

    stocks = []
    try:
        spot = provider.stock_spot()
        if not spot.empty and sectors:
            top_names = [s["name"] for s in sectors[:5]]
            candidates = []
            # 简化：从沪深300成分股中按板块名模糊匹配
            pool = spot[spot["name"].str.contains("|".join(top_names), regex=True, na=False)]
            for _, row in pool.head(20).iterrows():
                fin = provider.stock_financial(row["code"])
                if fin.get("pe"):
                    candidates.append({
                        "code": row["code"], "name": row["name"],
                        "roe": fin.get("roe", 0.0), "growth": fin.get("growth", 0.0),
                        "pe": fin.get("pe", 0.0), "pe_pct": fin.get("pe_pct"),
                        "dividend": fin.get("dividend", 0.0),
                    })
            stocks = rank_stocks(candidates, weights)[:10]
    except Exception as exc:  # noqa: BLE001
        logger.warning("选股失败: %s", exc)

    portfolio = build_portfolio(sectors[:4]) if sectors else {}

    data_until = trend.get("data_until", datetime.now().strftime("%Y-%m-%d"))
    warnings = _sufficiency_warnings(index_df, val_df, quotes, sectors, stocks)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trend": trend,
        "sectors": sectors[:10],
        "stocks": stocks,
        "portfolio": portfolio,
        "data_until": data_until,
        "data_quality": provider.quality_report(),
        "warnings": warnings,
    }


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
