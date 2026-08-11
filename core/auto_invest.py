"""AI 自动投资调度：启动执行一次 + 每个交易日收盘后自动执行。"""
import asyncio
from datetime import datetime

from core.logging import get_logger

logger = get_logger("core.auto_invest")


def is_trading_day(dt: datetime) -> bool:
    """简化交易日：周一至周五（不含法定节假日，已知限制）。"""
    return dt.weekday() < 5


def _do_invest(provider, account, store) -> dict:
    """执行一次分析 + 虚拟账户投资。复用 /api/analyze 端点逻辑。"""
    from core.analyze import run_analysis
    result = run_analysis(provider)
    portfolio = result.get("portfolio") or {}
    import re
    symbols = set()
    if portfolio.get("core"):
        m = re.search(r"(\d{6})", portfolio["core"].get("name", ""))
        if m:
            symbols.add(m.group(1))
    for sat in portfolio.get("satellite", []):
        m = re.search(r"(\d{6})", sat.get("name", ""))
        if m:
            symbols.add(m.group(1))
    prices = {}
    # 取实时价格：只写入有真实行情价的标的；缺失的标的保持在 prices 之外，
    # 这样 account.execute 会跳过买入，account.snapshot 会回退到成本价。
    try:
        spot = provider.stock_spot()
        if not spot.empty and "code" in spot.columns:
            cm = dict(zip(spot["code"], spot["price"]))
            for c in symbols:
                if c in cm:
                    prices[c] = float(cm[c])
        etf = provider.etf_spot()
        if not etf.empty and "code" in etf.columns:
            cm = dict(zip(etf["code"], etf["price"]))
            for c in symbols:
                if c in cm:
                    prices[c] = float(cm[c])
    except Exception as exc:  # noqa: BLE001
        logger.warning("自动投资取价失败: %s", exc)
    account.execute(portfolio, prices)
    account.snapshot(prices)
    return result


def run_auto_invest(provider, account, store) -> dict:
    """供后台调度与 /api/analyze 共用的执行入口。"""
    try:
        return _do_invest(provider, account, store)
    except Exception as exc:  # noqa: BLE001
        logger.error("自动投资失败: %s", exc)
        return {"error": str(exc)}
