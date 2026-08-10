"""FastAPI 后端：dashboard / analyze / chat / backtest / account。"""
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ai.chat import answer_question
from ai.interpret import (
    interpret_trend,
    plan_portfolio,
    recommend_sectors,
    recommend_stocks,
)
from ai.provider import get_client
from core.account import SimAccount
from core.analyze import run_analysis
from core.config import DB_PATH, load_weights
from core.data import DataProvider
from core.logging import get_logger
from core.rag import build_index, retrieve
from core.store import Store
from core.tune import run_iteration
from core.embedding import get_embedding_provider

logger = get_logger("api.main")

app = FastAPI(title="AI 智能投资助手")
_store = Store(DB_PATH)
_provider = DataProvider(_store)
_client = get_client()
_embed = get_embedding_provider()
_account = SimAccount(_store)


class ChatRequest(BaseModel):
    query: str
    symbol: str | None = None


def _safe(client, fn, *args, **kwargs):
    try:
        return fn(client, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AI 解读失败，降级: %s", exc)
        return {}


@app.get("/api/dashboard")
def dashboard():
    analysis = run_analysis(_provider)
    ai = _safe(_client, interpret_trend, analysis["trend"])
    acc = _account.period_stats()
    iters = _store.list_iters()
    periods = _store.list_periods()
    return {
        "analysis": analysis,
        "ai": ai,
        "account": acc,
        "iters": iters,
        "periods": periods,
        "data_until": analysis["data_until"],
    }


@app.post("/api/analyze")
def analyze():
    result = run_analysis(_provider)
    # 账户执行推荐并快照
    prices = _prices_for_portfolio(result["portfolio"])
    _account.execute(result["portfolio"], prices)
    _account.snapshot(prices)
    ai = {
        "trend": _safe(_client, interpret_trend, result["trend"]),
        "sectors": _safe(_client, recommend_sectors, result["sectors"]),
        "stocks": _safe(_client, recommend_stocks, result["stocks"]),
        "portfolio": _safe(_client, plan_portfolio, result["portfolio"]),
    }
    # 将 result 展开到顶层（trend/sectors/stocks/portfolio/warnings 等），
    # 同时保留嵌套的 analysis 供看板消费者使用。
    return {**result, "analysis": result, "ai": ai, "account": _account.period_stats(),
            "data_until": result["data_until"]}


@app.post("/api/chat")
def chat(req: ChatRequest):
    analysis = run_analysis(_provider)
    rag_hits = []
    if req.symbol:
        news = _provider.stock_news(req.symbol) + _provider.stock_notices(req.symbol)
        if news:
            build_index(_embed, news, _store, symbol=req.symbol)
            rag_hits = retrieve(_embed, req.query, _store, top_k=5)
    out = answer_question(_client, req.query, analysis, rag_hits)
    return {"answer": out.get("answer", ""), "references": out.get("references", []),
            "confidence": out.get("confidence", 0.0),
            "disclaimer": out.get("disclaimer", ""),
            "data_until": analysis["data_until"]}


@app.get("/api/backtest")
def backtest():
    try:
        result = run_iteration(_provider, _store)
    except Exception as exc:  # noqa: BLE001
        logger.error("迭代失败: %s", exc)
        result = {"status": "error", "reason": str(exc)}
    return {"result": result, "iters": _store.list_iters(),
            "weights": load_weights()}


@app.get("/api/account")
def account():
    _account.maybe_reset_period()
    return {"stats": _account.period_stats(),
            "periods": _store.list_periods(),
            "trades": _store.list_trades()[-50:]}


@app.get("/")
def index():
    return FileResponse("web/index.html")


def _prices_for_portfolio(portfolio: dict) -> dict[str, float]:
    symbols = []
    if portfolio.get("core"):
        symbols.append(_code(portfolio["core"]["name"]))
    for sat in portfolio.get("satellite", []):
        c = _code(sat["name"])
        if c:
            symbols.append(c)
    prices = {}
    try:
        spot = _provider.stock_spot()
        code_to_price = dict(zip(spot["code"], spot["price"]))
        etf = _provider.etf_spot()
        if not etf.empty and "代码" in etf.columns:
            code_to_price.update(dict(zip(etf["代码"], etf["最新价"])))
        for s in symbols:
            if s in code_to_price:
                prices[s] = float(code_to_price[s])
    except Exception as exc:  # noqa: BLE001
        logger.warning("取价失败: %s", exc)
    return prices


def _code(name: str):
    import re

    m = re.search(r"(\d{6})", name or "")
    return m.group(1) if m else None
