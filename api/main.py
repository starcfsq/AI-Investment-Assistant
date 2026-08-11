"""FastAPI 后端：dashboard / analyze / chat / backtest / account。"""
import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai.chat import answer_question
from ai.interpret import (
    DISCLAIMER,
    interpret_trend,
    plan_portfolio,
    recommend_sectors,
    recommend_stocks,
)
from core.account import SimAccount
from core.auto_invest import run_auto_invest as _auto_invest
from core.analyze import run_analysis
from core.config import DB_PATH, load_weights
from core.data import DataProvider
from core.history import build_history_summary
from core.logging import get_logger
from core.rag import build_index, retrieve
from core.simulation import run_year_simulation
from core.store import Store
from core.tune import run_iteration
from core.embedding import get_embedding_provider

logger = get_logger("api.main")

# 自动投资与 /api/analyze 互斥：共享 threading.Lock（跨线程安全，
# 兼容 asyncio.to_thread 与 FastAPI 线程池）。
_invest_lock = threading.Lock()
_sim_cache: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_auto_invest_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="AI 智能投资助手", lifespan=lifespan)


async def _auto_invest_loop():
    from core.auto_invest import is_trading_day
    from core.config import get_env

    enabled = get_env("AUTO_INVEST_ENABLED", "1") == "1"
    hh, mm = (get_env("AUTO_INVEST_TIME", "15:30") + ":00").split(":")[:2]
    while enabled:
        try:
            now = datetime.now()
            if is_trading_day(now) and now.strftime("%H:%M") >= f"{hh}:{mm}":
                # 仅在实际执行时持锁；休眠阶段不持锁，避免阻塞 /api/analyze。
                with _invest_lock:
                    await asyncio.to_thread(_auto_invest, _provider, _account, _store)
                await asyncio.sleep(60 * 60 * 12)  # 半天后再查
            else:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001
            logger.error("自动投资循环异常: %s", exc)
            await asyncio.sleep(60)


_store = Store(DB_PATH)
_provider = DataProvider(_store)
_client = None
_embed = get_embedding_provider()
_account = SimAccount(_store)


class ChatRequest(BaseModel):
    query: str
    symbol: str | None = None


def _get_client():
    global _client
    if _client is None:
        try:
            from ai.provider import get_client
            _client = get_client()
        except ValueError:
            _client = None
    return _client


def _safe(fn, *args, **kwargs):
    try:
        client = _get_client()
        if client is None:
            return {}
        return fn(client, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AI 解读失败，降级: %s", exc)
        return {}


@app.get("/api/dashboard")
def dashboard():
    _account.maybe_reset_period()
    analysis = run_analysis(_provider)
    ai = _safe(interpret_trend, analysis["trend"])
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
    with _invest_lock:
        _account.maybe_reset_period()
        result = _auto_invest(_provider, _account, _store)
    ai = {
        "trend": _safe(interpret_trend, result.get("trend")),
        "sectors": _safe(recommend_sectors, result.get("sectors")),
        "stocks": _safe(recommend_stocks, result.get("stocks")),
        "portfolio": _safe(plan_portfolio, result.get("portfolio")),
    }
    # 将 result 展开到顶层（trend/sectors/stocks/portfolio/warnings 等），
    # 同时保留嵌套的 analysis 供看板消费者使用。
    return {**result, "analysis": result, "ai": ai, "account": _account.period_stats(),
            "data_until": result.get("data_until")}


@app.get("/api/simulation")
def simulation():
    if _sim_cache:
        return _sim_cache
    # 用独立内存库跑模拟，避免把模拟交易写入真实虚拟账户（data/app.db）。
    from core.store import Store

    sim_store = Store(":memory:")
    out = run_year_simulation(_provider, sim_store)
    _sim_cache.update(out)
    return out


@app.post("/api/chat")
def chat(req: ChatRequest):
    analysis = run_analysis(_provider)
    history = build_history_summary(_store, _account)
    rag_hits = []
    if req.symbol:
        news = _provider.stock_news(req.symbol) + _provider.stock_notices(req.symbol)
        if news:
            build_index(_embed, news, _store, symbol=req.symbol)
            rag_hits = retrieve(_embed, req.query, _store, top_k=5)
    client = _get_client()
    if client is None:
        return {"answer": "AI 服务未配置 DEEPSEEK_API_KEY，请参考看板数据。",
                "references": [], "confidence": 0.0, "disclaimer": DISCLAIMER,
                "data_until": analysis["data_until"]}
    out = answer_question(client, req.query, analysis, rag_hits, history)
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


app.mount("/web", StaticFiles(directory="web"), name="web")
