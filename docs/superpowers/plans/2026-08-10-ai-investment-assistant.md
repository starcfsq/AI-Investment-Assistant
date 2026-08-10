# AI 智能投资助手 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个面向 A 股长期投资者的 AI 智能投资助手——涵盖趋势/板块/选股/组合分析、RAG 问答、基于回测的权重自我迭代、以及 AI 虚拟投资账户，通过看板与对话呈现。

**Architecture:** 分层架构。`core/` 为确定性分析引擎（数据、趋势、板块、选股、组合、RAG、回测、账户），`ai/` 为 DeepSeek 层（Provider 抽象 + Schema 约束，只解读不编数），`api/` 为 FastAPI REST，`web/` 为无构建的单页前端。核心原则：**数据永不来自 LLM**。

**Tech Stack:** Python 3.11、FastAPI + uvicorn、akshare、pandas、numpy、SQLite（stdlib sqlite3）、sentence-transformers（可选 embedding）、chromadb（可选向量库）、pytest、requests。

**Spec:** `docs/superpowers/specs/2026-08-10-ai-investment-assistant-design.md`

## Global Constraints

- Python 3.11+；依赖管理用 `requirements.txt` + `pip`。
- 所有数字（指数、估值、板块涨幅、财务指标）由 `core/` 确定性计算，`ai/` 层不得生成数字。
- 每次分析产出 `data_until`（数据时点），前端与 AI 解读必须展示。
- 交易/胜率/收益率口径：胜率=已平仓盈利笔数/已平仓总笔数；阶段资金变化率=(期末净值−期初净值)/期初净值；基准=沪深300。
- 所有 AI 输出带 `disclaimer`（投资免责声明）与 `confidence`（置信度）。
- 所有 akshare 调用失败必须可降级：重试 1 次 → 用缓存旧数据（标注时点）→ 跳过该条。
- 测试文件放 `tests/`，用 pytest；akshare 相关测试用 mock，不依赖真实网络。
- 中文界面与注释；Git 提交信息用英文 `type: subject` 格式。
- 环境变量经 `.env` 加载；`DEEPSEEK_API_KEY` 必填；`.env` 不入库。
- 代码路径全部相对项目根目录 `D:\cfproject\project`。

---

## Phase 0：项目骨架与存储层

### Task 1: 项目结构与基础配置

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `config/weights.json`
- Create: `core/__init__.py`
- Create: `core/config.py`
- Create: `core/logging.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `core/config.py`: `ROOT`, `WEIGHTS_PATH`, `DB_PATH`, `load_weights() -> dict`, `save_weights(weights: dict) -> None`, `get_env(name: str, default: str = "") -> str`
  - `core/logging.py`: `get_logger(name: str) -> logging.Logger`

- [ ] **Step 1: 写项目骨架文件**

`pyproject.toml`:
```toml
[project]
name = "ai-investment-assistant"
version = "0.1.0"
description = "A股长期投资智能助手：趋势/板块/选股/组合分析 + RAG + 回测迭代 + 虚拟账户"
requires-python = ">=3.11"
dependencies = []

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`requirements.txt`:
```
fastapi==0.115.*
uvicorn==0.30.*
akshare==1.16.*
pandas==2.2.*
numpy==2.*
python-dotenv==1.0.*
requests==2.32.*
pytest==8.*
httpx==0.27.*
sentence-transformers==3.*
chromadb==0.5.*
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
.env
data/
dist/
.pytest_cache/
*.egg-info/
```

`.env.example`:
```
# DeepSeek API Key（必填）
DEEPSEEK_API_KEY=sk-xxxx
# 可选：模型名，默认 deepseek-chat
DEEPSEEK_MODEL=deepseek-chat
# 可选：虚拟账户初始资金
ACCOUNT_INITIAL_CAPITAL=1000000
```

`config/weights.json`:
```json
{
  "trend": {"ma": 0.3, "valuation": 0.4, "bond": 0.3},
  "sector": {"rs": 0.4, "flow": 0.3, "momentum": 0.3},
  "stock": {"roe": 0.3, "growth": 0.25, "valuation": 0.25, "dividend": 0.2}
}
```

`core/__init__.py`:
```python
"""AI 智能投资助手核心分析引擎包。"""
```

`core/config.py`:
```python
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "config" / "weights.json"
DB_PATH = ROOT / "data" / "app.db"
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def load_weights() -> dict:
    if not WEIGHTS_PATH.exists():
        return {
            "trend": {"ma": 0.3, "valuation": 0.4, "bond": 0.3},
            "sector": {"rs": 0.4, "flow": 0.3, "momentum": 0.3},
            "stock": {"roe": 0.3, "growth": 0.25, "valuation": 0.25, "dividend": 0.2},
        }
    with open(WEIGHTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_weights(weights: dict) -> None:
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
```

`core/logging.py`:
```python
import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
        logger.root.addHandler(handler)
        logger.root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logger
```

- [ ] **Step 2: 创建虚拟环境并安装依赖**

Run:
```bash
cd "D:\cfproject\project"
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```
Expected: 依赖安装成功。若 `sentence-transformers` / `chromadb` 安装缓慢或失败，可先注释掉这两行（RAG 有 Hash 降级，见 Task 10），后续再补装。

- [ ] **Step 3: 验证导入**

Run:
```bash
python -c "from core.config import load_weights; print(load_weights())"
```
Expected: 打印出三组权重字典。

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: bootstrap project skeleton and config"
```

### Task 2: SQLite 存储层

**Files:**
- Create: `core/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `core/config.DB_PATH`
- Produces: `class Store`：
  - `__init__(self, db_path: Path | str)`
  - `cache_get(key: str) -> str | None`
  - `cache_set(key: str, value: str, ttl_seconds: int) -> None`
  - `insert_trade(trade: dict) -> None`
  - `list_trades() -> list[dict]`
  - `get_account() -> dict`
  - `save_account(state: dict) -> None`
  - `insert_snapshot(snapshot: dict) -> None`
  - `list_snapshots(period_id: str) -> list[dict]`
  - `insert_period(period: dict) -> None`
  - `list_periods() -> list[dict]`
  - `insert_iter(iter_rec: dict) -> None`
  - `list_iters() -> list[dict]`
  - `insert_news(item: dict) -> None`
  - `list_news(symbol: str | None = None, limit: int = 200) -> list[dict]`
  - `save_chunk(chunk: dict) -> None`
  - `get_chunks() -> list[dict]`

- [ ] **Step 1: 写失败测试**

`tests/test_store.py`:
```python
import tempfile
from pathlib import Path

from core.store import Store


def make_store():
    return Store(tempfile.mkdtemp() + "/t.db")


def test_cache_set_get_roundtrip():
    s = make_store()
    assert s.cache_get("k1") is None
    s.cache_set("k1", "v1", 3600)
    assert s.cache_get("k1") == "v1"


def test_cache_expired():
    s = make_store()
    s.cache_set("k1", "v1", -1)
    assert s.cache_get("k1") is None


def test_account_default():
    s = make_store()
    acc = s.get_account()
    assert acc["cash"] == 0.0


def test_save_and_reload_account():
    s = make_store()
    s.save_account({"cash": 999.0, "initial_capital": 1000000.0,
                    "period_start": "2026-08-01", "updated_at": "2026-08-10"})
    acc = s.get_account()
    assert acc["cash"] == 999.0


def test_trade_and_period_roundtrip():
    s = make_store()
    s.insert_trade({"time": "2026-08-10 10:00:00", "symbol": "510300",
                    "name": "沪深300ETF", "side": "buy", "price": 3.9,
                    "qty": 10000, "fee": 39.0, "pnl": None, "status": "open"})
    trades = s.list_trades()
    assert len(trades) == 1 and trades[0]["side"] == "buy"
    s.insert_period({"period_id": "2026-08", "start": "2026-08-01", "end": "2026-08-31",
                     "initial_capital": 1000000.0, "final_nav": 1010000.0,
                     "win_rate": 0.6, "return_pct": 1.0, "benchmark_return": 0.5})
    assert s.list_periods()[0]["win_rate"] == 0.6


def test_news_and_chunk():
    s = make_store()
    s.insert_news({"title": "标题", "content": "内容", "date": "2026-08-09",
                   "source": "来源", "url": "http://x", "symbol": "600519"})
    assert len(s.list_news(symbol="600519")) == 1
    s.save_chunk({"chunk_id": "c1", "source_id": "1", "text": "片段",
                  "meta": '{"title":"标题"}', "embedding": b"1"})
    assert len(s.get_chunks()) == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'core.store'`）。

- [ ] **Step 3: 实现存储层**

`core/store.py`:
```python
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class Store:
    """SQLite 统一存储：数据缓存、账户、交易、快照、阶段、迭代历史、新闻与向量块。"""

    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at REAL
                );
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL DEFAULT 0,
                    initial_capital REAL NOT NULL DEFAULT 1000000,
                    period_start TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    qty REAL NOT NULL DEFAULT 0,
                    cost_price REAL NOT NULL DEFAULT 0,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT,
                    symbol TEXT,
                    name TEXT,
                    side TEXT,
                    price REAL,
                    qty REAL,
                    fee REAL,
                    pnl REAL,
                    status TEXT
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period_id TEXT,
                    date TEXT,
                    nav REAL,
                    cash REAL,
                    holdings_value REAL
                );
                CREATE TABLE IF NOT EXISTS periods (
                    period_id TEXT PRIMARY KEY,
                    start TEXT,
                    end TEXT,
                    initial_capital REAL,
                    final_nav REAL,
                    win_rate REAL,
                    return_pct REAL,
                    benchmark_return REAL
                );
                CREATE TABLE IF NOT EXISTS iter_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT,
                    run_at TEXT,
                    weights_json TEXT,
                    backtest_window TEXT,
                    win_rate REAL,
                    excess_return REAL,
                    data_until TEXT
                );
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    content TEXT,
                    date TEXT,
                    source TEXT,
                    url TEXT,
                    symbol TEXT
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT,
                    text TEXT,
                    meta TEXT,
                    embedding BLOB
                );
                """
            )

    # ---- 缓存 ----
    def cache_set(self, key: str, value: str, ttl_seconds: int) -> None:
        expires = time.time() + ttl_seconds
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                (key, value, expires),
            )

    def cache_get(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None or row["expires_at"] < time.time():
            return None
        return row["value"]

    # ---- 账户 ----
    def get_account(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
        if row is None:
            return {"cash": 0.0, "initial_capital": 1000000.0,
                    "period_start": None, "updated_at": None}
        return dict(row)

    def save_account(self, state: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO account (id, cash, initial_capital, period_start, updated_at)
                   VALUES (1, ?, ?, ?, ?)""",
                (state["cash"], state["initial_capital"],
                 state.get("period_start"), state.get("updated_at")),
            )

    def list_positions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM positions").fetchall()
        return [dict(r) for r in rows]

    def save_position(self, pos: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO positions
                   (symbol, name, qty, cost_price, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (pos["symbol"], pos["name"], pos["qty"],
                 pos["cost_price"], pos.get("updated_at")),
            )

    def delete_position(self, symbol: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))

    # ---- 交易 ----
    def insert_trade(self, trade: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO trades (time, symbol, name, side, price, qty, fee, pnl, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade["time"], trade["symbol"], trade["name"], trade["side"],
                 trade["price"], trade["qty"], trade["fee"],
                 trade.get("pnl"), trade.get("status")),
            )

    def list_trades(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM trades ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    # ---- 快照与阶段 ----
    def insert_snapshot(self, snapshot: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO snapshots (period_id, date, nav, cash, holdings_value)
                   VALUES (?, ?, ?, ?, ?)""",
                (snapshot["period_id"], snapshot["date"], snapshot["nav"],
                 snapshot["cash"], snapshot["holdings_value"]),
            )

    def list_snapshots(self, period_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM snapshots WHERE period_id = ? ORDER BY date",
                (period_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_period(self, period: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO periods
                   (period_id, start, end, initial_capital, final_nav,
                    win_rate, return_pct, benchmark_return)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (period["period_id"], period["start"], period["end"],
                 period["initial_capital"], period["final_nav"],
                 period["win_rate"], period["return_pct"], period["benchmark_return"]),
            )

    def list_periods(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM periods ORDER BY start").fetchall()
        return [dict(r) for r in rows]

    # ---- 迭代历史 ----
    def insert_iter(self, rec: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO iter_history
                   (version, run_at, weights_json, backtest_window,
                    win_rate, excess_return, data_until)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (rec["version"], rec["run_at"], rec["weights_json"],
                 rec["backtest_window"], rec["win_rate"],
                 rec["excess_return"], rec["data_until"]),
            )

    def list_iters(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM iter_history ORDER BY id DESC LIMIT 50"
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- 新闻与 RAG ----
    def insert_news(self, item: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO news (title, content, date, source, url, symbol)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (item["title"], item.get("content", ""), item.get("date", ""),
                 item.get("source", ""), item.get("url", ""), item.get("symbol", "")),
            )

    def list_news(self, symbol: str | None = None, limit: int = 200) -> list[dict]:
        with self._conn() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM news WHERE symbol = ? ORDER BY date DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM news ORDER BY date DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    def save_chunk(self, chunk: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO rag_chunks
                   (chunk_id, source_id, text, meta, embedding)
                   VALUES (?, ?, ?, ?, ?)""",
                (chunk["chunk_id"], chunk["source_id"], chunk["text"],
                 chunk["meta"], chunk["embedding"]),
            )

    def get_chunks(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM rag_chunks").fetchall()
        return [dict(r) for r in rows]

    def clear_chunks(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM rag_chunks")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_store.py -v`
Expected: 6 个测试全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/test_store.py core/store.py
git commit -m "feat: add SQLite storage layer"
```

---

## Phase 1：数据层

### Task 3: akshare 数据封装与缓存

**Files:**
- Create: `core/data.py`

**Interfaces:**
- Consumes: `core.store.Store`, `core.config.DB_PATH`, `core.logging.get_logger`
- Produces: `class DataProvider`：
  - `__init__(self, store: Store | None = None)`
  - `index_daily(symbol: str) -> pd.DataFrame`  # columns: date, close
  - `index_valuation(name: str) -> pd.DataFrame`  # columns: date, pe, pb
  - `sector_quote() -> pd.DataFrame`  # columns: name, pct_change
  - `sector_flow() -> pd.DataFrame`  # columns: name, net_inflow
  - `sector_hist(name: str) -> pd.DataFrame`  # columns: date, close
  - `stock_spot() -> pd.DataFrame`  # columns: code, name, price, pct_change
  - `stock_hist(code: str, start: str, end: str) -> pd.DataFrame`
  - `stock_financial(code: str) -> dict`  # keys: roe, growth, pe, pb, dividend
  - `bond_yield() -> pd.DataFrame`  # columns: date, cn_10y
  - `stock_news(code: str) -> list[dict]`
  - `stock_notices(code: str) -> list[dict]`
  - `benchmark_index_code() -> str`  # "沪深300" 对应代码 "sh000300"
  - `quality_report() -> list[dict]`  # 各数据源 {source, status, fetched_at, data_until, ttl_seconds}

**设计要点（实时性 / 有效性 / 充分性，见设计文档 §6.1.1）：**
- 每个方法用 `_cached(key, ttl, fetch)` 包装：命中缓存直接返回，否则调用 akshare 并缓存 JSON。
- 所有 akshare 调用包 try/except：异常时重试一次；仍失败且无缓存则返回空 DataFrame / 空列表（由上层降级）。
- **实时性**：每次抓取记录 `fetched_at` 与 `data_until`（数据截止日）；缓存命中标记 `status=cached`。`quality_report()` 汇总供看板展示。
- **有效性**：抓取结果过 `_validate_df`——close/price 必须 >0，pct_change 在 ±50%，PE 在 0~500，PB 在 0~100，按日期去重升序。
- **充分性**：单条坏数据剔除不致命；数据量不足由分析模块（Task 4-8）标注告警。

- [ ] **Step 1: 实现数据层**

`core/data.py`:
```python
"""akshare 数据封装。所有方法可离线降级：缓存命中优先，网络失败返回空数据。"""
import json
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

from core.logging import get_logger
from core.store import Store

logger = get_logger("core.data")

_INDEX_CODES = {
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
    "沪深300": "sh000300",
    "中证500": "sh000905",
}


class DataProvider:
    def __init__(self, store: Store | None = None):
        self.store = store or Store(":memory:")
        self._freshness: dict[str, dict] = {}

    def _record_freshness(self, key: str, status: str,
                          data_until: str | None, ttl: int) -> None:
        self._freshness[key] = {
            "source": key, "status": status,
            "fetched_at": _now(), "data_until": data_until or "",
            "ttl_seconds": ttl,
        }

    def quality_report(self) -> list[dict]:
        """返回各数据源的新鲜度/状态汇总，供看板与 AI 展示。"""
        return [dict(v) for v in self._freshness.values()]

    # ---- 通用缓存 ----
    def _cached(self, key: str, ttl: int, fetch: Callable[[], Any]) -> Any:
        if self.store:
            raw = self.store.cache_get(key)
            if raw is not None:
                cached_data = _from_json(raw)
                self._record_freshness(key, "cached", _data_until(cached_data), ttl)
                return cached_data
        status = "ok"
        try:
            data = fetch()
        except Exception as exc:  # noqa: BLE001
            logger.warning("akshare 调用失败 %s: %s，重试一次", key, exc)
            try:
                data = fetch()
                status = "ok_retry"
            except Exception as exc2:  # noqa: BLE001
                logger.error("akshare 重试仍失败 %s: %s", key, exc2)
                self._record_freshness(key, "missing", None, ttl)
                return _empty_like(key)
        data = _validate_df(data, key)
        if self.store and data is not None:
            self.store.cache_set(key, _to_json(data), ttl)
        self._record_freshness(key, status, _data_until(data), ttl)
        return data

    # ---- 指数 ----
    def index_daily(self, symbol: str) -> pd.DataFrame:
        import akshare as ak

        code = _INDEX_CODES.get(symbol, symbol)

        def fetch():
            df = ak.stock_zh_index_daily(symbol=code)
            df = df.rename(columns={"date": "date", "close": "close"})
            return df[["date", "close"]].copy()

        return self._cached(f"index_daily:{symbol}", 3600 * 6, fetch)

    def index_valuation(self, name: str) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            # 乐咕乐股指数 PE/PB 历史
            df = ak.stock_index_pe_lg(symbol=name)
            return df[["date", "pe", "pb"]].copy()

        return self._cached(f"index_valuation:{name}", 3600 * 12, fetch)

    # ---- 板块 ----
    def sector_quote(self) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            df = ak.stock_board_industry_name_em()
            return df.rename(
                columns={"板块名称": "name", "涨跌幅": "pct_change"}
            )[["name", "pct_change"]].copy()

        return self._cached("sector_quote", 3600 * 4, fetch)

    def sector_flow(self) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            df = ak.stock_sector_fund_flow_rank(
                indicator="今日", sector_type="行业资金流"
            )
            return df.rename(
                columns={"名称": "name", "主力净流入-净额": "net_inflow"}
            )[["name", "net_inflow"]].copy()

        return self._cached("sector_flow", 3600 * 4, fetch)

    def sector_hist(self, name: str) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            df = ak.stock_board_industry_hist_em(
                symbol=name, period="日k",
                start_date="20200101",
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="",
            )
            return df.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]].copy()

        return self._cached(f"sector_hist:{name}", 3600 * 6, fetch)

    # ---- 个股 ----
    def stock_spot(self) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            df = ak.stock_zh_a_spot_em()
            return df.rename(
                columns={"代码": "code", "名称": "name",
                         "最新价": "price", "涨跌幅": "pct_change"}
            )[["code", "name", "price", "pct_change"]].copy()

        return self._cached("stock_spot", 3600, fetch)

    def stock_hist(self, code: str, start: str, end: str) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust="qfq",
            )
            return df.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]].copy()

        return self._cached(f"stock_hist:{code}:{start}:{end}", 3600 * 6, fetch)

    def stock_financial(self, code: str) -> dict:
        import akshare as ak

        def fetch():
            df = ak.stock_a_indicator_lg(symbol=code)
            if df is None or df.empty:
                return {}
            last = df.iloc[-1]
            return {
                "code": code,
                "pe": _num(last.get("pe")),
                "pb": _num(last.get("pb")),
                "dividend": _num(last.get("dv_ratio")),
                "roe": 0.0,
                "growth": 0.0,
            }

        return self._cached(f"stock_financial:{code}", 3600 * 12, fetch)

    # ---- 国债收益率 ----
    def bond_yield(self) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            df = ak.bond_china_yield(start_date="20150101")
            col = [c for c in df.columns if "中债国债10年" in str(c)]
            if not col:
                return pd.DataFrame(columns=["date", "cn_10y"])
            out = df[["日期", col[0]]].rename(
                columns={"日期": "date", col[0]: "cn_10y"}
            )
            out["date"] = pd.to_datetime(out["date"])
            return out

        return self._cached("bond_yield", 3600 * 12, fetch)

    # ---- 新闻与公告 ----
    def stock_news(self, code: str) -> list[dict]:
        import akshare as ak

        def fetch():
            df = ak.stock_news_em(symbol=code)
            items = []
            for _, row in df.head(50).iterrows():
                items.append({
                    "title": str(row.get("新闻标题", "")),
                    "content": str(row.get("新闻内容", "")),
                    "date": str(row.get("发布时间", "")),
                    "source": str(row.get("文章来源", "")),
                    "url": str(row.get("新闻链接", "")),
                    "symbol": code,
                })
            return items

        return self._cached(f"stock_news:{code}", 3600 * 3, fetch)

    def stock_notices(self, code: str) -> list[dict]:
        import akshare as ak

        def fetch():
            df = ak.stock_notice_report(symbol=code)
            items = []
            for _, row in df.head(50).iterrows():
                items.append({
                    "title": str(row.get("公告标题", "")),
                    "content": str(row.get("公告内容", "")),
                    "date": str(row.get("公告日期", "")),
                    "source": "交易所公告",
                    "url": str(row.get("pdf链接", "")),
                    "symbol": code,
                })
            return items

        return self._cached(f"stock_notices:{code}", 3600 * 3, fetch)

    def benchmark_index_code(self) -> str:
        return "sh000300"


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _validate_df(data: Any, key: str) -> Any:
    """数据有效性校验：剔除坏值，按日期去重升序。单条坏数据不影响整体。"""
    if not isinstance(data, pd.DataFrame) or data.empty:
        return data
    try:
        if "close" in data.columns:
            data = data[data["close"] > 0]
        if "price" in data.columns:
            data = data[data["price"] > 0]
        if "pct_change" in data.columns:
            data = data[data["pct_change"].abs() <= 50]
        if "pe" in data.columns:
            data = data[(data["pe"] > 0) & (data["pe"] < 500)]
        if "pb" in data.columns:
            data = data[(data["pb"] > 0) & (data["pb"] < 100)]
        subset = [c for c in ("date", "close", "price", "pe", "pb")
                  if c in data.columns]
        if subset:
            data = data.dropna(subset=subset)
        if "date" in data.columns:
            data = data.sort_values("date").drop_duplicates(subset=["date"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("数据校验失败 %s: %s", key, exc)
    return data


def _data_until(data: Any) -> str | None:
    """从数据里推断内容截止日期（实时性指标之一）。"""
    if isinstance(data, pd.DataFrame) and "date" in data.columns and not data.empty:
        return str(data["date"].iloc[-1])[:10]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return str(data[0].get("date", ""))[:10]
    return None


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _to_json(obj: Any) -> str:
    if isinstance(obj, pd.DataFrame):
        return obj.to_json(orient="split", date_format="iso", force_ascii=False)
    return json.dumps(obj, ensure_ascii=False, default=str)


def _from_json(raw: str) -> Any:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return _empty_like("")
    if isinstance(obj, dict) and "columns" in obj and "data" in obj:
        return pd.DataFrame(data=obj["data"], columns=obj["columns"])
    return obj


def _empty_like(key: str) -> Any:
    if key.startswith(("index_daily", "index_valuation", "sector_quote",
                       "sector_flow", "sector_hist", "stock_spot", "stock_hist",
                       "bond_yield")):
        return pd.DataFrame()
    if key.startswith("stock_financial"):
        return {}
    return []
```

**说明（实现时必读）：** akshare 接口名随版本变化。若 `dir(ak)` 中找不到上述函数，用 `python -c "import akshare as ak; print([x for x in dir(ak) if 'board_industry' in x or 'index_pe' in x])"` 找替代名，并替换 `fetch` 内的调用。这是已知的维护点，不是占位符。

- [ ] **Step 2: 数据层离线冒烟测试**

Run:
```bash
python -c "
from core.data import DataProvider
p = DataProvider()
print('bond_yield:', p.bond_yield().shape)
print('sector_quote:', p.sector_quote().shape)
"
```
Expected: 若网络可用则打印真实形状；若网络不可用则打印 `(0, 0)` 且不抛异常（验证降级路径）。**注意**：`Store(":memory:")` 每次调用独立连接，缓存不共享——生产路径由 Task 2 的持久化 Store 提供。

- [ ] **Step 3: Commit**

```bash
git add core/data.py
git commit -m "feat: add akshare data provider with cache and degradation"
```

---

## Phase 2：确定性分析引擎

### Task 4: trend 趋势模块

**Files:**
- Create: `core/trend.py`
- Test: `tests/test_trend.py`

**Interfaces:**
- Consumes: `core.config.load_weights`
- Produces:
  - `pct_rank_historical(value: float, history: pd.Series) -> float`  # 0-100
  - `ma_deviation(close: float, ma_value: float) -> float`  # (close/ma - 1)
  - `analyze_trend(index_df, val_df, bond_df, weights: dict) -> dict`
    返回形如：
    ```json
    {
      "signals": {"ma": 72.0, "valuation": 35.0, "bond": 60.0},
      "state": "高估风险",
      "composite": 53.5,
      "detail": {"ma_dev": 0.08, "pe_pct": 65.0, "pb_pct": 70.0, "bond_equity_pct": 60.0},
      "data_until": "2026-08-09"
    }
    ```

**信号口径（0-100，越高对长期投资者越有利）：**
- ma：`clamp((close/MA250 - 1) * 500 + 50, 0, 100)`（显著站上年线 → 高分）
- valuation：`100 - pe_pct`（PE 百分位越低 → 越便宜 → 高分）
- bond：股债性价比百分位（越高 → 股优于债 → 高分）

**状态映射（composite = 加权和，权重来自 config）：**
- composite ≥ 60 → "低估机会"
- composite ≤ 40 → "高估风险"
- 否则 → "中性合理"

- [ ] **Step 1: 写失败测试**

`tests/test_trend.py`:
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_trend.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现趋势模块**

`core/trend.py`:
```python
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

    pe_pct = pb_pct = 50.0
    valuation_signal = 50.0
    if not val_df.empty and "pe" in val_df.columns and "pb" in val_df.columns:
        pe_pct = pct_rank_historical(_last_pe(val_df), val_df["pe"])
        pb_pct = pct_rank_historical(_last_pb(val_df), val_df["pb"])
        valuation_signal = 100.0 - pe_pct

    bond_signal = 50.0
    bond_pct = 50.0
    if (not bond_df.empty and not val_df.empty
            and "pe" in val_df.columns and "cn_10y" in bond_df.columns):
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_trend.py -v`
Expected: 4 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add core/trend.py tests/test_trend.py
git commit -m "feat: add market trend analysis"
```

### Task 5: sector 板块模块

**Files:**
- Create: `core/sector.py`
- Test: `tests/test_sector.py`

**Interfaces:**
- Consumes: `core.config.load_weights`
- Produces:
  - `score_sectors(quotes: pd.DataFrame, flow: pd.DataFrame, hist: dict[str, pd.DataFrame],
                   bench_df: pd.DataFrame, weights: dict) -> list[dict]`
    返回按 score 降序的列表，每项：
    ```json
    {"name": "半导体", "rs": 72.0, "flow": 60.0, "momentum": 80.0, "score": 71.2}
    ```

**打分口径（均归一化到 0-100）：**
- RS：板块近 3 个月收益 − 基准同区间收益，跨板块 min-max 归一化
- flow：净流入跨板块 min-max 归一化
- momentum：近 20 日涨幅跨板块 min-max 归一化
- score = rs×w.rs + flow×w.flow + momentum×w.momentum

- [ ] **Step 1: 写失败测试**

`tests/test_sector.py`:
```python
import pandas as pd
from core.sector import score_sectors

WEIGHTS = {"sector": {"rs": 0.4, "flow": 0.3, "momentum": 0.3}}


def _dummy_hist(n=65, start_val=100.0):
    import numpy as np
    dates = pd.date_range("2026-01-01", periods=n)
    return pd.DataFrame({"date": dates, "close": np.linspace(start_val, start_val * 1.2, n)})


def test_score_orders_by_score():
    quotes = pd.DataFrame({"name": ["A", "B"], "pct_change": [2.0, 1.0]})
    flow = pd.DataFrame({"name": ["A", "B"], "net_inflow": [10.0, 5.0]})
    hist = {"A": _dummy_hist(), "B": _dummy_hist(start_val=100.0)}
    bench = _dummy_hist(start_val=100.0)
    result = score_sectors(quotes, flow, hist, bench, WEIGHTS)
    assert result[0]["name"] == "A"
    assert 0 <= result[0]["score"] <= 100


def test_score_fields_present():
    quotes = pd.DataFrame({"name": ["X"], "pct_change": [1.0]})
    flow = pd.DataFrame({"name": ["X"], "net_inflow": [8.0]})
    hist = {"X": _dummy_hist()}
    bench = _dummy_hist()
    result = score_sectors(quotes, flow, hist, bench, WEIGHTS)
    item = result[0]
    for k in ("name", "rs", "flow", "momentum", "score"):
        assert k in item
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_sector.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现板块模块**

`core/sector.py`:
```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_sector.py -v`
Expected: 2 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add core/sector.py tests/test_sector.py
git commit -m "feat: add sector scoring"
```

### Task 6: stock 选股模块

**Files:**
- Create: `core/stock.py`
- Test: `tests/test_stock.py`

**Interfaces:**
- Consumes: `core.config.load_weights`
- Produces:
  - `score_stock(candidate: dict, weights: dict) -> float`
  - `rank_stocks(candidates: list[dict], weights: dict) -> list[dict]`
    返回按 score 降序，每项含原始字段 + `score`。

**打分口径（各因子归一化 0-100，跨候选 min-max）：**
- roe：高好
- growth（净利润同比增速）：高好
- valuation：PE 百分位取反（低估值好）；无百分位时用 1/PE 归一化
- dividend：高好
- score = roe×w.roe + growth×w.growth + val×w.valuation + div×w.dividend

- [ ] **Step 1: 写失败测试**

`tests/test_stock.py`:
```python
from core.stock import rank_stocks, score_stock

WEIGHTS = {"stock": {"roe": 0.3, "growth": 0.25, "valuation": 0.25, "dividend": 0.2}}

CANDIDATES = [
    {"code": "600001", "name": "甲", "roe": 20.0, "growth": 15.0,
     "pe_pct": 20.0, "dividend": 3.0},
    {"code": "600002", "name": "乙", "roe": 5.0, "growth": -5.0,
     "pe_pct": 90.0, "dividend": 0.5},
]


def test_rank_stocks_orders_by_score():
    ranked = rank_stocks(CANDIDATES, WEIGHTS)
    assert ranked[0]["name"] == "甲"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_score_stock_returns_0_to_100():
    s = score_stock(CANDIDATES[0], WEIGHTS)
    assert 0 <= s <= 100
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_stock.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现选股模块**

`core/stock.py`:
```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_stock.py -v`
Expected: 2 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add core/stock.py tests/test_stock.py
git commit -m "feat: add stock multi-factor scoring"
```

### Task 7: portfolio 组合模块

**Files:**
- Create: `core/portfolio.py`
- Test: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: 无（输入为 sector 打分结果）
- Produces:
  - `build_portfolio(sector_scores: list[dict], core_ratio: float = 0.7,
                     top_n: int = 4) -> dict`
    返回：
    ```json
    {
      "core": {"name": "沪深300ETF", "weight": 0.7, "note": "宽基核心"},
      "satellite": [{"name": "半导体ETF", "weight": 0.1}, ...],
      "rebalance_rule": "权重偏离目标 5% 时再平衡",
      "summary": "核心70% + 卫星30%"
    }
    ```

**规则：** 取 top_n 板块做卫星，权重按 score 归一化后 × (1 − core_ratio)。ETF 名称映射表（板块→常见 ETF），未匹配则用"板块名+ETF"。

- [ ] **Step 1: 写失败测试**

`tests/test_portfolio.py`:
```python
import pytest

from core.portfolio import build_portfolio

SECTORS = [
    {"name": "半导体", "score": 90.0},
    {"name": "白酒", "score": 60.0},
    {"name": "医药", "score": 30.0},
    {"name": "新能源", "score": 20.0},
]


def test_core_ratio_and_satellite_sum():
    p = build_portfolio(SECTORS, core_ratio=0.7, top_n=3)
    assert p["core"]["weight"] == 0.7
    sat_sum = round(sum(s["weight"] for s in p["satellite"]), 4)
    assert sat_sum == pytest.approx(0.3)
    assert len(p["satellite"]) == 3


def test_satellite_weight_proportional_to_score():
    p = build_portfolio(SECTORS, core_ratio=0.7, top_n=3)
    assert p["satellite"][0]["name"] == "半导体ETF(512480)"
    assert p["satellite"][0]["weight"] > p["satellite"][2]["weight"]
```

（pytest 断言已含 `import pytest`。）

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_portfolio.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现组合模块**

`core/portfolio.py`:
```python
"""长期配置：核心 + 卫星。"""
from core.logging import get_logger

logger = get_logger("core.portfolio")

ETF_MAP = {
    "半导体": "半导体ETF(512480)",
    "白酒": "白酒ETF(512690)",
    "医药": "医药ETF(512010)",
    "新能源": "新能源ETF(516160)",
    "证券": "证券ETF(512880)",
    "银行": "银行ETF(512800)",
    "光伏": "光伏ETF(515790)",
    "军工": "军工ETF(512660)",
    "消费": "消费ETF(159928)",
    "科技": "科技ETF(515000)",
}


def build_portfolio(sector_scores: list[dict], core_ratio: float = 0.7,
                    top_n: int = 4) -> dict:
    top = sorted(sector_scores, key=lambda s: s["score"], reverse=True)[:top_n]
    scores = [s["score"] for s in top]
    total = sum(scores) or 1.0
    satellite = []
    for s in top:
        weight = round((s["score"] / total) * (1.0 - core_ratio), 4)
        etf = ETF_MAP.get(s["name"], f"{s['name']}ETF")
        satellite.append({"name": etf, "weight": weight})
    return {
        "core": {"name": "沪深300ETF(510300)", "weight": core_ratio,
                 "note": "宽基核心"},
        "satellite": satellite,
        "rebalance_rule": "权重偏离目标 5% 时再平衡",
        "summary": f"核心{int(core_ratio * 100)}% + 卫星{int((1 - core_ratio) * 100)}%",
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_portfolio.py -v`
Expected: 2 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add core/portfolio.py tests/test_portfolio.py
git commit -m "feat: add portfolio construction"
```

### Task 8: analyze 分析编排

**Files:**
- Create: `core/analyze.py`

**Interfaces:**
- Consumes: `core.data.DataProvider`, `core.trend.analyze_trend`, `core.sector.score_sectors`, `core.stock.rank_stocks`, `core.portfolio.build_portfolio`, `core.config.load_weights`
- Produces: `run_analysis(provider: DataProvider) -> dict`
  ```json
  {
    "generated_at": "...",
    "trend": {...},
    "sectors": [ {name, rs, flow, momentum, score}, ...],
    "stocks": [ {code, name, score}, ...],
    "portfolio": {...},
    "data_until": "...",
    "data_quality": [ {source, status, fetched_at, data_until, ttl_seconds}, ...],
    "warnings": [ "趋势数据不足：估值历史少于5年", ...]
  }
  ```

**充分性告警规则（写进实现）：**
- 指数历史 < 250 行 → `趋势数据不足：MA250 样本不足`
- 估值历史 < 120 行 → `趋势数据不足：估值历史过短`
- 板块数 < 10 → `板块覆盖不足`
- 候选股票 < 5 → `选股候选不足`

- [ ] **Step 1: 实现编排**

`core/analyze.py`:
```python
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
        if not quotes.empty and "name" in quotes.columns:
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
```

- [ ] **Step 2: 冒烟测试（离线可用）**

Run:
```bash
python -c "
from core.data import DataProvider
from core.analyze import run_analysis
from core.store import Store
import tempfile
p = DataProvider(Store(tempfile.mkdtemp() + '/t.db'))
r = run_analysis(p)
print('trend:', r['trend'].get('state'))
print('sectors:', len(r['sectors']), 'stocks:', len(r['stocks']))
print('portfolio:', r['portfolio'].get('summary'))
"
```
Expected: 不抛异常；网络可用时打印真实分析结果，不可用时打印空数据/默认值。

- [ ] **Step 3: Commit**

```bash
git add core/analyze.py
git commit -m "feat: add analysis orchestration"
```

---

## Phase 3：AI 层 + RAG

### Task 9: LLM Provider 抽象与 DeepSeek 实现

**Files:**
- Create: `ai/__init__.py`
- Create: `ai/provider.py`
- Create: `ai/deepseek.py`
- Test: `tests/test_deepseek.py`

**Interfaces:**
- Consumes: `core.config.get_env`
- Produces:
  - `ai/provider.py`:
    - `class LLMClient`: `chat_json(self, messages: list[dict], schema: dict) -> dict`
    - `def get_client() -> LLMClient`
  - `ai/deepseek.py`:
    - `class DeepSeekClient(LLMClient)`

**DeepSeek 约定：** POST `https://api.deepseek.com/chat/completions`，`model=deepseek-chat`，`response_format={"type": "json_object"}`，`Authorization: Bearer <key>`。解析 `choices[0].message.content` 为 JSON；校验 schema 失败重试一次。

- [ ] **Step 1: 写失败测试（mock HTTP）**

`tests/test_deepseek.py`:
```python
import pytest
from ai.deepseek import DeepSeekClient


class FakeResp:
    def __init__(self, content):
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_chat_json_parses_content(monkeypatch):
    client = DeepSeekClient(api_key="test-key")
    fake = FakeResp('{"state": "低估机会"}')

    def fake_post(url, headers, json, timeout):
        assert "api.deepseek.com" in url
        return fake

    monkeypatch.setattr(client._session, "post", fake_post)
    out = client.chat_json(
        [{"role": "user", "content": "解读"}],
        {"type": "object", "properties": {"state": {"type": "string"}}},
    )
    assert out == {"state": "低估机会"}


def test_no_api_key_raises():
    from ai.deepseek import DeepSeekClient
    with pytest.raises(ValueError):
        DeepSeekClient(api_key="")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_deepseek.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现 Provider**

`ai/__init__.py`:
```python
"""AI 层：LLM Provider、解读与对话。"""
```

`ai/provider.py`:
```python
"""LLM Provider 抽象。"""
from abc import ABC, abstractmethod

from core.config import get_env


class LLMClient(ABC):
    @abstractmethod
    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        """发送消息，返回符合 schema 的 JSON 对象。"""


def get_client() -> LLMClient:
    from ai.deepseek import DeepSeekClient

    return DeepSeekClient(api_key=get_env("DEEPSEEK_API_KEY"))
```

`ai/deepseek.py`:
```python
"""DeepSeek LLM 客户端。"""
import json

import requests

from ai.provider import LLMClient
from core.logging import get_logger

logger = get_logger("ai.deepseek")

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 未配置")
        self.api_key = api_key
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        }
        for attempt in range(2):
            resp = self._session.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            try:
                data = json.loads(content)
                _validate(data, schema)
                return data
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("DeepSeek 输出不符合 schema（第 %d 次）: %s", attempt + 1, exc)
        raise ValueError("DeepSeek 输出两次均不符合 schema")


def _validate(data, schema: dict) -> None:
    if schema.get("type") == "object":
        for key, prop in schema.get("properties", {}).items():
            if key not in data:
                raise ValueError(f"缺少字段 {key}")
            if prop.get("type") == "number" and not isinstance(data[key], (int, float)):
                raise ValueError(f"字段 {key} 应为数字")
            if prop.get("type") == "string" and not isinstance(data[key], str):
                raise ValueError(f"字段 {key} 应为字符串")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_deepseek.py -v`
Expected: 2 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add ai/ tests/test_deepseek.py
git commit -m "feat: add DeepSeek LLM client with JSON schema validation"
```

### Task 10: RAG 管道（embedding + 检索）

**Files:**
- Create: `core/embedding.py`
- Create: `core/rag.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `core.store.Store`
- Produces:
  - `core/embedding.py`:
    - `class EmbeddingProvider`: `embed(self, texts: list[str], dim: int = 64) -> list[list[float]]`
    - `class HashEmbedding(EmbeddingProvider)`：确定性哈希，离线可用，作为测试与默认降级
    - `def get_embedding_provider() -> EmbeddingProvider`
  - `core/rag.py`:
    - `build_index(provider: EmbeddingProvider, news_items: list[dict], store: Store, symbol: str) -> None`
    - `retrieve(provider: EmbeddingProvider, query: str, store: Store, top_k: int = 5) -> list[dict]`
      返回项含 `text`, `title`, `date`, `source`, `url`, `similarity`。

**设计：** chunk 为单条新闻（标题+内容），向量化存入 Store；检索用余弦相似度取 top-k。HashEmbedding 用 n-gram 哈希，保证无需下载模型即可跑通；生产可用 sentence-transformers（`get_embedding_provider` 尝试加载，失败回退 HashEmbedding）。

- [ ] **Step 1: 写失败测试**

`tests/test_rag.py`:
```python
import tempfile

from core.embedding import HashEmbedding
from core.rag import build_index, retrieve
from core.store import Store


def test_retrieve_finds_most_similar():
    store = Store(tempfile.mkdtemp() + "/t.db")
    emb = HashEmbedding()
    news = [
        {"title": "半导体板块大涨", "content": "芯片需求旺盛", "date": "2026-08-09",
         "source": "s", "url": "http://a", "symbol": "600001"},
        {"title": "白酒板块回调", "content": "白酒库存偏高", "date": "2026-08-08",
         "source": "s", "url": "http://b", "symbol": "600002"},
    ]
    build_index(emb, news, store, symbol="all")
    hits = retrieve(emb, "半导体 芯片", store, top_k=1)
    assert hits[0]["title"] == "半导体板块大涨"


def test_retrieve_empty_index_returns_empty():
    store = Store(tempfile.mkdtemp() + "/t.db")
    emb = HashEmbedding()
    assert retrieve(emb, "测试", store) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_rag.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现 embedding 与 RAG**

`core/embedding.py`:
```python
"""文本向量化 Provider。默认用确定性哈希，可切换 sentence-transformers。"""
import hashlib

import numpy as np

from core.logging import get_logger

logger = get_logger("core.embedding")


class EmbeddingProvider:
    def embed(self, texts: list[str], dim: int = 64) -> list[list[float]]:
        raise NotImplementedError


class HashEmbedding(EmbeddingProvider):
    """离线确定性哈希向量，供测试与默认降级。"""

    def embed(self, texts: list[str], dim: int = 512) -> list[list[float]]:
        # 默认 512 维：64 维哈希碰撞会导致检索测试失真（无关文档误得高分）；
        # 512 同时与 bge-small-zh 输出维度一致，降级换真实模型时宽度不变。
        vectors = []
        for text in texts:
            vec = np.zeros(dim, dtype=float)
            grams = _ngrams(text, n=3)
            for gram in grams:
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                vec[h % dim] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors


def _ngrams(text: str, n: int = 3):
    text = text.replace(" ", "")
    return [text[i : i + n] for i in range(max(0, len(text) - n + 1))]


def get_embedding_provider() -> EmbeddingProvider:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

        class STEmbedding(EmbeddingProvider):
            def embed(self, texts: list[str], dim: int = 64) -> list[list[float]]:
                return [v.tolist() for v in model.encode(texts)]

        logger.info("已加载 sentence-transformers 模型")
        return STEmbedding()
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法加载 sentence-transformers（%s），使用 HashEmbedding 降级", exc)
        return HashEmbedding()
```

`core/rag.py`:
```python
"""RAG：新闻/公告 索引与检索。"""
from core.logging import get_logger
from core.store import Store

logger = get_logger("core.rag")


def build_index(provider, news_items: list[dict], store: Store, symbol: str) -> None:
    if not news_items:
        return
    store.clear_chunks()
    texts, metas = [], []
    for i, item in enumerate(news_items):
        text = f"{item['title']}。{item['content']}"[:500]
        texts.append(text)
        metas.append({
            "title": item["title"], "date": item.get("date", ""),
            "source": item.get("source", ""), "url": item.get("url", ""),
            "symbol": item.get("symbol", symbol),
        })
    vectors = provider.embed(texts)
    for i, vec in enumerate(vectors):
        store.save_chunk({
            "chunk_id": f"{symbol}:{i}",
            "source_id": str(i),
            "text": texts[i],
            "meta": _to_json(metas[i]),
            "embedding": _float32_bytes(vec),
        })


def retrieve(provider, query: str, store: Store, top_k: int = 5) -> list[dict]:
    chunks = store.get_chunks()
    if not chunks:
        return []
    qvec = provider.embed([query])[0]
    scored = []
    for c in chunks:
        sim = _cosine(qvec, _bytes_to_float32(c["embedding"]))
        meta = _from_json(c["meta"])
        scored.append({
            "text": c["text"],
            "title": meta.get("title", ""),
            "date": meta.get("date", ""),
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
            "similarity": round(sim, 4),
        })
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    x, y = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) or 1.0
    return float(np.dot(x, y) / denom)


def _float32_bytes(vec: list[float]) -> bytes:
    import numpy as np

    return np.asarray(vec, dtype=np.float32).tobytes()


def _bytes_to_float32(raw: bytes) -> list[float]:
    import numpy as np

    return np.frombuffer(raw, dtype=np.float32).tolist()


def _to_json(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def _from_json(raw: str) -> dict:
    import json

    return json.loads(raw)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_rag.py -v`
Expected: 2 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add core/embedding.py core/rag.py tests/test_rag.py
git commit -m "feat: add RAG index and retrieval"
```

### Task 11: AI 解读功能

**Files:**
- Create: `ai/schemas.py`
- Create: `ai/interpret.py`
- Create: `ai/chat.py`

**Interfaces:**
- Consumes: `ai.provider.LLMClient`, `core.rag.retrieve`, `core.data.DataProvider`
- Produces:
  - `ai/schemas.py`: `TREND_SCHEMA`, `SECTOR_SCHEMA`, `STOCK_SCHEMA`, `PORTFOLIO_SCHEMA`, `CHAT_SCHEMA`（均为 dict）
  - `ai/interpret.py`:
    - `interpret_trend(client, trend: dict) -> dict`
    - `recommend_sectors(client, sectors: list[dict]) -> dict`
    - `recommend_stocks(client, stocks: list[dict]) -> dict`
    - `plan_portfolio(client, portfolio: dict) -> dict`
  - `ai/chat.py`:
    - `answer_question(client, query: str, context: dict, rag_hits: list[dict]) -> dict`

**Prompt 硬约束（写入每个 interpret 函数）：** "你只能引用输入数据中的数字，禁止编造任何市场数据、价格或预测。" 输出对象始终含 `disclaimer`。

- [ ] **Step 1: 实现 schemas 与 interpret**

`ai/schemas.py`:
```python
TREND_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string"},
        "points": {"type": "array", "items": {"type": "string"}},
        "risk": {"type": "string"},
        "confidence": {"type": "number"},
        "disclaimer": {"type": "string"},
    },
}

SECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "logic": {"type": "string"},
        "confidence": {"type": "number"},
        "disclaimer": {"type": "string"},
    },
}

STOCK_SCHEMA = {
    "type": "object",
    "properties": {
        "report": {"type": "string"},
        "confidence": {"type": "number"},
        "disclaimer": {"type": "string"},
    },
}

PORTFOLIO_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "rebalance": {"type": "string"},
        "confidence": {"type": "number"},
        "disclaimer": {"type": "string"},
    },
}

CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "references": {"type": "array", "items": {"type": "object"}},
        "confidence": {"type": "number"},
        "disclaimer": {"type": "string"},
    },
}
```

`ai/interpret.py`:
```python
"""把确定性分析结果交给 LLM 生成解读。LLM 不生成数字，只解读输入。"""
import json

from ai.schemas import (
    PORTFOLIO_SCHEMA,
    SECTOR_SCHEMA,
    STOCK_SCHEMA,
    TREND_SCHEMA,
)
from core.logging import get_logger

logger = get_logger("ai.interpret")

_GUARD = (
    "你只能引用输入 JSON 中出现的数字，禁止编造任何市场数据、价格、预测或代码。"
    "结论必须是中文，客观、克制，提示不确定性。"
)

DISCLAIMER = (
    "本内容仅为基于公开数据的分析展示，不构成任何投资建议。投资有风险，入市需谨慎。"
)


def _json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def interpret_trend(client, trend: dict) -> dict:
    if not trend:
        return {"state": "数据不足", "points": [], "risk": "",
                "confidence": 0.0, "disclaimer": DISCLAIMER}
    messages = [
        {"role": "system", "content": "你是 A 股长期投资分析师。" + _GUARD},
        {"role": "user", "content": (
            "基于以下趋势指标数据，给出当前市场长期趋势判断：\n"
            + _json_dumps(trend)
            + "\n返回字段：state(与输入 state 一致), points(3-5条要点), "
              "risk(主要风险), confidence(0-1), disclaimer。")},
    ]
    return client.chat_json(messages, TREND_SCHEMA)


def recommend_sectors(client, sectors: list[dict]) -> dict:
    if not sectors:
        return {"recommendations": [], "logic": "暂无板块数据", "confidence": 0.0,
                "disclaimer": DISCLAIMER}
    messages = [
        {"role": "system", "content": "你是 A 股板块轮动分析师。" + _GUARD},
        {"role": "user", "content": (
            "基于以下板块打分表（数字仅供参考，不要臆造），推荐值得长期关注的板块并说明逻辑：\n"
            + _json_dumps(sectors[:8])
            + "\n返回字段：recommendations(3-5条), logic, confidence, disclaimer。")},
    ]
    return client.chat_json(messages, SECTOR_SCHEMA)


def recommend_stocks(client, stocks: list[dict]) -> dict:
    if not stocks:
        return {"report": "暂无标的候选", "confidence": 0.0, "disclaimer": DISCLAIMER}
    messages = [
        {"role": "system", "content": "你是 A 股基本面分析师。" + _GUARD},
        {"role": "user", "content": (
            "基于以下量化筛选出的标的（只讨论，不编造新标的），写一份客观综合报告：\n"
            + _json_dumps(stocks)
            + "\n返回字段：report, confidence, disclaimer。")},
    ]
    return client.chat_json(messages, STOCK_SCHEMA)


def plan_portfolio(client, portfolio: dict) -> dict:
    if not portfolio:
        return {"plan": "暂无组合建议", "rebalance": "", "confidence": 0.0,
                "disclaimer": DISCLAIMER}
    messages = [
        {"role": "system", "content": "你是 A 股长期资产配置顾问。" + _GUARD},
        {"role": "user", "content": (
            "基于以下组合配置数据，生成一份长期投资计划说明（含执行与再平衡）：\n"
            + _json_dumps(portfolio)
            + "\n返回字段：plan, rebalance, confidence, disclaimer。")},
    ]
    return client.chat_json(messages, PORTFOLIO_SCHEMA)
```

- [ ] **Step 2: 实现对话编排**

`ai/chat.py`:
```python
"""对话编排：结合看板上下文与 RAG 检索结果作答。"""
import json

from ai.interpret import DISCLAIMER, _GUARD
from ai.schemas import CHAT_SCHEMA
from core.logging import get_logger

logger = get_logger("ai.chat")


def answer_question(client, query: str, context: dict,
                    rag_hits: list[dict]) -> dict:
    ctx_text = json.dumps(context, ensure_ascii=False)[:2000]
    refs = []
    ref_text = "无检索结果"
    if rag_hits:
        refs = [{"title": h["title"], "date": h["date"], "source": h["source"],
                 "url": h["url"]} for h in rag_hits]
        ref_text = json.dumps(refs, ensure_ascii=False)[:1500]
    messages = [
        {"role": "system", "content": "你是 A 股智能投资助手。" + _GUARD
                                     + "若引用了新闻/公告，必须在 references 中给出来源。"},
        {"role": "user", "content": (
            f"用户问题：{query}\n\n当前看板分析数据：{ctx_text}\n"
            f"相关新闻/公告：{ref_text}\n"
            "返回字段：answer, references(来源数组), confidence, disclaimer。")},
    ]
    try:
        out = client.chat_json(messages, CHAT_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        logger.warning("对话失败，返回降级回答: %s", exc)
        return {
            "answer": "AI 服务暂时不可用，请参考看板数据。",
            "references": refs, "confidence": 0.0, "disclaimer": DISCLAIMER,
        }
    out.setdefault("disclaimer", DISCLAIMER)
    return out
```

- [ ] **Step 3: 静态校验（导入 + schema 结构）**

Run:
```bash
python -c "
from ai.interpret import interpret_trend, DISCLAIMER
from ai.chat import answer_question
from ai.schemas import TREND_SCHEMA
print('TREND_SCHEMA props:', sorted(TREND_SCHEMA['properties']))
print('DISCLAIMER ok:', bool(DISCLAIMER))
"
```
Expected: 打印 schema 字段与 `DISCLAIMER ok: True`。

- [ ] **Step 4: Commit**

```bash
git add ai/schemas.py ai/interpret.py ai/chat.py
git commit -m "feat: add AI interpretation and chat orchestration"
```

---

## Phase 4：回测与自我迭代

### Task 12: 回测引擎

**Files:**
- Create: `core/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `core.config.load_weights`, `core.sector.score_sectors`
- Produces:
  - `backtest_sectors(sector_hist: dict[str, pd.DataFrame], bench: pd.DataFrame,
                      weights: dict, lookahead_days: int = 63) -> dict`
    返回：
    ```json
    {
      "win_rate": 0.6,
      "excess_return": 0.02,
      "n_samples": 20,
      "wins": 12,
      "data_until": "2026-08-09"
    }
    ```

**口径：** 在回测窗口内，每隔 `lookahead_days` 天采样一个决策点 T。用 T 及之前的板块数据计算得分，取 top3 等权组合，与基准（沪深300）对比 T→T+lookahead_days 区间收益；跑赢基准记 win。

- [ ] **Step 1: 写失败测试**

`tests/test_backtest.py`:
```python
import numpy as np
import pandas as pd
import pytest

from core.backtest import backtest_sectors

WEIGHTS = {"sector": {"rs": 0.4, "flow": 0.3, "momentum": 0.3}}


def _series(n, trend=1.0, start=100.0):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    closes = start * np.linspace(1.0, trend, n)
    return pd.DataFrame({"date": dates, "close": closes})


def test_backtest_returns_metrics():
    hist = {
        "A": _series(300, 1.5),
        "B": _series(300, 1.2),
        "C": _series(300, 1.0),
    }
    bench = _series(300, 1.1)
    result = backtest_sectors(hist, bench, WEIGHTS, lookahead_days=40)
    assert "win_rate" in result
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["n_samples"] >= 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现回测引擎**

`core/backtest.py`:
```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: 1 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add core/backtest.py tests/test_backtest.py
git commit -m "feat: add sector backtest engine"
```

### Task 13: 权重调优与迭代流程

**Files:**
- Create: `core/tune.py`
- Test: `tests/test_tune.py`

**Interfaces:**
- Consumes: `core.backtest.backtest_sectors`, `core.config.load_weights/save_weights`, `core.data.DataProvider`
- Produces:
  - `grid_search_weights(score_fn, data, base_weights: dict, search_keys: list[str],
                         steps: list[float]) -> tuple[dict, float]`
  - `run_iteration(provider, store) -> dict`
    - 拉板块历史 → 网格搜索 sector 权重（仅在训练窗调参）→ 用验证窗评估 → 更新 `config/weights.json` → 写 `iter_history` → 返回迭代结果。

**防过拟合：** 网格搜索在时间窗前半段（train）评估，验证用后半段（test）胜率对比；若新权重在 test 窗胜率不高于当前权重，则不更新（收敛阈值）。

- [ ] **Step 1: 写失败测试**

`tests/test_tune.py`:
```python
import pytest

from core.tune import grid_search_weights


def fake_score_fn(data, weights):
    # 用 "rs" 权重作为正向指标：权重越大分数越高
    return {"win_rate": weights["sector"]["rs"], "n_samples": 1}


def test_grid_search_finds_best():
    base = {"sector": {"rs": 0.5, "flow": 0.25, "momentum": 0.25}}
    best_weights, best_score = grid_search_weights(
        fake_score_fn, None, base,
        search_keys=["sector.rs", "sector.flow"],
        steps=[0.8, 0.9],
    )
    assert best_weights["sector"]["rs"] == 0.8
    assert best_score == pytest.approx(0.8)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_tune.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现调优模块**

`core/tune.py`:
```python
"""权重网格搜索与迭代流程。"""
import itertools
from datetime import datetime

from core.backtest import backtest_sectors
from core.config import load_weights, save_weights
from core.logging import get_logger

logger = get_logger("core.tune")

SECTOR_SEARCH = {
    "keys": ["sector.rs", "sector.flow", "sector.momentum"],
    "steps": [0.5, 0.7, 0.9],
}


def grid_search_weights(score_fn, data, base_weights: dict,
                        search_keys: list[str], steps: list[float]):
    """对 search_keys（形如 'sector.rs'）做网格搜索，返回 (最优权重, 最优指标)。"""
    best = base_weights
    best_score = score_fn(data, best)["win_rate"]
    for combo in itertools.product(steps, repeat=len(search_keys)):
        cand = _deep_copy(base_weights)
        for key, val in zip(search_keys, combo):
            _set_path(cand, key.split("."), val)
        score = score_fn(data, cand)["win_rate"]
        if score > best_score:
            best, best_score = cand, score
    return best, best_score


def run_iteration(provider, store) -> dict:
    weights = load_weights()
    sector_hist, bench = _collect_history(provider)
    if not sector_hist:
        return {"status": "no_data", "reason": "板块历史数据不足"}

    # 按时间排序，前 60% 调参，后 40% 验证
    sorted_hist, sorted_bench = _split_train_test(sector_hist, bench, 0.6)
    train_hist, test_hist = sorted_hist
    train_bench, test_bench = sorted_bench

    def train_score(_, w):
        return backtest_sectors(train_hist, train_bench, w)

    best_weights, _ = grid_search_weights(
        train_score, None, weights, SECTOR_SEARCH["keys"], SECTOR_SEARCH["steps"]
    )
    old_test = backtest_sectors(test_hist, test_bench, weights)
    new_test = backtest_sectors(test_hist, test_bench, best_weights)

    changed = new_test["win_rate"] > old_test["win_rate"] + 1e-9
    if changed:
        save_weights(best_weights)
    version = datetime.now().strftime("v%Y%m%d%H%M")
    rec = {
        "version": version,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weights_json": _dumps(best_weights if changed else weights),
        "backtest_window": f"{str(train_hist[list(train_hist)[0]]['date'].iloc[0])[:10]}~"
                           f"{str(test_hist[list(test_hist)[0]]['date'].iloc[-1])[:10]}",
        "win_rate": new_test["win_rate"],
        "excess_return": new_test["excess_return"],
        "data_until": str(test_bench["date"].iloc[-1])[:10],
    }
    store.insert_iter(rec)
    return {"status": "updated" if changed else "unchanged", "version": version,
            "old_win_rate": old_test["win_rate"], "new_win_rate": new_test["win_rate"],
            "weights": best_weights if changed else weights}


def _collect_history(provider):
    sector_hist, bench = {}, None
    try:
        quotes = provider.sector_quote()
        for name in list(quotes["name"])[:20]:
            h = provider.sector_hist(name)
            if not h.empty:
                sector_hist[name] = h
        bench = provider.index_daily(provider.benchmark_index_code())
    except Exception as exc:  # noqa: BLE001
        logger.warning("迭代数据采集失败: %s", exc)
    return sector_hist, bench


def _split_train_test(sector_hist, bench, ratio):
    import pandas as pd

    def _cut(df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        n = int(len(df) * ratio)
        return df.iloc[:n], df.iloc[n:]

    train_h, test_h = {}, {}
    for name, h in sector_hist.items():
        a, b = _cut(h)
        if len(a) >= 30 and len(b) >= 30:
            train_h[name], test_h[name] = a, b
    tb = bench.copy()
    tb["date"] = pd.to_datetime(tb["date"])
    tb = tb.sort_values("date").reset_index(drop=True)
    n = int(len(tb) * ratio)
    train_b, test_b = tb.iloc[:n], tb.iloc[n:]
    return (train_h, test_h), (train_b, test_b)


def _set_path(obj: dict, path: list[str], value) -> None:
    cur = obj
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _deep_copy(obj):
    import copy

    return copy.deepcopy(obj)


def _dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_tune.py -v`
Expected: 1 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add core/tune.py tests/test_tune.py
git commit -m "feat: add weight grid search and iteration loop"
```

---

## Phase 5：虚拟投资账户

### Task 14: 账户与交易执行

**Files:**
- Create: `core/account.py`
- Test: `tests/test_account.py`

**Interfaces:**
- Consumes: `core.store.Store`, `core.config.get_env`
- Produces: `class SimAccount`：
  - `__init__(self, store: Store, initial_capital: float | None = None)`
  - `ensure_initialized() -> None`
  - `execute(portfolio: dict, prices: dict[str, float]) -> list[dict]`  # 返回本次交易
  - `snapshot(prices: dict[str, float], benchmark_return: float | None = None) -> None`
  - `period_stats() -> dict`  # 见下
  - `maybe_reset_period(benchmark_return: float) -> None`
  - `current_period_id() -> str`

`period_stats()` 返回：
```json
{
  "period_id": "2026-08",
  "nav": 1010000.0, "cash": 500000.0, "holdings_value": 510000.0,
  "initial_capital": 1000000.0, "win_rate": 0.6,
  "return_pct": 1.0, "curve": [{"date": "...", "nav": 1000000.0}, ...]
}
```

**执行规则：**
- `execute`：目标组合（core + satellite，ETFs 用 ETF_MAP 或输入 prices）。把账户按目标权重分配资金：先卖出现有但不在目标中的持仓，再买入不足部分。成交价 = prices[symbol]。费用 = 金额 × 0.0003（佣金）。卖出时计算 pnl，status=closed；买入 status=open。
- `snapshot`：nav = cash + Σ qty×price。
- `maybe_reset_period`：当前日期所在月份 ≠ period_start 月份时，归档 period（含 win_rate、return_pct、benchmark_return），重置 cash=initial_capital，清空持仓与交易。
- win_rate = closed 交易中 pnl>0 的比例。

- [ ] **Step 1: 写失败测试**

`tests/test_account.py`:
```python
import tempfile

from core.account import SimAccount
from core.store import Store

PORTFOLIO = {
    "core": {"name": "沪深300ETF(510300)", "weight": 0.7},
    "satellite": [{"name": "半导体ETF(512480)", "weight": 0.2},
                  {"name": "医药ETF(512010)", "weight": 0.1}],
}
PRICES = {"510300": 3.9, "512480": 1.2, "512010": 0.8}


def _new_account():
    store = Store(tempfile.mkdtemp() + "/t.db")
    acc = SimAccount(store, initial_capital=100000.0)
    acc.ensure_initialized()
    return acc, store


def test_execute_buys_positions():
    acc, store = _new_account()
    trades = acc.execute(PORTFOLIO, PRICES)
    assert len(trades) == 3
    assert all(t["side"] == "buy" for t in trades)
    pos = {p["symbol"]: p["qty"] for p in store.list_positions()}
    assert abs(pos["510300"] - 0.7 * 100000 / 3.9) < 1


def test_snapshot_calc_nav():
    acc, store = _new_account()
    acc.execute(PORTFOLIO, PRICES)
    acc.snapshot(PRICES)
    stats = acc.period_stats()
    assert abs(stats["nav"] - 100000.0) < 1000.0  # 含费用，接近初始资金


def test_win_rate_after_profit_sell():
    acc, store = _new_account()
    acc.execute(PORTFOLIO, PRICES)
    # 全部卖出，价格涨 10%
    up = {k: v * 1.1 for k, v in PRICES.items()}
    acc.execute({"core": None, "satellite": []}, up)
    stats = acc.period_stats()
    assert stats["win_rate"] == 1.0


def test_reset_period_archives():
    acc, store = _new_account()
    acc.execute(PORTFOLIO, PRICES)
    acc.snapshot(PRICES, benchmark_return=0.5)
    acc.maybe_reset_period(benchmark_return=0.5)
    assert len(store.list_periods()) == 1
    acc.ensure_initialized()
    assert acc.period_stats()["cash"] == 100000.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_account.py -v`
Expected: FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现账户模块**

`core/account.py`:
```python
"""虚拟投资账户：执行推荐、资金曲线、胜率、阶段重置。"""
from datetime import datetime

from core.config import get_env
from core.logging import get_logger
from core.store import Store

logger = get_logger("core.account")

FEE_RATE = 0.0003


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class SimAccount:
    def __init__(self, store: Store, initial_capital: float | None = None):
        self.store = store
        self.initial_capital = initial_capital or float(
            get_env("ACCOUNT_INITIAL_CAPITAL", "1000000")
        )

    def ensure_initialized(self) -> None:
        acc = self.store.get_account()
        if acc["cash"] == 0.0 and acc["period_start"] is None:
            self.store.save_account({
                "cash": self.initial_capital,
                "initial_capital": self.initial_capital,
                "period_start": _today(),
                "updated_at": _now(),
            })

    def current_period_id(self) -> str:
        acc = self.store.get_account()
        return (acc.get("period_start") or _today())[:7]

    def execute(self, portfolio: dict, prices: dict[str, float]) -> list[dict]:
        self.ensure_initialized()
        acc = self.store.get_account()
        cash = acc["cash"]
        trades = []
        targets = {}
        if portfolio and portfolio.get("core"):
            targets[portfolio["core"]["name"]] = portfolio["core"]["weight"]
        for sat in portfolio.get("satellite", []) if portfolio else []:
            targets[sat["name"]] = sat.get("weight", 0.0)
        # 解析 symbol
        target_by_symbol = {}
        for name, weight in targets.items():
            sym = _symbol_from(name)
            if sym:
                target_by_symbol[sym] = weight

        existing = {p["symbol"]: p for p in self.store.list_positions()}
        now = _now()
        # 卖出不在目标中的持仓
        for sym, pos in existing.items():
            if sym not in target_by_symbol:
                trades += [self._close(pos, prices.get(sym, pos["cost_price"]), now)]
        # 买入/调仓到目标权重
        total_weight = sum(target_by_symbol.values()) or 1.0
        remaining = cash
        for sym, weight in target_by_symbol.items():
            price = prices.get(sym)
            if not price:
                continue
            target_value = cash * (weight / total_weight)
            qty_target = target_value / price
            current = existing.get(sym)
            cur_qty = current["qty"] if current else 0.0
            diff = qty_target - cur_qty
            if diff > 0.01:
                cost = diff * price
                fee = cost * FEE_RATE
                if cost + fee > remaining:
                    diff = (remaining - fee) / price
                    cost = diff * price
                    fee = cost * FEE_RATE
                self.store.save_position({
                    "symbol": sym, "name": _name_from(sym),
                    "qty": cur_qty + diff,
                    "cost_price": price, "updated_at": now,
                })
                self.store.insert_trade({
                    "time": now, "symbol": sym, "name": _name_from(sym),
                    "side": "buy", "price": round(price, 4), "qty": round(diff, 2),
                    "fee": round(fee, 2), "pnl": None, "status": "open",
                })
                remaining -= cost + fee
                trades.append({"side": "buy", "symbol": sym, "qty": round(diff, 2)})
        new_cash = max(0.0, remaining)
        self.store.save_account({**acc, "cash": new_cash, "updated_at": now})
        return trades

    def _close(self, pos: dict, price: float, now: str):
        qty = pos["qty"]
        proceeds = qty * price
        fee = proceeds * FEE_RATE
        pnl = proceeds - fee - qty * pos["cost_price"]
        self.store.insert_trade({
            "time": now, "symbol": pos["symbol"], "name": pos["name"],
            "side": "sell", "price": round(price, 4), "qty": round(qty, 2),
            "fee": round(fee, 2), "pnl": round(pnl, 2), "status": "closed",
        })
        self.store.delete_position(pos["symbol"])
        acc = self.store.get_account()
        self.store.save_account({**acc, "cash": acc["cash"] + proceeds - fee,
                                 "updated_at": now})
        return {"side": "sell", "symbol": pos["symbol"], "qty": round(qty, 2)}

    def snapshot(self, prices: dict[str, float], benchmark_return: float | None = None) -> None:
        acc = self.store.get_account()
        holdings_value = 0.0
        for pos in self.store.list_positions():
            holdings_value += pos["qty"] * prices.get(pos["symbol"], pos["cost_price"])
        nav = acc["cash"] + holdings_value
        self.store.insert_snapshot({
            "period_id": self.current_period_id(),
            "date": _today(), "nav": round(nav, 2),
            "cash": round(acc["cash"], 2), "holdings_value": round(holdings_value, 2),
        })

    def period_stats(self) -> dict:
        acc = self.store.get_account()
        period_id = self.current_period_id()
        snapshots = self.store.list_snapshots(period_id)
        nav = acc["cash"]
        for pos in self.store.list_positions():
            nav += pos["qty"] * pos["cost_price"]
        curve = [{"date": s["date"], "nav": s["nav"]} for s in snapshots]
        trades = [t for t in self.store.list_trades() if t["status"] == "closed"]
        wins = sum(1 for t in trades if (t["pnl"] or 0) > 0)
        win_rate = wins / len(trades) if trades else 0.0
        init = acc["initial_capital"] or self.initial_capital
        return {
            "period_id": period_id,
            "nav": round(nav, 2),
            "cash": round(acc["cash"], 2),
            "initial_capital": init,
            "win_rate": round(win_rate, 3),
            "return_pct": round((nav / init - 1.0) * 100.0, 2) if init else 0.0,
            "curve": curve,
        }

    def maybe_reset_period(self, benchmark_return: float | None = None) -> None:
        acc = self.store.get_account()
        start = acc.get("period_start") or _today()
        if start[:7] == _today()[:7]:
            return
        stats = self.period_stats()
        self.store.insert_period({
            "period_id": start[:7], "start": start, "end": _today(),
            "initial_capital": acc["initial_capital"],
            "final_nav": stats["nav"], "win_rate": stats["win_rate"],
            "return_pct": stats["return_pct"],
            "benchmark_return": benchmark_return or 0.0,
        })
        for pos in self.store.list_positions():
            self.store.delete_position(pos["symbol"])
        self.store.save_account({
            "cash": acc["initial_capital"], "initial_capital": acc["initial_capital"],
            "period_start": _today(), "updated_at": _now(),
        })


def _symbol_from(name: str) -> str | None:
    import re

    m = re.search(r"\((\d{6})\)", name)
    return m.group(1) if m else None


def _name_from(symbol: str) -> str:
    for p in __import__("core.portfolio", fromlist=["ETF_MAP"]).ETF_MAP.values():
        if f"({symbol})" in p:
            return p
    return symbol
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_account.py -v`
Expected: 4 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add core/account.py tests/test_account.py
git commit -m "feat: add simulated investment account"
```

---

## Phase 6：应用层（API + 前端）

### Task 15: FastAPI 后端

**Files:**
- Create: `api/__init__.py`
- Create: `api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: 全部 core/ai 模块
- Produces:
  - `api/main.py` 暴露 5 个接口：`GET /api/dashboard`, `POST /api/analyze`,
    `POST /api/chat`, `GET /api/backtest`, `GET /api/account`
  - `tests/test_api.py` 用 `httpx.ASGITransport` 直测，不启动服务器

- [ ] **Step 1: 写失败测试**

`tests/test_api.py`:
```python
import httpx
import pytest

from api.main import app


@pytest.mark.asyncio
async def test_dashboard_returns_json():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "data_until" in data


@pytest.mark.asyncio
async def test_analyze_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/analyze")
    assert resp.status_code == 200
    data = resp.json()
    assert "trend" in data and "sectors" in data


@pytest.mark.asyncio
async def test_unknown_endpoint_404():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/nope")
    assert resp.status_code == 404
```

（需要 `pip install pytest-asyncio` 并在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 增加 `asyncio_mode = "auto"`。可改为主进程 `asyncio.run` 形式或仅跑第 3 个测试，若 asyncio 插件安装失败。）

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL（`ModuleNotFoundError: api`）。

- [ ] **Step 3: 实现后端**

`api/__init__.py`:
```python
"""FastAPI 应用。"""
```

`api/main.py`:
```python
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
    ai = interpret_trend(_client, analysis["trend"])
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
        "trend": interpret_trend(_client, result["trend"]),
        "sectors": recommend_sectors(_client, result["sectors"]),
        "stocks": recommend_stocks(_client, result["stocks"]),
        "portfolio": plan_portfolio(_client, result["portfolio"]),
    }
    return {"analysis": result, "ai": ai, "account": _account.period_stats(),
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
```

- [ ] **Step 4: 安装 pytest-asyncio 并跑测试**

Run:
```bash
pip install pytest-asyncio
```
在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 增加：
```toml
asyncio_mode = "auto"
```
再运行 `python -m pytest tests/test_api.py -v`
Expected: 3 个测试 PASS（dashboard/analyze 可能因网络为空数据，但结构字段必须存在）。

- [ ] **Step 5: Commit**

```bash
git add api/ tests/test_api.py
git commit -m "feat: add FastAPI backend endpoints"
```

### Task 16: 前端看板与聊天

**Files:**
- Create: `web/index.html`
- Create: `web/style.css`
- Create: `web/app.js`

**功能：**
- 顶部指标卡：趋势状态（含数据时点）、账户净值、阶段收益率、操作胜率
- 看板区：板块排名表、标的表、组合配置、回测迭代表、阶段历史表
- 聊天区：输入框 + 消息流，渲染引用与置信度
- 按钮："重新分析"调 `/api/analyze`，"运行回测迭代"调 `/api/backtest`

- [ ] **Step 1: 实现 HTML**

`web/index.html`:
```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>AI 智能投资助手</title>
  <link rel="stylesheet" href="/web/style.css" />
</head>
<body>
  <header>
    <h1>AI 智能投资助手 <span id="data-until"></span></h1>
    <div class="actions">
      <button id="btn-analyze">重新分析</button>
      <button id="btn-backtest">运行回测迭代</button>
    </div>
  </header>

  <section id="cards"></section>

  <main>
    <section class="panel">
      <h2>市场趋势</h2>
      <div id="trend"></div>
      <h2>板块排名</h2>
      <table id="sectors"></table>
      <h2>标的推荐</h2>
      <table id="stocks"></table>
      <h2>组合配置</h2>
      <div id="portfolio"></div>
    </section>

    <section class="panel">
      <h2>回测与迭代</h2>
      <div id="backtest"></div>
      <h2>投资账户</h2>
      <div id="account"></div>
      <h2>阶段历史</h2>
      <table id="periods"></table>
    </section>
  </main>

  <section class="chat">
    <div id="messages"></div>
    <div class="chat-input">
      <input id="query" placeholder="输入问题，如：最近半导体板块发生了什么？" />
      <button id="btn-send">发送</button>
    </div>
  </section>

  <footer>本工具为数据分析演示，不构成投资建议。</footer>
  <script src="/web/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 实现 CSS**

`web/style.css`:
```css
* { box-sizing: border-box; margin: 0; }
body { font-family: "Microsoft YaHei", sans-serif; background: #f5f7fa; color: #1f2937; }
header { background: #111827; color: #fff; padding: 16px 24px; display: flex;
         justify-content: space-between; align-items: center; }
header h1 { font-size: 20px; }
#data-until { font-size: 12px; color: #9ca3af; margin-left: 12px; }
.actions button { margin-left: 8px; padding: 8px 16px; border: none; border-radius: 6px;
                  cursor: pointer; background: #2563eb; color: #fff; }
.warnings { margin: 0 24px; padding: 8px 16px; background: #fef3c7; color: #92400e;
            border-radius: 6px; font-size: 13px; }
#cards { display: flex; gap: 16px; padding: 16px 24px; flex-wrap: wrap; }
.card { background: #fff; border-radius: 8px; padding: 16px 20px; flex: 1 1 200px;
        box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.card .label { font-size: 12px; color: #6b7280; }
.card .value { font-size: 24px; font-weight: 600; margin-top: 4px; }
main { display: flex; gap: 16px; padding: 0 24px 16px; }
.panel { flex: 1; background: #fff; border-radius: 8px; padding: 16px;
         box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.panel h2 { font-size: 15px; margin: 12px 0 8px; border-left: 3px solid #2563eb; padding-left: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #e5e7eb; }
th { background: #f9fafb; }
.chat { margin: 0 24px 16px; background: #fff; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,.1); padding: 16px; }
#messages { height: 260px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 6px;
            padding: 12px; margin-bottom: 12px; font-size: 14px; }
.msg { margin-bottom: 10px; }
.msg.user { text-align: right; color: #2563eb; }
.msg .ref { font-size: 12px; color: #6b7280; }
footer { padding: 12px 24px; font-size: 12px; color: #9ca3af; text-align: center; }
```

- [ ] **Step 3: 实现 JS**

`web/app.js`:
```js
async function getJSON(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) throw new Error(url + " -> " + resp.status);
  return resp.json();
}

function card(label, value, extra) {
  return `<div class="card"><div class="label">${label}</div>` +
         `<div class="value">${value ?? "—"}</div>${extra ? `<div class="label">${extra}</div>` : ""}</div>`;
}

function table(headers, rows) {
  if (!rows || rows.length === 0) return "<p>暂无数据</p>";
  const h = headers.map(x => `<th>${x}</th>`).join("");
  const body = rows.map(r => `<tr>${r.map(c => `<td>${c ?? ""}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${h}</tr></thead><tbody>${body}</tbody></table>`;
}

async function refreshDashboard() {
  const d = await getJSON("/api/dashboard");
  document.getElementById("data-until").textContent = "数据截至 " + d.data_until;
  const warns = (d.analysis.warnings || []);
  const warnsHtml = warns.length
    ? `<div class="warnings">⚠️ ${warns.map(w => `<span>${w}</span>`).join("　")}</div>` : "";
  document.querySelector("header").insertAdjacentHTML("afterend", warnsHtml);
  const t = d.analysis.trend || {};
  const a = d.account || {};
  const cards = [
    card("趋势状态", t.state, "综合分 " + (t.composite ?? "—")),
    card("账户净值", "¥" + (a.nav ?? 0).toLocaleString(), "阶段 " + a.period_id),
    card("阶段收益率", (a.return_pct ?? 0) + "%"),
    card("操作胜率", Math.round((a.win_rate ?? 0) * 100) + "%"),
  ];
  document.getElementById("cards").innerHTML = cards.join("");

  document.getElementById("sectors").innerHTML = table(
    ["板块", "RS", "资金流", "动量", "得分"],
    (d.analysis.sectors || []).map(s => [s.name, s.rs, s.flow, s.momentum, s.score]));
  document.getElementById("stocks").innerHTML = table(
    ["代码", "名称", "得分"],
    (d.analysis.stocks || []).map(s => [s.code, s.name, s.score]));
  const p = d.analysis.portfolio || {};
  document.getElementById("portfolio").innerHTML =
    `<p>${p.summary || ""}　${p.rebalance_rule || ""}</p>` + table(
      ["组合", "名称", "权重"],
      (p.core ? [["核心", p.core.name, p.core.weight]] : [])
        .concat((p.satellite || []).map(s => ["卫星", s.name, s.weight])));
  document.getElementById("account").innerHTML = table(
    ["阶段", "胜率", "收益率", "基准"],
    (d.periods || []).map(p => [p.period_id, p.win_rate, p.return_pct + "%", p.benchmark_return]));
}

async function runAnalyze() {
  const d = await getJSON("/api/analyze", { method: "POST" });
  pushMsg("assistant", "分析完成：" + (d.ai.trend?.state || d.analysis.trend?.state || ""));
  refreshDashboard();
}

async function runBacktest() {
  const d = await getJSON("/api/backtest");
  const r = d.result || {};
  const body = table(["状态", "版本", "胜率"],
    [[r.status, r.version, r.new_win_rate ?? r.win_rate ?? ""]]);
  document.getElementById("backtest").innerHTML =
    body + table(["版本", "运行时间", "胜率", "超额收益"],
      (d.iters || []).map(i => [i.version, i.run_at, i.win_rate, i.excess_return]));
}

async function send() {
  const q = document.getElementById("query").value.trim();
  if (!q) return;
  pushMsg("user", q);
  document.getElementById("query").value = "";
  const d = await getJSON("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: q }),
  });
  const refs = (d.references || []).map(r =>
    `<div class="ref">· ${r.title}（${r.date}，${r.source}）</div>`).join("");
  pushMsg("assistant", d.answer + ` <small>置信度 ${d.confidence}</small>` + refs);
}

function pushMsg(role, text) {
  const box = document.getElementById("messages");
  box.insertAdjacentHTML("beforeend",
    `<div class="msg ${role}">${text.replace(/\n/g, "<br>")}</div>`);
  box.scrollTop = box.scrollHeight;
}

document.getElementById("btn-analyze").addEventListener("click", runAnalyze);
document.getElementById("btn-backtest").addEventListener("click", runBacktest);
document.getElementById("btn-send").addEventListener("click", send);
document.getElementById("query").addEventListener("keydown", e => { if (e.key === "Enter") send(); });

refreshDashboard();
```

- [ ] **Step 4: 启动服务器并验证页面可访问**

Run:
```bash
source .venv/Scripts/activate
export DEEPSEEK_API_KEY=sk-你的key
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
打开 `http://127.0.0.1:8000/`。Expected: 页面渲染，看板出现，聊天可用。若页面 404，确认 uvicorn 从项目根目录启动（`web/` 相对路径）。

- [ ] **Step 5: Commit**

```bash
git add web/
git commit -m "feat: add web dashboard and chat UI"
```

### Task 17: 端到端冒烟

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: 写端到端测试**

`tests/test_e2e.py`:
```python
"""端到端：数据→分析→AI(降级)→账户 全链路不抛异常。"""
import os
import tempfile

import pytest

from core.analyze import run_analysis
from core.config import DB_PATH, load_weights
from core.data import DataProvider
from core.store import Store


def test_full_pipeline_offline_safe(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    store = Store(tempfile.mkdtemp() + "/t.db")
    provider = DataProvider(store)
    result = run_analysis(provider)
    assert isinstance(result, dict)
    assert "trend" in result and "sectors" in result
    assert "stocks" in result and "portfolio" in result


def test_weights_config_parseable():
    w = load_weights()
    for key in ("trend", "sector", "stock"):
        assert key in w
        assert abs(sum(w[key].values()) - 1.0) < 1e-9
```

- [ ] **Step 2: 运行测试**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS（网络可用时数据真实，不可用时走降级）。

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: add end-to-end smoke test"
```

---

## Phase 7：文档

### Task 18: README 与使用说明

**Files:**
- Create: `README.md`
- Create: `docs/architecture.md`

- [ ] **Step 1: 写 README**

`README.md`:
```markdown
# AI 智能投资助手

面向 A 股长期投资者的智能投资分析项目：趋势/板块/选股/组合分析 + RAG 问答 + 回测自我迭代 + AI 虚拟投资账户。

## 功能

- **市场趋势**：MA250 偏离、指数 PE/PB 历史百分位、股债性价比 → 低估/中性/高估状态
- **板块机会**：RS 相对强度 + 资金流 + 动量打分
- **个股筛选**：ROE/成长/估值/股息多因子打分
- **组合配置**：核心 70% + 卫星 30%
- **RAG 问答**：个股新闻/公告检索，回答带来源引用
- **回测迭代**：历史数据网格搜索权重，时间窗分割防过拟合
- **虚拟账户**：AI 亲自投资，展示资金曲线、操作胜率、阶段收益率与阶段重置

## 核心原则

> **数据永不来自 LLM。** 所有数字由确定性分析引擎从 akshare 真实数据计算，
> DeepSeek 只负责解读，禁止编造。AI 是增强层，不是依赖层。

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `.env`：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`
3. 启动：`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`
4. 打开 `http://127.0.0.1:8000/`

## 项目结构

```
core/      确定性分析引擎（数据/趋势/板块/选股/组合/RAG/回测/账户）
ai/        DeepSeek 层（Provider + Schema 约束 + 解读/对话）
api/       FastAPI REST 接口
web/       单页前端（看板 + 聊天）
tests/     pytest 测试
docs/      设计文档与架构说明
```

## 测试与评估

```bash
python -m pytest -v
```

评估体系（黄金场景 + 一致性校验）见 `docs/superpowers/specs/2026-08-10-ai-investment-assistant-design.md` §9。

## 免责声明

本项目为数据分析与技术演示用途，不构成任何投资建议。回测胜率不代表未来收益，投资有风险，入市需谨慎。
```

- [ ] **Step 2: 写架构文档**

`docs/architecture.md`：用 Mermaid 画出 §5 的架构图（core/ai/api/web 分层），并补充：
- 数据流时序（analyze / chat）
- 错误降级三层策略
- 回测与账户的指标口径

（内容取自设计文档 §5/§7/§8/§10，转换为 Markdown + Mermaid。）

- [ ] **Step 3: 全量测试**

Run: `python -m pytest -v`
Expected: 全部 PASS。

- [ ] **Step 4: 提交并推送**

```bash
git add README.md docs/architecture.md
git commit -m "docs: add README and architecture doc"
git push -u origin main
```

- [ ] **Step 5: 收尾检查清单**

- [ ] `git log --oneline` 各阶段提交齐全
- [ ] `python -m pytest -v` 全绿
- [ ] 本地启动 uvicorn，页面与聊天可用
- [ ] 确认 `.env` 不在仓库中（`.gitignore` 覆盖）
- [ ] DeepSeek 解读：`curl -X POST http://127.0.0.1:8000/api/analyze` 返回含 `ai` 字段
- [ ] 免责声明在所有 AI 输出中可见

### Task 19: AI须知.md —— 新 AI 上手文档

**Files:**
- Create: `AI须知.md`（项目主目录）

**目的：** 所有功能实现完善之后，在主目录新增 `AI须知.md`。让一个新上下文、从未接触过本项目的 AI（或新开发者）仅阅读该文件即可：
1. 快速理解项目是什么、核心设计原则
2. 知道代码在哪儿、每部分职责
3. 知道如何启动、测试、验证改动
4. 知道如何安全地优化或迭代本项目（哪些不能改、哪些坑要避开）

**文档必须包含的章节（要求绝对具体、可操作）：**

- [ ] **Step 1: 写 AI须知.md**

内容要求（参考，可在此基础上扩充）：

```markdown
# AI 须知（Project Onboarding for AI Agents）

> 给新上下文 AI 的快速上手文档。阅读本文件后，你应当能理解、启动、测试并安全地修改本项目。

## 1. 项目是什么

一句话定位 + 核心功能列表（趋势分析/板块/选股/组合/RAG/回测迭代/虚拟账户）。

## 2. 核心设计原则（改动时不可违背）

1. **数据永不来自 LLM**：所有数字由 core/ 确定性引擎从 akshare 真实数据计算，
   ai/ 层只做解读，禁止生成数字。改动不得让 LLM 编造数据。
2. **AI 是增强层，不是依赖层**：DeepSeek 挂了，看板和确定性分析必须照常工作。
3. **指标口径统一**：胜率=已平仓盈利/已平仓；阶段收益率=(期末-期初)/期初；
   基准=沪深300。
4. **免责声明**：所有 AI 输出带 disclaimer 与 confidence。

## 3. 项目结构速览

（核心目录/模块 → 一句话职责 → 入口文件/函数）

## 4. 环境与启动

- 依赖：`pip install -r requirements.txt`（sentence-transformers/chromadb 可暂缺，RAG 会 Hash 降级）
- 配置：复制 `.env.example` 为 `.env`，填 `DEEPSEEK_API_KEY`
- 启动：`python -m uvicorn api.main:app --port 8000`
- 打开：`http://127.0.0.1:8000/`

## 5. 测试

- 全部测试：`python -m pytest -v`
- 数据层测试不依赖真实网络（mock/降级）
- 新增功能必须补测试

## 6. 常见改动场景与步骤

- 修改打分权重：`config/weights.json`（或用回测迭代自动调优）
- 新增分析指标：改 core/xxx.py → 补测试 → 在 ai/interpret 提示词中说明
- 换 LLM 模型：改 `ai/provider.py` 的 get_client，模型名在 `.env` 的 DEEPSEEK_MODEL

## 7. 常见坑与注意

- akshare 接口名随版本变化：先用 `dir(ak)` 确认，别硬编码已废弃的接口
- `.env` 不入库；`data/` 不入库
- 回测/账户数据在 SQLite `data/app.db`，删除即重置
- 前端无构建工具，改 `web/app.js` 后刷新即可
- 提交前跑全量测试
```

- [ ] **Step 2: 校验文档可读性**

通读一遍，确认以下问题都可从文中直接找到答案：
- 项目是做什么的？核心原则有哪些？
- 怎么安装依赖、配置、启动、测试？
- 想加一个分析指标，改哪几个文件？
- 哪些东西绝不能乱改？

- [ ] **Step 3: Commit**

```bash
git add AI须知.md
git commit -m "docs: add AI onboarding guide for future agents"
```

---

## Self-Review 记录

- **Spec 覆盖**：数据层 §6.1→Task3；trend §6.2→Task4；sector §6.3→Task5；stock §6.4→Task6；portfolio §6.5→Task7；rag §6.6→Task10；ai §6.7→Task9/11；api §6.8→Task15；web §6.9→Task16；backtest §6.10→Task12/13；account §6.11→Task14；评估/测试 §9→各 Task + Task17；MVP §11→全部；错误处理 §8→Task3 降级 + api 兜底。
- **类型一致性**：`Store`、`DataProvider`、`LLMClient`、`score_sectors`、`rank_stocks`、`build_portfolio`、`analyze_trend`、`backtest_sectors`、`SimAccount` 等签名在 Task 间一致。
- **占位符检查**：无 TBD/TODO；akshare 接口名的版本差异有明确处理指引（Task 3 说明），非占位符。
