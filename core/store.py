import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class Store:
    """SQLite 统一存储：数据缓存、账户、交易、快照、阶段、迭代历史、新闻与向量块。"""

    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._mem_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            # SQLite 的内存库只存在于其连接生命周期内；每次新建连接都会得到一个
            # 全新空库，导致 schema 丢失。因此对 :memory: 保持单一持久连接。
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.row_factory = sqlite3.Row
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
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
