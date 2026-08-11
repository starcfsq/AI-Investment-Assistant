# AI 自动投资 + 一年模拟回测 + 收益曲线 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 系统内嵌定时自动投资虚拟账户；新增一年历史模拟引擎与收益曲线前端展示。

**Architecture:** 把 `core/analyze.py` 抽出纯函数 `analyze_at(snapshot, weights)`（不访问网络，实时与模拟共用）；新增 `core/simulation.py`（HistoryProvider 历史视图 + 月度调仓 + SimAccount 模拟 + 每日净值）；`api/main.py` 用 FastAPI lifespan 启动自动投资后台任务，新增 `GET /api/simulation`；前端 Canvas 绘制净值 vs 基准曲线与详细操作表。

**Tech Stack:** Python 3.12 / FastAPI / pandas / akshare>=1.18.84 / 原生 JS + Canvas（无构建工具）

## Global Constraints

- akshare 版本下限：`>=1.18.84`（`requirements.txt` 已锁定）
- 数据永不来自 LLM：所有数字由确定性引擎从历史/实时数据计算，`_GUARD` 不变
- 东财行情接口当前被限流，板块 K 线用同花顺 `stock_board_industry_index_ths`、ETF 历史用新浪 `fund_etf_hist_sina(symbol='sh'+code)`
- 虚拟账户口径不变：操作胜率 = 已平仓 `pnl>0` ÷ 已平仓总笔数；阶段收益率 = (期末净值−期初资金)÷期初资金；基准 = 沪深300
- 全量测试提交前必须全绿（当前基线 69）

---

### Task 1: 重构 `core/analyze.py` 抽 `analyze_at` 纯函数

**Files:**
- Modify: `core/analyze.py`
- Test: `tests/test_analyze.py`

**Interfaces:**
- Produces: `analyze_at(data: dict, weights: dict) -> dict` — data 含 `index_df/val_df/bond_df/quotes/flow/hist/bench`（DataFrame/ dict），返回 `{"trend", "sectors", "portfolio", "warnings"}`。不访问网络。
- `run_analysis(provider)` 改为：拉数据 → 调 `analyze_at` → 追加实时个股选股。

- [ ] **Step 1: 写失败测试**

在 `tests/test_analyze.py` 追加：

```python
def test_analyze_at_pure_computation():
    from core.analyze import analyze_at
    import pandas as pd
    index = pd.DataFrame({"date": ["2026-08-10"], "close": [4000.0]})
    data = {
        "index_df": index,
        "val_df": pd.DataFrame({"date": ["2026-08-10"], "pe": [13.0], "pb": [1.4]}),
        "bond_df": pd.DataFrame({"date": ["2026-08-10"], "cn_10y": [2.5]}),
        "quotes": pd.DataFrame({"name": ["医疗服务"], "pct_change": [2.0]}),
        "flow": pd.DataFrame(),
        "hist": {"医疗服务": pd.DataFrame({"date": ["2026-08-10"], "close": [100.0]})},
        "bench": index,
    }
    from core.config import load_weights
    out = analyze_at(data, load_weights())
    assert set(out) == {"trend", "sectors", "portfolio", "warnings"}
    assert "trend" in out and "sectors" in out and "portfolio" in out
```

- [ ] **Step 2: 运行确认失败**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_analyze.py::test_analyze_at_pure_computation -v`
Expected: FAIL（`ImportError: cannot import name 'analyze_at'`）

- [ ] **Step 3: 重构实现**

`core/analyze.py` 新增纯函数（置于 `run_analysis` 之前），并把 `run_analysis` 改为先拉数据再调它：

```python
def analyze_at(data: dict, weights: dict) -> dict:
    """给定数据切片 + 权重，计算趋势/板块/组合。纯函数，不访问网络。"""
    index_df = data.get("index_df") or pd.DataFrame()
    val_df = data.get("val_df") or pd.DataFrame()
    bond_df = data.get("bond_df") or pd.DataFrame()
    quotes = data.get("quotes") or pd.DataFrame()
    flow = data.get("flow") or pd.DataFrame()
    hist = data.get("hist") or {}
    bench = data.get("bench") or pd.DataFrame()
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
```

改写 `run_analysis`（拉数据 + 调 `analyze_at` + 实时选股），保持返回键不变：

```python
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
    data_until = base["trend"].get("data_until",
                                   datetime.now().strftime("%Y-%m-%d"))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trend": base["trend"], "sectors": base["sectors"], "stocks": stocks,
        "portfolio": base["portfolio"], "data_until": data_until,
        "data_quality": provider.quality_report(), "warnings": base["warnings"],
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
```

- [ ] **Step 4: 运行确认通过**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_analyze.py -v`
Expected: 全部 PASS（含既有 3 个候选池测试）

- [ ] **Step 5: 跑全量离线子集确认无回归**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_analyze.py tests/test_data.py tests/test_sector.py tests/test_portfolio.py -q`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add core/analyze.py tests/test_analyze.py
git commit -m "refactor: extract analyze_at pure function from run_analysis"
```

---

### Task 2: `core/simulation.py` — HistoryProvider（历史视图）

**Files:**
- Create: `core/simulation.py`
- Test: `tests/test_simulation.py`

**Interfaces:**
- Produces: `class HistoryProvider` — `load(provider, lookback_days=365)` 加载历史到内存；`snapshot_at(date: str) -> dict` 返回截至 date 的 `analyze_at` 数据切片（`index_df/val_df/bond_df/quotes/flow/hist/bench`）。`etf_close(code6: str) -> pd.DataFrame` 返回该 ETF 的 `date/close` 历史。

- [ ] **Step 1: 写失败测试**

`tests/test_simulation.py`：

```python
import pandas as pd
import pytest


def _fake_provider():
    """内存 fake，snapshot_at 只做日期切片，不访问网络。"""
    class P:
        pass
    p = P
    p.index_df = pd.DataFrame({"date": pd.to_datetime(
        ["2026-01-05", "2026-02-02", "2026-03-02"]), "close": [1.0, 2.0, 3.0]})
    return p


def test_history_snapshot_slices_by_date():
    from core.simulation import _slice_by_date
    df = pd.DataFrame({"date": pd.to_datetime(
        ["2026-01-05", "2026-02-02", "2026-03-02"]), "close": [1.0, 2.0, 3.0]})
    out = _slice_by_date(df, "2026-02-01")
    assert list(out["close"]) == [1.0]


def test_snapshot_at_keeps_bench_parallel():
    from core.simulation import _snapshot_at
    index = pd.DataFrame({"date": pd.to_datetime(
        ["2026-01-05", "2026-02-02", "2026-03-02"]), "close": [1.0, 2.0, 3.0]})
    data = {"index_df": index, "val_df": pd.DataFrame(),
            "bond_df": pd.DataFrame(),
            "quotes": pd.DataFrame(), "flow": pd.DataFrame(),
            "hist": {}, "bench": index}
    snap = _snapshot_at(data, "2026-02-01")
    assert list(snap["index_df"]["close"]) == [1.0]
    assert list(snap["bench"]["close"]) == [1.0]
```

- [ ] **Step 2: 运行确认失败**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_simulation.py -v`
Expected: FAIL（`ImportError: cannot import name '_slice_by_date'`）

- [ ] **Step 3: 实现 HistoryProvider 与切片**

`core/simulation.py`：

```python
"""一年模拟引擎：历史数据视图 + 月度调仓 + 虚拟账户模拟。"""
import pandas as pd
from datetime import datetime, timedelta

from core.logging import get_logger

logger = get_logger("core.simulation")


def _slice_by_date(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """返回 df 中 date <= 指定日的行（date 列先转 datetime）。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out[out["date"] <= pd.to_datetime(date)].reset_index(drop=True)


def _snapshot_at(data: dict, date: str) -> dict:
    """对 data 中每个 DataFrame 按 date 切片，dict 值(hist)递归处理。"""
    snap = {}
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            snap[key] = _slice_by_date(value, date)
        elif isinstance(value, dict):
            snap[key] = {k: _slice_by_date(v, date) for k, v in value.items()
                         if isinstance(v, pd.DataFrame)}
        else:
            snap[key] = value
    return snap


class HistoryProvider:
    """一次性加载历史数据，提供任意历史日期的数据切片（防前视）。"""

    def __init__(self, provider, store, lookback_days: int = 365):
        self.provider = provider
        self.store = store
        self._etf_cache: dict[str, pd.DataFrame] = {}
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")
        self._index = provider.index_daily("沪深300")
        self._val = provider.index_valuation("沪深300")
        self._bond = provider.bond_yield()
        self._bench = provider.index_daily(provider.benchmark_index_code())
        self._quotes = provider.sector_quote()
        # flow 在历史时点不可得，用空表
        self._flow = pd.DataFrame()
        self._hist = {}
        if not self._quotes.empty:
            for name in list(self._quotes["name"])[:30]:
                try:
                    h = provider.sector_hist(name)
                    if not h.empty:
                        self._hist[name] = h
                except Exception as exc:  # noqa: BLE001
                    logger.warning("模拟板块历史失败 %s: %s", name, exc)
        self._data = {
            "index_df": self._index, "val_df": self._val, "bond_df": self._bond,
            "quotes": self._quotes, "flow": self._flow, "hist": self._hist,
            "bench": self._bench,
        }

    def snapshot_at(self, date: str) -> dict:
        return _snapshot_at(self._data, date)

    def etf_close(self, code6: str) -> pd.DataFrame:
        """新浪 ETF 历史收盘（东财被限流时可用），结果缓存。code6 如 '510300'。"""
        if code6 in self._etf_cache:
            return self._etf_cache[code6]
        import akshare as ak
        prefix = "sh" if code6.startswith(("5", "6")) else "sz"
        df = ak.fund_etf_hist_sina(symbol=prefix + code6)
        out = df[["date", "close"]].copy()
        out["date"] = pd.to_datetime(out["date"])
        self._etf_cache[code6] = out
        return out
```

- [ ] **Step 4: 运行确认通过**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_simulation.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add core/simulation.py tests/test_simulation.py
git commit -m "feat: add HistoryProvider historical snapshot view"
```

---

### Task 3: `run_year_simulation` 核心（月度调仓 + 净值 + 统计）

**Files:**
- Modify: `core/simulation.py`
- Test: `tests/test_simulation.py`

**Interfaces:**
- Consumes: `HistoryProvider.snapshot_at` / `etf_close`、`analyze_at`、`SimAccount`
- Produces: `run_year_simulation(provider, store, lookback_days=365) -> dict`（结构见 spec §4）

- [ ] **Step 1: 写失败测试**

`tests/test_simulation.py` 追加（用小型构造数据，不访问网络）：

```python
def test_monthly_rebalance_dates():
    from core.simulation import _monthly_rebalance_dates
    dates = pd.to_datetime([
        "2026-01-05", "2026-01-28", "2026-01-29", "2026-01-30",
        "2026-02-02", "2026-02-27", "2026-03-02", "2026-03-31"])
    out = _monthly_rebalance_dates(dates)
    # 每月最后交易日
    assert out == ["2026-01-30", "2026-02-27", "2026-03-31"]


def test_run_year_simulation_returns_structure():
    from core.simulation import run_year_simulation
    from core.store import Store
    import tempfile
    store = Store(tempfile.mkdtemp() + "/t.db")
    provider = _FakeSimProvider()
    out = run_year_simulation(provider, store, lookback_days=90)
    assert {"stats", "curve", "trades", "rebalances"} <= set(out)
    assert out["stats"]["total_return"] is not None
    assert len(out["curve"]) >= 1


class _FakeSimProvider:
    """小型历史数据的 fake provider，供模拟测试离线使用。"""

    def __init__(self):
        import pandas as pd
        dates = pd.to_datetime(["2026-01-05", "2026-02-02", "2026-03-02"])
        self.index = pd.DataFrame({"date": dates,
                                   "close": [4000.0, 4100.0, 4200.0]})
        self.val = pd.DataFrame({"date": dates,
                                 "pe": [13.0, 13.5, 14.0], "pb": [1.4, 1.4, 1.5]})
        self.bond = pd.DataFrame({"date": dates, "cn_10y": [2.5, 2.5, 2.6]})
        self.quotes = pd.DataFrame({"name": ["医疗服务"], "pct_change": [2.0]})
        self.hist = {"医疗服务": pd.DataFrame(
            {"date": dates, "close": [100.0, 105.0, 110.0]})}

    def index_daily(self, symbol="沪深300"):
        return self.index

    def index_valuation(self, name):
        return self.val

    def bond_yield(self):
        return self.bond

    def sector_quote(self):
        return self.quotes

    def sector_flow(self):
        return pd.DataFrame()

    def sector_hist(self, name):
        return self.hist.get(name, pd.DataFrame())

    def benchmark_index_code(self):
        return "sh000300"

    def etf_close(self, code6):
        import pandas as pd
        return pd.DataFrame({
            "date": pd.to_datetime(["2026-01-05", "2026-02-02", "2026-03-02"]),
            "close": [3.9, 4.0, 4.1],
        })

- [ ] **Step 2: 运行确认失败**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_simulation.py::test_run_year_simulation_returns_structure -v`
Expected: FAIL（`ImportError: cannot import name 'run_year_simulation'`）

- [ ] **Step 3: 实现**

`core/simulation.py` 追加：

```python
def _monthly_rebalance_dates(all_dates) -> list[str]:
    """返回每月最后一个交易日的 date 字符串列表（升序）。"""
    dates = sorted(pd.to_datetime(all_dates).tolist())
    months = {}
    for d in dates:
        months.setdefault(d.strftime("%Y-%m"), d)
    return [v.strftime("%Y-%m-%d") for v in months.values()]


def run_year_simulation(provider, store, lookback_days: int = 365) -> dict:
    hp = HistoryProvider(provider, store, lookback_days)
    bench = hp._bench
    if bench.empty:
        return {"stats": {}, "curve": [], "trades": [], "rebalances": [],
                "error": "基准历史不足"}
    all_dates = pd.to_datetime(bench["date"])
    rebalance_dates = _monthly_rebalance_dates(all_dates)
    weights = __import__("core.config", fromlist=["load_weights"]).load_weights()
    account = __import__("core.account", fromlist=["SimAccount"]).SimAccount(store)
    account.ensure_initialized()
    rebalances = []
    for rd in rebalance_dates:
        snap = hp.snapshot_at(rd)
        base = __import__("core.analyze", fromlist=["analyze_at"]).analyze_at(snap, weights)
        port = base["portfolio"]
        if not port:
            continue
        prices = {}
        for name in [port["core"]["name"]] + [s["name"] for s in port.get("satellite", [])]:
            code = re.search(r"(\d{6})", name or "")
            if code:
                closes = hp.etf_close(code.group(1))
                c = closes[closes["date"] <= pd.to_datetime(rd)]
                if not c.empty:
                    prices[code.group(1)] = float(c["close"].iloc[-1])
        account.execute(port, prices)
        rebalances.append({"date": rd, "weights": {
            (port["core"]["name"]): port["core"]["weight"]} | {
            s["name"]: s["weight"] for s in port.get("satellite", [])}})
    return _build_result(account, store, hp, bench, rebalances, all_dates)


def _build_result(account, store, hp, bench, rebalances, all_dates) -> dict:
    curve = []
    trades = [dict(t) for t in store.list_trades()]
    acc = store.get_account()
    cash0 = float(acc.get("cash", 0.0))
    for d in pd.to_datetime(all_dates):
        dstr = d.strftime("%Y-%m-%d")
        acc = store.get_account()
        cash = float(acc.get("cash", 0.0))
        holdings = 0.0
        for pos in store.list_positions():
            closes = hp.etf_close(pos["symbol"])
            c = closes[closes["date"] <= d]
            price = float(c["close"].iloc[-1]) if not c.empty else pos["cost_price"]
            holdings += pos["qty"] * price
        nav = cash + holdings
        b = bench[bench["date"] <= d]
        bn = float(b["close"].iloc[-1]) / float(bench["close"].iloc[0]) if not b.empty and len(bench) else 1.0
        curve.append({"date": dstr, "nav": round(nav, 2),
                      "benchmark": round(bn, 4)})
    return {
        "stats": _stats_from(curve, trades),
        "curve": curve, "trades": trades, "rebalances": rebalances,
    }


def _stats_from(curve, trades) -> dict:
    if not curve:
        return {}
    first_nav = curve[0]["nav"] or 1.0
    last_nav = curve[-1]["nav"]
    total_return = round(last_nav / first_nav - 1.0, 4)
    closed = [t for t in trades if t.get("status") == "closed"]
    wins = sum(1 for t in closed if (t.get("pnl") or 0) > 0)
    win_rate = round(wins / len(closed), 3) if closed else 0.0
    peak = curve[0]["nav"]
    mdd = 0.0
    for p in curve:
        peak = max(peak, p["nav"])
        if peak > 0:
            mdd = min(mdd, p["nav"] / peak - 1.0)
    return {"total_return": total_return,
            "benchmark_return": round(curve[-1]["benchmark"] - 1.0, 4),
            "win_rate": win_rate,
            "excess_return": round(total_return - (curve[-1]["benchmark"] - 1.0), 4),
            "max_drawdown": round(mdd, 4),
            "n_trades": len([t for t in trades if t.get("side") == "buy"])}
```

> 注：Task 3 的 `run_year_simulation` 直接依赖 `HistoryProvider`；`etf_close` 为网络调用，测试用 `_FakeSimProvider` 的 fake 版覆盖。若网络接口被限流导致失败，各调仓点跳过对应持仓取价（沿用降级，不崩溃）。

- [ ] **Step 4: 运行确认通过**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_simulation.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add core/simulation.py tests/test_simulation.py
git commit -m "feat: add year-long simulation engine with monthly rebalancing"
```

---

### Task 4: 自动投资调度 `core/auto_invest.py`

**Files:**
- Create: `core/auto_invest.py`
- Test: `tests/test_auto_invest.py`

**Interfaces:**
- Produces: `is_trading_day(dt) -> bool`（周一至周五）；`run_auto_invest(provider, account, store) -> dict`（执行一次分析+投资，复用 analyze 端点逻辑，返回结果）。

- [ ] **Step 1: 写失败测试**

`tests/test_auto_invest.py`：

```python
from datetime import datetime
import pytest


def test_is_trading_day_weekday():
    from core.auto_invest import is_trading_day
    assert is_trading_day(datetime(2026, 8, 10))  # 周一
    assert is_trading_day(datetime(2026, 8, 14))  # 周五
    assert not is_trading_day(datetime(2026, 8, 15))  # 周六
    assert not is_trading_day(datetime(2026, 8, 16))  # 周日


def test_run_auto_invest_returns_result(monkeypatch):
    from core.auto_invest import run_auto_invest
    called = {}
    def fake_run(provider, account, store):
        called["ok"] = True
        return {"trend": {"state": "低估"}}
    import core.auto_invest as ai
    monkeypatch.setattr(ai, "_do_invest", fake_run)
    out = run_auto_invest(None, None, None)
    assert called.get("ok")
    assert out["trend"]["state"] == "低估"
```

- [ ] **Step 2: 运行确认失败**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_auto_invest.py -v`
Expected: FAIL（`ImportError: cannot import name 'is_trading_day'`）

- [ ] **Step 3: 实现**

`core/auto_invest.py`：

```python
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
    prices = {}
    portfolio = result.get("portfolio") or {}
    if portfolio.get("core"):
        import re
        m = re.search(r"(\d{6})", portfolio["core"].get("name", ""))
        if m:
            prices[m.group(1)] = 0.0
    for sat in portfolio.get("satellite", []):
        import re
        m = re.search(r"(\d{6})", sat.get("name", ""))
        if m:
            prices.setdefault(m.group(1), 0.0)
    # 取实时价格
    try:
        spot = provider.stock_spot()
        if not spot.empty and "code" in spot.columns:
            cm = dict(zip(spot["code"], spot["price"]))
            for c in list(prices):
                if c in cm:
                    prices[c] = float(cm[c])
        etf = provider.etf_spot()
        if not etf.empty and "code" in etf.columns:
            cm = dict(zip(etf["code"], etf["price"]))
            for c in list(prices):
                if c in cm and not prices.get(c):
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
```

- [ ] **Step 4: 运行确认通过**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_auto_invest.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add core/auto_invest.py tests/test_auto_invest.py
git commit -m "feat: add auto-invest worker logic with trading-day check"
```

---

### Task 5: 接线 `api/main.py`（lifespan 调度 + `/api/simulation`）

**Files:**
- Modify: `api/main.py`
- Test: `tests/test_api.py`、`tests/test_chat_api.py`

**Interfaces:**
- Consumes: `core.auto_invest.run_auto_invest`、`core.simulation.run_year_simulation`
- Produces: 新增 `GET /api/simulation`；lifespan 启动后台自动投资任务；`POST /api/analyze` 改用 `run_auto_invest`。

- [ ] **Step 1: 写失败测试**

`tests/test_api.py` 追加（monkeypatch 引擎避免真实计算）：

```python
@pytest.mark.asyncio
async def test_simulation_endpoint(monkeypatch):
    import api.main as api
    monkeypatch.setattr(
        api, "run_year_simulation",
        lambda *a, **k: {"stats": {"total_return": 0.1}, "curve": [],
                         "trades": [], "rebalances": []})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/simulation")
    assert resp.status_code == 200
    assert "stats" in resp.json()
```

- [ ] **Step 2: 运行确认失败**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_api.py::test_simulation_endpoint -v`
Expected: FAIL（404）

- [ ] **Step 3: 实现**

`api/main.py` 改动：
- 顶部 import：`import threading`、`from contextlib import asynccontextmanager`、`from core.auto_invest import run_auto_invest as _auto_invest`、`from core.simulation import run_year_simulation`
- 用 lifespan 启动后台任务（替换 `app = FastAPI(...)`）。互斥用 **`threading.Lock`**（跨线程安全，兼容 to_thread 与 FastAPI 线程池）：

```python
_invest_lock = threading.Lock()
_sim_cache: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_auto_invest_loop())
    yield
    task.cancel()


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
```

- `POST /api/analyze` 改用 `_auto_invest(...)` 并加锁；`GET /api/simulation`：

```python
@app.post("/api/analyze")
def analyze():
    with _invest_lock:
        result = _auto_invest(_provider, _account, _store)
    ai = { "trend": _safe(interpret_trend, result.get("trend")),
           "sectors": _safe(recommend_sectors, result.get("sectors")),
           "stocks": _safe(recommend_stocks, result.get("stocks")),
           "portfolio": _safe(plan_portfolio, result.get("portfolio")) }
    return {**result, "analysis": result, "ai": ai, "account": _account.period_stats(),
            "data_until": result.get("data_until")}


@app.get("/api/simulation")
def simulation():
    if _sim_cache:
        return _sim_cache
    out = run_year_simulation(_provider, _store)
    _sim_cache.update(out)
    return out
```

> 注：`_auto_invest` 为 `core.auto_invest.run_auto_invest` 的别名（import 时 `from core.auto_invest import run_auto_invest as _auto_invest`）。原 `analyze` 端点的 `_prices_for_portfolio` 取价逻辑已并入 `core.auto_invest._do_invest`，删除端点内重复实现。原 `_prices_for_portfolio`/`_code` 函数若无其他引用则删除。

- [ ] **Step 4: 运行确认通过**

Run: `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_api.py::test_simulation_endpoint -v`
Expected: PASS

- [ ] **Step 5: 跑全量确认 lifespan 改动不破坏现有端点**

Run: `HF_HUB_OFFLINE=1 DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest tests/test_api.py tests/test_chat_api.py -q`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add api/main.py tests/test_api.py
git commit -m "feat: wire auto-invest scheduler and /api/simulation endpoint"
```

---

### Task 6: 前端一年模拟区块（Canvas 曲线 + 操作表）

**Files:**
- Modify: `web/index.html`、`web/app.js`、`web/style.css`

**Interfaces:**
- Consumes: `GET /api/simulation` 返回结构
- Produces: 看板"一年模拟"区块：Canvas 折线图（净值 vs 基准）、统计卡片、详细操作表、调仓记录。

- [ ] **Step 1: `web/index.html` 追加区块**

在 `<script src="app.js"></script>` 之前插入：

```html
<section id="sim-section">
  <h2>一年模拟（按系统推荐方法）</h2>
  <button id="btn-sim">加载一年模拟</button>
  <canvas id="sim-chart" width="900" height="320" style="max-width:100%;border:1px solid #ddd;"></canvas>
  <div id="sim-stats" class="cards"></div>
  <h3>详细操作</h3>
  <div id="sim-trades"></div>
  <h3>调仓记录</h3>
  <div id="sim-rebalances"></div>
</section>
```

- [ ] **Step 2: `web/app.js` 追加渲染逻辑**

```js
async function loadSimulation() {
  const d = await getJSON("/api/simulation");
  if (d.error) { document.getElementById("sim-trades").innerHTML = "<p>" + d.error + "</p>"; return; }
  const st = d.stats || {};
  document.getElementById("sim-stats").innerHTML = [
    card("总收益", (st.total_return * 100).toFixed(2) + "%"),
    card("基准收益", (st.benchmark_return * 100).toFixed(2) + "%"),
    card("交易笔数", st.n_trades),
  ].join("");
  drawSimChart(d.curve);
  document.getElementById("sim-trades").innerHTML = table(
    ["时间", "方向", "名称", "价格", "数量", "盈亏"],
    (d.trades || []).map(t => [t.time, t.side, t.name, t.price, t.qty, t.pnl]));
  document.getElementById("sim-rebalances").innerHTML = table(
    ["日期", "权重"],
    (d.rebalances || []).map(r => [r.date, JSON.stringify(r.weights)]));
}

function drawSimChart(curve) {
  const c = document.getElementById("sim-chart");
  if (!c || !curve || curve.length === 0) return;
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 30;
  ctx.clearRect(0, 0, W, H);
  const xs = curve.map((_, i) => i), ys = curve.map(p => p.nav);
  const yMax = Math.max(...ys) * 1.05, yMin = Math.min(...ys) * 0.95;
  const px = i => pad + i / (xs.length - 1) * (W - 2 * pad);
  const py = v => H - pad - (v - yMin) / (yMax - yMin) * (H - 2 * pad);
  ctx.strokeStyle = "#1f77b4"; ctx.beginPath();
  curve.forEach((p, i) => i === 0 ? ctx.moveTo(px(i), py(p.nav)) : ctx.lineTo(px(i), py(p.nav)));
  ctx.stroke();
  ctx.strokeStyle = "#999"; ctx.beginPath();
  curve.forEach((p, i) => i === 0 ? ctx.moveTo(px(i), py(p.benchmark)) : ctx.lineTo(px(i), py(p.benchmark)));
  ctx.stroke();
  ctx.fillStyle = "#333";
  ctx.fillText("模拟净值", pad + 4, pad + 12);
  ctx.fillText("沪深300", pad + 4, pad + 26);
}

document.getElementById("btn-sim").addEventListener("click", loadSimulation);
```

- [ ] **Step 3: 提交**

```bash
git add web/index.html web/app.js web/style.css
git commit -m "feat: render year simulation curve and trade table in dashboard"
```

---

### Task 7: 全量验证 + AI 须知同步

- [ ] **Step 1: 同步 `AI须知.md`**

在 §3 结构速览的 `core/` 增加：`simulation.py` 一年模拟引擎；`auto_invest.py` 自动投资调度。§7 坑 1 补充 ETF 历史用新浪回退、板块历史同花顺。

- [ ] **Step 2: 全量测试**

Run: `HF_HUB_OFFLINE=1 DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest -q`
Expected: 全部 PASS（基线 69 + 新增 ≈ 79+）

- [ ] **Step 3: 启动服务实测**

Run: `HF_HUB_OFFLINE=1 USE_MOCK_LLM=1 DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m uvicorn api.main:app --port 8001`
Expected: `/api/dashboard` 正常；`GET /api/simulation` 返回 `curve/trades/stats`；启动日志出现一次自动投资执行。

- [ ] **Step 4: 提交**

```bash
git add "AI须知.md"
git commit -m "docs: sync AI onboarding with simulation & auto-invest"
```
