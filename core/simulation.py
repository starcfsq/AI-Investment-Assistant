"""一年模拟引擎：历史数据视图 + 月度调仓 + 虚拟账户模拟。"""
import re
from datetime import datetime, timedelta

import pandas as pd

from core.account import SimAccount
from core.analyze import analyze_at
from core.config import load_weights
from core.logging import get_logger

logger = get_logger("core.simulation")


def _slice_by_date(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """返回 df 中 date <= 指定日的行（date 列先转 datetime）。

    无 date 列的实时快照（如板块行情 quotes/资金流 flow）不按日期切片，原样返回。
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "date" not in out.columns:
        return out
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
        """ETF 历史收盘，结果缓存。code6 如 '510300'。

        优先用 provider.etf_close（测试/fake 可离线覆盖），否则走新浪接口
        （东财被限流时的回退），结果统一为 date/close。
        """
        if code6 in self._etf_cache:
            return self._etf_cache[code6]
        if hasattr(self.provider, "etf_close"):
            out = self.provider.etf_close(code6)
        else:
            import akshare as ak
            prefix = "sh" if code6.startswith(("5", "6")) else "sz"
            df = ak.fund_etf_hist_sina(symbol=prefix + code6)
            out = df[["date", "close"]].copy()
        out = out[["date", "close"]].copy()
        out["date"] = pd.to_datetime(out["date"])
        self._etf_cache[code6] = out
        return out


def _monthly_rebalance_dates(all_dates) -> list[str]:
    """返回每月最后一个交易日的 date 字符串列表（升序）。"""
    dates = sorted(pd.to_datetime(all_dates).tolist())
    months: dict[str, pd.Timestamp] = {}
    for d in dates:
        # 升序遍历，后出现的同日/月覆盖，最终保留每月最后交易日
        months[d.strftime("%Y-%m")] = d
    return [v.strftime("%Y-%m-%d") for v in months.values()]


def run_year_simulation(provider, store, lookback_days: int = 365) -> dict:
    """年度模拟：按每月最后交易日调仓，逐日估值画净值曲线，输出统计/交易/调仓。

    全程使用 HistoryProvider.snapshot_at 的日期切片数据，杜绝前视；
    ETF 取价失败时跳过对应持仓（沿用降级，不崩溃）。
    """
    hp = HistoryProvider(provider, store, lookback_days)
    bench = hp._bench
    if bench.empty:
        return {"stats": {}, "curve": [], "trades": [], "rebalances": [],
                "error": "基准历史不足"}
    all_dates = pd.to_datetime(bench["date"])
    rebalance_dates = _monthly_rebalance_dates(all_dates)
    weights = load_weights()
    account = SimAccount(store)
    account.ensure_initialized()
    rebalances = []
    for rd in rebalance_dates:
        snap = hp.snapshot_at(rd)
        base = analyze_at(snap, weights)
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
    """按全部基准日生成净值/基准曲线，聚合交易与统计。"""
    curve = []
    trades = [dict(t) for t in store.list_trades()]
    bench = bench.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    base_close = float(bench["close"].iloc[0]) if len(bench) else 1.0
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
        bn = float(b["close"].iloc[-1]) / base_close if not b.empty and len(bench) else 1.0
        curve.append({"date": dstr, "nav": round(nav, 2),
                      "benchmark": round(bn, 4)})
    return {
        "stats": _stats_from(curve, trades),
        "curve": curve, "trades": trades, "rebalances": rebalances,
    }


def _stats_from(curve, trades) -> dict:
    """从净值曲线与交易记录计算虚拟账户统计（定义与虚拟账户一致）。"""
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
