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
