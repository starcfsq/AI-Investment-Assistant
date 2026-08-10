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

    def etf_spot(self) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            df = ak.fund_etf_spot_em()
            return df.rename(
                columns={"代码": "code", "名称": "name",
                         "最新价": "price"}
            )[["code", "name", "price"]].copy()

        return self._cached("etf_spot", 3600, fetch)

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
                       "bond_yield", "etf_spot")):
        return pd.DataFrame()
    if key.startswith("stock_financial"):
        return {}
    return []
