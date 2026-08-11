"""akshare 数据封装。所有方法可离线降级：缓存命中优先，网络失败返回空数据。"""
import json
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

from core.logging import get_logger
from core.store import Store

logger = get_logger("core.data")

# 抓取重试：3 次 + 指数退避（0.5s → 1s → 2s），应对瞬时网络抖动/接口限流
_FETCH_RETRIES = 3
_FETCH_BACKOFF_BASE = 0.5

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
        # 板块历史熔断：东财连续失败后直接走同花顺，避免每个板块反复等待东财超时
        self._em_sector_hist_failures = 0
        self._sector_hist_fallback = False

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
        data = None
        last_exc = None
        for attempt in range(1, _FETCH_RETRIES + 1):
            try:
                data = fetch()
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _FETCH_RETRIES:
                    delay = _FETCH_BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "akshare 调用失败 %s（第 %d/%d 次）: %s，%.1fs 后重试",
                        key, attempt, _FETCH_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
        if data is None:
            logger.error("akshare 获取失败 %s: %s", key, last_exc)
            self._record_freshness(key, "missing", None, ttl)
            return _empty_like(key)
        if last_exc is not None:
            status = "ok_retry"
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
            # 乐咕乐股 PE/PB 历史分两张表返回（新版列名为中文），合并为 date/pe/pb
            pe = ak.stock_index_pe_lg(symbol=name)
            pb = ak.stock_index_pb_lg(symbol=name)
            out = _merge_pe_pb(pe, pb)
            if out.empty:
                raise RuntimeError("估值数据列为空或缺少 PE/PB")
            return out

        return self._cached(f"index_valuation:{name}", 3600 * 12, fetch)

    # ---- 板块 ----
    def sector_quote(self) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            # 东财优先，失败回退同花顺板块汇总
            errors = []
            try:
                out = _normalize_sector_quote(ak.stock_board_industry_name_em(), "em")
                if out is not None:
                    return out
                errors.append("东财: 列结构异常")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"东财: {exc}")
            try:
                out = _normalize_sector_quote(
                    ak.stock_board_industry_summary_ths(), "ths"
                )
                if out is not None:
                    return out
            except Exception as exc:  # noqa: BLE001
                errors.append(f"同花顺: {exc}")
            raise RuntimeError("板块行情源均不可用: " + "; ".join(errors))

        return self._cached("sector_quote", 3600 * 4, fetch)

    def sector_flow(self) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            # 东财优先，失败回退同花顺板块汇总（净流入）
            errors = []
            try:
                out = _normalize_sector_flow(
                    ak.stock_sector_fund_flow_rank(
                        indicator="今日", sector_type="行业资金流"
                    ), "em",
                )
                if out is not None:
                    return out
                errors.append("东财: 列结构异常")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"东财: {exc}")
            try:
                out = _normalize_sector_flow(
                    ak.stock_board_industry_summary_ths(), "ths"
                )
                if out is not None:
                    return out
            except Exception as exc:  # noqa: BLE001
                errors.append(f"同花顺: {exc}")
            raise RuntimeError("板块资金流源均不可用: " + "; ".join(errors))

        return self._cached("sector_flow", 3600 * 4, fetch)

    def sector_hist(self, name: str) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            # 东财优先；熔断开启或东财连续失败后直接走同花顺行业指数 K 线
            if not self._sector_hist_fallback:
                try:
                    out = _normalize_sector_hist(
                        ak.stock_board_industry_hist_em(
                            symbol=name, period="日k",
                            start_date="20200101",
                            end_date=datetime.now().strftime("%Y%m%d"),
                            adjust="",
                        ), "em",
                    )
                    if out is not None:
                        self._em_sector_hist_failures = 0
                        return out
                except Exception as exc:  # noqa: BLE001
                    self._em_sector_hist_failures += 1
                    logger.warning("板块历史东财失败 %s（第 %d 次），回退同花顺: %s",
                                   name, self._em_sector_hist_failures, exc)
                    if self._em_sector_hist_failures >= 2:
                        self._sector_hist_fallback = True
                        logger.warning("板块历史东财熔断，后续直接使用同花顺")
            out = _normalize_sector_hist(
                ak.stock_board_industry_index_ths(
                    symbol=name, start_date="20200101",
                    end_date=datetime.now().strftime("%Y%m%d"),
                ), "ths",
            )
            if out is None:
                raise RuntimeError("板块历史源均不可用")
            return out

        return self._cached(f"sector_hist:{name}", 3600 * 6, fetch)

    # ---- 个股 ----
    def stock_spot(self) -> pd.DataFrame:
        import akshare as ak

        def fetch():
            # 多源回退：东财优先，失败自动切新浪，降低限流/断连导致的获取失败
            errors = []
            for source, fn in (("东财", ak.stock_zh_a_spot_em),
                               ("新浪", ak.stock_zh_a_spot)):
                try:
                    out = _normalize_spot(fn())
                    if out is not None:
                        return out
                    errors.append(f"{source}: 列结构异常")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{source}: {exc}")
            raise RuntimeError("行情源均不可用: " + "; ".join(errors))

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
            # 东财 stock_value_em 优先（含 PE/PB），失败回退百度估值（仅 PE）
            code6 = _clean_code(code)
            try:
                return _extract_financial(
                    code6, ak.stock_value_em(symbol=code6), "em"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("stock_value_em 失败 %s，回退百度: %s", code6, exc)
            df = ak.stock_zh_valuation_baidu(
                symbol=code6, indicator="市盈率(TTM)", period="近一年"
            )
            return _extract_financial(code6, df, "baidu")

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


def _merge_pe_pb(pe_df: pd.DataFrame, pb_df: pd.DataFrame) -> pd.DataFrame:
    """把乐咕 PE/PB 两张表（新版中文列名）合并为 date/pe/pb。

    PE 列优先取「静态市盈率」，缺失时回退「滚动市盈率」；
    缺关键列返回空 DataFrame，由调用方决定是否降级。
    """
    pe_col = "静态市盈率" if "静态市盈率" in pe_df.columns else "滚动市盈率"
    if ("日期" not in pe_df.columns or pe_col not in pe_df.columns
            or "日期" not in pb_df.columns or "市净率" not in pb_df.columns):
        return pd.DataFrame(columns=["date", "pe", "pb"])
    pe = pe_df[["日期", pe_col]].rename(columns={"日期": "date", pe_col: "pe"})
    pb = pb_df[["日期", "市净率"]].rename(columns={"日期": "date", "市净率": "pb"})
    return pe.merge(pb, on="date").copy()


def _clean_code(code: Any) -> str:
    """归一化个股代码：去 sh/sz/bj 市场前缀、去浮点尾巴、补零到 6 位。

    兼容东财(纯数字)/新浪(带前缀)两种源的 code 表示，供 stock_financial 使用。
    """
    s = str(code).strip()
    s = re.sub(r"^(sh|sz|bj)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.0$", "", s)
    return s.zfill(6)


def _extract_financial(code: str, df: pd.DataFrame, source: str) -> dict:
    """从行情/估值表提取选股所需字段。source: em(东财 stock_value_em) / baidu(百度)。

    roe/growth 上游无数据源时诚实置 0（不影响数据真实性原则）；空表返回 {}。
    """
    if df is None or df.empty:
        return {}
    last = df.iloc[-1]
    if source == "em":
        return {
            "code": code,
            "pe": _num(last.get("PE(TTM)") or last.get("PE(静)")),
            "pb": _num(last.get("市净率")),
            "dividend": 0.0, "roe": 0.0, "growth": 0.0,
        }
    return {
        "code": code,
        "pe": _num(last.get("value")),
        "pb": 0.0,
        "dividend": 0.0, "roe": 0.0, "growth": 0.0,
    }


def _normalize_spot(df: pd.DataFrame) -> pd.DataFrame | None:
    """行情多源归一化：中文列 → code/name/price/pct_change。

    缺关键列或空数据返回 None（触发多源回退），而非抛 KeyError。
    code 统一为 6 位数字字符串（兼容东财/新浪的 int/float/str 表示）。
    """
    cols = {"代码": "code", "名称": "name", "最新价": "price", "涨跌幅": "pct_change"}
    if df is None or df.empty or not all(c in df.columns for c in cols):
        return None
    out = df.rename(columns=cols)[["code", "name", "price", "pct_change"]].copy()
    # 新浪源带市场前缀(sh/sz/bj)，东财源为纯数字；统一为 6 位数字字符串
    out["code"] = (
        out["code"].astype(str)
        .str.replace(r"^(sh|sz|bj)", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    return out


def _normalize_sector_quote(df: pd.DataFrame, source: str) -> pd.DataFrame | None:
    """板块行情归一化为 name/pct_change。source: em(东财) / ths(同花顺汇总)。"""
    if df is None or df.empty:
        return None
    if source == "em":
        if "板块名称" not in df.columns or "涨跌幅" not in df.columns:
            return None
        return df.rename(columns={"板块名称": "name", "涨跌幅": "pct_change"})[
            ["name", "pct_change"]].copy()
    if "板块" not in df.columns or "涨跌幅" not in df.columns:
        return None
    return df.rename(columns={"板块": "name", "涨跌幅": "pct_change"})[
        ["name", "pct_change"]].copy()


def _normalize_sector_flow(df: pd.DataFrame, source: str) -> pd.DataFrame | None:
    """板块资金流归一化为 name/net_inflow。source: em(东财) / ths(同花顺汇总)。"""
    if df is None or df.empty:
        return None
    if source == "em":
        if "名称" not in df.columns or "主力净流入-净额" not in df.columns:
            return None
        return df.rename(columns={"名称": "name", "主力净流入-净额": "net_inflow"})[
            ["name", "net_inflow"]].copy()
    if "板块" not in df.columns or "净流入" not in df.columns:
        return None
    return df.rename(columns={"板块": "name", "净流入": "net_inflow"})[
        ["name", "net_inflow"]].copy()


def _normalize_sector_hist(df: pd.DataFrame, source: str) -> pd.DataFrame | None:
    """板块K线归一化为 date/close。source: em(东财) / ths(同花顺行业指数)。"""
    if df is None or df.empty:
        return None
    if source == "em":
        if "日期" not in df.columns or "收盘" not in df.columns:
            return None
        return df.rename(columns={"日期": "date", "收盘": "close"})[
            ["date", "close"]].copy()
    if "日期" not in df.columns or "收盘价" not in df.columns:
        return None
    return df.rename(columns={"日期": "date", "收盘价": "close"})[
        ["date", "close"]].copy()


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
