"""core.data 纯函数测试：估值合并、行情多源归一化（离线可跑）。"""
import pandas as pd
import pytest

from core.data import (
    _clean_code,
    _extract_financial,
    _merge_pe_pb,
    _normalize_sector_flow,
    _normalize_sector_hist,
    _normalize_sector_quote,
    _normalize_spot,
)


def test_merge_pe_pb_combines_date_pe_pb():
    pe = pd.DataFrame({"日期": ["2026-08-08", "2026-08-09"],
                       "静态市盈率": [12.5, 13.0]})
    pb = pd.DataFrame({"日期": ["2026-08-08", "2026-08-09"],
                       "市净率": [1.4, 1.5]})
    out = _merge_pe_pb(pe, pb)
    assert list(out.columns) == ["date", "pe", "pb"]
    assert out["pe"].tolist() == [12.5, 13.0]
    assert out["pb"].tolist() == [1.4, 1.5]
    assert out["date"].tolist() == ["2026-08-08", "2026-08-09"]


def test_merge_pe_pb_rolls_back_to_rolling_pe():
    # 新版 pe 表若无静态市盈率列，回退到滚动市盈率
    pe = pd.DataFrame({"日期": ["2026-08-08"], "滚动市盈率": [15.2]})
    pb = pd.DataFrame({"日期": ["2026-08-08"], "市净率": [1.2]})
    out = _merge_pe_pb(pe, pb)
    assert out["pe"].tolist() == [15.2]


def test_merge_pe_pb_missing_required_returns_empty():
    pe = pd.DataFrame({"日期": ["2026-08-08"]})  # 无 PE 列
    pb = pd.DataFrame({"日期": ["2026-08-08"], "市净率": [1.2]})
    assert _merge_pe_pb(pe, pb).empty


def test_normalize_spot_maps_chinese_columns():
    df = pd.DataFrame({"代码": ["000001"], "名称": ["平安银行"],
                       "最新价": [10.5], "涨跌幅": [1.2]})
    out = _normalize_spot(df)
    assert list(out.columns) == ["code", "name", "price", "pct_change"]
    assert out.iloc[0]["code"] == "000001"
    assert out.iloc[0]["name"] == "平安银行"
    assert out.iloc[0]["price"] == 10.5


def test_normalize_spot_missing_column_returns_none():
    # 源缺关键列时返回 None（触发多源回退），而不是 KeyError
    df = pd.DataFrame({"代码": ["000001"]})
    assert _normalize_spot(df) is None


def test_normalize_spot_strips_market_prefix():
    # 新浪源带市场前缀(sh/sz/bj)，需统一为纯 6 位数字供下游使用
    df = pd.DataFrame({"代码": ["sh600000", "sz000001", "bj920000"],
                       "名称": ["浦发银行", "平安银行", "万达轴承"],
                       "最新价": [10.0, 11.0, 12.0], "涨跌幅": [1.0, 2.0, 3.0]})
    out = _normalize_spot(df)
    assert out["code"].tolist() == ["600000", "000001", "920000"]


def test_extract_financial_em():
    df = pd.DataFrame({"数据日期": ["2026-08-10"], "PE(TTM)": [20.4], "市净率": [6.2]})
    out = _extract_financial("600519", df, "em")
    assert out["code"] == "600519"
    assert out["pe"] == 20.4
    assert out["pb"] == 6.2


def test_extract_financial_baidu_fallback():
    df = pd.DataFrame({"date": ["2026-08-10"], "value": [20.39]})
    out = _extract_financial("600519", df, "baidu")
    assert out["pe"] == 20.39
    assert out["pb"] == 0.0


def test_extract_financial_empty_returns_empty():
    assert _extract_financial("600519", pd.DataFrame(), "em") == {}


def test_clean_code_normalizes_prefix_and_padding():
    assert _clean_code("sh601212") == "601212"
    assert _clean_code("sz000001") == "000001"
    assert _clean_code("bj920000") == "920000"
    assert _clean_code("601212") == "601212"
    assert _clean_code("600000.0") == "600000"  # 浮点字符串
    assert _clean_code(1) == "000001"           # int 补零


def test_normalize_sector_quote_em():
    df = pd.DataFrame({"板块名称": ["医疗服务"], "涨跌幅": [2.5]})
    out = _normalize_sector_quote(df, "em")
    assert out["name"].tolist() == ["医疗服务"]
    assert out["pct_change"].tolist() == [2.5]


def test_normalize_sector_quote_ths():
    df = pd.DataFrame({"板块": ["贵金属"], "涨跌幅": [-5.03]})
    out = _normalize_sector_quote(df, "ths")
    assert out["name"].tolist() == ["贵金属"]
    assert out["pct_change"].tolist() == [-5.03]


def test_normalize_sector_flow_em():
    df = pd.DataFrame({"名称": ["医疗服务"], "主力净流入-净额": [123.5]})
    out = _normalize_sector_flow(df, "em")
    assert out["name"].tolist() == ["医疗服务"]
    assert out["net_inflow"].tolist() == [123.5]


def test_normalize_sector_flow_ths():
    df = pd.DataFrame({"板块": ["贵金属"], "净流入": [-48.37]})
    out = _normalize_sector_flow(df, "ths")
    assert out["name"].tolist() == ["贵金属"]
    assert out["net_inflow"].tolist() == [-48.37]


def test_normalize_sector_hist_em():
    df = pd.DataFrame({"日期": ["2026-08-10"], "收盘": [100.0]})
    out = _normalize_sector_hist(df, "em")
    assert list(out.columns) == ["date", "close"]


def test_normalize_sector_hist_ths():
    df = pd.DataFrame({"日期": ["2026-08-10"], "收盘价": [20554.083]})
    out = _normalize_sector_hist(df, "ths")
    assert list(out.columns) == ["date", "close"]
    assert out["close"].tolist() == [20554.083]


def test_normalize_sector_missing_column_returns_none():
    assert _normalize_sector_quote(pd.DataFrame({"板块": ["贵金属"]}), "em") is None
    assert _normalize_sector_flow(pd.DataFrame({"板块": ["贵金属"]}), "em") is None
    assert _normalize_sector_hist(pd.DataFrame({"日期": ["x"]}), "em") is None
