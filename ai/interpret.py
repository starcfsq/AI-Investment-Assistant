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
