"""演示用 Mock LLM 客户端。

不访问任何网络 API。从输入消息中提取确定性分析数据，按 schema 生成中文解读，
用于 USE_MOCK_LLM=1 时在无 Key 环境下预览「AI 解读」效果。
仍遵守核心原则：只转述输入数据，不生成任何市场数字。
"""
import json

from ai.provider import LLMClient

DISCLAIMER = ("本内容仅为基于公开数据的分析展示，不构成任何投资建议。"
              "投资有风险，入市需谨慎。")
CONFIDENCE = 0.7


class MockLLMClient(LLMClient):
    """模拟 DeepSeek：从消息里提取数据并填充 schema 输出。"""

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_content = m.get("content", "")
                break
        data = _extract_json(user_content)
        kind = _schema_kind(schema)
        return _build(kind, data)


def _extract_json(text: str):
    """提取消息里第一个 JSON 对象或数组（解读函数把数据内嵌在消息中）。"""
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return {}


def _schema_kind(schema: dict) -> str:
    props = set(schema.get("properties", {}).keys())
    if "points" in props and "state" in props:
        return "trend"
    if "recommendations" in props:
        return "sectors"
    if "report" in props:
        return "stocks"
    if "plan" in props and "rebalance" in props:
        return "portfolio"
    if "answer" in props:
        return "chat"
    return "generic"


def _build(kind: str, data):
    if kind == "trend":
        d = data if isinstance(data, dict) else {}
        state = d.get("state", "数据不足")
        detail = d.get("detail", {})
        return {
            "state": state,
            "points": [
                f"当前市场长期趋势状态为「{state}」",
                f"指数相对 250 日长期均线偏离 {detail.get('ma_dev', '—')}",
                f"估值历史百分位 PE {detail.get('pe_pct', '—')} / PB {detail.get('pb_pct', '—')}",
                f"股债性价比百分位 {detail.get('bond_equity_pct', '—')}",
            ],
            "risk": "以上基于公开历史数据，市场存在不确定性，请结合自身风险承受能力判断",
            "confidence": CONFIDENCE,
            "disclaimer": DISCLAIMER,
        }
    if kind == "sectors":
        sectors = data if isinstance(data, list) else []
        names = [s.get("name", "?") for s in sectors[:5]] or ["暂无足够板块数据"]
        return {
            "recommendations": [f"关注板块：{n}" for n in names],
            "logic": "基于 RS 相对强度、资金流与动量综合打分的排序结果",
            "confidence": CONFIDENCE,
            "disclaimer": DISCLAIMER,
        }
    if kind == "stocks":
        stocks = data if isinstance(data, list) else []
        if stocks:
            top = stocks[0]
            body = (f"当前量化筛选综合得分最高的是 {top.get('name', '?')}（{top.get('code', '?')}），"
                    f"得分 {top.get('score', '—')}。该排序综合了 ROE、成长、估值与股息因子。")
        else:
            body = "暂无足够的个股候选数据，无法给出标的推荐。"
        return {
            "report": body,
            "confidence": CONFIDENCE,
            "disclaimer": DISCLAIMER,
        }
    if kind == "portfolio":
        p = data if isinstance(data, dict) else {}
        summary = p.get("summary", "暂无组合建议")
        return {
            "plan": f"建议采用「{summary}」结构：宽基核心打底、高得分板块卫星增强。",
            "rebalance": p.get("rebalance_rule", "定期再平衡"),
            "confidence": CONFIDENCE,
            "disclaimer": DISCLAIMER,
        }
    if kind == "chat":
        d = data if isinstance(data, dict) else {}
        trend = d.get("trend", {})
        state = trend.get("state", "数据不足")
        until = d.get("data_until", "")
        return {
            "answer": f"根据当前看板分析，市场长期趋势为「{state}」（数据截至 {until}）。"
                      f"如需针对具体标的的新闻与公告解读，请在输入框填入对应股票代码。",
            "references": [],
            "confidence": CONFIDENCE,
            "disclaimer": DISCLAIMER,
        }
    return {"disclaimer": DISCLAIMER, "confidence": CONFIDENCE}
