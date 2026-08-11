"""对话编排：结合看板上下文与 RAG 检索结果作答。"""
import json

from ai.interpret import DISCLAIMER, _GUARD
from ai.schemas import CHAT_SCHEMA
from core.logging import get_logger

logger = get_logger("ai.chat")


def answer_question(client, query: str, context: dict,
                    rag_hits: list[dict], history: dict | None = None) -> dict:
    ctx_text = json.dumps(context, ensure_ascii=False)[:2000]
    refs = []
    ref_text = "无检索结果"
    if rag_hits:
        refs = [{"title": h["title"], "date": h["date"], "source": h["source"],
                 "url": h["url"]} for h in rag_hits]
        ref_text = json.dumps(refs, ensure_ascii=False)[:1500]
    hist_text = "无历史数据"
    if history:
        hist_text = json.dumps(history, ensure_ascii=False)[:1500]
    messages = [
        {"role": "system", "content": "你是 A 股智能投资助手。" + _GUARD
                                     + "若引用了新闻/公告，必须在 references 中给出来源。"},
        {"role": "user", "content": (
            f"用户问题：{query}\n\n当前看板分析数据：{ctx_text}\n"
            f"虚拟账户历史（来自真实运行数据，回答涉及用户账户/操作时可引用其中数字，"
            f"禁止编造历史数字或市场数据）：{hist_text}\n"
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
