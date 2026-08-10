"""Mock LLM 客户端测试：验证输出结构符合 schema，且不访问网络。"""
from ai.mock import MockLLMClient
from ai.schemas import CHAT_SCHEMA, PORTFOLIO_SCHEMA, SECTOR_SCHEMA, STOCK_SCHEMA, TREND_SCHEMA


def _trend_message():
    return [{"role": "user", "content": (
        "基于以下趋势指标数据，给出当前市场长期趋势判断：\n"
        '{"state": "低估机会", "detail": {"ma_dev": 0.05, "pe_pct": 20.0, '
        '"pb_pct": 25.0, "bond_equity_pct": 80.0}, "composite": 65.0, "data_until": "2026-08-10"}'
        "\n返回字段：state, points, risk, confidence, disclaimer。")}]


def test_mock_trend_fills_schema():
    client = MockLLMClient()
    out = client.chat_json(_trend_message(), TREND_SCHEMA)
    assert out["state"] == "低估机会"
    assert len(out["points"]) >= 3
    assert "disclaimer" in out and "confidence" in out


def test_mock_all_schemas_return_required_keys():
    client = MockLLMClient()
    for schema in (TREND_SCHEMA, SECTOR_SCHEMA, STOCK_SCHEMA, PORTFOLIO_SCHEMA, CHAT_SCHEMA):
        out = client.chat_json([{"role": "user", "content": "{}"}], schema)
        for key in schema["properties"]:
            assert key in out, f"{key} missing for {schema}"


def test_mock_does_not_fabricate_numbers():
    client = MockLLMClient()
    out = client.chat_json([{"role": "user", "content": '{"state": "高估风险"}'}], TREND_SCHEMA)
    assert out["state"] == "高估风险"  # 转述输入，不编造
