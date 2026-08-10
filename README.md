# AI 智能投资助手

面向 A 股长期投资者的智能投资分析项目：趋势/板块/选股/组合分析 + RAG 问答 + 回测自我迭代 + AI 虚拟投资账户。

## 功能

- **市场趋势**：MA250 偏离、指数 PE/PB 历史百分位、股债性价比 → 低估/中性/高估状态
- **板块机会**：RS 相对强度 + 资金流 + 动量打分
- **个股筛选**：ROE/成长/估值/股息多因子打分
- **组合配置**：核心 70% + 卫星 30%
- **RAG 问答**：个股新闻/公告检索，回答带来源引用
- **回测迭代**：历史数据网格搜索权重，时间窗分割防过拟合
- **虚拟账户**：AI 亲自投资，展示资金曲线、操作胜率、阶段收益率与阶段重置

## 核心原则

> **数据永不来自 LLM。** 所有数字由确定性分析引擎从 akshare 真实数据计算，
> DeepSeek 只负责解读，禁止编造。AI 是增强层，不是依赖层。

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `.env`：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`
3. 启动：`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`
4. 打开 `http://127.0.0.1:8000/`

## 项目结构

```
core/      确定性分析引擎（数据/趋势/板块/选股/组合/RAG/回测/账户）
ai/        DeepSeek 层（Provider + Schema 约束 + 解读/对话）
api/       FastAPI REST 接口
web/       单页前端（看板 + 聊天）
tests/     pytest 测试
docs/      设计文档与架构说明
```

## 测试与评估

```bash
python -m pytest -v
```

评估体系（黄金场景 + 一致性校验）见 `docs/superpowers/specs/2026-08-10-ai-investment-assistant-design.md` §9。

## 免责声明

本项目为数据分析与技术演示用途，不构成任何投资建议。回测胜率不代表未来收益，投资有风险，入市需谨慎。
