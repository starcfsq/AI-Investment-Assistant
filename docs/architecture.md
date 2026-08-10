# 架构说明

本文件描述 AI 智能投资助手的系统架构，包括分层结构、数据流时序、错误降级策略与回测/账户指标口径。内容对应设计文档 `docs/superpowers/specs/2026-08-10-ai-investment-assistant-design.md` §5/§7/§8/§10。

## 1. 分层架构

系统分为 `web / api / core / ai` 四层。核心设计原则：**数据永不来自 LLM** —— 所有数字由确定性分析引擎（`core/`）从 akshare 真实数据计算，DeepSeek（`ai/`）只负责把结构化结果翻译为自然语言解读，是增强层而非依赖层。

```mermaid
flowchart TB
    subgraph WEB["web/ 前端（单页 HTML + JS + CSS）"]
        W_DASH["看板区<br/>趋势状态卡片 · 板块排名表 · 标的推荐表 · 配置建议"]
        W_CHAT["聊天区<br/>回答渲染 · 来源引用 · 置信度标记"]
        W_BT["回测胜率区<br/>胜率曲线 · 迭代历史 · 当前权重版本"]
        W_ACC["投资账户区<br/>当前净值 · 现金 · 持仓表 · 资金曲线 · 阶段历史"]
    end

    subgraph API["api/ FastAPI REST"]
        A_DASH["GET /api/dashboard<br/>看板聚合数据"]
        A_AN["POST /api/analyze<br/>完整分析链路 + AI 解读"]
        A_CHAT["POST /api/chat<br/>对话问答（上下文 + RAG 检索）"]
        A_BT["GET /api/backtest<br/>回测迭代 + 权重"]
        A_ACC["GET /api/account<br/>账户状态 + 资金曲线 + 阶段"]
    end

    subgraph CORE["core/ 确定性分析引擎"]
        C_DATA["data/ akshare 抓取 + SQLite 缓存 + 新鲜度/校验"]
        C_TREND["trend/ MA250 偏离 · PE/PB 百分位 · 股债性价比"]
        C_SECTOR["sector/ RS 相对强度 · 资金流 · 动量"]
        C_STOCK["stock/ ROE · 成长 · 估值 · 股息 多因子"]
        C_PORT["portfolio/ 核心70% + 卫星30% 配比"]
        C_RAG["rag/ 新闻/公告 → 分块 → 向量化 → 检索"]
        C_BT["backtest/ 回测引擎 · 胜率统计 · 权重调优迭代"]
        C_ACC["account/ 虚拟账户 · 交易执行 · 胜率 · 阶段重置"]
        C_STORE["store/ SQLite 统一存储（缓存/账户/交易/快照/迭代/向量块）"]
    end

    subgraph AI["ai/ DeepSeek 层"]
        AI_PROV["provider/ LLMClient 抽象（DeepSeek 实现）"]
        AI_INT["interpret/ 趋势解读 · 板块推荐 · 标的报告 · 规划建议"]
        AI_CHAT["chat/ 对话编排（上下文 + RAG 检索 + 引用）"]
        AI_SCHEMA["schemas/ JSON Schema 约束与校验"]
    end

    WEB -->|"REST API"| API
    API --> CORE
    CORE --> C_STORE
    CORE -->|"结构化分析结果 JSON"| AI
    CORE -->|"RAG 检索片段"| AI
    AI -->|"解读/回答 + 置信度 + 免责声明"| API
```

### 各层职责

| 层 | 职责 | 关键文件 |
|---|---|---|
| `web/` | 单页前端：看板 + 聊天 + 回测胜率 + 投资账户 | `web/index.html` `web/app.js` `web/style.css` |
| `api/` | FastAPI REST 接口，编排分析、账户执行与 AI 解读；AI 失败时降级返回纯确定性结果 | `api/main.py` |
| `core/` | 确定性分析引擎 + 数据层 + 存储 | `core/data.py` `core/trend.py` `core/sector.py` `core/stock.py` `core/portfolio.py` `core/rag.py` `core/backtest.py` `core/tune.py` `core/account.py` `core/analyze.py` `core/store.py` |
| `ai/` | DeepSeek 层：Provider 抽象、JSON Schema 约束、解读/对话 | `ai/provider.py` `ai/deepseek.py` `ai/interpret.py` `ai/chat.py` `ai/schemas.py` |
| `config/` | 权重配置（网格搜索/调优的产物，随迭代版本更新） | `config/weights.json` |
| `tests/` | pytest 单元测试 + 端到端 | `tests/*.py` |

## 2. 数据流时序

### 2.1 完整分析 `POST /api/analyze`

```mermaid
sequenceDiagram
    participant U as 前端 web/
    participant A as api/ FastAPI
    participant C as core/ 分析引擎
    participant R as core/rag
    participant L as ai/ DeepSeek

    U->>A: POST /api/analyze
    A->>C: run_analysis(provider)
    C->>C: 数据层抓取/读缓存（akshare → SQLite）
    C->>C: trend 趋势指标 + 状态（低估/中性/高估）
    C->>C: sector 板块打分排序
    C->>C: stock 标的筛选打分
    C->>C: portfolio 核心+卫星配置
    C-->>A: 结构化分析结果 JSON（含 data_until / data_quality / warnings）
    A->>A: 账户执行推荐 + 每日净值快照
    A->>L: interpret_trend / recommend_sectors / recommend_stocks / plan_portfolio
    L-->>A: 自然语言解读（置信度 + 免责声明）
    A-->>U: 分析结果 + AI 解读 + 账户统计 + data_until
```

### 2.2 对话问答 `POST /api/chat`

```mermaid
sequenceDiagram
    participant U as 前端 web/
    participant A as api/ FastAPI
    participant C as core/ 分析引擎
    participant R as core/rag
    participant L as ai/ DeepSeek

    U->>A: POST /api/chat {query, symbol?}
    A->>C: run_analysis(provider)（携带看板上下文）
    alt 提供了 symbol
        A->>C: 抓取该标的新闻/公告（stock_news + stock_notices）
        A->>R: build_index → retrieve(query, top_k=5)
        R-->>A: 相关片段（标题/日期/来源元数据）
    end
    A->>L: answer_question(query, analysis, rag_hits)
    L-->>A: answer + references + confidence + disclaimer
    A-->>U: 回答 + 引用 + 置信度 + data_until
```

## 3. 错误降级（三层策略）

原则：**AI 是增强层，不是依赖层。** 任何一层失败都不应使整个请求失败。

| 故障 | 处理 |
|---|---|
| **数据层：akshare 网络/接口失败** | 重试 1 次 → 仍失败则读 SQLite 缓存旧数据（标注数据时点与"缓存命中"）→ 仍不可用则返回空数据并记录 `status=missing`，该项跳过 |
| **AI 层：DeepSeek 超时/调用失败** | `_safe()` 捕获异常并降级：跳过 AI 解读（`{}`），返回纯确定性分析结果，看板正常渲染 |
| **业务层：数据缺失/样本不足** | 跳过该条数据，其余正常返回；在 `warnings` 中显式标注（如"指数 MA250 样本不足"、"板块覆盖不足"、"选股候选不足"），AI 解读据此降低置信度 |

数据层每类数据记录 `fetched_at`（抓取时刻）、`data_until`（数据截止日）、`status`（`ok` / `cached` / `ok_retry` / `missing`）与 `ttl_seconds`，汇总为 `quality_report()` 随分析结果返回；所有分析输出与看板展示 `data_until`，前端标注"数据截至 <时间>"。

## 4. 回测与账户指标口径

### 4.1 回测指标（`core/backtest.py` + `core/tune.py`）

回测引擎以沪深300 为基准，按 `lookahead_days=63`（约一个季度）滑动重放：在每个决策时刻 `T` 仅使用 `T` 之前可得的历史板块数据计算打分，选出得分最高的 3 个板块组成等权组合，评估其在 `T+63` 日内的实际收益。

| 指标 | 口径 |
|---|---|
| **胜率 `win_rate`** | `组合区间收益 > 同期沪深300 区间收益` 的期数 ÷ 有效期数（`n_samples`）。胜场数记录为 `wins` |
| **平均超额收益 `excess_return`** | 各期 `(组合区间收益 − 基准区间收益)` 的算术平均 |
| **权重调优** | `tune.grid_search_weights` 对 `sector.rs / sector.flow / sector.momentum` 在 `{0.5, 0.7, 0.9}` 网格搜索，以训练窗胜率为目标；时间窗前 60% 调参、后 40% 验证，验证窗胜率提升才更新 `config/weights.json`（收敛阈值 `> 1e-9`），否则保持原权重 |
| **迭代记录** | 每次 `run_iteration` 写 `iter_history`：版本号、运行时间、权重、回测窗口、胜率、超额收益、数据截止日 |

防前视偏差：决策时刻只用当时可得数据；训练/验证时间窗分割；IC 监控因子有效性；权重更新设收敛阈值（变化过小则不更新）。

### 4.2 虚拟账户指标（`core/account.py`）

账户以虚拟资金（默认 100 万，`ACCOUNT_INITIAL_CAPITAL` 可配）执行自身推荐：每次 `analyze` 按推荐组合权重生成买卖指令，含简化交易成本（佣金费率 `FEE_RATE=0.0003`），持仓按最新行情更新市值，并记录每日净值快照。

| 指标 | 口径 |
|---|---|
| **当前净值 `nav`** | `现金 + Σ(持仓数量 × 最新价)`；最新价缺失时以成本价兜底 |
| **操作胜率 `win_rate`** | 已平仓交易中 `pnl > 0` 的笔数 ÷ 已平仓总笔数；未平仓不计入 |
| **阶段收益率 `return_pct`** | `(阶段末净值 − 阶段初资金) ÷ 阶段初资金`，同阶段展示沪深300 涨跌幅作基准（`benchmark_return`） |
| **阶段重置** | 默认每月为一个阶段；跨月时归档阶段历史（胜率、资金变化率、曲线、基准），清空持仓，资金重置为初始值 |

与回测的关系：`backtest` 为历史离线回测（评估打分权重、驱动调优），`account` 为当前在线模拟盘（执行推荐、跟踪实际绩效），两者指标口径统一（胜率、相对基准），互为补充。

## 5. 日志与可观测性

- **结构化日志**：`core/logging.py` 统一格式 `时间 | 级别 | 模块 | 消息`，`api.main` 对 AI 解读失败降级打 `warning`、对迭代失败打 `error`。
- **数据溯源**：数据层记录每个数据源的新鲜度/状态；分析结果带 `generated_at`、`data_until` 与 `data_quality` 报告。
- **前端标注**：看板与 AI 输出展示"数据截至 <时间>"，AI 解读在数据不足时降低置信度并说明。

## 6. 免责声明

本项目为数据分析与技术演示用途，不构成任何投资建议。回测胜率不代表未来收益，虚拟账户为模拟执行不含真实下单，投资有风险，入市需谨慎。
