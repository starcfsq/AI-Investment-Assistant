# AI 须知（Project Onboarding for AI Agents）

> 给新上下文 AI 的快速上手文档。阅读本文件后，你应当能理解、启动、测试并安全地修改本项目。
> 本文件面向**从未接触过本项目的 AI/开发者**。所有路径、命令、函数名均以当前 `main` 分支真实代码为准。
> 写文档时请保持本文与代码同步：改动涉及本节内容时，记得同步更新。

---

## 1. 项目是什么

**一句话定位**：面向 A 股长期投资者的 AI 智能投资分析助手——用确定性引擎从 akshare 真实数据算出全部数字，再由 DeepSeek 把结果翻译成自然语言解读，并带虚拟投资账户、回测自我迭代与 RAG 问答。

**核心功能**：
- **市场趋势**：MA250 偏离 + 指数 PE/PB 历史百分位 + 股债性价比 → 低估/中性/高估状态（`core/trend.py`）
- **板块机会**：RS 相对强度 + 资金流 + 动量打分排序（`core/sector.py`）
- **个股筛选**：ROE/成长/估值/股息多因子打分（`core/stock.py`）
- **组合配置**：核心 70% + 卫星 30%，卫星按板块得分归一化（`core/portfolio.py`）
- **RAG 问答**：个股新闻/公告 → 分块向量化 → 检索 → 回答带来源引用（`core/rag.py` + `ai/chat.py`）
- **回测迭代**：用历史板块数据网格搜索打分权重，时间窗分割防过拟合（`core/backtest.py` + `core/tune.py`）
- **虚拟账户**：AI 用虚拟资金执行自身推荐，展示资金曲线、操作胜率、阶段收益率与阶段重置（`core/account.py`）

---

## 2. 核心设计原则（改动时不可违背）

1. **数据永不来自 LLM。** 所有数字（指数、估值、板块涨幅、财务指标、回测胜率）由 `core/` 确定性引擎从 akshare 真实数据计算，`ai/` 层只把结构化结果翻译为自然语言解读，**禁止编造任何数字**。LLM 的 system 提示词里有硬性护栏（`ai/interpret.py` 的 `_GUARD`："只能引用输入 JSON 中出现的数字，禁止编造任何市场数据、价格、预测或代码"）。改动时不得让 LLM 生成数字或接触原始行情。
2. **AI 是增强层，不是依赖层。** DeepSeek 挂了（超时/调用失败/无 Key），看板和确定性分析必须照常工作。`api/main.py` 的 `_safe()` 捕获 AI 异常并降级为空解读；`ai/chat.py` 返回降级回答。数据层 akshare 失败也有三级降级（见 §8）。**新增 AI 调用不得让请求整体失败。**
3. **指标口径统一**（改任何一处都要保持另一处一致）：
   - 虚拟账户操作胜率 = 已平仓交易中 `pnl > 0` 的笔数 ÷ 已平仓总笔数（未平仓不计入）
   - 回测胜率 = 组合区间收益跑赢同期沪深300 的期数 ÷ 有效期数
   - 阶段收益率 = (阶段末净值 − 阶段初资金) ÷ 阶段初资金
   - 基准 = 沪深300（`core/data.py` `benchmark_index_code()` 返回 `sh000300`）
4. **免责声明与置信度。** 所有 AI 输出（趋势解读/板块推荐/标的报告/组合建议/对话回答）都必须带 `disclaimer` 与 `confidence` 字段（Schema 强制，见 `ai/schemas.py`）。

---

## 3. 项目结构速览

```
web/        单页前端（无构建工具）：index.html / app.js / style.css
api/        FastAPI REST 接口：main.py（5 个接口 + /）
core/       确定性分析引擎（全部数字在此产生）
  data.py       akshare 数据封装 + SQLite 缓存 + 新鲜度/校验（DataProvider）
  trend.py      大盘趋势：analyze_trend()
  sector.py     板块打分：score_sectors()
  stock.py      选股打分：rank_stocks()
  portfolio.py  核心+卫星配置：build_portfolio()，含 ETF_MAP（板块→可交易 ETF 名）
  rag.py        新闻/公告 RAG：build_index() / retrieve()
  backtest.py   历史回测：backtest_sectors()
  tune.py       权重网格搜索：grid_search_weights() / run_iteration()
  account.py    虚拟账户：SimAccount（execute / snapshot / period_stats / maybe_reset_period）
  analyze.py    串起整条分析链路：run_analysis(provider)
  store.py      SQLite 统一存储：Store 类（缓存/账户/持仓/交易/快照/阶段/迭代历史/新闻/向量块）
  embedding.py  文本向量化：get_embedding_provider()（sentence-transformers，缺则 HashEmbedding 降级）
  config.py     路径与配置：DB_PATH / WEIGHTS_PATH / load_weights() / save_weights() / get_env()
  logging.py    结构化日志
ai/          DeepSeek 层（只解读，不产生数字）
  provider.py   LLMClient 抽象 + get_client()
  deepseek.py   DeepSeekClient 实现（HTTP 调用 + JSON Schema 校验）
  schemas.py    各功能输出 JSON Schema（TREND/SECTOR/STOCK/PORTFOLIO/CHAT_SCHEMA）
  interpret.py  4 个解读函数：interpret_trend / recommend_sectors / recommend_stocks / plan_portfolio
  chat.py       RAG 对话编排：answer_question()
config/      weights.json 打分权重（trend/sector/stock 三组，回测迭代的产物）
tests/       pytest（34 个测试，离线可跑）
data/        SQLite 数据库 app.db（运行时生成，gitignore）
docs/        设计文档（specs）与架构说明（architecture.md）
```

**API 一览**（`api/main.py`）：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 返回 `web/index.html` |
| GET | `/api/dashboard` | 看板聚合数据（分析结果 + AI 趋势解读 + 账户统计 + 迭代历史 + 阶段历史） |
| POST | `/api/analyze` | 触发完整分析链路，并让虚拟账户执行推荐 + 每日净值快照，返回分析结果 + AI 解读 |
| POST | `/api/chat` | 对话问答，`{query, symbol?}`；提供 symbol 时抓新闻/公告做 RAG 检索，回答带引用 |
| GET | `/api/backtest` | 运行一次权重调优迭代，返回最新迭代结果 + 迭代历史 + 当前权重 |
| GET | `/api/account` | 触发阶段重置检查，返回账户统计 + 阶段历史 + 最近 50 笔交易 |

---

## 4. 环境与启动

**前置**：Python 3.11+，本项目在 Windows + Git Bash 下开发（venv 位于 `.venv/`）。

```bash
# 1. 激活虚拟环境（Windows Git Bash）
source .venv/Scripts/activate
# 或全程用 .venv/Scripts/python.exe 代替 python

# 2. 安装依赖
pip install -r requirements.txt
```

依赖说明（`requirements.txt`）：
- `sentence-transformers` **可暂缺**：装不上时 `core/embedding.py` 自动降级为 `HashEmbedding`（确定性哈希向量），RAG 仍可用。
- `chromadb` **已声明但当前代码未使用**（RAG 向量块实际存 SQLite `rag_chunks` 表，见 `core/store.py`）。不要因为"它没被 import"就误判为失效。
- `akshare` 是唯一数据源，**无需密钥**；DeepSeek 才需要 API Key。

**配置**：复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`（无 Key 时看板仍能跑，只是 AI 解读降级）。

```bash
cp .env.example .env   # 然后编辑 .env
```

`core/config.py` 在 import 时会自动 `load_dotenv()`，无需手动加载。可用环境变量：
- `DEEPSEEK_API_KEY`：DeepSeek 密钥（必填，否则 `get_client()` 抛 `ValueError`；测试时用 `test` 占位即可）
- `DEEPSEEK_MODEL`：DeepSeek 模型名，默认 `deepseek-chat`（`get_client()` 会读取）
- `ACCOUNT_INITIAL_CAPITAL`：虚拟账户初始资金，默认 1000000

**启动**：

```bash
python -m uvicorn api.main:app --port 8000
# 打开 http://127.0.0.1:8000/
```

前端无构建工具：直接改 `web/app.js` / `web/index.html` / `web/style.css`，刷新浏览器即可。

---

## 5. 测试

```bash
# 全部测试（共 34 个）
DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest -v
```

要点：
- **数据层测试不依赖真实网络**：`tests/conftest.py` 在收集前 `os.environ.setdefault("DEEPSEEK_API_KEY", "test")`；`core/data.py` 是缓存优先 + 网络失败返回空数据（`status=missing`），`test_e2e.py` 明确验证"离线全链路不抛异常"。
- 测试覆盖：趋势/板块/选股/组合/RAG/回测/权重调优/账户/存储/DeepSeek 客户端/API 端点/端到端。
- **新增功能必须补测试**：改 `core/` 逻辑 → 在 `tests/test_xxx.py` 加用例（构造小型 `pd.DataFrame` 直接喂纯函数即可，不需要真实网络）。
- 提交前跑一次全量测试，必须全绿（当前基线 34 passed）。

---

## 6. 常见改动场景与步骤

**① 修改打分权重（手工）**
- 直接编辑 `config/weights.json`。三组权重各自归一化到 1.0：`trend`（ma/valuation/bond）、`sector`（rs/flow/momentum）、`stock`（roe/growth/valuation/dividend）。
- `test_e2e.py::test_weights_config_parseable` 会校验三组权重之和为 1.0，别改成不归一。

**② 用回测自动调权（推荐，别手改）**
- 权重是 `core/tune.py::run_iteration` 的产物：对 `sector.rs/flow/momentum` 在 `{0.5, 0.7, 0.9}` 网格搜索，前 60% 时间窗调参、后 40% 验证，仅当验证窗胜率提升超过 `1e-9` 才 `save_weights()`。
- 手动跑一次：启动服务后点前端"运行回测迭代"按钮，或 `curl http://127.0.0.1:8000/api/backtest`。每次迭代写 `iter_history` 表（可看历史版本对比）。

**③ 新增一个分析指标**
1. 在对应 `core/xxx.py` 里实现纯函数（输入 DataFrame/dict，输出数字，不做 LLM 调用）。
2. 在 `tests/test_xxx.py` 补测试（构造小型数据验证口径）。
3. 在 `ai/interpret.py` 的对应提示词中把新字段写进 user 消息，并保持 `ai/schemas.py` 的 Schema 字段同步。
4. 若新指标影响前端展示，在 `web/app.js` 对应区域渲染。
5. 跑全量测试。

**④ 换 LLM 模型 / 供应商**
- 供应商：`ai/provider.py::get_client()` 返回 `DeepSeekClient`，切换供应商就是换这里的实现（实现 `LLMClient` 抽象即可，接口只有 `chat_json(messages, schema)`）。
- 模型名：在 `.env` 里设 `DEEPSEEK_MODEL`（`get_client()` 已读取，默认 `deepseek-chat`），或改 `ai/deepseek.py` 的 `DEFAULT_MODEL`。

**⑤ 前端改动**
- `web/app.js` 无构建、无框架，原生 JS + `fetch`。改完刷新浏览器即可。接口契约见 §3 的 API 表。

**⑥ 想清空运行状态（重置虚拟账户/缓存/回测历史）**
- 删除 `data/app.db`（服务停止时）再重启，所有表会重建。见 §7。

---

## 7. 常见坑与注意（改代码前必读）

1. **akshare 接口名随版本变化**：`core/data.py` 里硬编码了 12 个 akshare 接口（`stock_zh_index_daily`、`stock_index_pe_lg`、`stock_board_industry_name_em`、`stock_sector_fund_flow_rank`、`stock_board_industry_hist_em`、`stock_zh_a_spot_em`、`fund_etf_spot_em`、`stock_zh_a_hist`、`stock_a_indicator_lg`、`bond_china_yield`、`stock_news_em`、`stock_notice_report`）。升级 akshare 或报"接口不存在"时，**先用 `dir(ak)` 确认新接口名/新列名，再改代码**，别硬编码已废弃的接口。每个接口的返回列名也做了 `.rename(...)`，新版本列名变了会静默返回空数据（`status=missing`），此时看板会显示数据不足，不是崩溃。
2. **`.env` 不入库；`data/` 不入库**（`.gitignore` 已覆盖）。**绝不要把 `DEEPSEEK_API_KEY` 提交进 git**，也绝不提交 `data/app.db`。
3. **`data/app.db` 是运行时状态的唯一来源**，包含：数据缓存 `cache`、虚拟账户 `account`、持仓 `positions`、交易 `trades`、净值快照 `snapshots`、阶段历史 `periods`、回测迭代历史 `iter_history`、新闻 `news`、RAG 向量块 `rag_chunks`。**删除 `data/app.db` 即重置全部运行状态**（账户归零、缓存清空、迭代历史消失）。
4. **虚拟账户当前净值用成本价，快照曲线用市价**：`SimAccount.period_stats()` 里 `nav = cash + Σ qty × cost_price`（持仓按**成本价**估值）；而 `snapshot()` 里 `holdings_value` 用 `prices.get(symbol, cost_price)`（**优先市价**，缺价才回落成本价）。因此"当前净值"和"资金曲线最新点"口径不同，是**有意设计**，别"顺手改统一"。改它会破坏 `tests/test_account.py` 的回归断言。
5. **RAG 索引只保留最近一次查询的标的**：`core/rag.py::build_index()` 开头调用 `store.clear_chunks()` 清空旧块再写入——它只保存**最近查询的那个 symbol** 的新闻/公告。这是简化设计，不是 bug。若要支持多标的并行检索需改造存储结构。
6. **Windows Git Bash 里 curl 内联中文 JSON 会 400**：`curl -X POST ... -d '{"query":"最近半导体板块发生了什么"}'` 在 Git Bash 下常返回 `400 {"detail":"There was an error parsing the body"}`，是 shell 编码问题不是服务 bug。**改用 `curl --data @body.json`（JSON 存文件）或 Python/HTTP 客户端**。
7. **portfolio 卫星项 `name` 是可交易 ETF 字符串**（含 6 位代码），如 `"半导体ETF(512480)"`，来自 `core/portfolio.py` 的 `ETF_MAP`。虚拟账户用它解析出 6 位 symbol 去取价、下单（Task 7 契约）。**别把 `name` 改成纯中文板块名**，否则账户无法取价/下单。
8. **`DEEPSEEK_MODEL` 已接线**：`.env` 里设 `DEEPSEEK_MODEL` 即可换模型（默认 `deepseek-chat`）；`DeepSeekClient` 对 HTTP 错误或非 JSON 输出会重试一次再抛错（调用方 `_safe` 兜底降级）。
9. **改动影响判定**：任何改动上线前先确认不违背 §2 的四条核心原则；改 `core/` 数学逻辑后跑全量测试，改 `ai/` 提示词/Schema 后至少跑 `test_deepseek.py` 与 `test_e2e.py`。
10. **提交前跑全量测试**（§5），不要只跑自己改的用例。

---

## 8. 数据流与降级机制（理解用）

**完整分析链路 `POST /api/analyze`**：
`web/` → `api/main.py` → `core/analyze.py::run_analysis(provider)` → 依次跑 趋势 → 板块 → 选股 → 组合 → 汇总 `{trend, sectors, stocks, portfolio, data_until, data_quality, warnings}` → `api/main.py` 让虚拟账户 `execute()` + `snapshot()` → 把结构化结果交给 `ai/interpret.py` 的 4 个解读函数 → 返回分析结果 + AI 解读 + 账户统计。

**对话链路 `POST /api/chat`**：`api/main.py` → `run_analysis()` 带看板上下文 → （可选）按 `symbol` 抓新闻/公告 → `build_index()` + `retrieve(top_k=5)` → `ai/chat.py::answer_question()` → 返回 `answer + references + confidence + disclaimer`。

**三层降级**（`core/data.py::_cached` + `api/main.py::_safe` + `core/analyze.py::_sufficiency_warnings`）：
| 故障 | 处理 |
|---|---|
| akshare 网络/接口失败 | 缓存优先：未过期缓存直接返回（`status=cached`，不再发网络请求）；缓存未命中则抓取，失败重试 1 次（`status=ok_retry`）；仍失败返回空数据并标记 `status=missing`，该项跳过 |
| DeepSeek 超时/调用失败 | `api/main.py::_safe()` 捕获，跳过 AI 解读返回 `{}`；`ai/chat.py` 返回固定降级回答；看板照常渲染确定性结果 |
| 数据缺失/样本不足 | 跳过该条，其余正常；在 `warnings` 中显式标注（"指数 MA250 样本不足"、"板块覆盖不足"、"选股候选不足"），AI 解读据此降低置信度 |

**数据质量保证**（`core/data.py`）：
- **实时性**：每类数据按 TTL 缓存（`core/data.py` 各方法里 `3600 * n` 秒：指数日线 6h、估值 12h、板块行情/资金流 4h、板块历史 6h、个股/ETF 行情 1h、个股历史 6h、财务 12h、国债 12h、新闻/公告 3h），每次拉取记录 `fetched_at` 与 `data_until`（内容截止日），汇总为 `quality_report()` 随分析结果返回；前端与 AI 输出都展示"数据截至 <时间>"。
- **有效性**：`_validate_df()` 剔除坏值（价格为负、涨跌幅超 ±50%、PE 超 0~500、PB 超 0~100），按日期去重升序。单条坏数据不影响整体。
- **充分性**：`core/analyze.py::_sufficiency_warnings()` 对最小数据量做显式告警而非静默出错。

**回测口径**（`core/backtest.py` + `core/tune.py`）：以沪深300 为基准，`lookahead_days=63`（约一季度）滑动重放，决策时刻只用当时可得数据（防前视偏差）；胜率=跑赢基准期数/有效期数；网格搜索 + 60/40 时间窗分割防过拟合，收敛阈值 `> 1e-9`。

---

## 免责声明

本项目为数据分析与技术演示用途，不构成任何投资建议。回测胜率不代表未来收益，虚拟账户为模拟执行不含真实下单，投资有风险，入市需谨慎。
