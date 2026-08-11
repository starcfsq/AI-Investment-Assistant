# 历史感知对话（History-Aware Chat）设计文档

- 日期：2026-08-11
- 定位：让 `/api/chat` 对话问答能读取虚拟账户的历史投资信息，结合当前市场分析给出推荐
- 关联：`docs/superpowers/specs/2026-08-10-ai-investment-assistant-design.md`

## 1. 背景与目标

现状：`/api/chat` 只把 `run_analysis()` 的**当前市场分析**（trend/sectors/stocks/portfolio/warnings）作为上下文传给 LLM。系统虽然把历史投资信息持久化在 `data/app.db`（`trades`/`positions`/`snapshots`/`periods`/`iter_history`），但**对话层完全读不到**。用户问"结合我之前的操作，现在该买什么"、"我过去操作胜率如何"，LLM 在 `_GUARD`（"只能引用输入 JSON 中出现的数字，禁止编造"）护栏下拿不到任何历史数据，只能降级回答。

**目标**：把虚拟账户的历史投资信息（持仓、交易、胜率、资金曲线、回测迭代）以结构化摘要注入对话上下文，让 LLM 在给出推荐时能看到用户历史，且不违反 §2 核心原则（数字由确定性引擎从 SQLite 真实数据算出，LLM 只解读）。

## 2. 需求范围

**自动注入全部对话**：每次 `/api/chat` 都自动携带历史摘要，不按意图判断。

历史摘要包含六项信息（均设数量上限防 token 爆炸）：
1. **账户状态**：现金、初始资金、当前净值、持仓数、当前阶段 ID
2. **当前持仓**：符号/名称/数量/成本价（上限 10 条）
3. **最近交易**：时间/标的/方向/价格/盈亏（上限 20 条，倒序）
4. **胜率/阶段**：当前阶段 `win_rate`/`return_pct` + 最近 5 期阶段结算
5. **资金曲线**：取 3 个点——最新快照点、首次快照点、历史最高/最低 nav 极值点
6. **回测迭代**：最近 5 次（版本/胜率/超额收益）

**不做**：
- 不改 `ai/interpret.py` 四个独立解读函数（它们面向板块/选股/组合，不该掺入账户状态）
- 不改前端（`/api/dashboard` 已展示账户与阶段历史）
- 不新增独立复盘接口（YAGNI）
- 不改变 `_GUARD` 护栏语义，仅补充"历史字段同样属于可引用的输入数字"

## 3. 方案对比

| 方案 | 做法 | 结论 |
|---|---|---|
| A. 新增 `core/history.py` + 只改 chat 链路 | 纯函数聚合历史，`chat()` 并入 context，`answer_question` 扩展 prompt | **采纳**：改动面最小、职责清晰、可单测 |
| B. 在 `run_analysis()` 里附历史摘要 | 让所有解读函数也看到历史 | 否决：让板块/选股解读与账户绑定，口径混乱 |
| C. 在 `chat()` 内联聚合 | 不新增模块 | 否决：职责混杂、难测试 |

## 4. 架构与数据流

新增 `core/history.py`：

```text
core/history.py
  build_history_summary(store, account) -> dict
```

- 输入：`Store`、`SimAccount`（复用现有 `period_stats()` 语义）
- 输出：结构化 dict，字段见 §2，全部从 SQLite 读取/聚合，**无 LLM 调用**
- 单条数据全取回再由 Python 聚合（数据量小，无需 SQL 聚合），设上限截断

改动 `api/main.py::chat()`：

```python
analysis = run_analysis(_provider)
history = build_history_summary(_store, _account)
...
out = answer_question(client, req.query, analysis, rag_hits, history)
```

改动 `ai/chat.py::answer_question()`：

```python
def answer_question(client, query, context, rag_hits, history=None):
    # history 并入 user 消息："以下是你的虚拟账户历史（来自真实运行数据）..."
```

数据流：

```text
web/ 对话框
  → POST /api/chat {query, symbol?}
  → run_analysis()         当前市场分析（现状）
  → build_history_summary() 虚拟账户历史摘要（新增）
  → (可选) RAG 检索          个股新闻/公告
  → answer_question()       LLM 结合市场 + 历史 作答
  → 返回 answer/references/confidence/disclaimer
```

## 5. 错误处理与护栏

- **history 构建失败**：`build_history_summary` 内部 try/except，失败返回 `{}`，不阻塞对话（沿现有 `_safe` 降级思路）
- **LLM 失败**：仍走 `answer_question` 现有降级回答
- **护栏**：历史数字全部来自 SQLite 真实运行数据（确定性），LLM 只解读；prompt 明确"历史来自真实运行数据，可引用其中数字，禁止编造历史数字或市场数据"；输出仍强制 `disclaimer` + `confidence`（Schema 不变）

## 6. 测试

新增 `tests/test_history.py`：
- 构造内存 Store + SimAccount，填充 `account`/`positions`/`trades`/`snapshots`/`periods`/`iter_history`
- 验证输出结构：各字段存在、持仓上限 10、交易上限 20、阶段上限 5、迭代上限 5
- 空库返回空摘要（各字段为默认值/空列表），不抛异常
- 超量数据正确截断

`tests/test_e2e.py` 补用例：
- 带 history 的 `answer_question` 调用不抛异常、返回结构完整

**验证**：全量测试 `DEEPSEEK_API_KEY=test .venv/Scripts/python.exe -m pytest -v`，基线 34 + 新增，须全绿。

## 7. 数据量预估

历史摘要文本量：持仓 ≤10 条、交易 ≤20 条、阶段 ≤5 期、迭代 ≤5 次、曲线 3 点。按平均每条约 40-60 token，合计约 2-3k token，与 `ctx_text` 现有 2000 字符截断同量级，可接受。若未来数据膨胀，可在此处收紧上限。
