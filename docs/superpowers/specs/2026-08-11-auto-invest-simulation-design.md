# AI 自动投资 + 一年模拟回测 + 收益曲线 设计文档

- 日期：2026-08-11
- 定位：把虚拟账户的"手动点运行分析"改为 AI 自动投资；提供过去一年按系统推荐方法投资的模拟操作与收益曲线
- 关联：`docs/superpowers/specs/2026-08-10-ai-investment-assistant-design.md`、`docs/superpowers/specs/2026-08-11-history-aware-chat-design.md`

## 1. 背景与目标

现状：虚拟账户仅在用户手动调用 `POST /api/analyze` 时执行推荐。用户希望：
1. **账户权限交给 AI**——系统自动分析并按推荐投资，无需手动干预
2. **一年历史模拟**——用过去一年的真实历史数据，按系统推荐方法（趋势/板块/组合）模拟投资，给出详细操作与收益曲线
3. **前端展示**——收益曲线（净值 vs 基准）+ 详细操作表

**目标**：系统内嵌定时调度自动投资；新增 `core/simulation.py` 一年模拟引擎；新增 `GET /api/simulation`；前端 Canvas 展示收益曲线与操作表。全部数字由确定性引擎从历史数据计算，LLM 只解读，不参与决策（§2 核心原则）。

## 2. 需求范围

**做**：
- AI 自动投资：启动执行一次 + 每个交易日收盘后自动执行（复用 `/api/analyze` 链路）
- 一年模拟：月度调仓，每月末用截至当时的历史数据重跑系统推荐并模拟调仓
- 收益曲线 + 详细操作表前端展示（Canvas 手绘，零依赖）
- `GET /api/simulation` 端点（带缓存）

**不做**（YAGNI）：
- 不做真实下单/券商对接（虚拟账户仍是模拟）
- 不做节假日日历（交易日简化为周一至周五，已知限制）
- 不做个股选股进组合执行（组合仍是核心宽基 ETF + 卫星板块 ETF，与现状一致）
- 不引入前端构建工具/图表库（保持无构建、离线优先）

## 3. ① AI 自动投资

**实现**：`api/main.py` 用 FastAPI **lifespan** 启动后台 `asyncio` 任务 `AutoInvestWorker`。

**调度**：
- 启动时立即执行一次（用户打开看板即可看到 AI 已自动操作）
- 之后每个交易日（周一至周五）15:30 自动执行一次
- 通过 `core/config.py::get_env` 配置：`AUTO_INVEST_ENABLED`(默认 `1`)、`AUTO_INVEST_TIME`(默认 `15:30`)

**执行体**（复用现有 analyze 端点逻辑，抽为可复用函数）：
```
run_analysis(_provider) → _prices_for_portfolio(portfolio) → _account.execute(portfolio, prices) → _account.snapshot(prices)
```
**并发防护**：模块级 `asyncio.Lock`，自动任务与手动 `POST /api/analyze` 互斥，防重入。
**失败降级**：自动执行任何异常 → `logger.error` 记录，不中断服务；下次定时点重试。
**幂等**：同一交易日重复触发时，`SimAccount.execute` 按目标权重调仓（已有逻辑），不会重复买入同一仓位。

## 4. ② 一年模拟引擎 `core/simulation.py`（新增）

**入口**：`run_year_simulation(provider, store, lookback_days: int = 365) -> dict`

**返回结构**：
```python
{
  "generated_at": str,
  "start": str, "end": str,            # 模拟区间
  "stats": {"total_return": float, "annualized": float,
            "win_rate": float, "excess_return": float, "max_drawdown": float,
            "n_trades": int, "benchmark_return": float},
  "curve": [{"date": str, "nav": float, "benchmark": float}],  # 净值 vs 基准(归一化)
  "trades": [{"date": str, "side": str, "name": str, "symbol": str,
              "price": float, "qty": float, "pnl": float|None}],  # 详细操作
  "rebalances": [{"date": str, "weights": {name: weight}}],      # 每次调仓的目标权重
}
```

**数据加载与历史视图**（防前视关键）：
- 加载过去 `lookback_days` 的**全量**历史到内存（一次性），构建 `HistoryProvider`——提供 `snapshot_at(date)` 返回"截至 date 可用"的数据切片：
  - 指数日线（沪深300，基准 + 趋势 MA）
  - 指数估值 PE/PB 历史（乐咕，趋势估值信号）
  - 国债收益率历史（趋势债券信号）
  - 板块历史 K 线（同花顺 `stock_board_industry_index_ths`，90 板块 × 全区间，板块 RS/动量）
- **板块资金流（flow）在历史时点不可得**（`summary_ths` 只有当前值）——模拟打分退化为 RS + 动量两因子：把 sector 权重 `{rs:0.4, flow:0.3, momentum:0.3}` 去掉 flow 后归一化为 `{rs: 0.4/0.7, momentum: 0.3/0.7}` 使用，并在输出中标注 `flow_not_available: true`。这是"历史时点确实无此数据"的诚实降级，符合防前视。
- 持仓取价与每日净值用 ETF 历史收盘价（`fund_etf_hist_em`，portfol 卫星项均为 ETF）。

**分析复用**：把 `core/analyze.py` 的"单次完整分析"抽为纯函数 `analyze_at(snapshot, weights)`（输入历史切片 + 权重，输出趋势/板块/组合），现有 `run_analysis(provider)` 与模拟引擎共用；模拟在每月末用 `snapshot_at(t)` 调用它，保证与实时分析同一套推荐方法。

**调仓与执行**：
- 按自然月取调仓点（每月最后一个交易日），首个调仓点需有 ≥ 30 个交易日历史（防样本不足）
- 每月末 `t`：用 `snapshot_at(t)` 跑趋势/板块/组合（复用 `core/analyze.py` 的纯函数，传入历史切片），得到目标组合
- 用 `SimAccount.execute(portfolio, prices_at_t)` 模拟调仓（复用现有 execute 逻辑，含费用/加权成本）
- **每日净值**：区间内每个交易日，`nav = cash + Σ qty × close_at(d)`，基准同日归一化到 1.0

**性能**：板块历史与 ETF 历史加载较重，结果缓存到 Store（`simulation_result` 键，TTL 24h），API 返回缓存；首次计算预计 30s–2min，前端加载提示。

## 5. ③ API 端点

`GET /api/simulation` → 返回 `run_year_simulation` 结果（缓存命中直接返回；未命中计算后缓存）。参数：`?days=365`（可选，默认 365）。

## 6. ④ 前端展示

`web/index.html` + `web/app.js` + `web/style.css`（无构建、原生 JS）：
- 看板新增"**一年模拟**"区块（按钮触发加载 + 自动加载缓存）
- **Canvas 手绘折线图**：X 轴日期、Y 轴净值；两条线（模拟净值、沪深300 基准），图例/坐标标注
- **统计卡片**：总收益、年化、胜率、超额收益、最大回撤、交易笔数
- **详细操作表**：日期/方向/名称/代码/价格/数量/盈亏
- **调仓记录**：每次调仓的目标权重表

## 7. 错误处理与护栏

- 模拟引擎任何异常 → 返回 `{"error": ...}`，看板提示"模拟数据暂不可用"，不影响其他功能
- 历史数据不足（某板块 < 30 交易日）→ 该板块跳过，其余正常；`stats` 中标注样本说明
- 自动投资失败 → 记录日志，下次定时重试
- 所有输出数字来自真实历史数据计算，无 LLM 参与决策；前端展示不涉及推荐，无需额外免责声明字段，但保留页面底部既有免责声明

## 8. 测试

新增 `tests/test_simulation.py`：
- 历史视图切片：`snapshot_at(t)` 只含 `date <= t`（防前视）
- 月度调仓点生成：自然月最后一个交易日
- 完整模拟：构造小型历史 DataFrame（指数/板块/ETF），跑 `run_year_simulation`，验证返回结构、净值单调计算、操作记录
- 数据不足：板块历史过短 → 跳过不崩溃

`tests/test_api.py` 补：`GET /api/simulation` 返回结构（monkeypatch 引擎避免真实计算耗时）。

**验证**：全量测试须全绿（当前基线 69）。

## 9. 数据量预估

模拟一次：指数/估值/国债/90 板块 K 线/核心+卫星 ETF 历史 ≈ 数百 MB 内存级（DataFrame 常驻），可接受；计算结果缓存于 Store，避免重复计算。
