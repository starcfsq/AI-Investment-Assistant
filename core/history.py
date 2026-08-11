"""把虚拟账户历史聚合为对话可用的结构化摘要。

所有数字来自 data/app.db 的真实运行数据（确定性），LLM 只解读、可引用其中数字，
不参与计算。各块设数量上限，防止历史上下文过于臃肿挤占 LLM 注意力。
"""
from core.logging import get_logger

logger = get_logger("core.history")

MAX_POSITIONS = 10
MAX_TRADES = 20
MAX_PERIODS = 5
MAX_ITERS = 5


def build_history_summary(store, account) -> dict:
    """聚合账户状态/持仓/交易/阶段/资金曲线/回测迭代。

    任何失败都返回空 dict（{}），不阻塞对话链路。
    """
    try:
        stats = account.period_stats()
        period_id = account.current_period_id()
        positions_raw = store.list_positions()
        snapshots = store.list_snapshots(period_id)
        return {
            "account": {
                "period_id": period_id,
                "cash": stats["cash"],
                "nav": stats["nav"],
                "initial_capital": stats["initial_capital"],
                "position_count": len(positions_raw),
            },
            "positions": [
                {"symbol": p["symbol"], "name": p["name"], "qty": p["qty"],
                 "cost_price": p["cost_price"]}
                for p in positions_raw[:MAX_POSITIONS]
            ],
            "recent_trades": [
                {"time": t["time"], "symbol": t["symbol"], "name": t["name"],
                 "side": t["side"], "price": t["price"], "qty": t["qty"],
                 "pnl": t["pnl"]}
                for t in store.list_trades()[::-1][:MAX_TRADES]
            ],
            "period": {
                "win_rate": stats["win_rate"],
                "return_pct": stats["return_pct"],
            },
            "periods": [
                {"period_id": p["period_id"], "start": p["start"], "end": p["end"],
                 "win_rate": p["win_rate"], "return_pct": p["return_pct"],
                 "benchmark_return": p["benchmark_return"]}
                for p in store.list_periods()[-MAX_PERIODS:]
            ],
            "curve": _curve_points(snapshots),
            "iters": [
                {"version": i["version"], "run_at": i["run_at"],
                 "win_rate": i["win_rate"], "excess_return": i["excess_return"]}
                for i in store.list_iters()[:MAX_ITERS]
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("历史摘要构建失败，对话降级为无历史: %s", exc)
        return {}


def _curve_points(snapshots: list[dict]) -> dict:
    """资金曲线取 3 个点：首次、最新、最高/最低 nav 极值。"""
    if not snapshots:
        return {"first": None, "latest": None, "max": None, "min": None}
    pts = [{"date": s["date"], "nav": s["nav"]} for s in snapshots]
    return {
        "first": pts[0],
        "latest": pts[-1],
        "max": max(pts, key=lambda x: x["nav"]),
        "min": min(pts, key=lambda x: x["nav"]),
    }
