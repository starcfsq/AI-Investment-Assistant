"""虚拟投资账户：执行推荐、资金曲线、胜率、阶段重置。"""
from datetime import datetime

from core.config import get_env
from core.logging import get_logger
from core.store import Store

logger = get_logger("core.account")

FEE_RATE = 0.0003


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class SimAccount:
    def __init__(self, store: Store, initial_capital: float | None = None):
        self.store = store
        self.initial_capital = initial_capital or float(
            get_env("ACCOUNT_INITIAL_CAPITAL", "1000000")
        )

    def ensure_initialized(self) -> None:
        acc = self.store.get_account()
        if acc["cash"] == 0.0 and acc["period_start"] is None:
            self.store.save_account({
                "cash": self.initial_capital,
                "initial_capital": self.initial_capital,
                "period_start": _today(),
                "updated_at": _now(),
            })

    def current_period_id(self) -> str:
        acc = self.store.get_account()
        return (acc.get("period_start") or _today())[:7]

    def execute(self, portfolio: dict, prices: dict[str, float]) -> list[dict]:
        self.ensure_initialized()
        acc = self.store.get_account()
        cash = acc["cash"]
        trades = []
        targets = {}
        if portfolio and portfolio.get("core"):
            targets[portfolio["core"]["name"]] = portfolio["core"]["weight"]
        for sat in portfolio.get("satellite", []) if portfolio else []:
            targets[sat["name"]] = sat.get("weight", 0.0)
        # 解析 symbol
        target_by_symbol = {}
        for name, weight in targets.items():
            sym = _symbol_from(name)
            if sym:
                target_by_symbol[sym] = weight

        existing = {p["symbol"]: p for p in self.store.list_positions()}
        now = _now()
        # 卖出不在目标中的持仓
        for sym, pos in existing.items():
            if sym not in target_by_symbol:
                trades += [self._close(pos, prices.get(sym, pos["cost_price"]), now)]
        # 买入/调仓到目标权重（基于卖出后的现金，卖出所得须可用于买入）
        cash = self.store.get_account()["cash"]
        total_weight = sum(target_by_symbol.values()) or 1.0
        remaining = cash
        for sym, weight in target_by_symbol.items():
            price = prices.get(sym)
            if not price:
                continue
            target_value = cash * (weight / total_weight)
            qty_target = target_value / price
            current = existing.get(sym)
            cur_qty = current["qty"] if current else 0.0
            diff = qty_target - cur_qty
            if diff > 0.01:
                cost = diff * price
                fee = cost * FEE_RATE
                if cost + fee > remaining:
                    diff = (remaining - fee) / price
                    cost = diff * price
                    fee = cost * FEE_RATE
                self.store.save_position({
                    "symbol": sym, "name": _name_from(sym),
                    "qty": cur_qty + diff,
                    "cost_price": price, "updated_at": now,
                })
                self.store.insert_trade({
                    "time": now, "symbol": sym, "name": _name_from(sym),
                    "side": "buy", "price": round(price, 4), "qty": round(diff, 2),
                    "fee": round(fee, 2), "pnl": None, "status": "open",
                })
                remaining -= cost + fee
                trades.append({"side": "buy", "symbol": sym, "qty": round(diff, 2)})
        new_cash = max(0.0, remaining)
        self.store.save_account({**acc, "cash": new_cash, "updated_at": now})
        return trades

    def _close(self, pos: dict, price: float, now: str):
        qty = pos["qty"]
        proceeds = qty * price
        fee = proceeds * FEE_RATE
        pnl = proceeds - fee - qty * pos["cost_price"]
        self.store.insert_trade({
            "time": now, "symbol": pos["symbol"], "name": pos["name"],
            "side": "sell", "price": round(price, 4), "qty": round(qty, 2),
            "fee": round(fee, 2), "pnl": round(pnl, 2), "status": "closed",
        })
        self.store.delete_position(pos["symbol"])
        acc = self.store.get_account()
        self.store.save_account({**acc, "cash": acc["cash"] + proceeds - fee,
                                 "updated_at": now})
        return {"side": "sell", "symbol": pos["symbol"], "qty": round(qty, 2)}

    def snapshot(self, prices: dict[str, float], benchmark_return: float | None = None) -> None:
        acc = self.store.get_account()
        holdings_value = 0.0
        for pos in self.store.list_positions():
            holdings_value += pos["qty"] * prices.get(pos["symbol"], pos["cost_price"])
        nav = acc["cash"] + holdings_value
        self.store.insert_snapshot({
            "period_id": self.current_period_id(),
            "date": _today(), "nav": round(nav, 2),
            "cash": round(acc["cash"], 2), "holdings_value": round(holdings_value, 2),
        })

    def period_stats(self) -> dict:
        acc = self.store.get_account()
        period_id = self.current_period_id()
        snapshots = self.store.list_snapshots(period_id)
        nav = acc["cash"]
        for pos in self.store.list_positions():
            nav += pos["qty"] * pos["cost_price"]
        curve = [{"date": s["date"], "nav": s["nav"]} for s in snapshots]
        trades = [t for t in self.store.list_trades() if t["status"] == "closed"]
        wins = sum(1 for t in trades if (t["pnl"] or 0) > 0)
        win_rate = wins / len(trades) if trades else 0.0
        init = acc["initial_capital"] or self.initial_capital
        return {
            "period_id": period_id,
            "nav": round(nav, 2),
            "cash": round(acc["cash"], 2),
            "holdings_value": round(nav - acc["cash"], 2),
            "initial_capital": init,
            "win_rate": round(win_rate, 3),
            "return_pct": round((nav / init - 1.0) * 100.0, 2) if init else 0.0,
            "curve": curve,
        }

    def maybe_reset_period(self, benchmark_return: float | None = None) -> None:
        acc = self.store.get_account()
        start = acc.get("period_start") or _today()
        if start[:7] == _today()[:7]:
            return
        stats = self.period_stats()
        self.store.insert_period({
            "period_id": start[:7], "start": start, "end": _today(),
            "initial_capital": acc["initial_capital"],
            "final_nav": stats["nav"], "win_rate": stats["win_rate"],
            "return_pct": stats["return_pct"],
            "benchmark_return": benchmark_return or 0.0,
        })
        for pos in self.store.list_positions():
            self.store.delete_position(pos["symbol"])
        self.store.save_account({
            "cash": acc["initial_capital"], "initial_capital": acc["initial_capital"],
            "period_start": _today(), "updated_at": _now(),
        })


def _symbol_from(name: str) -> str | None:
    import re

    m = re.search(r"\((\d{6})\)", name)
    return m.group(1) if m else None


def _name_from(symbol: str) -> str:
    for p in __import__("core.portfolio", fromlist=["ETF_MAP"]).ETF_MAP.values():
        if f"({symbol})" in p:
            return p
    return symbol
