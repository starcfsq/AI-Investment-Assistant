"""权重网格搜索与迭代流程。"""
import itertools
from datetime import datetime

from core.backtest import backtest_sectors
from core.config import load_weights, save_weights
from core.logging import get_logger

logger = get_logger("core.tune")

SECTOR_SEARCH = {
    "keys": ["sector.rs", "sector.flow", "sector.momentum"],
    "steps": [0.5, 0.7, 0.9],
}


def grid_search_weights(score_fn, data, base_weights: dict,
                        search_keys: list[str], steps: list[float]):
    """对 search_keys（形如 'sector.rs'）做网格搜索，返回 (最优权重, 最优指标)。"""
    best = base_weights
    best_score = score_fn(data, best)["win_rate"]
    for combo in itertools.product(steps, repeat=len(search_keys)):
        cand = _deep_copy(base_weights)
        for key, val in zip(search_keys, combo):
            _set_path(cand, key.split("."), val)
        score = score_fn(data, cand)["win_rate"]
        if score > best_score:
            best, best_score = cand, score
    return best, best_score


def run_iteration(provider, store) -> dict:
    weights = load_weights()
    sector_hist, bench = _collect_history(provider)
    if not sector_hist:
        return {"status": "no_data", "reason": "板块历史数据不足"}
    if bench is None or len(bench) < 60:
        return {"status": "no_data", "reason": "基准数据不足"}

    # 按时间排序，前 60% 调参，后 40% 验证
    sorted_hist, sorted_bench = _split_train_test(sector_hist, bench, 0.6)
    train_hist, test_hist = sorted_hist
    train_bench, test_bench = sorted_bench
    if not train_hist:
        return {"status": "no_data", "reason": "训练窗口数据不足"}

    def train_score(_, w):
        return backtest_sectors(train_hist, train_bench, w)

    best_weights, _ = grid_search_weights(
        train_score, None, weights, SECTOR_SEARCH["keys"], SECTOR_SEARCH["steps"]
    )
    old_test = backtest_sectors(test_hist, test_bench, weights)
    new_test = backtest_sectors(test_hist, test_bench, best_weights)

    changed = new_test["win_rate"] > old_test["win_rate"] + 1e-9
    if changed:
        save_weights(best_weights)
    version = datetime.now().strftime("v%Y%m%d%H%M")
    rec = {
        "version": version,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weights_json": _dumps(best_weights if changed else weights),
        "backtest_window": f"{str(train_hist[list(train_hist)[0]]['date'].iloc[0])[:10]}~"
                           f"{str(test_hist[list(test_hist)[0]]['date'].iloc[-1])[:10]}",
        "win_rate": new_test["win_rate"],
        "excess_return": new_test["excess_return"],
        "data_until": str(test_bench["date"].iloc[-1])[:10],
    }
    store.insert_iter(rec)
    return {"status": "updated" if changed else "unchanged", "version": version,
            "old_win_rate": old_test["win_rate"], "new_win_rate": new_test["win_rate"],
            "weights": best_weights if changed else weights}


def _collect_history(provider):
    sector_hist, bench = {}, None
    try:
        quotes = provider.sector_quote()
        for name in list(quotes["name"])[:20]:
            h = provider.sector_hist(name)
            if not h.empty:
                sector_hist[name] = h
        bench = provider.index_daily(provider.benchmark_index_code())
    except Exception as exc:  # noqa: BLE001
        logger.warning("迭代数据采集失败: %s", exc)
    return sector_hist, bench


def _split_train_test(sector_hist, bench, ratio):
    import pandas as pd

    def _cut(df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        n = int(len(df) * ratio)
        return df.iloc[:n], df.iloc[n:]

    train_h, test_h = {}, {}
    for name, h in sector_hist.items():
        a, b = _cut(h)
        if len(a) >= 30 and len(b) >= 30:
            train_h[name], test_h[name] = a, b
    tb = bench.copy()
    tb["date"] = pd.to_datetime(tb["date"])
    tb = tb.sort_values("date").reset_index(drop=True)
    n = int(len(tb) * ratio)
    train_b, test_b = tb.iloc[:n], tb.iloc[n:]
    return (train_h, test_h), (train_b, test_b)


def _set_path(obj: dict, path: list[str], value) -> None:
    cur = obj
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _deep_copy(obj):
    import copy

    return copy.deepcopy(obj)


def _dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
