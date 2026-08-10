import pytest

from core.tune import grid_search_weights


def fake_score_fn(data, weights):
    # 用 "rs" 权重作为正向指标：权重越大分数越高
    return {"win_rate": weights["sector"]["rs"], "n_samples": 1}


def test_grid_search_finds_best():
    base = {"sector": {"rs": 0.5, "flow": 0.25, "momentum": 0.25}}
    best_weights, best_score = grid_search_weights(
        fake_score_fn, None, base,
        search_keys=["sector.rs", "sector.flow"],
        steps=[0.8, 0.9],
    )
    # 网格选中的仍是 rs=0.9 的候选；返回前 sector 组已归一化为和 1.0
    assert best_score == pytest.approx(0.9)
    s = best_weights["sector"]
    assert abs(sum(s.values()) - 1.0) < 1e-6
    assert s["rs"] > s["flow"] > s["momentum"]


def test_run_iteration_normalizes_sector_weights():
    from core.tune import grid_search_weights, _deep_copy

    base = {"sector": {"rs": 0.5, "flow": 0.3, "momentum": 0.2}}

    def score_fn(_, w):
        return {"win_rate": w["sector"]["rs"], "n_samples": 1}

    best, _ = grid_search_weights(score_fn, None, _deep_copy(base),
                                  ["sector.rs", "sector.flow", "sector.momentum"],
                                  [0.9, 0.5, 0.5])
    s = best["sector"]
    assert abs(sum(s.values()) - 1.0) < 1e-6


def test_run_iteration_no_data_when_bench_missing(monkeypatch):
    from core.tune import run_iteration

    class FakeProvider:
        def sector_quote(self):
            import pandas as pd
            return pd.DataFrame({"name": ["A", "B"]})
        def sector_hist(self, name):
            import pandas as pd
            return pd.DataFrame()
        def index_daily(self, code):
            return None
        def benchmark_index_code(self):
            return "sh000300"

    store = None
    result = run_iteration(FakeProvider(), store)
    assert result["status"] == "no_data"
