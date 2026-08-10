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
    assert best_weights["sector"]["rs"] == 0.9
    assert best_score == pytest.approx(0.9)
