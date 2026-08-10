import pytest

from core.portfolio import build_portfolio

SECTORS = [
    {"name": "半导体", "score": 90.0},
    {"name": "白酒", "score": 60.0},
    {"name": "医药", "score": 30.0},
    {"name": "新能源", "score": 20.0},
]


def test_core_ratio_and_satellite_sum():
    p = build_portfolio(SECTORS, core_ratio=0.7, top_n=3)
    assert p["core"]["weight"] == 0.7
    sat_sum = round(sum(s["weight"] for s in p["satellite"]), 4)
    assert sat_sum == pytest.approx(0.3)
    assert len(p["satellite"]) == 3


def test_satellite_weight_proportional_to_score():
    p = build_portfolio(SECTORS, core_ratio=0.7, top_n=3)
    assert p["satellite"][0]["name"] == "半导体"
    assert p["satellite"][0]["weight"] > p["satellite"][2]["weight"]
