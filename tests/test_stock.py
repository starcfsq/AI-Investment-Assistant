from core.stock import rank_stocks, score_stock

WEIGHTS = {"stock": {"roe": 0.3, "growth": 0.25, "valuation": 0.25, "dividend": 0.2}}

CANDIDATES = [
    {"code": "600001", "name": "甲", "roe": 20.0, "growth": 15.0,
     "pe_pct": 20.0, "dividend": 3.0},
    {"code": "600002", "name": "乙", "roe": 5.0, "growth": -5.0,
     "pe_pct": 90.0, "dividend": 0.5},
]


def test_rank_stocks_orders_by_score():
    ranked = rank_stocks(CANDIDATES, WEIGHTS)
    assert ranked[0]["name"] == "甲"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_score_stock_returns_0_to_100():
    s = score_stock(CANDIDATES[0], WEIGHTS)
    assert 0 <= s <= 100
