"""core.analyze 选股候选生成测试。"""
import pandas as pd


def _spot():
    return pd.DataFrame({
        "name": ["湖南白银", "国际医学", "浦发银行", "平安银行"],
        "code": ["002716", "000516", "600000", "000001"],
        "price": [10.0, 12.0, 8.0, 11.0],
        "pct_change": [1.0, -2.0, 0.5, 5.0],
    })


def test_candidate_pool_matches_sector_names():
    from core.analyze import _candidate_pool

    pool = _candidate_pool(_spot(), ["医疗服务", "白银"])
    names = set(pool["name"].tolist())
    assert "湖南白银" in names  # 股票名含板块词"白银"
    assert "国际医学" not in names  # 不含"医疗服务"，不应误匹配


def test_candidate_pool_falls_back_when_no_match():
    from core.analyze import _candidate_pool

    # 板块名匹配不到任何股票 → 回退全市场活跃候选(按涨跌幅降序)
    pool = _candidate_pool(_spot(), ["医疗服务"])
    assert len(pool) == 4
    assert pool.iloc[0]["code"] == "000001"  # 涨跌幅 5.0 最高排前


def test_candidate_pool_empty_spot_returns_empty():
    from core.analyze import _candidate_pool

    empty = _spot().iloc[0:0]
    assert _candidate_pool(empty, ["医疗服务"]).empty


def test_analyze_at_pure_computation():
    from core.analyze import analyze_at
    import pandas as pd
    index = pd.DataFrame({"date": ["2026-08-10"], "close": [4000.0]})
    data = {
        "index_df": index,
        "val_df": pd.DataFrame({"date": ["2026-08-10"], "pe": [13.0], "pb": [1.4]}),
        "bond_df": pd.DataFrame({"date": ["2026-08-10"], "cn_10y": [2.5]}),
        "quotes": pd.DataFrame({"name": ["医疗服务"], "pct_change": [2.0]}),
        "flow": pd.DataFrame(),
        "hist": {"医疗服务": pd.DataFrame({"date": ["2026-08-10"], "close": [100.0]})},
        "bench": index,
    }
    from core.config import load_weights
    out = analyze_at(data, load_weights())
    assert set(out) == {"trend", "sectors", "portfolio", "warnings"}
    assert "trend" in out and "sectors" in out and "portfolio" in out
