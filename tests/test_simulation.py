import pandas as pd
import pytest


def _fake_provider():
    """内存 fake，snapshot_at 只做日期切片，不访问网络。"""
    class P:
        pass
    p = P
    p.index_df = pd.DataFrame({"date": pd.to_datetime(
        ["2026-01-05", "2026-02-02", "2026-03-02"]), "close": [1.0, 2.0, 3.0]})
    return p


def test_history_snapshot_slices_by_date():
    from core.simulation import _slice_by_date
    df = pd.DataFrame({"date": pd.to_datetime(
        ["2026-01-05", "2026-02-02", "2026-03-02"]), "close": [1.0, 2.0, 3.0]})
    out = _slice_by_date(df, "2026-02-01")
    assert list(out["close"]) == [1.0]


def test_snapshot_at_keeps_bench_parallel():
    from core.simulation import _snapshot_at
    index = pd.DataFrame({"date": pd.to_datetime(
        ["2026-01-05", "2026-02-02", "2026-03-02"]), "close": [1.0, 2.0, 3.0]})
    data = {"index_df": index, "val_df": pd.DataFrame(),
            "bond_df": pd.DataFrame(),
            "quotes": pd.DataFrame(), "flow": pd.DataFrame(),
            "hist": {}, "bench": index}
    snap = _snapshot_at(data, "2026-02-01")
    assert list(snap["index_df"]["close"]) == [1.0]
    assert list(snap["bench"]["close"]) == [1.0]
