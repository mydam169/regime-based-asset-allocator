"""Tests for src/regime_mapping.py"""
import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.regime_mapping import map_regime_to_weights


SPEC = {
    0: {"index_fund": 0.6, "treasury_fund": 0.3, "gold_fund": 0.1},
    1: {"index_fund": 0.2, "treasury_fund": 0.5, "gold_fund": 0.3},
}

def _dates(n=6):
    return pd.date_range("2020-01-31", periods=n, freq="ME")


def test_output_columns():
    w = map_regime_to_weights([0, 1, 0], SPEC, date_index=_dates(3))
    assert list(w.columns) == ["regime_label", "index_fund",
                                "treasury_fund", "gold_fund"]


def test_weights_sum_to_one():
    regimes = [0, 0, 1, 1, 0, 1]
    w = map_regime_to_weights(regimes, SPEC, date_index=_dates(6))
    sums = w[["index_fund", "treasury_fund", "gold_fund"]].sum(axis=1)
    assert np.allclose(sums, 1.0)


def test_series_input_matches_array_input():
    dates   = _dates(4)
    regimes = [0, 1, 1, 0]
    w_arr = map_regime_to_weights(regimes, SPEC, date_index=dates)
    w_ser = map_regime_to_weights(pd.Series(regimes, index=dates), SPEC)
    assert w_arr.equals(w_ser)


def test_correct_weights_assigned():
    dates   = _dates(2)
    w = map_regime_to_weights([0, 1], SPEC, date_index=dates)
    assert w.iloc[0]["index_fund"]    == 0.6
    assert w.iloc[1]["index_fund"]    == 0.2
    assert w.iloc[0]["treasury_fund"] == 0.3
    assert w.iloc[1]["gold_fund"]     == 0.3


def test_unknown_regime_raises():
    with pytest.raises(KeyError):
        map_regime_to_weights([0, 2], SPEC, date_index=_dates(2))
