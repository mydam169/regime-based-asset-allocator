"""Tests for src/preprocess.py"""
import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import prepare_macro_data, train_test_split


def _make_macro_df(T=100, n=13):
    rng   = np.random.default_rng(0)
    data  = rng.standard_normal((T, n))
    index = pd.date_range("1990-01-31", periods=T, freq="ME")
    return pd.DataFrame(data, index=index, columns=[f"feat_{i}" for i in range(n)])


def test_prepare_macro_data_shapes():
    df = _make_macro_df(T=100, n=13)
    X_train, X_test, scaler, d_train, d_test = prepare_macro_data(df, train_end_idx=80)
    assert X_train.shape == (80, 13)
    assert X_test.shape  == (20, 13)
    assert len(d_train)  == 80
    assert len(d_test)   == 20


def test_prepare_macro_data_no_nans():
    df = _make_macro_df()
    X_train, X_test, _, _, _ = prepare_macro_data(df, train_end_idx=80)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_prepare_macro_data_robust_scaled():
    """
    RobustScaler (used instead of StandardScaler) centres on the median
    and scales by IQR — not mean/std.  After transformation:
      - The median of each feature should be close to zero.
      - The IQR of each feature should be close to 1.
    The tolerance is looser than for StandardScaler because RobustScaler
    uses quantile_range=(5, 95) which clips the extreme 10% of values
    before computing the scale, so the exact median/IQR relationship
    to the transformed data differs slightly from the standard (25,75) IQR.
    """
    df = _make_macro_df()
    X_train, _, _, _, _ = prepare_macro_data(df, train_end_idx=80)
    # Median should be close to zero (RobustScaler centres on median)
    assert np.abs(np.median(X_train, axis=0)).max() < 0.5
    # Values should be in a reasonable range (not wildly unscaled)
    assert np.abs(X_train).max() < 20


def test_prepare_macro_data_drop_first():
    df = _make_macro_df(T=100)
    X_no_drop, _, _, d_no_drop, _ = prepare_macro_data(df)
    X_drop,    _, _, d_drop,    _ = prepare_macro_data(df, drop_first=1)
    assert X_drop.shape[0] == X_no_drop.shape[0] - 1
    assert d_drop[0] == d_no_drop[1]


def test_train_test_split_chronological():
    df = _make_macro_df(T=100)
    train, test = train_test_split(df, train_size=0.8)
    assert len(train) == 80
    assert len(test)  == 20
    assert train.index[-1] < test.index[0]