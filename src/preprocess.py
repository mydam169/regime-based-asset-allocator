"""
Preprocess monthly macroeconomic data into model-ready arrays.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def prepare_macro_data(macro_df, train_end_idx: int = None, drop_first: int = 0):
    """
    Standardize macro indicators and split into train / test arrays.

    Parameters
    ----------
    macro_df      : pd.DataFrame (T, n) — raw macro indicators
    train_end_idx : int — last training index; if None, use all data
    drop_first    : int — rows to skip at the start (1 for VAR(1), 0 for HMM)
                    accounts for lag consumption in MSVAR

    Returns
    -------
    X_train      : (T_train, n) standardized training array
    X_test       : (T_test, n) standardized test array, or None
    scaler       : fitted StandardScaler
    dates_train  : DatetimeIndex
    dates_test   : DatetimeIndex or None
    """
    if drop_first:
        macro_df = macro_df.iloc[drop_first:]

    values = macro_df.values.astype(float)
    dates  = macro_df.index

    if train_end_idx is None:
        train_end_idx = len(values)

    X_raw_train = values[:train_end_idx]
    X_raw_test  = values[train_end_idx:] if train_end_idx < len(values) else None

    # scaler  = StandardScaler()
    scaler = RobustScaler(quantile_range=(5, 95))  # more robust to outliers in macro data
    X_train = scaler.fit_transform(X_raw_train)
    X_test  = scaler.transform(X_raw_test) if X_raw_test is not None else None

    dates_train = dates[:train_end_idx]
    dates_test  = dates[train_end_idx:] if X_raw_test is not None else None

    return X_train, X_test, scaler, dates_train, dates_test


def train_test_split(df, train_size: float = 0.8):
    """
    Chronological train / test split for a DataFrame.

    Parameters
    ----------
    df         : pd.DataFrame with DatetimeIndex
    train_size : float in (0, 1)

    Returns
    -------
    df_train, df_test : pd.DataFrame
    """
    split_idx = int(len(df) * train_size)
    return df.iloc[:split_idx], df.iloc[split_idx:]
