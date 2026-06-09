"""
Preprocess monthly macroeconomic data into model-ready arrays.

Pipeline
--------
1. prepare_macro_data()
2. apply_pca()
3. train_test_split()
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

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
    X_train      : (T_train, n) robust-scaled training array
    X_test       : (T_test, n) robust-scaled test array, or None
    scaler       : fitted RobustScaler
    dates_train  : DatetimeIndex aligned with X_train rows
    dates_test   : DatetimeIndex aligned with X_test rows, or None
    """
    if drop_first:
        macro_df = macro_df.iloc[drop_first:]

    values = macro_df.values.astype(float)
    dates  = macro_df.index

    if train_end_idx is None:
        train_end_idx = len(values)

    X_raw_train = values[:train_end_idx]
    X_raw_test  = values[train_end_idx:] if train_end_idx < len(values) else None

    # Use robust scaler to clip the most extreme 10% of values in macro data
    scaler = RobustScaler(quantile_range=(5, 95))  
    X_train = scaler.fit_transform(X_raw_train)
    X_test  = scaler.transform(X_raw_test) if X_raw_test is not None else None

    dates_train = dates[:train_end_idx]
    dates_test  = dates[train_end_idx:] if X_raw_test is not None else None

    return X_train, X_test, scaler, dates_train, dates_test



def apply_pca(X_train: np.ndarray, 
              X_test: np.ndarray = None, 
              n_components: int = None, 
              variance_threshold: float = 0.85):
    """
    Apply PCA to reduce dimensionality of macro indicators while retaining variance.
    Needed to avoid overfitting and reduce computational cost of MSVAR, especially 
    when implementing walk-forward prediction to simulate real-time trading scenarios.

    Parameters
    ----------
    X_train            : (T_train, n) robust-scaled array of training features
    X_test             : (T_test, n) robust-scaled array of test features, or None
    n_components       : int — number of PCA components to keep; if None, use variance_threshold
    variance_threshold : float in (0,1) — minimum variance to retain; used if n_components is None
    
    Returns
    -------
    X_train_pca : (T_train, k) PCA-transformed training array
    X_test_pca  : (T_test, k) PCA-transformed test array, or None
    pca         : fitted PCA object
    """
    n = X_train.shape[1]

    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
    else:
        pca_full = PCA(n_components=n, random_state=42)
        pca_full.fit(X_train)
        cumvar = pca_full.explained_variance_ratio_.cumsum()

        # Find the number of components needed to reach the variance threshold
        n_keep = int((cumvar < variance_threshold).sum()) + 1
        n_keep = max(n_keep, 2)  # Keep at least 2 components to avoid degenerate cases

        pca = PCA(n_components=n_keep, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
    
    # Project test data using the training-fitted PCA 
    X_test_pca = pca.transform(X_test) if X_test is not None else None

    # Diagnostic checks
    cumvar_kept = pca.explained_variance_ratio_.cumsum()
    print(f"\n── PCA summary ──")
    print(f"  Input dimensions  : {n}")
    print(f"  Components kept   : {pca.n_components_}  "
          f"(threshold = {variance_threshold:.0%})")
    print(f"  Variance explained: {cumvar_kept[-1]:.1%}")
    print(f"\n  Per-component breakdown:")
    for i, (v, cv) in enumerate(
        zip(pca.explained_variance_ratio_, cumvar_kept), start=1
    ):
        bar = "█" * int(v * 40)
        print(f"    PC{i}: {v:5.1%}  cumulative {cv:5.1%}  {bar}")

    print(f"\n  Projected train shape : {X_train_pca.shape}")
    if X_test_pca is not None:
        print(f"  Projected test shape  : {X_test_pca.shape}")

    return X_train_pca, X_test_pca, pca


def pca_loading_table(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    """
    Return a tidy DataFrame of PCA component loadings.

    Each column is one principal component (PC1, PC2, …).
    Each row is one original macro indicator.
    Large absolute values indicate that indicator contributes strongly
    to that component.

    Typical economic interpretation (may vary by sample):
      PC1 — broad real-activity factor
            (high loadings on INDPRO, W875RX1; negative on UNRATE, ICSA)
      PC2 — financial stress factor
            (high loadings on VIX, BAA10Y, T10Y2Y)
      PC3 — inflation / monetary factor
            (high loadings on CPI, FEDFUNDS, M2SL, oil)
      PC4 — sentiment / housing factor
            (high loadings on UMCSENT, HOUST)

    Parameters
    ----------
    pca          : fitted PCA object returned by apply_pca()
    feature_names: list of original indicator names (macro_df.columns)

    Returns
    -------
    pd.DataFrame — shape (n_features, n_components)
    """
    return pd.DataFrame(
        pca.components_.T,
        index   = feature_names,
        columns = [f"PC{i+1}" for i in range(pca.n_components_)],
    )

def train_test_split(df: pd.DataFrame, train_size: float = 0.8) -> tuple:
    """
    Chronological train / test split for a DataFrame.

    Parameters
    ----------
    df         : pd.DataFrame with DatetimeIndex
    train_size : float in (0, 1) - fraction of rows used for training

    Returns
    -------
    df_train, df_test : pd.DataFrame
    """
    split_idx = int(len(df) * train_size)
    return df.iloc[:split_idx], df.iloc[split_idx:]
