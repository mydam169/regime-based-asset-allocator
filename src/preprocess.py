"""
Preprocess monthly macroeconomic data into model-ready arrays.

Pipeline
--------
1. prepare_macro_data()
2. apply_dim_reduction()
3. train_test_split()
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression



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
        pca_full = PCA(n_components=variance_threshold, random_state=42)
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

def apply_pls(X_train: np.ndarray,
              X_test:  np.ndarray = None,
              Y_train: np.ndarray = None,
              n_components: int   = 4) -> tuple:
    """
    Apply Partial Least Squares (PLS) dimensionality reduction.

    PLS finds latent directions in X that maximally covary with Y (asset returns).
    Unlike PCA — which maximises variance in X alone — PLS finds macro directions
    that are predictive of asset return distributions, aligning preprocessing
    directly to the portfolio objective.

    Reference: Kelly & Pruitt (2013, 2015) show that PLS outperforms PCA when
    only a subset of X is predictively relevant for Y — exactly the setting here
    (13 macro indicators, only a few directions matter for 3-asset returns).

    Note on objective leakage
    -------------------------
    PLS preprocessing sees asset returns Y_train during fitting.  This is not
    look-ahead bias in the time-series sense (PLS is fitted on training data only),
    but it does mean the preprocessing step is shaped by the portfolio objective.
    In a pure macro regime detection study this would be a concern; in a portfolio
    allocation study it is defensible and arguably correct.

    Parameters
    ----------
    X_train      : (T_train, n) robust-scaled macro indicator array
    X_test       : (T_test,  n) robust-scaled macro indicator array, or None
    Y_train      : (T_train, q) asset return array.  If None, falls back to PCA.
    n_components : int — number of PLS components (default 4).
                   Choose by cross-validation on held-out return data, not by
                   variance threshold.  Kelly & Pruitt (2015) show that using
                   too many components hurts out-of-sample performance.

    Returns
    -------
    X_train_pls : (T_train, n_components) PLS-projected training array
    X_test_pls  : (T_test,  n_components) PLS-projected test array, or None
    pls         : fitted PLSRegression object
    """
    if Y_train is None:
        import warnings
        warnings.warn(
            "Y_train not provided — falling back to PCA. "
            "Pass asset return array to use PLS.",
            UserWarning
        )
        return apply_pca(X_train, X_test, n_components=n_components)

    # Align lengths (Y_train may have fewer rows if asset data starts later)
    T = min(len(X_train), len(Y_train))
    X_fit = X_train[:T]
    Y_fit = Y_train[:T]

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_fit, Y_fit)

    X_train_pls = pls.transform(X_train)
    X_test_pls  = pls.transform(X_test) if X_test is not None else None

    # ── Diagnostics ──────────────────────────────────────────────────────────
    # Compute how much variance in X is explained by PLS components
    # (for comparison with PCA explained variance)
    X_reconstructed = pls.inverse_transform(X_train_pls)
    ss_res = np.sum((X_train - X_reconstructed) ** 2)
    ss_tot = np.sum((X_train - X_train.mean(axis=0)) ** 2)
    x_var_explained = 1 - ss_res / ss_tot

    print(f"\n── PLS summary ──")
    print(f"  Input dimensions  : {X_train.shape[1]}")
    print(f"  Components        : {n_components}")
    print(f"  X variance explained (approx): {x_var_explained:.1%}")
    print(f"  Projected train shape : {X_train_pls.shape}")
    if X_test_pls is not None:
        print(f"  Projected test shape  : {X_test_pls.shape}")

    return X_train_pls, X_test_pls, pls


def pls_loading_table(pls: PLSRegression, feature_names: list[str]) -> pd.DataFrame:
    """
    Return a tidy DataFrame of PLS component loadings.

    Each column is one PLS component (PLS1, PLS2, …).
    Each row is one original macro indicator.
    Large absolute values indicate that indicator contributes strongly
    to that component.

    Parameters
    ----------
    pls          : fitted PLSRegression object returned by apply_pls()
    feature_names: list of original indicator names (macro_df.columns)

    Returns
    -------
    pd.DataFrame — shape (n_features, n_components)
    """
    return pd.DataFrame(
        pls.x_loadings_,
        index   = feature_names,
        columns = [f"PLS{i+1}" for i in range(pls.n_components)],
    )


def apply_dim_reduction(X_train: np.ndarray,
                        X_test:  np.ndarray = None,
                        method:  str        = 'pls',
                        Y_train: np.ndarray = None,
                        n_components: int   = 4,
                        variance_threshold: float = 0.85,
                        winsor_clip:        float = 4.0) -> tuple:
    """
    Unified dimensionality reduction interface.

    Wraps apply_pca(), apply_pls(), and the no-reduction case so that the
    preprocessing choice is a single parameter switch throughout the codebase.
    Winsorisation (±winsor_clip std) is applied after any reduction.

    Parameters
    ----------
    X_train            : (T_train, n) robust-scaled input array
    X_test             : (T_test,  n) robust-scaled input array, or None
    method             : 'pca' | 'pls' | 'none'
                         'pca'  — unsupervised PCA (default)
                         'pls'  — supervised PLS; requires Y_train
                         'none' — no reduction; returns X_train unchanged
    Y_train            : (T_train, q) asset return array (required for 'pls')
    n_components       : int — explicit component count for PCA (overrides threshold)
    variance_threshold : float — PCA cumulative variance target (default 0.85)
    n_pls_components   : int — number of PLS components (default 4)
    winsor_clip        : float — clip at ±winsor_clip after reduction (default 4.0)

    Returns
    -------
    X_reduced  : (T_train, k) reduced training array
    X_test_red : (T_test,  k) reduced test array, or None
    reducer    : fitted reducer object (PCA, PLSRegression, or None)
    """
    if method == 'pca':
        X_red, X_test_red, reducer = apply_pca(
            X_train, X_test,
            n_components=n_components,
            variance_threshold=variance_threshold
        )
    elif method == 'pls':
        X_red, X_test_red, reducer = apply_pls(
            X_train, X_test,
            Y_train=Y_train,
            n_components=n_components
        )
    elif method == 'none':
        X_red     = X_train.copy()
        X_test_red = X_test.copy() if X_test is not None else None
        reducer   = None
        print(f"  No dimensionality reduction: shape {X_red.shape}")
    else:
        raise ValueError(f"method must be 'pca', 'pls', or 'none', got '{method}'")

    # Winsorise after reduction (handles extreme outliers like April 2020 in PCA space)
    if winsor_clip is not None and winsor_clip > 0:
        X_red = np.clip(X_red, -winsor_clip, winsor_clip)
        if X_test_red is not None:
            X_test_red = np.clip(X_test_red, -winsor_clip, winsor_clip)

    return X_red, X_test_red, reducer

def loading_table(reducer, feature_names: list[str]) -> pd.DataFrame:
    """
    Return loading table for either a PCA or PLSRegression object.
    Dispatches to pca_loading_table or pls_loading_table automatically.
    """
    if isinstance(reducer, PCA):
        return pca_loading_table(reducer, feature_names)
    elif isinstance(reducer, PLSRegression):
        return pls_loading_table(reducer, feature_names)
    elif reducer is None:
        raise ValueError("No reducer fitted — dim_reduction='none' has no loadings.")
    else:
        raise TypeError(f"Unsupported reducer type: {type(reducer)}")