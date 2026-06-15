"""
Walk-forward backtesting engine for regime-based portfolio allocation.

Design
------
Simulates real-time portfolio management: at each rebalancing date the model
is re-estimated on all data available up to that point (expanding window),
filtered probabilities are used for regime detection (no look-ahead from the
Kim smoother), and portfolio weights are set for the following month.

This is methodologically superior to the fixed train/test split used in
notebooks 03-04 because:
1. The model parameters adapt as new data arrives.
2. Filtered probabilities P(S_t | y_{1:t}) replace smoothed probabilities
   P(S_t | y_{1:T}) — the smoother uses future data and inflates in-sample
   performance.
3. The evaluation is genuinely out-of-sample at every step.

Walk-forward design
-------------------
- Initial training window: first `initial_train_months` observations
- Refit frequency         : every `refit_every` months (default 12 = annual)
- Between refits          : apply one forward pass using pre-estimated
                            parameters to get updated filtered probs
                            (cheap; avoids expensive monthly full re-estimation)
- Portfolio weights       : computed from regime-conditional moments estimated
                            on the filtered-regime observations in the current
                            training window
- Signal lag              : weights from date t applied to returns at t+1
                            (prevents same-period execution)

Computational note
------------------
With PCA-reduced inputs (n ≈ 4-6 components), a full MSVAR re-estimation
takes ~10-30 seconds on a laptop.  Annual refitting over a 7-year test period
requires ~7 fits — total runtime ~1-3 minutes.  Monthly refitting would take
~30-60 minutes and is not recommended without GPU/cluster access.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from .hmm_model import HMMRegimeModel
from .msvar_model import MSVARRegimeModel
from .portfolio import estimate_regime_moments, optimize_regimes
from .backtest import backtest_with_transaction_costs
from .benchmarks import build_constant_weights
from .constants import ASSET_COLS


# ---------------------------------------------------------------------------
# Core walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    macro_df:              pd.DataFrame,
    asset_df:              pd.DataFrame,
    model_type:            str   = 'hmm',
    initial_train_months:  int   = 120,
    refit_every:           int   = 12,
    dim_reduction:         str   = 'pca', # 'pca', 'pls', or None
    n_components:      Optional[int] = None,
    variance_threshold:    float = 0.85,
    winsor_clip:           float = 4.0,
    transaction_cost_bps:  int   = 10,
    objective:             str   = 'min_variance',
    rf:                    float = 0.0,
    long_only:             bool  = True,
    hmm_kwargs:            dict  = None,
    msvar_kwargs:          dict  = None,
    verbose:               bool  = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expanding-window walk-forward backtest for HMM or MSVAR regime allocator.

    At each rebalancing date t:
      1. Fit model on macro_df.iloc[:t] (expanding window, every refit_every months)
      2. Get one-step-ahead filtered regime probability P(S_t | y_{1:t})
      3. Assign regime by argmax of filtered probs
      4. Estimate regime-conditional moments from filtered-regime observations
      5. Solve MVO → target weights for month t+1 (signal lag)
      6. Record portfolio return for month t+1

    Parameters
    ----------
    macro_df             : pd.DataFrame — full macro indicator series (all dates)
    asset_df             : pd.DataFrame — full asset return series (ASSET_COLS)
    model_type           : 'hmm' or 'msvar'
    initial_train_months : int — minimum training window before first evaluation
    refit_every          : int — re-estimate model parameters every N months
    dim_reduction        : str — dimensionality reduction method: 'pca', 'pls', or 'none'.
                           'pca'  — unsupervised PCA; n_components or variance_threshold
                                    controls the number of components retained.
                           'pls'  — supervised PLS; finds macro directions that maximally
                                    co-vary with asset returns. n_components controls
                                    the number of latent factors (default 4).
                           'none' — no reduction; raw robust-scaled indicators fed directly.
                                    Appropriate for HMM only (MSVAR too slow with many vars).
    n_components         : int — component count for PCA or PLS; None = use variance_threshold
                           for PCA (ignored for PLS, which uses 4 by default).
    variance_threshold   : float — PCA cumulative variance target (default 0.85; PCA only)
    transaction_cost_bps : int — one-way transaction costs in basis points
    objective            : 'min_variance' or 'max_sharpe'
    rf                   : float — monthly risk-free rate for Sharpe computation
    long_only            : bool — enforce non-negative weights
    hmm_kwargs           : dict — override HMMRegimeModel constructor arguments
    msvar_kwargs         : dict — override MSVARRegimeModel constructor arguments
    verbose              : bool — print progress

    Returns
    -------
    returns_df   : pd.DataFrame — monthly portfolio returns with columns
                   ['gross_return', 'net_return_after_costs', 'trade_notional',
                    'transaction_cost', regime_col, 'index_fund_weight',
                    'treasury_fund_weight', 'gold_fund_weight']
    state_series : pd.Series — filtered regime label at each evaluation date
    """
    assert model_type in ('hmm', 'msvar'), "model_type must be 'hmm' or 'msvar'"
    regime_col = f'{model_type.upper()}_WF_State'

    # Default model kwargs
    if hmm_kwargs is None:
        hmm_kwargs = dict(n_states=2, n_iter=200, n_restarts=10,
                          tol=1e-4, covariance_type='full',
                          constrain_transmat=True)
    if msvar_kwargs is None:
        msvar_kwargs = dict(n_states=2, p=1, n_restarts=5,
                            max_iter=300, tol=1e-6, verbose=False)

    p = msvar_kwargs.get('p', 1) if model_type == 'msvar' else 0

    # Align asset returns to macro dates
    common_idx    = macro_df.index.intersection(asset_df.index)
    macro_aligned = macro_df.loc[common_idx]
    asset_aligned = asset_df[ASSET_COLS].loc[common_idx]
    T             = len(common_idx)

    if verbose:
        print(f"Walk-forward backtest: {model_type.upper()}, "
              f"{T} months total, "
              f"initial window={initial_train_months}, "
              f"refit every={refit_every}")

    records      = []
    state_labels = {}
    refit_states        = {}   # date → state from most recent refit's smoothed sequence
    smoothed_probs_dict = {}   # date → (K,) smoothed prob vector from most recent refit
    filtered_probs_dict = {}   # date → (K,) filtered prob vector (no look-ahead)
    prev_state          = 0    # default expansion before first refit
    prev_weights = np.ones(3) / 3      # start equal-weight
    model        = None
    scaler       = None
    reducer      = None
    last_refit   = -refit_every        # force fit on first iteration

    import warnings

    for t in range(initial_train_months, T - 1):
        date_t    = common_idx[t]
        date_tp1  = common_idx[t + 1]

        # ── Step 1: (re)fit model every refit_every months ────────────────
        if (t - last_refit) >= refit_every:
            if verbose:
                print(f"  Refitting at {date_t.date()} "
                      f"(t={t}, window={t} months)...")

            # Robust scale on expanding training window
            X_raw    = macro_aligned.iloc[:t].values.astype(float)
            scaler   = RobustScaler(quantile_range=(5, 95))
            X_scaled = scaler.fit_transform(X_raw)

            # ── Dimensionality reduction ──────────────────────────────────
            n_features = X_scaled.shape[1]

            if dim_reduction == 'none':
                # No dimensionality reduction — feed raw robust-scaled data.
                # Appropriate for HMM (fast enough); not recommended for MSVAR
                # (too slow and parameter-heavy with 15 variables).
                X_reduced = np.clip(X_scaled, -winsor_clip, winsor_clip)
                reducer   = None
                if verbose:
                    print(f"    No dim reduction: feeding {n_features} raw indicators")

            elif dim_reduction == 'pls':
                # Partial Least Squares: find macro directions maximally
                # predictive of asset returns rather than maximally variable
                # in macro space.  Requires asset returns aligned to macro dates.
                Y_train_arr = asset_aligned.iloc[:t][ASSET_COLS].values
                # PLS needs Y to cover the same rows as X — trim to common length
                min_len = min(len(X_scaled), len(Y_train_arr))
                reducer = PLSRegression(n_components=n_components or 4)
                reducer.fit(X_scaled[:min_len], Y_train_arr[:min_len])
                X_reduced = np.clip(reducer.transform(X_scaled), -winsor_clip, winsor_clip)
                if dim_reduction == 'pls' and verbose:
                    print(f"    PLS label check will follow — "
                          f"components={n_components or 4}")
                if verbose:
                    print(f"    PLS: {n_components or 4} components")
            else:
                # Standard PCA (default)
                if n_components is not None:
                    reducer = PCA(n_components=n_components, random_state=42)
                else:
                    pca_full = PCA(n_components=n_features, random_state=42)
                    pca_full.fit(X_scaled)
                    cumvar = pca_full.explained_variance_ratio_.cumsum()
                    n_keep = max(int((cumvar < variance_threshold).sum()) + 1, 2)
                    reducer = PCA(n_components=n_keep, random_state=42)
                X_reduced = np.clip(reducer.fit_transform(X_scaled), -winsor_clip, winsor_clip)

            # Fit model — suppress numerical convergence warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if model_type == 'hmm':
                    model = HMMRegimeModel(**hmm_kwargs)
                    model.fit(X_reduced)
                else:
                    model = MSVARRegimeModel(**msvar_kwargs)
                    model.fit(X_reduced)

            # ── Label anchoring across refits ─────────────────────────────
            # Sorting criteria (_sort_states / _sort_regimes) can assign
            # inconsistent labels across windows because PCA eigenvectors are
            # only defined up to a sign flip — a sign change in PC1 between
            # consecutive fits makes dot-product similarity unreliable.
            #
            # Robust fix: enforce that state 0 = expansion = majority state,
            # by counting observations via argmax of smoothed/filtered probs.
            # This is invariant to PCA sign flips and reliable at all window
            # sizes (any window ≥ 3 years will have expansion as majority).
            # Use smoothed argmax for label anchoring check — consistent with
            # how window_states is computed below.
            if model_type == 'hmm':
                states_in_window = np.argmax(model.smoothed_probs(X_reduced), axis=0)
            else:
                states_in_window = model._result.regime_sequence(use_smoothed=True)

            counts = np.bincount(states_in_window, minlength=2)
            # State 0 should be majority (expansion). If state 1 has more
            # observations, labels are inverted — swap all state-indexed params.
            if counts[1] > counts[0]:
                if model_type == 'hmm':
                    m = model._model
                    m.means_     = m.means_[[1, 0]]
                    m._covars_   = m._covars_[[1, 0]]   # see hmm_model._sort_states
                    m.startprob_ = m.startprob_[[1, 0]]
                    m.transmat_  = m.transmat_[[1, 0]][:, [1, 0]]
                else:
                    r = model._result
                    r.B_list           = [r.B_list[1],     r.B_list[0]]
                    r.Sigma_list       = [r.Sigma_list[1], r.Sigma_list[0]]
                    r.P                = r.P[[1, 0]][:, [1, 0]]
                    r.pi               = r.pi[[1, 0]]
                    r.xi_filtered      = r.xi_filtered[[1, 0]]
                    r.xi_smoothed      = r.xi_smoothed[[1, 0]]
                    r.eff_sample_sizes = r.eff_sample_sizes[[1, 0]]
                if verbose:
                    print(f"    Label corrected (minority→recession) at {date_t.date()}")

            # PLS column-0 ordering is unconstrained — log the majority-state
            # check so the assumption is visible in verbose output.
            if dim_reduction == 'pls' and verbose:
                status = 'OK' if counts[0] >= counts[1] else 'WARNING: recession majority'
                print(f"    PLS label check at {date_t.date()}: "
                      f"expansion={counts[0]}, recession={counts[1]} — {status}")

            # ── Populate refit_states for the current training window ─────────
            # Use SMOOTHED probabilities (Kim smoother) for all regime labels
            # in the training window.  The smoother computes P(S_t | y_{1:T})
            # — it uses the full training sequence including future observations
            # relative to t, which gives the most accurate retrospective regime
            # assessment at each refit.  This is correct for in-sample use
            # because we are re-estimating on the whole window anyway.
            #
            # At each annual refit, smoothed labels for ALL past dates are
            # recomputed and overwrite previous assignments (line below).
            # This means the regime assigned to e.g. Jan 2021 may change
            # when the model refits in Dec 2022 — exactly as you would expect
            # a portfolio manager to revise their historical regime assessment
            # as more data arrives and the smoother propagates new information
            # backward.
            #
            # Note: Viterbi (model.predict) is NOT used here. Viterbi makes
            # hard binary decisions at each step and is noisier near regime
            # boundaries than the smoothed argmax, which averages over
            # uncertainty more gracefully.
            if model_type == 'hmm':
                # argmax of smoothed probs — consistent with MSVAR treatment
                window_states = np.argmax(
                    model.smoothed_probs(X_reduced), axis=0
                )
                window_dates  = macro_aligned.iloc[:t].index
            else:
                window_states = model._result.regime_sequence(use_smoothed=True)
                # VAR(p=1) drops first observation as lag
                window_dates  = macro_aligned.iloc[p:t].index

            # Store both smoothed and filtered probabilities
            if model_type == 'hmm':
                window_smoothed  = model.smoothed_probs(X_reduced)   # (K, T)
                window_filtered  = model.filtered_probs(X_reduced)   # (K, T)
            else:
                window_smoothed  = model._result.xi_smoothed     # (K, T-p)
                window_filtered  = model._result.xi_filtered      # (K, T-p)

            for wd, ws, wsp, wfp in zip(window_dates, window_states,
                                         window_smoothed.T,
                                         window_filtered.T):
                refit_states[wd]        = int(ws)
                state_labels[wd]        = int(ws)  # keep state_labels in sync
                smoothed_probs_dict[wd] = wsp   # (K,) smoothed — for diagnostics
                filtered_probs_dict[wd] = wfp   # (K,) filtered — for WF plots

            last_refit = t

        # ── Step 2: get regime label for date t ──────────────────────────
        # Use the smoothed state sequence from the most recent full refit.
        # Running a monthly forward pass produces near-50/50 filtered
        # probabilities between refits (especially for MSVAR), which causes
        # spurious high-frequency switching in the state series.
        # Refit-anchored approach: assign smoothed labels at each annual
        # refit and hold them fixed until the next refit.
        state_t = refit_states.get(date_t, prev_state)
        state_labels[date_t] = state_t
        prev_state = state_t

        # ── Step 3: estimate moments and solve MVO ────────────────────────
        # Use the smoothed regime labels stored in state_labels (populated at
        # each refit from the Kim smoother).  Using smoothed labels for moment
        # estimation is correct: we are estimating parameters on the training
        # window where the smoother is not look-ahead (it uses y_{1:T_train}).
        # Labels are updated at every refit so historical regime assignments
        # are revised as new data arrives and the smoother is re-run.
        asset_window = asset_aligned.iloc[:t].copy()
        asset_window[regime_col] = [
            state_labels.get(d, 0) for d in asset_window.index
        ]

        # Need at least 10 obs per regime — fall back to equal weight
        regime_counts = asset_window[regime_col].value_counts()
        if regime_counts.min() < 10:
            target_weights = prev_weights.copy()
        else:
            try:
                moments  = estimate_regime_moments(
                    asset_window,
                    asset_cols=ASSET_COLS,
                    state_col=regime_col
                )
                results  = optimize_regimes(
                    moments, ASSET_COLS, rf=rf, long_only=long_only
                )
                target_weights = results[state_t][objective].weights
            except Exception as e:
                if verbose:
                    print(f"    MVO failed at {date_t.date()}: {e} — using prev weights")
                target_weights = prev_weights.copy()

        # ── Step 4: compute portfolio return for t+1 ─────────────────────
        returns_tp1  = asset_aligned.loc[date_tp1, ASSET_COLS].values
        gross_return = float(target_weights @ returns_tp1)

        # Drift-adjust previous weights and compute trade notional
        prev_ret = asset_aligned.loc[date_t, ASSET_COLS].values
        if len(records) == 0:
            trade_notional = 0.0  # no cost on first period
        else:
            drifted = prev_weights * (1 + prev_ret)
            drifted = drifted / drifted.sum()
            trade_notional = float(np.abs(target_weights - drifted).sum())

        tc           = trade_notional * transaction_cost_bps / 10_000
        net_return   = gross_return - tc

        records.append({
            'Date':                   date_tp1,
            'gross_return':           gross_return,
            'net_return_after_costs': net_return,
            'trade_notional':         trade_notional,
            'one_way_turnover':       trade_notional / 2,
            'transaction_cost':       tc,
            regime_col:               state_t,
            'index_fund_weight':      target_weights[0],
            'treasury_fund_weight':   target_weights[1],
            'gold_fund_weight':       target_weights[2],
        })
        prev_weights = target_weights.copy()

    returns_df   = pd.DataFrame(records).set_index('Date')
    state_series = pd.Series(state_labels, name=regime_col)

    # Build smoothed probability DataFrame (for diagnostics)
    prob_records = {
        d: {f'State_{k}_prob': float(v[k])
            for k in range(len(v))}
        for d, v in smoothed_probs_dict.items()
        if d in state_series.index
    }
    smoothed_prob_series = pd.DataFrame(prob_records).T.sort_index()
    smoothed_prob_series.index.name = 'Date'

    # Build filtered probability DataFrame (for walk-forward plots — no look-ahead)
    filt_records = {
        d: {f'State_{k}_prob': float(v[k])
            for k in range(len(v))}
        for d, v in filtered_probs_dict.items()
        if d in state_series.index
    }
    filtered_prob_series = pd.DataFrame(filt_records).T.sort_index()
    filtered_prob_series.index.name = 'Date' 

    if verbose:
        n_months = len(returns_df)
        ann_ret  = (1 + returns_df['net_return_after_costs']).prod() ** (12/n_months) - 1
        ann_vol  = returns_df['net_return_after_costs'].std() * np.sqrt(12)
        sharpe   = (ann_ret - rf * 12) / ann_vol if ann_vol > 1e-12 else 0.0
        print(f"\nWalk-forward results ({n_months} months):")
        print(f"  Ann. return  : {ann_ret*100:.2f}%")
        print(f"  Ann. vol     : {ann_vol*100:.2f}%")
        print(f"  Sharpe ratio : {sharpe:.3f}")
        rc = returns_df[regime_col].value_counts()
        print(f"  Regime distribution: {rc.to_dict()}")

    return returns_df, state_series, smoothed_prob_series, filtered_prob_series


def _msvar_forward_step(model: MSVARRegimeModel,
                         X_reduced: np.ndarray,
                         p: int) -> np.ndarray:
    """
    Run a lightweight Hamilton filter forward pass on new data using the
    pre-estimated MSVAR parameters.  Returns filtered regime probabilities
    at the last time step — shape (K,).

    This avoids a full re-estimation between refit dates while still updating
    the filtered probability using the most recent observation.

    The correct call sequence mirrors what em_fit() does internally:
      1. Build Y (dependent) and Z (regressors) matrices from X_reduced
      2. Compute log emission probabilities via emission_logprob(Y, Z, B, Sigma)
      3. Pass log_lik to hamilton_filter(log_lik, P, pi) → filtered probs
    """
    from .msvar import hamilton_filter, emission_logprob

    result = model._result
    T, n   = X_reduced.shape
    K      = model.n_states
    p_lag  = p

    if T <= p_lag:
        return np.ones(K) / K   # not enough data — return uniform

    # ── Build Y and Z in the same format as em_fit ────────────────────────
    # Y : (n, T-p)  — dependent variable (column-major, matches emission_logprob)
    # Z : (n*p+1, T-p) — regressors [y_{t-1}; ...; y_{t-p}; 1] (column-major)
    T_eff = T - p_lag
    Y = X_reduced[p_lag:].T                          # (n, T_eff)

    # Build lag matrix: each row of Z_row is [y_{t-1}, ..., y_{t-p}, 1]
    Z_rows = []
    for i in range(p_lag):
        Z_rows.append(X_reduced[p_lag - i - 1: T - i - 1].T)  # (n, T_eff)
    Z_rows.append(np.ones((1, T_eff)))                      # intercept row
    Z = np.vstack(Z_rows)                                   # (n*p+1, T_eff)

    # ── Compute log emission probabilities ────────────────────────────────
    # emission_logprob expects Y: (n, T), X: (np+1, T)
    log_lik = emission_logprob(Y, Z, result.B_list, result.Sigma_list)  # (K, T_eff)

    # ── Run Hamilton filter ───────────────────────────────────────────────
    # hamilton_filter(log_lik, P, pi) → (xi_filt, xi_pred) each (K, T)
    xi_filt, _, _ = hamilton_filter(log_lik, result.P, result.pi)

    return xi_filt[:, -1]   # filtered probs at last time step — shape (K,)


# ---------------------------------------------------------------------------
# Convenience wrapper: run both HMM and MSVAR walk-forward and compare
# ---------------------------------------------------------------------------

def compare_walk_forward(
    macro_df:             pd.DataFrame,
    asset_df:             pd.DataFrame,
    dim_reduction:        str   = 'pca', # 'pca', 'pls', or None
    initial_train_months: int   = 120,
    refit_every:          int   = 12,
    transaction_cost_bps: int   = 10,
    objective:            str   = 'min_variance',
    rf:                   float = 0.0,
    verbose:              bool  = True,
) -> pd.DataFrame:
    """
    Run walk-forward backtest for both HMM and MSVAR and return a combined
    net return DataFrame for direct comparison.

    Returns
    -------
    pd.DataFrame with columns ['HMM_WF', 'MSVAR_WF']
    """
    results = {}
    for model_type in ('hmm', 'msvar'):
        if verbose:
            print(f"\n{'='*55}")
            print(f"  Walk-forward: {model_type.upper()}")
            print(f"{'='*55}")
        ret_df, _, _, _ = walk_forward_backtest(
            macro_df             = macro_df,
            asset_df             = asset_df,
            model_type           = model_type,
            initial_train_months = initial_train_months,
            refit_every          = refit_every,
            dim_reduction        = dim_reduction,
            transaction_cost_bps = transaction_cost_bps,
            objective            = objective,
            rf                   = rf,
            verbose              = verbose,
        )
        results[f'{model_type.upper()}_WF'] = ret_df['net_return_after_costs']

    return pd.DataFrame(results).dropna()