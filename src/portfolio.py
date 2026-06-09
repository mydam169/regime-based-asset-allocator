"""
Regime-conditional portfolio optimization via mean-variance (MVO).

Uses cvxpy for explicit, readable constraint specification.

Shrinkage
---------
- Ledoit-Wolf shrinkage on covariance matrices  (all regimes)
- Bayes-Stein shrinkage on expected return vectors (regime-conditional means
  are shrunk toward the grand mean; particularly important for the recession
  regime which typically has only 15-20 observations)

Public API
----------
Core solver         : _solve_mvo()
Moment estimation   : estimate_regime_moments()       — regime-conditional
                      estimate_unconditional_moments() — full-sample (benchmark)
Optimization loops  : optimize_regimes()              — GMV + max-Sharpe per regime
Benchmark builder   : build_unconditional_portfolio() — fixed-weight MVO benchmark
Frontier tracer     : MeanVariancePortfolio           — parametric EF + summary table
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
from sklearn.covariance import LedoitWolf

from .plots import _fmt_matrix


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """Output of a single MVO solve."""
    weights    : np.ndarray
    exp_return : float
    volatility : float
    sharpe     : float
    regime     : int        # None for regime-agnostic (unconditional) portfolios
    portfolio  : str
    asset_names: list

    def summary(self):
        ann  = 12
        sqan = np.sqrt(ann)
        print(f"\n── {self.portfolio}  |  Regime {self.regime} ──")
        print(f"  Expected return (ann.) : {self.exp_return * ann  * 100:+.2f}%")
        print(f"  Volatility     (ann.) : {self.volatility * sqan  * 100:.2f}%")
        print(f"  Sharpe ratio          : {self.sharpe     * sqan:.4f}")
        print(f"  Weights:")
        for name, w in zip(self.asset_names, self.weights):
            print(f"    {name:<14}: {w:.4f}  ({w*100:.1f}%)")


# ---------------------------------------------------------------------------
# MVO core solver  (private — called by all public functions)
# ---------------------------------------------------------------------------

def _constraints(w, n, long_only, max_weight):
    """Standard MVO constraint set: full investment + optional long-only + cap."""
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    if max_weight < 1.0:
        cons.append(w <= max_weight)
    return cons


def _solve_mvo(mu, Sigma, objective='min_variance',
               rf=0.0, gamma=3.0,
               long_only=True, max_weight=1.0) -> np.ndarray:
    """
    Solve a single MVO problem using cvxpy + CLARABEL solver.

    Objectives
    ----------
    'min_variance'  — global minimum-variance portfolio (GMV).
                      No return target; minimises w'Σw subject to
                      full-investment and optional long-only constraints.

    'max_sharpe'    — maximum Sharpe ratio (tangency) portfolio.
                      Solved via the Cornuejols-Tütüncü variable substitution
                      y = w/(w'(μ−rf)), κ = 1/(w'(μ−rf)) which converts the
                      fractional program into a convex QP.

    Returns
    -------
    (n,) optimal weight vector.
    Falls back to equal weights if the solver fails.
    """
    n     = len(mu)
    Sigma = 0.5 * (np.array(Sigma) + np.array(Sigma).T)   # enforce symmetry

    if objective == 'min_variance':
        w    = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)),
                          _constraints(w, n, long_only, max_weight))

    elif objective == 'max_sharpe':
        # Tobin / Cornuejols-Tütüncü substitution:
        #   y = w / (w'(μ-rf)),   κ = 1 / (w'(μ-rf))
        # Minimise y'Σy  s.t.  (μ-rf)'y = 1,  Σy = κ,  κ ≥ 0
        # Recover weights as w = y / κ
        y, kap    = cp.Variable(n), cp.Variable()
        excess_mu = mu - rf
        cons = [excess_mu @ y == 1, cp.sum(y) == kap, kap >= 0]
        if long_only:
            cons.append(y >= 0)
        if max_weight < 1.0:
            cons.append(y <= max_weight * kap)
        prob = cp.Problem(cp.Minimize(cp.quad_form(y, Sigma)), cons)
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ('optimal', 'optimal_inaccurate') and kap.value > 1e-10:
            w_opt = np.clip(y.value / kap.value,
                            0 if long_only else -np.inf, max_weight)
            return w_opt / w_opt.sum()
        print("  Warning: max_sharpe solver failed — returning equal weights.")
        return np.ones(n) / n

    else:
        raise ValueError(
            f"Unknown objective '{objective}'. "
            f"Valid options: 'min_variance', 'max_sharpe'."
        )

    prob.solve(solver=cp.CLARABEL)
    if prob.status in ('optimal', 'optimal_inaccurate') and w.value is not None:
        w_opt = np.clip(w.value, 0 if long_only else -np.inf, max_weight)
        return w_opt / w_opt.sum()

    print(f"  Warning: solver status '{prob.status}' — returning equal weights.")
    return np.ones(n) / n


# ---------------------------------------------------------------------------
# Bayes-Stein mean shrinkage
# ---------------------------------------------------------------------------

def bayes_stein_shrinkage(mu_k: np.ndarray,
                           sigma_k: np.ndarray,
                           n_obs: int,
                           mu_global: np.ndarray) -> tuple:
    """
    Shrink a regime-conditional sample mean toward the grand mean.

    Reference: Jorion (1986), "Bayes-Stein Estimation for Portfolio Analysis",
    Journal of Financial and Quantitative Analysis.

    Motivation: When the number of observations in a regime is small, the sample mean is an unreliable estimate
    of the true conditional mean.  The Bayes-Stein estimator treats mu_global
    as an informative prior and blends the sample mean toward it with intensity
    phi inversely proportional to the number of observations:

        mu_BS = (1 - phi) * mu_sample  +  phi * mu_global

    where phi → 1 when n_obs is small (heavy shrinkage to the prior) and
    phi → 0 when n_obs is large (sample mean dominates).

    Parameters
    ----------
    mu_k      : (n,) regime-conditional sample mean
    sigma_k   : (n,n) regime-conditional Ledoit-Wolf covariance
    n_obs     : number of asset-return observations in regime k
    mu_global : (n,) grand mean across all regimes (the shrinkage target),
                computed as an observation-count-weighted average

    Returns
    -------
    mu_bs : (n,) shrunk mean estimate used for portfolio optimisation
    phi   : float in [0, 1] — shrinkage intensity;
            phi ≈ 1 → dominated by prior (few observations)
            phi ≈ 0 → dominated by sample mean (many observations)
    """
    n    = len(mu_k)
    diff = mu_k - mu_global

    # Mahalanobis distance of mu_k from the prior mu_global
    try:
        sigma_inv = np.linalg.inv(sigma_k)
        mahal     = float(diff @ sigma_inv @ diff)
    except np.linalg.LinAlgError:
        # Sigma is singular (can happen with very few observations);
        # fall back to Euclidean distance
        mahal = float(diff @ diff)

    # Jorion (1986) eq. 14:  phi = (n+2) / (n_obs * mahal + n + 2)
    # Clipped to [0, 1] for numerical safety
    phi  = (n + 2) / (n_obs * mahal + n + 2)
    phi  = float(np.clip(phi, 0.0, 1.0))

    mu_bs = (1.0 - phi) * mu_k + phi * mu_global
    return mu_bs, phi


# ---------------------------------------------------------------------------
# Regime-conditional moment estimation
# ---------------------------------------------------------------------------

def estimate_regime_moments(df_train, asset_cols, state_col='state',
                             use_bayes_stein: bool = True) -> dict:
    """
    Estimate state-conditional means and covariance matrices.

    Covariance : Ledoit-Wolf shrinkage (scikit-learn).
    Mean       : Bayes-Stein shrinkage toward the grand mean (Jorion 1986).
                 Controlled by use_bayes_stein (default True).

    The output dict carries both the shrunk mean ('mu', used for MVO) and
    the raw sample mean ('mu_sample', kept for diagnostics), plus the
    Bayes-Stein shrinkage intensity ('mu_bs_phi') so callers can see how
    heavily each regime's mean was adjusted.

    Parameters
    ----------
    df_train        : pd.DataFrame with state and asset return columns
    asset_cols      : list of str — asset return column names
    state_col       : str — name of the integer state column
    use_bayes_stein : bool — apply Bayes-Stein to means (default True)

    Returns
    -------
    dict {k: {'mu', 'mu_sample', 'mu_bs_phi',
              'sigma', 'sigma_raw', 'lw_shrinkage', 'n_obs'}}
    """
    # ── Pass 1: raw moment estimates per regime ───────────────────────────────
    raw = {}
    for k in sorted(df_train[state_col].unique()):
        R_k      = df_train.loc[df_train[state_col] == k, asset_cols].values
        T_k, _   = R_k.shape
        mu_k     = R_k.mean(axis=0)
        lw       = LedoitWolf().fit(R_k)
        raw[k]   = {
            'R':           R_k,
            'mu_sample':   mu_k,
            'sigma':       lw.covariance_,
            'sigma_raw':   np.cov(R_k, rowvar=False),
            'lw_shrinkage': lw.shrinkage_,
            'n_obs':       T_k,
        }

    # ── Grand mean: observation-count-weighted pooled mean ───────────────────
    # Used as the Bayes-Stein shrinkage target.
    # Weighting by n_obs gives more influence to the dominant (expansion) regime.
    total_obs = sum(v['n_obs'] for v in raw.values())
    mu_global = sum(
        v['n_obs'] * v['mu_sample'] for v in raw.values()
    ) / total_obs

    # ── Pass 2: apply Bayes-Stein, build output dict ─────────────────────────
    moments = {}
    for k, r in raw.items():
        if use_bayes_stein:
            mu_final, phi = bayes_stein_shrinkage(
                r['mu_sample'], r['sigma'], r['n_obs'], mu_global
            )
        else:
            mu_final, phi = r['mu_sample'], 0.0

        moments[k] = {
            'mu':           mu_final,      # shrunk mean  → used for MVO
            'mu_sample':    r['mu_sample'],# raw sample mean → diagnostics only
            'mu_bs_phi':    phi,           # Bayes-Stein intensity
            'sigma':        r['sigma'],    # Ledoit-Wolf covariance
            'sigma_raw':    r['sigma_raw'],
            'lw_shrinkage': r['lw_shrinkage'],
            'n_obs':        r['n_obs'],
        }

        # ── Per-regime diagnostics ────────────────────────────────────────────
        print(f"\n── Regime {k}  ({r['n_obs']} observations) ──")
        print(f"  Mean returns — sample vs Bayes-Stein (annualized %):")
        for col, m_raw, m_bs in zip(
            asset_cols,
            r['mu_sample'] * 12 * 100,
            mu_final       * 12 * 100,
        ):
            print(f"    {col:<16}: sample {m_raw:+.2f}%  →  BS {m_bs:+.2f}%")
        if use_bayes_stein:
            intensity = 'heavy' if phi > 0.5 else 'mild'
            print(f"  Bayes-Stein φ  : {phi:.4f}  ({intensity} shrinkage)")
        print(f"  Ledoit-Wolf α  : {r['lw_shrinkage']:.4f}")
        print(f"  Implied annual volatilities (shrunk Σ):")
        for col, v in zip(
            asset_cols,
            np.sqrt(np.diag(r['sigma']) * 12) * 100,
        ):
            print(f"    {col:<16}: {v:.2f}%")

    return moments


def shrinkage_diagnostics(moments, asset_cols):
    """Print the difference between Ledoit-Wolf and raw covariance per regime."""
    print("\n── Ledoit-Wolf shrinkage effect (shrunk Σ − raw Σ) ──")
    for k, m in moments.items():
        diff = m['sigma'] - m['sigma_raw']
        print(f"\nRegime {k}  (α = {m['lw_shrinkage']:.4f}):")
        print(_fmt_matrix(diff, list(asset_cols)))


# ---------------------------------------------------------------------------
# Unconditional moment estimation  (regime-agnostic benchmark)
# ---------------------------------------------------------------------------

def estimate_unconditional_moments(df_returns, asset_cols) -> dict:
    """
    Estimate full-sample means and covariance ignoring regime labels.

    Purpose
    -------
    Provides the inputs for the regime-agnostic MVO benchmark portfolio.
    This is the most direct test of whether regime detection adds value:
    if the regime-switching strategies cannot beat a fixed portfolio that
    is optimally constructed on unconditional moments, the regime detection
    machinery adds no incremental value.

    Covariance: Ledoit-Wolf shrinkage on the full training sample.
    Mean: raw sample mean (no Bayes-Stein — there is no prior to shrink toward
          when regimes are ignored).

    Parameters
    ----------
    df_returns : pd.DataFrame of asset returns (no state column needed)
    asset_cols : list of str

    Returns
    -------
    dict with keys 'mu', 'sigma', 'sigma_raw', 'lw_shrinkage', 'n_obs'
    """
    R    = df_returns[list(asset_cols)].values
    T, _ = R.shape
    mu   = R.mean(axis=0)
    lw   = LedoitWolf().fit(R)

    print(f"\n── Unconditional moments  ({T} observations, all regimes pooled) ──")
    print(f"  Mean returns (annualized %):")
    for col, m in zip(asset_cols, mu * 12 * 100):
        print(f"    {col:<16}: {m:+.2f}%")
    print(f"  Ledoit-Wolf α  : {lw.shrinkage_:.4f}")
    print(f"  Implied annual volatilities:")
    for col, v in zip(asset_cols, np.sqrt(np.diag(lw.covariance_) * 12) * 100):
        print(f"    {col:<16}: {v:.2f}%")

    return {
        'mu':           mu,
        'sigma':        lw.covariance_,
        'sigma_raw':    np.cov(R, rowvar=False),
        'lw_shrinkage': lw.shrinkage_,
        'n_obs':        T,
    }


def build_unconditional_portfolio(df_train_returns, asset_cols,
                                   rf: float = 0.0,
                                   long_only: bool = True,
                                   max_weight: float = 1.0,
                                   objective: str = 'max_sharpe') -> 'PortfolioResult':
    """
    Build a fixed-weight MVO portfolio from unconditional moments.

    This portfolio is estimated once on training data and held fixed
    throughout the test period — it is the regime-agnostic MVO benchmark.

    The comparison to include in the backtest notebook is:
        HMM GMV     — regime-switching, minimum variance
        HMM MVO     — regime-switching, maximum Sharpe
        MSVAR GMV   — regime-switching, minimum variance
        MSVAR MVO   — regime-switching, maximum Sharpe
        Uncond. MVO — fixed weights, unconditional max-Sharpe  ← this function
        60/40       — fixed weights, naive
        Equal-weight— fixed weights, naive
        Buy & Hold  — fixed weights, naive

    Parameters
    ----------
    df_train_returns : pd.DataFrame of training-period asset returns
                       (no state column; covers all regimes pooled)
    asset_cols       : list of str
    rf               : monthly risk-free rate (default 0)
    long_only        : bool (default True)
    max_weight       : per-asset upper bound (default 1.0 = unconstrained)
    objective        : 'max_sharpe' or 'min_variance' (default 'max_sharpe')

    Returns
    -------
    PortfolioResult with regime=None
    """
    mom     = estimate_unconditional_moments(df_train_returns, asset_cols)
    w       = _solve_mvo(mom['mu'], mom['sigma'], objective=objective,
                         rf=rf, long_only=long_only, max_weight=max_weight)
    exp_ret = float(w @ mom['mu'])
    vol     = float(np.sqrt(w @ mom['sigma'] @ w))
    sharpe  = (exp_ret - rf) / vol if vol > 1e-12 else 0.0

    result = PortfolioResult(
        weights     = w,
        exp_return  = exp_ret,
        volatility  = vol,
        sharpe      = sharpe,
        regime      = None,
        portfolio   = f'Unconditional MVO ({objective})',
        asset_names = list(asset_cols),
    )
    result.summary()
    return result


# ---------------------------------------------------------------------------
# Regime-loop MVO
# ---------------------------------------------------------------------------

def optimize_regimes(moments, asset_names,
                     rf=0.0, long_only=True, max_weight=1.0) -> dict:
    """
    Solve MVO for each regime for the two reported objectives.

    Objectives
    ----------
    'min_variance' — GMV: reported as the primary result throughout the paper.
    'max_sharpe'   — Tangency: reported in the Appendix (MVO comparison).

    Note: 'max_return' (mean-variance utility) has been removed.  It was
    computed in the original code but never exported to CSV, never used in
    backtesting, and not reported in the paper.  Removing it keeps the
    results dict clean and avoids confusion.

    Parameters
    ----------
    moments     : dict from estimate_regime_moments()
    asset_names : array-like of str
    rf          : monthly risk-free rate
    long_only   : bool
    max_weight  : per-asset upper bound

    Returns
    -------
    dict {k: {'min_variance': PortfolioResult, 'max_sharpe': PortfolioResult}}
    """
    results = {}
    for k, m in moments.items():
        results[k] = {}
        for obj in ('min_variance', 'max_sharpe'):
            w_opt   = _solve_mvo(m['mu'], m['sigma'], objective=obj,
                                  rf=rf, long_only=long_only,
                                  max_weight=max_weight)
            exp_ret = float(w_opt @ m['mu'])
            vol     = float(np.sqrt(w_opt @ m['sigma'] @ w_opt))
            sharpe  = (exp_ret - rf) / vol if vol > 1e-12 else 0.0
            res     = PortfolioResult(
                w_opt, exp_ret, vol, sharpe, k, obj, list(asset_names)
            )
            results[k][obj] = res
            res.summary()
    return results

