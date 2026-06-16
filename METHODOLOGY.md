# Methodology

This document covers the full design rationale for the preprocessing pipeline, regime model estimation, portfolio optimization, and walk-forward backtesting framework. For results, see [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).

------------------------------------------------------------------------

## 1. Data

**Macroeconomic indicators (15 series, FRED, Jan 1990 – Dec 2025):** industrial production, real personal income, unemployment rate, initial jobless claims, CPI, oil price, VIX, BAA-10Y credit spread, 10Y-2Y yield curve slope, federal funds rate, consumer sentiment, housing starts, M2 money supply, real retail sales, nonfarm payrolls.

All series are transformed to stationarity (log-differences or first differences, selected by ADF/KPSS test) and robust-scaled using `RobustScaler` with a 5–95 percentile range. Robust scaling rather than standard scaling limits the influence of extreme observations — notably the April 2020 COVID outlier, which would otherwise dominate the standardization.

**Assets:** VFINX (S&P 500 index fund), VUSTX (long-term Treasury fund), gold (USD/oz) — monthly total returns from 1990.

------------------------------------------------------------------------

## 2. Dimensionality reduction

### Why it is necessary

Both regime models are estimated on latent macro components rather than the full 15 raw indicators. The motivation is parameter count and EM convergence stability.

With 15 indicators, a 2-state Gaussian HMM requires estimating 15 means and 15 variances per regime under diagonal covariance — 60 parameters from roughly 350 training observations. The MSVAR(1) is far worse:

| Component                                                  | Parameters |
|------------------------------------------------------------|------------|
| VAR coefficient matrices B (2 regimes × 15 vars × 16 cols) | 480        |
| Covariance matrices Σ (2 regimes × 15×16/2)                | 240        |
| Transition matrix                                          | 2          |
| **Total**                                                  | **722**    |

The recession regime covariance block alone (120 parameters) is identified from only \~20 recession observations — a ratio that guarantees degenerate M-step estimates and numerically unstable EM convergence. Reducing to 4 latent components brings the MSVAR total to 60 parameters, well within the identifiable range even for the recession regime.

Beyond parameter count, correlated inputs inflate the condition number of the within-regime scatter matrix, making Cholesky factorization in the emission likelihood unstable. Orthogonal or near-orthogonal components (from PCA or PLS) eliminate this problem.

### Method choice: PLS over PCA

PCA finds directions in macro space that maximize variance in the indicators, regardless of their relevance to asset returns. PLS finds directions that maximally co-vary with asset returns, aligning preprocessing directly to the portfolio objective. This distinction matters when only a subset of macro variation is return-predictive — exactly the case here, where 15 correlated macro series contain significant idiosyncratic noise.

The choice is motivated by Kelly & Pruitt (2015), who show that PLS outperforms PCA as a dimensionality reduction step when the target Y is spanned by only a low-dimensional subspace of X. With three asset return series and 15 macro indicators, this condition is plausibly satisfied.

**Important:** preferring PLS over PCA is not justified by PCA explaining more variance (it always will, by construction) or by VIF analysis (VIF below 5 in this dataset indicates moderate, not severe, multicollinearity). The sole empirical justification is the walk-forward backtest outcome: HMM-PLS Sortino 1.463 vs HMM-PCA 1.305. A single evaluation window is limited evidence and should be interpreted cautiously.

### Implementation details

-   PCA: variance threshold 85%, minimum 2 components retained
-   PLS: 4 components, fitted on the training-window macro and asset return series jointly; Y is the raw asset return matrix (no standardization required since all three series are monthly returns on the same scale)
-   Both: winsorization at ±4 standard deviations after projection, applied independently to training and test projections

------------------------------------------------------------------------

## 3. Regime detection

### Models

**Gaussian HMM** (`src/hmm_model.py`): Baum-Welch EM with 10 random restarts; diagonal covariance (orthogonal PCA/PLS components justify removing cross-component correlations, reducing parameters from n(n+1)/2 to n per regime); transition matrix initialized to a business-cycle prior (p_stay ≈ 0.95 for expansion, 0.90 for recession) to prevent degenerate EM starts where one state absorbs all observations.

**MSMH-VAR(1)** (`src/msvar.py`): Hamilton filter (forward pass) + Kim smoother (backward pass) within an EM loop, implemented from scratch — no existing Python or R package was available at the time of writing. Numerical safeguards include log-space forward recursion, Cholesky-based emission log-likelihoods, and positive definite regularization in the M-step covariance update. 5 random restarts.

### State ordering

After fitting, states are ordered by the trace of the regime-conditional covariance matrix $tr(\Sigma_k)$. The state with the higher total variance is labeled recession (state 1). This criterion is sign-invariant — unlike sorting by the mean of a specific component, it does not depend on the orientation of the PCA or PLS axes, which can flip sign between refit windows. It is equivalent to sorting by the second moment scalar $s_k = tr(\Sigma_k) + \lVert μ_k \lVert^2$ when means are small relative to variances, which is the case here after robust scaling.

### Filtered vs. smoothed probabilities

The Kim smoother produces smoothed probabilities $P(S_t = k | y_{1:T})$, which condition on the full observation sequence including future data. These are appropriate for retrospective analysis but introduce look-ahead bias in out-of-sample evaluation. The walk-forward engine uses filtered probabilities $P(S_t = k | y_{1:t})$, computed via a manual forward pass (Hamilton filter), which condition only on data available at time t.

------------------------------------------------------------------------

## 4. Portfolio optimization

Regime-conditional asset return moments are estimated separately for each detected regime using the asset returns falling in that regime over the training window:

-   **Covariance:** Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`), which provides a well-conditioned estimator even when the number of observations per regime is small relative to the number of assets.
-   **Mean:** Bayes-Stein shrinkage toward the grand (pooled) mean (Jorion 1986), with shrinkage intensity φ inversely proportional to the number of observations in the regime. For the recession regime (\~20 observations), φ is typically large, pulling the conditional mean strongly toward the unconditional mean and preventing extreme return forecasts from driving extreme positions.

**Objective:** Global minimum variance (GMV) is the primary objective. GMV is confirmed superior to max-Sharpe MVO across all model variants (HMM Sortino 1.294 vs 0.825; MSVAR 1.241 vs 0.763). The underperformance of MVO is consistent with estimation error in the conditional means dominating any return-forecasting benefit. GMV avoids the mean-estimation problem entirely.

**Solver:** cvxpy with CLARABEL. Constraints: full-investment and long-only (weights lie between 0 and 1 and sum to 1).

------------------------------------------------------------------------

## 5. Walk-forward backtesting

### Design overview

| Parameter               | Value                            |
|-------------------------|----------------------------------|
| Initial training window | 346 months (Feb 1990 – Nov 2018) |
| Test period             | Jan 2019 – Dec 2025 (84 months)  |
| Refit frequency         | Annual (every 12 months)         |
| Regime probability type | Filtered (no look-ahead)         |
| Signal lag              | 1 month                          |
| Transaction costs       | 10 bps one-way, drift-adjusted   |

### How it differs from a fixed train/test split

A conventional fixed split estimates all model components once on the training window and applies them unchanged to the test period. This has two problems: (1) parameters become stale as new data arrives — in particular, regime-conditional asset return moments (means and covariances) are never updated to reflect the post-training return environment; (2) smoothed probabilities are typically used for the regime sequence, introducing look-ahead bias from the Kim smoother's backward pass.

The walk-forward design addresses both. At each annual refit date, the following are all re-estimated from scratch on the full expanded dataset available at that point:

1.  `RobustScaler` — refitted on all macro data through the refit date
2.  PLS or PCA reducer — refitted on the expanded scaled macro data (PLS also sees the expanded asset return history)
3.  Regime model (HMM or MSVAR) — re-estimated via EM on the projected components
4.  Regime-conditional asset return moments — re-estimated on asset returns falling in each regime over the full training history through the refit date

This means that as the test period progresses, the portfolio optimizer's view of expected returns and covariances in each regime continuously incorporates new information, rather than remaining anchored to a 2018 estimate.

### Between-refit regime assignment

In the months between annual refits, the current regime is identified by the filtered probability $P(S_t = k | y_{1:t})$ from the most recently fitted model, evaluated on the new macro observation at time t. The regime label is the argmax of the filtered probability vector. This is the correct choice for a real trading context: the portfolio manager observes macro data through t and must make an allocation decision without any future information.

### Signal lag

Portfolio weights implied by the regime signal at time t are applied to asset returns at time t+1. This reflects the operational reality that identifying the current regime, computing optimal weights, and executing trades cannot all occur within the same month-end window.

### Transaction costs

At each rebalancing date (both annual refits and monthly regime updates), the portfolio has drifted from its target weights due to differential asset price movements since the last rebalance. The drift-adjusted portfolio weight for asset j at time t is:

$$
  w_j^{drift} = w_j^{t-1} * (1 + r_j^{t-1}) / \Sigma_i w_i^{t-1} * (1 + r_i^{t-1})
$$

Transaction costs of 10 bps one-way are applied to the traded notional: $$
    cost_t = 0.001 \times \Sigma_j \Vert w_j^{target} - w_j^{drift}\Vert
$$ This formulation avoids overstating costs when turnover is low (e.g. when the regime signal is unchanged month-over-month and only drift correction is needed).

------------------------------------------------------------------------

## 6. Known limitations

-   **Three-asset universe.** With only equities, Treasuries, and gold, naive equal weight performs competitively — permanent gold exposure is the dominant driver of downside protection. A broader universe (commodities, REITs, credit) would provide a more demanding test of regime-based allocation.
-   **Single evaluation window.** The walk-forward covers one historical path (2019–2025). The PLS advantage over PCA is empirically supported but should be interpreted cautiously — one backtest is limited evidence for a structural claim.
-   **Regime inflation with PLS.** PLS classifies more periods as recessionary than NBER dates would suggest, because it optimizes for macro-return co-movement rather than the textbook recession definition. This is defensible for a portfolio model but should be noted when comparing regime sequences to external benchmarks.
-   **Parameter uncertainty.** Regime-conditional moments are treated as point estimates; accounting for estimation uncertainty via Bayesian posterior predictive distributions would produce more conservative allocations, particularly for the recession regime with \~20 observations per refit window.