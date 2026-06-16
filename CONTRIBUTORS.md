# Contributors

## My Dam — [\@mydam169](https://github.com/mydam169)

**Dimensionality reduction analysis** (`notebooks/03_dimensionality_reduction.ipynb`, `src/preprocess.py`)

-    Conducted VIF, PCA, and PLS analysis on 15 macro indicators to select the preprocessing method for the walk-forward engine and concluded PLS is preferred over PCA on the basis of portfolio-objective alignment;

-   Implemented `apply_pls()`, `pls_loading_table()`, and the unified `apply_dim_reduction()` and `loading_table()` interfaces in `preprocess.py`

-   Extended robust scaling from `StandardScaler` to `RobustScaler` (5–95 percentile range) to reduce sensitivity to macro outliers (COVID April 2020)

**Gaussian HMM implementation** (`src/hmm_model.py`, `notebooks/04_hmm_experiments.ipynb`)

-    Wrapped `hmmlearn` with a clean interface compatible with the MSVAR model for model-agnostic downstream use

-   Implemented a manual forward algorithm for filtered probabilities (distinct from hmmlearn's smoothed `predict_proba`); added `_sort_states()` for consistent regime labeling across restarts based on regime-conditional covariance trace

**MSMH-VAR(1) implementation** (`src/msvar.py`, `src/msvar_model.py`, `notebooks/05_msvar_experiments.ipynb`) -

-   Built the full MSMH-VAR(1) estimation engine from scratch (no existing Python/R package available)

-   Implemented the Hamilton filter (forward pass) and Kim smoother (backward pass) within an EM loop - Includes numerical safeguards: log-space Bayes updates, Cholesky-based emission log-likelihoods, PD regularization in the M-step, and multiple random restarts to avoid local optima

**Portfolio optimization engine** (`src/portfolio.py`)

-   Implemented regime-conditional mean-variance optimization via `cvxpy`

-   Two objectives are considered: global minimum variance-GMV (primary) and maximum Sharpe (MVO) with Ledoit-Wolf shrinkage for covariance estimation; Bayes-Stein shrinkage for expected returns (Jorion 1986). GMV is confirmed superior to MVO across all model variants in all backtests

**Walk-forward backtesting engine** (`src/walk_forward.py`, `notebooks/07_walk_forward.ipynb`)

-   Designed and implemented the primary evaluation framework: expanding-window walk-forward with annual refit, filtered (not smoothed) regime probabilities at each rebalancing date, and label-consistency enforcement across refit windows

-   Implemented three dimensionality-reduction branches (`dim_reduction='pls'|'pca'|'none'`) within the walk-forward loop, with per-window PLS fitting against the aligned asset return series to avoid look-ahead bias

------------------------------------------------------------------------

## Asad Ali Akhtar — [\@AsadAliAkhtar](https://github.com/AsadAliAkhtar)

**Data extraction** (`notebooks/01_data_extraction.ipynb`): Extracted13 macroeconomic indicators and three asset price series (VFINX, VUSTX, gold) from various sources (Sample: January 1990 – December 2025)

**Data wrangling and preprocessing** (`notebooks/02_data_wrangling.ipynb`)

-   Conducted ADF/KPSS stationarity tests on all indicators and selected appropriate transformations (log-difference, first difference, or level)

-   Computed monthly asset returns via percentage change on price series

------------------------------------------------------------------------

## Khang Manh Bui

**Fixed-split backtesting engine** (`src/backtest.py`, `notebooks/05_backtesting.ipynb`)

-   Designed and implemented the production backtest with drift-adjusted transaction costs -

-   Ensured one-month signal lag to prevent look-ahead bias; 10 bps one-way transaction costs

-   Added tax sensitivity layer approximating capital gains drag from rebalancing

NOTE: Walk-forward replaces fixed-split as the primary backtesting method in this repo.