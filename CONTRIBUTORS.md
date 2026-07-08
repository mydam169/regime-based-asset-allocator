# Contributors

## My Dam — [\@mydam169](https://github.com/mydam169)

**Dimensionality reduction analysis** (`notebooks/03_dimensionality_reduction.ipynb`, `src/preprocess.py`)

Conducted VIF, PCA, and PLS analysis on 15 macro indicators to select the preprocessing method for the walk-forward engine and concluded PLS is preferred over PCA on the basis of portfolio-objective alignment

**Gaussian HMM implementation** (`src/hmm_model.py`, `notebooks/04_hmm_experiments.ipynb`)

-   Wrapped `hmmlearn` with a clean interface compatible with the MSVAR model 

-   Implemented a manual forward algorithm for filtered probabilities (distinct from hmmlearn's smoothed `predict_proba`); added a regime-sorting method (`_sort_states()`) for consistent regime labeling across restarts based on regime-conditional covariance trace

**MSMH-VAR(1) implementation** (`src/msvar.py`, `src/msvar_model.py`, `notebooks/05_msvar_experiments.ipynb`) -

-   Built the full MSMH-VAR(1) estimation engine from scratch (no existing Python/R package available)

-   Implemented the Hamilton filter (forward pass) and Kim smoother (backward pass) within an EM loop

**Portfolio optimization engine** (`src/portfolio.py`)

-   Implemented regime-conditional mean-variance optimization via `cvxpy`
-   Two objectives are considered: global minimum variance-GMV (primary) and maximum Sharpe (MVO) with Ledoit-Wolf shrinkage for covariance estimation and Bayes-Stein shrinkage for expected returns (Jorion 1986). GMV is confirmed superior to MVO across all model variants in all backtests

**Walk-forward backtesting engine** (`src/walk_forward.py`, `notebooks/07_walk_forward.ipynb`)

Implemented expanding-window walk-forward backtesting with annual refit to avoid look-ahead bias.



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
