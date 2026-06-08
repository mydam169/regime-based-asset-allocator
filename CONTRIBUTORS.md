# Contributors

## My Dam — [@mydam169](https://github.com/mydam169)

**MSMH-VAR(1) implementation** (`src/msvar.py`, `src/msvar_model.py`, `notebooks/04_msvar_experiments.ipynb`)
- Built the full MSMH-VAR(1) estimation engine from scratch — no existing Python or R package was available
- Implemented the Hamilton filter (forward pass) and Kim smoother (backward pass) within an EM loop
- Includes numerical safeguards: log-space Bayes updates, Cholesky-based emission log-likelihoods, PD regularization in the M-step, and multiple random restarts to avoid local optima

**Gaussian HMM implementation** (`src/hmm_model.py`, `notebooks/03_hmm_experiments.ipynb`)
- Wrapped `hmmlearn` with a clean interface compatible with the MSVAR model for model-agnostic downstream use
- Implemented a manual forward algorithm for filtered probabilities (distinct from hmmlearn's smoothed `predict_proba`)

**Portfolio optimization engine** (`src/portfolio.py`)
- Implemented regime-conditional mean-variance optimization via `cvxpy`
- Three objectives: global minimum variance, maximum Sharpe (Tobin substitution), and mean-variance utility
- Ledoit-Wolf shrinkage for covariance estimation; Bayes-Stein shrinkage for expected returns
- Includes `MeanVariancePortfolio` class with parametric efficient frontier tracing

**Robustness checks**
- Implemented K=3 regime robustness check
- Added GMV vs. MVO comparison (Appendix A) confirming GMV is superior regardless of regime model

---

## Asad Ali Akhtar — [@AsadAliAkhtar](https://github.com/AsadAliAkhtar)

**Data extraction** (`notebooks/01_data_extraction.ipynb`)
- Pulled 13 macroeconomic indicators from FRED API and three asset price series (VFINX, VUSTX, gold) from yfinance
- Sample: January 1990 – December 2025

**Data wrangling and preprocessing** (`notebooks/02_data_wrangling.ipynb`, `src/preprocess.py`)
- Conducted ADF/KPSS stationarity tests on all indicators and selected appropriate transformations (log-difference, first difference, or level)
- Applied StandardScaler standardization to produce the model-ready `macro_clean.csv`
- Computed monthly asset returns via percentage change on price series

---

## Khang Manh Bui

**Backtesting engine** (`src/backtest.py`, `notebooks/05_backtesting.ipynb`)
- Designed and implemented the production backtest with drift-adjusted transaction costs
- One-month signal lag to prevent look-ahead bias; 10 bps one-way transaction costs
- Tax sensitivity layer approximating capital gains drag from rebalancing
- Out-of-sample evaluation: December 2018 – December 2025