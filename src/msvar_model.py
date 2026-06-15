"""
Estimate hidden regimes using a Markov-Switching VAR (MSMH-VAR).

Mirrors the interface of HMMRegimeModel in hmm_model.py so downstream
code (portfolio optimization, backtesting) is model-agnostic.

Architecture
------------
msvar.py       — pure implementation: Hamilton filter, Kim smoother,
                 EM loop, em_fit(), MSVARResult, helpers.  No class.
msvar_model.py — MSVARRegimeModel class only.  Imports em_fit() from
                 msvar.py exactly as hmm_model.py imports GaussianHMM
                 from hmmlearn.

Key difference from HMM
-----------------------
VAR(p) consumes p observations as lags so the effective sample size is
T - p.  Regime labels are therefore aligned to dates_train[p:], not
dates_train.  This is handled automatically — the caller just needs to
use dates_train[result.p:] when building the state series.

State labeling
--------------
_sort_regimes() inside em_fit() reorders states after every restart so
state 0 = expansion (low PC1 unconditional mean) and state 1 = recession
(high PC1 unconditional mean).  No manual label flip is needed in the
notebook — this matches the HMMRegimeModel convention exactly.
"""

import numpy as np
from .msvar import em_fit


class MSVARRegimeModel:
    """
    MSMH-VAR for macroeconomic regime detection.

    Wraps em_fit() and exposes the same interface as HMMRegimeModel so
    downstream code (portfolio optimization, backtesting) is model-agnostic.

    Parameters
    ----------
    n_states     : int   — number of regimes (default 2)
    p            : int   — VAR lag order (default 1)
    n_restarts   : int   — random restarts for EM (default 5)
    max_iter     : int   — max EM iterations per restart (default 300)
    tol          : float — convergence tolerance on log-likelihood
    random_state : int   — seed for reproducibility
    verbose      : bool  — print EM progress
    """

    def __init__(self, n_states: int = 2, p: int = 1,
                 n_restarts: int = 5, max_iter: int = 300,
                 tol: float = 1e-6, random_state: int = 42,
                 verbose: bool = True):
        self.n_states     = n_states
        self.p            = p
        self.n_restarts   = n_restarts
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state
        self.verbose      = verbose
        self._result      = None

    def fit(self, X: np.ndarray) -> "MSVARRegimeModel":
        """
        Fit MSMH-VAR via EM (Hamilton filter + Kim smoother).

        Parameters
        ----------
        X : (T, n_components) PCA-projected and winsorized training array.
            n_components is the number of PCA components, not the original
            number of macro indicators.  State ordering (state 0 = expansion,
            state 1 = recession) is handled automatically by _sort_regimes()
            inside em_fit().

        Returns
        -------
        self
        """
        self._result = em_fit(
            data         = X,
            K            = self.n_states,
            p            = self.p,
            n_restarts   = self.n_restarts,
            max_iter     = self.max_iter,
            tol          = self.tol,
            random_state = self.random_state,
            verbose      = self.verbose,
        )
        # Print state labeling confirmation — mirrors HMMRegimeModel output
        for k in range(self.n_states):
            intercept = self._result.B_list[k][0, 0]
            label     = "Expansion" if k == 0 else "Recession"
            print(f"State {k} = {label}  (PC1 intercept: {intercept:.3f})")
        print(f"Transition matrix:\n{np.round(self._result.P, 4)}")
        print(f"State durations (months): "
              f"{[f'{d:.1f}' for d in self._result.expected_durations]}")
        return self

    def predict_states(self, use_smoothed: bool = True) -> np.ndarray:
        """
        Most-probable state sequence via argmax — shape (T - p,).

        Parameters
        ----------
        use_smoothed : bool — use smoothed probs (True) or filtered (False).
                       Use filtered for walk-forward / out-of-sample evaluation
                       to avoid look-ahead bias.
        """
        self._check_fitted()
        return self._result.regime_sequence(use_smoothed=use_smoothed)

    def smoothed_probs(self) -> np.ndarray:
        """
        Smoothed probabilities P(S_t=k | y_{1:T}) — shape (K, T-p).

        Uses the full observation sequence.  Appropriate for in-sample
        analysis and retrospective regime labeling.
        """
        self._check_fitted()
        return self._result.xi_smoothed

    def filtered_probs(self) -> np.ndarray:
        """
        Filtered probabilities P(S_t=k | y_{1:t}) — shape (K, T-p).

        Uses only information up to time t.  Correct choice for walk-forward
        and out-of-sample evaluation — avoids look-ahead bias from the Kim
        smoother's backward pass.
        """
        self._check_fitted()
        return self._result.xi_filtered

    def score(self) -> float:
        """Total log-likelihood of the fitted model."""
        self._check_fitted()
        return self._result.log_likelihood

    def summary(self):
        """Print estimation summary."""
        self._check_fitted()
        return self._result.summary()

    # ── Aliases for interface consistency with HMMRegimeModel ────────────────

    def predict(self, use_smoothed: bool = True) -> np.ndarray:
        """Alias for predict_states()."""
        return self.predict_states(use_smoothed=use_smoothed)

    def predict_probabilities(self) -> np.ndarray:
        """Alias for smoothed_probs()."""
        return self.smoothed_probs()

    # ── Properties matching HMMRegimeModel attribute names ───────────────────

    @property
    def n_components(self) -> int:
        """Number of regimes."""
        return self.n_states

    @property
    def transmat_(self) -> np.ndarray:
        """
        Transition matrix (K, K) — matches HMMRegimeModel attribute name.
        transmat_[i, j] = P(S_t = j | S_{t-1} = i).
        """
        self._check_fitted()
        return self._result.P

    @property
    def means_(self) -> np.ndarray:
        """
        Regime-conditional VAR intercepts — shape (K, n_components).

        Returns the intercept column B[:, 0] for each regime in PC space.
        Mirrors HMMRegimeModel.means_ shape (K, n) for interface consistency.

        Note: for the full VAR coefficient matrices use self._result.B_list.
        """
        self._check_fitted()
        return np.array([self._result.B_list[k][:, 0]
                         for k in range(self.n_states)])

    @property
    def covars_(self) -> np.ndarray:
        """
        Regime-conditional covariance matrices — shape (K, n, n).
        Mirrors HMMRegimeModel.covars_ for interface consistency.
        """
        self._check_fitted()
        return np.array(self._result.Sigma_list)

    @property
    def expected_durations(self) -> np.ndarray:
        """Expected duration in each regime (months)."""
        self._check_fitted()
        return self._result.expected_durations

    def _check_fitted(self):
        if self._result is None:
            raise RuntimeError("Call fit() before using the model.")
