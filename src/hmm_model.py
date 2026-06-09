"""
Estimate hidden regimes using a Gaussian Hidden Markov Model.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMRegimeModel:
    """
    Gaussian HMM for macroeconomic regime detection.

    Uses multiple random restarts to avoid local optima in the EM algorithm.
    Exposes a common interface with MSVARRegimeModel so downstream code
    (portfolio optimization, backtesting) is model-agnostic.

    Covariance type
    ---------------
    After PCA, components are orthogonal by construction, which justifies
    using 'diag' covariance (no cross-component correlations).  This reduces
    covariance parameters from n(n+1)/2 to n per regime — critical for the
    recession regime which typically has only 15-20 observations.  With n=6
    PCA components, 'full' covariance requires 21 parameters per regime from
    ~20 recession observations, which causes degeneracy (one regime collapses
    to a single observation).  'diag' requires only 6 parameters per regime.

    Transition matrix initialization
    ---------------------------------
    The default 'stmc' init_params randomizes the transition matrix, which
    can lead to degenerate solutions where one regime absorbs all observations.
    Setting constrain_transmat=True initializes the transition matrix to a
    reasonable business-cycle prior (high persistence, ~5% monthly transition
    probability) and removes 'tm' from init_params so EM refines but does not
    randomly reinitialize the transition matrix or start probabilities.

    State labeling
    --------------
    The HMM assigns state labels (0, 1) arbitrarily — there is no guarantee
    that state 0 corresponds to expansion.  After fitting, _sort_states()
    reorders states so that state 0 is always expansion (lowest mean of PC1,
    which loads positively on unemployment) and state 1 is always recession.
    This ensures consistent labeling across restarts and random seeds.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 200,
                 n_restarts: int = 10, tol: float = 1e-4,
                 random_state: int = 42,
                 covariance_type: str = 'diag', # change from diag to full 
                 constrain_transmat: bool = True):
        self.n_states           = n_states
        self.n_iter             = n_iter
        self.n_restarts         = n_restarts
        self.tol                = tol
        self.random_state       = random_state
        self.covariance_type    = covariance_type
        self.constrain_transmat = constrain_transmat
        self._model             = None

    def fit(self, X: np.ndarray) -> "HMMRegimeModel":
        """
        Fit a Gaussian HMM via Baum-Welch EM with multiple restarts.

        Parameters
        ----------
        X : (T, n_components) PCA-projected and winsorized training array.
            Note: n here is the number of PCA components, not original
            indicators.  PC1 must be the dominant business-cycle component
            (positive loading on unemployment) for state sorting to work
            correctly — this is guaranteed when apply_pca() is run on the
            standard 13-indicator macro dataset.

        Returns
        -------
        self
        """
        best_model, best_loglik = None, -np.inf

        # Remove 'tm' from init_params when constraining so EM starts from
        # our prior for both the transition matrix and start probabilities.
        init_params = 'mc' if self.constrain_transmat else 'stmc'

        for i in range(self.n_restarts):
            model = GaussianHMM(
                n_components    = self.n_states,
                covariance_type = self.covariance_type,
                n_iter          = self.n_iter,
                tol             = self.tol,
                init_params     = init_params,
                random_state    = self.random_state + i,
            )

            if self.constrain_transmat:
                # Business-cycle prior: high persistence, small transition prob.
                # Expansion: ~20 months average duration (p_stay = 0.95)
                # Recession: ~10 months average duration (p_stay = 0.90)
                # EM will move away from these freely — they just prevent
                # degenerate starts where one state absorbs all observations.
                K   = self.n_states
                off = 0.05 / (K - 1)
                P   = np.full((K, K), off)
                np.fill_diagonal(P, 1.0 - off * (K - 1))
                model.transmat_  = P
                # Start in expansion with high probability
                model.startprob_ = np.array(
                    [1.0 - 0.1 * (k / (K - 1)) for k in range(K)]
                )
                model.startprob_ /= model.startprob_.sum()

            try:
                model.fit(X)
                loglik = model.score(X)
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_model  = model
            except Exception as e:
                print(f"Restart {i} failed: {e}")

        if best_model is None:
            raise RuntimeError("All restarts failed — check input data.")

        # ── Sort states so state 0 = expansion, state 1 = recession ──────────
        # Sorting is based on the mean of PC1, which loads positively on
        # unemployment.  The state with the lower PC1 mean is expansion.
        best_model = self._sort_states(best_model)

        print(f"Best log-likelihood (per sample): {best_loglik:.4f}")
        print(f"Transition matrix:\n{best_model.transmat_}")
        print(f"State durations (months): "
              f"{[f'{1/(1-p):.1f}' for p in best_model.transmat_.diagonal()]}")
        print(f"State 0 = Expansion  (PC1 mean: {best_model.means_[0, 0]:.3f})")
        print(f"State 1 = Recession  (PC1 mean: {best_model.means_[1, 0]:.3f})")
        self._model = best_model
        return self

    def _sort_states(self, model: GaussianHMM) -> GaussianHMM:
        """
        Reorder states so state 0 = expansion (low PC1) and state 1 = recession
        (high PC1).

        PC1 is the dominant business-cycle factor with a strong positive loading
        on unemployment rate.  Sorting by PC1 mean gives a consistent, economically
        interpretable labeling regardless of the random initialization order.

        All model parameters that are indexed by state are permuted:
        means_, covars_, startprob_, transmat_.
        """
        # Ascending sort on PC1 mean: lowest = expansion (state 0)
        order = np.argsort(model.means_[:, 0])

        if np.array_equal(order, np.arange(self.n_states)):
            return model   # already correctly ordered — nothing to do

        # Permute all state-indexed parameters
        model.means_     = model.means_[order]
        model.covars_    = model._covars_[order]
        model.startprob_ = model.startprob_[order]

        # Transition matrix: permute both rows and columns
        model.transmat_  = model.transmat_[order][:, order]

        return model

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """Most-likely state sequence via Viterbi algorithm."""
        self._check_fitted()
        return self._model.predict(X)

    def smoothed_probs(self, X: np.ndarray) -> np.ndarray:
        """Smoothed probabilities P(S_t=k | y_{1:T}) — shape (K, T)."""
        self._check_fitted()
        return self._model.predict_proba(X).T

    def filtered_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Filtered probabilities P(S_t=k | y_{1:t}) — shape (K, T).

        Implemented via a manual forward pass because hmmlearn's predict_proba
        returns smoothed (full-sequence) probabilities, not filtered ones.
        Filtered probabilities are the correct choice for out-of-sample /
        walk-forward evaluation since they only use information up to time t.
        """
        self._check_fitted()
        T, _ = X.shape
        K    = self.n_states

        log_emit = self._model._compute_log_likelihood(X)
        emit     = np.exp(log_emit)
        alpha    = np.zeros((T, K))

        alpha[0] = self._model.startprob_ * emit[0]
        alpha[0] /= alpha[0].sum()

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self._model.transmat_) * emit[t]
            s = alpha[t].sum()
            alpha[t] = alpha[t] / s if s > 0 else np.ones(K) / K

        return alpha.T

    def score(self, X: np.ndarray) -> float:
        """Total log-likelihood — delegates to hmmlearn GaussianHMM.score()."""
        self._check_fitted()
        return self._model.score(X)

    # Aliases for backward compatibility with notebook code
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict_states()."""
        return self.predict_states(X)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        return self.smoothed_probs(X)

    @property
    def n_components(self) -> int:
        return self.n_states

    @property
    def transmat_(self):
        self._check_fitted()
        return self._model.transmat_

    @property
    def means_(self):
        self._check_fitted()
        return self._model.means_

    @property
    def covars_(self):
        self._check_fitted()
        return self._model.covars_

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("Call fit() before using the model.")
