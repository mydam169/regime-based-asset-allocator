"""
MSMH-VAR: Markov-Switching Mean-Heteroskedastic Vector Autoregression
======================================================================
Implements the full EM estimation loop following:
  - Hamilton (1989)  — filter
  - Kim (1994)       — smoother
  - Krolzig (1997)   — MSVAR framework

Modules
-------
1. build_regressors       — construct Y, X matrices from raw data
2. emission_logprob       — log p(y_t | S_t=k, past) for all t, k
3. hamilton_filter        — forward pass: filtered probabilities + log-likelihood
4. kim_smoother           — backward pass: smoothed probabilities + joint probs
5. mstep_var_params       — weighted OLS update for B^(k) and Sigma^(k)
6. mstep_transition       — update transition matrix P
7. ergodic_distribution   — stationary distribution of P (init for filter)
8. em_fit                 — full EM loop
9. MSVARResult            — result container with diagnostics
10. simulate_msvar         — data simulator for testing
"""

import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
from dataclasses import dataclass, field
from typing import Optional
import warnings


# =============================================================================
# MODULE 1 — Build Regressors
# =============================================================================

def build_regressors(data: np.ndarray, p: int):
    """
    Construct the Y and X matrices for a VAR(p) model.

    Parameters
    ----------
    data : (T_raw, n) array — raw multivariate time series
    p    : int             — number of VAR lags

    Returns
    -------
    Y : (n, T) array     — dependent variable matrix (columns = time)
    X : (np+1, T) array  — regressor matrix: [1, y_{t-1}, ..., y_{t-p}]
    T : int              — number of usable observations (T_raw - p)
    n : int              — number of variables
    """
    data = np.asarray(data, dtype=float)
    T_raw, n = data.shape
    T = T_raw - p

    Y = data[p:].T                          # (n, T)
    X = np.zeros((n * p + 1, T))
    X[0, :] = 1.0                           # intercept row
    for lag in range(1, p + 1):
        rows = slice((lag - 1) * n + 1, lag * n + 1)
        X[rows, :] = data[p - lag: T_raw - lag].T

    return Y, X, T, n


# =============================================================================
# MODULE 2 — Emission Log-Probabilities
# =============================================================================

def emission_logprob(Y: np.ndarray, X: np.ndarray,
                     B_list: list, Sigma_list: list) -> np.ndarray:
    """
    Compute log p(y_t | S_t = k, past) for all t and k.

    Parameters
    ----------
    Y        : (n, T)
    X        : (np+1, T)
    B_list   : list of K arrays, each (n, np+1) — VAR coefficients per regime
    Sigma_list: list of K arrays, each (n, n)   — covariance per regime

    Returns
    -------
    log_lik : (K, T) array of log-densities
    """
    K = len(B_list)
    n, T = Y.shape
    log_lik = np.zeros((K, T))

    for k in range(K):
        residuals = Y - B_list[k] @ X          # (n, T)
        # Use Cholesky for numerical stability
        try:
            L = linalg.cholesky(Sigma_list[k], lower=True)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            L_inv_resid = linalg.solve_triangular(L, residuals, lower=True)  # (n, T)
            mahal = np.sum(L_inv_resid ** 2, axis=0)                         # (T,)
            log_lik[k, :] = -0.5 * (n * np.log(2 * np.pi) + log_det + mahal)
        except linalg.LinAlgError:
            # Sigma is singular — fill with -inf to signal degenerate regime
            log_lik[k, :] = -np.inf
            warnings.warn(f"Regime {k}: Sigma is not positive definite.")

    return log_lik   # (K, T)


# =============================================================================
# MODULE 3 — Hamilton Filter (E-step Forward Pass)
# =============================================================================

def hamilton_filter(log_lik: np.ndarray, P: np.ndarray,
                    pi: np.ndarray) -> tuple:
    """
    Hamilton (1989) filter: forward pass of the EM E-step.

    Parameters
    ----------
    log_lik : (K, T) — log emission probabilities from emission_logprob
    P       : (K, K) — row-stochastic transition matrix, P[i,j] = p(S_t=j|S_{t-1}=i)
    pi      : (K,)   — initial regime distribution (use ergodic_distribution)

    Returns
    -------
    xi_filt    : (K, T) — filtered probabilities P(S_t=k | y_{1:t})
    xi_pred    : (K, T) — predicted probabilities P(S_t=k | y_{1:t-1})
    log_likelihood : float — total log-likelihood sum_t log p(y_t | y_{1:t-1})
    """
    K, T = log_lik.shape
    xi_filt = np.zeros((K, T))
    xi_pred = np.zeros((K, T))
    log_likelihood = 0.0

    xi_prev = pi.copy()   # xi_{0|0} = pi

    for t in range(T):
        # --- Prediction step ---
        # xi_{t|t-1} = P^T @ xi_{t-1|t-1}
        xi_pred_t = P.T @ xi_prev              # (K,)
        xi_pred[:, t] = xi_pred_t

        # --- Update step (Bayes) ---
        # Work in log-space for numerical stability
        log_joint = log_lik[:, t] + np.log(xi_pred_t + 1e-300)
        log_norm = _logsumexp(log_joint)       # log p(y_t | y_{1:t-1})

        xi_filt_t = np.exp(log_joint - log_norm)
        xi_filt_t /= xi_filt_t.sum()          # renormalize for floating point
        xi_filt[:, t] = xi_filt_t

        log_likelihood += log_norm
        xi_prev = xi_filt_t

    return xi_filt, xi_pred, log_likelihood


def _logsumexp(log_x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = log_x.max()
    return c + np.log(np.sum(np.exp(log_x - c)))


# =============================================================================
# MODULE 4 — Kim Smoother (E-step Backward Pass)
# =============================================================================

def kim_smoother(xi_filt: np.ndarray, xi_pred: np.ndarray,
                 P: np.ndarray) -> tuple:
    """
    Kim (1994) smoother: backward pass of the EM E-step.

    Parameters
    ----------
    xi_filt : (K, T) — filtered probabilities from hamilton_filter
    xi_pred : (K, T) — predicted probabilities from hamilton_filter
    P       : (K, K) — row-stochastic transition matrix

    Returns
    -------
    xi_smooth      : (K, T)   — smoothed probabilities P(S_t=k | y_{1:T})
    xi_joint_smooth: (K, K, T-1) — joint P(S_{t-1}=j, S_t=k | y_{1:T})
                     axis: [from_regime j, to_regime k, time t]
    """
    K, T = xi_filt.shape
    xi_smooth = np.zeros((K, T))
    xi_joint_smooth = np.zeros((K, K, T - 1))

    # Initialize: smoothed = filtered at T
    xi_smooth[:, T - 1] = xi_filt[:, T - 1]

    for t in range(T - 2, -1, -1):
        # Kim smoother recursion:
        # xi_{t|T}^j = xi_{t|t}^j * sum_k [ p_{jk} * xi_{t+1|T}^k / xi_{t+1|t}^k ]
        denom = xi_pred[:, t + 1]                            # (K,) — predicted at t+1
        denom_safe = np.where(denom > 1e-300, denom, 1e-300)  # avoid division by zero

        # ratio[k] = xi_{t+1|T}^k / xi_{t+1|t}^k
        ratio = xi_smooth[:, t + 1] / denom_safe             # (K,)

        # backward_factor[j] = sum_k p_{jk} * ratio[k]
        backward_factor = P @ ratio                           # (K,)  (P is K×K, rows=from)

        xi_smooth_t = xi_filt[:, t] * backward_factor
        # Normalize
        s = xi_smooth_t.sum()
        if s > 1e-300:
            xi_smooth_t /= s
        xi_smooth[:, t] = xi_smooth_t

        # Joint smoothed probability: xi_{t,t+1|T}^{j,k}
        # P(S_t=j, S_{t+1}=k | y_{1:T}) = xi_{t|t}^j * p_{jk} * xi_{t+1|T}^k / xi_{t+1|t}^k
        joint = (xi_filt[:, t][:, None] *         # (K,1): xi_{t|t}^j
                 P *                               # (K,K): p_{jk}
                 (xi_smooth[:, t + 1] /            # (K,): ratio broadcast over j
                  denom_safe)[None, :])
        joint /= (joint.sum() + 1e-300)            # normalize to sum=1 over (j,k)
        xi_joint_smooth[:, :, t] = joint

    return xi_smooth, xi_joint_smooth


# =============================================================================
# MODULE 5 — M-step: Weighted OLS for VAR Parameters
# =============================================================================

def mstep_var_params(Y: np.ndarray, X: np.ndarray,
                     xi_smooth: np.ndarray,
                     n: int, min_weight: float = 1e-6) -> tuple:
    """
    M-step: update VAR coefficients B^(k) and covariance Sigma^(k)
    via weighted OLS using smoothed regime probabilities as weights.

    Parameters
    ----------
    Y          : (n, T)
    X          : (np+1, T)
    xi_smooth  : (K, T) — smoothed probabilities
    n          : int    — number of variables
    min_weight : float  — minimum effective weight to avoid degeneracy

    Returns
    -------
    B_list     : list of K arrays (n, np+1)
    Sigma_list : list of K arrays (n, n)
    eff_sizes  : (K,) array — effective sample sizes sum_t xi_{t|T}^k
    """
    K, T = xi_smooth.shape
    q = X.shape[0]           # np+1
    B_list = []
    Sigma_list = []
    eff_sizes = xi_smooth.sum(axis=1)   # (K,)

    for k in range(K):
        w = xi_smooth[k, :]    # (T,) weights

        # Weighted sufficient statistics
        # S_xy = sum_t w_t * y_t * x_t^T  =  Y @ diag(w) @ X^T
        S_xy = (Y * w[None, :]) @ X.T         # (n, q)
        S_xx = (X * w[None, :]) @ X.T         # (q, q)

        # Weighted OLS: B^k = S_xy @ S_xx^{-1}
        try:
            B_k = S_xy @ linalg.inv(S_xx + 1e-10 * np.eye(q))
        except linalg.LinAlgError:
            B_k = np.zeros((n, q))
            warnings.warn(f"Regime {k}: singular S_xx, using zero B.")

        # Weighted residuals and covariance
        resid = Y - B_k @ X                   # (n, T)
        # Sigma^k = sum_t w_t * e_t * e_t^T  /  sum_t w_t
        eff = max(eff_sizes[k], min_weight)
        Sigma_k = (resid * w[None, :]) @ resid.T / eff

        # Ensure symmetry and positive definiteness via regularization
        Sigma_k = 0.5 * (Sigma_k + Sigma_k.T)
        min_eig = np.linalg.eigvalsh(Sigma_k).min()
        if min_eig < 1e-8:
            Sigma_k += (abs(min_eig) + 1e-8) * np.eye(n)

        B_list.append(B_k)
        Sigma_list.append(Sigma_k)

    return B_list, Sigma_list, eff_sizes


# =============================================================================
# MODULE 6 — M-step: Transition Matrix Update
# =============================================================================

def mstep_transition(xi_smooth: np.ndarray,
                     xi_joint_smooth: np.ndarray) -> np.ndarray:
    """
    M-step: update transition matrix P.

    P_new[j, k] = sum_t xi_{t-1,t|T}^{j,k}  /  sum_t xi_{t-1|T}^j

    Parameters
    ----------
    xi_smooth       : (K, T)
    xi_joint_smooth : (K, K, T-1) — joint[j, k, t] = P(S_t=j, S_{t+1}=k | y_{1:T})

    Returns
    -------
    P_new : (K, K) row-stochastic transition matrix
    """
    # Numerator: sum over t=0..T-2 of joint probability (j->k)
    numerator = xi_joint_smooth.sum(axis=2)          # (K, K)

    # Denominator: total smoothed time in regime j (over t=0..T-2)
    denominator = xi_smooth[:, :-1].sum(axis=1)      # (K,)
    denominator = np.where(denominator > 1e-300, denominator, 1e-300)

    P_new = numerator / denominator[:, None]

    # Renormalize rows to sum exactly to 1
    row_sums = P_new.sum(axis=1, keepdims=True)
    P_new /= np.where(row_sums > 0, row_sums, 1.0)

    return P_new


# =============================================================================
# MODULE 7 — Ergodic Distribution
# =============================================================================

def ergodic_distribution(P: np.ndarray) -> np.ndarray:
    """
    Compute the stationary (ergodic) distribution of transition matrix P.
    Solves pi^T P = pi^T, sum(pi) = 1 via left eigenvector decomposition.

    Parameters
    ----------
    P : (K, K) row-stochastic matrix

    Returns
    -------
    pi : (K,) stationary distribution
    """
    K = P.shape[0]
    # Left eigenvectors of P^T  <=>  right eigenvectors of P
    eigvals, eigvecs = np.linalg.eig(P.T)

    # Find eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi)
    pi /= pi.sum()

    # Fallback: if distribution is degenerate, use uniform
    if np.any(pi < 0) or not np.isfinite(pi).all():
        pi = np.ones(K) / K

    return pi


# =============================================================================
# MODULE 8 — EM Fit
# =============================================================================

def em_fit(data: np.ndarray, K: int, p: int,
           n_restarts: int = 5,
           max_iter: int = 500,
           tol: float = 1e-6,
           random_state: Optional[int] = None,
           verbose: bool = False) -> 'MSVARResult':
    """
    Fit MSMH-VAR(K, p) model via EM algorithm with multiple random restarts.

    Parameters
    ----------
    data        : (T_raw, n) — multivariate time series
    K           : int        — number of regimes
    p           : int        — VAR lag order
    n_restarts  : int        — number of random initializations
    max_iter    : int        — maximum EM iterations per restart
    tol         : float      — convergence tolerance on log-likelihood
    random_state: int        — seed for reproducibility
    verbose     : bool       — print progress

    Returns
    -------
    MSVARResult object with the best (highest log-lik) solution
    """
    rng = np.random.default_rng(random_state)
    Y, X, T, n = build_regressors(data, p)

    best_result = None
    best_loglik = -np.inf

    for restart in range(n_restarts):
        if verbose:
            print(f"\n--- Restart {restart + 1}/{n_restarts} ---")

        # --- Initialize parameters ---
        B_list, Sigma_list, P = _initialize_params(Y, X, K, n, p, T, rng)

        log_lik_history = []
        prev_loglik = -np.inf

        for iteration in range(max_iter):
            # ---- E-step ----
            log_emit = emission_logprob(Y, X, B_list, Sigma_list)
            pi = ergodic_distribution(P)
            xi_filt, xi_pred, loglik = hamilton_filter(log_emit, P, pi)
            xi_smooth, xi_joint = kim_smoother(xi_filt, xi_pred, P)

            log_lik_history.append(loglik)

            # Convergence check
            delta = loglik - prev_loglik
            if verbose and (iteration % 10 == 0):
                print(f"  iter {iteration:4d} | log-lik = {loglik:.6f} | Δ = {delta:.2e}")

            if iteration > 0 and abs(delta) < tol:
                if verbose:
                    print(f"  Converged at iteration {iteration}.")
                break

            if delta < -1e-6 and iteration > 2:
                warnings.warn(f"Restart {restart}: log-lik decreased by {delta:.2e} at iter {iteration}. "
                              "Possible numerical issue.")

            prev_loglik = loglik

            # ---- M-step ----
            B_list, Sigma_list, eff_sizes = mstep_var_params(Y, X, xi_smooth, n)
            P = mstep_transition(xi_smooth, xi_joint)

        # Keep best restart
        if loglik > best_loglik:
            best_loglik = loglik
            best_result = MSVARResult(
                B_list=B_list,
                Sigma_list=Sigma_list,
                P=P,
                pi=ergodic_distribution(P),
                xi_filtered=xi_filt,
                xi_smoothed=xi_smooth,
                log_likelihood=loglik,
                log_lik_history=log_lik_history,
                n_iter=len(log_lik_history),
                K=K, p=p, n=n, T=T,
                eff_sample_sizes=eff_sizes,
                data=data,
            )

    # Apply label ordering: sort regimes by mean return of first variable
    best_result = _sort_regimes(best_result)

    return best_result


def _initialize_params(Y, X, K, n, p, T, rng):
    """
    K-means-style initialization on VAR residuals.
    Falls back to random perturbation if K-means assignment is degenerate.
    """
    from sklearn.cluster import KMeans

    # Fit plain VAR (OLS) to get residuals
    B_ols = Y @ X.T @ linalg.inv(X @ X.T + 1e-8 * np.eye(X.shape[0]))
    resid = Y - B_ols @ X   # (n, T)

    # Cluster residuals into K groups
    try:
        km = KMeans(n_clusters=K, n_init=5, random_state=int(rng.integers(1e6)))
        labels = km.fit_predict(resid.T)   # (T,)
    except Exception:
        labels = rng.integers(0, K, size=T)

    # Build initial params from cluster assignments
    B_list = []
    Sigma_list = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < n + 1:
            # Not enough points: use overall stats with perturbation
            idx = np.arange(T)

        w = np.zeros(T)
        w[idx] = 1.0
        w /= w.sum()
        xi_init = np.tile(w, (K, 1))
        B_k, Sigma_k, _ = mstep_var_params(Y, X,
                                            np.eye(K)[k:k+1].T * w[None, :],
                                            n)
        # Actually compute per-regime
        S_xy = (Y * w[None, :]) @ X.T
        S_xx = (X * w[None, :]) @ X.T
        B_k = S_xy @ linalg.inv(S_xx + 1e-8 * np.eye(X.shape[0]))
        resid_k = Y - B_k @ X
        Sigma_k = (resid_k * w[None, :]) @ resid_k.T
        Sigma_k = 0.5 * (Sigma_k + Sigma_k.T)
        Sigma_k += 1e-4 * np.eye(n)   # small ridge for PD
        B_list.append(B_k)
        Sigma_list.append(Sigma_k)

    # Random transition matrix with high diagonal
    P = rng.dirichlet(np.ones(K) * 0.5, size=K)
    # Inflate diagonal for persistence
    P = P * 0.3 + 0.7 * np.eye(K)
    P /= P.sum(axis=1, keepdims=True)

    return B_list, Sigma_list, P


def _avg_correlation(cov: np.ndarray) -> float:
    """
    Average absolute off-diagonal correlation of a covariance matrix.
    Returns 0 for 1x1 matrices.  Used by _sort_regimes().
    """
    n = cov.shape[0]
    if n == 1:
        return 0.0
    std = np.sqrt(np.diag(cov))
    std = np.where(std > 1e-12, std, 1e-12)
    corr = cov / np.outer(std, std)
    mask = ~np.eye(n, dtype=bool)
    return float(np.abs(corr[mask]).mean())


def _sort_regimes(result: 'MSVARResult') -> 'MSVARResult':
    """
    Sort regimes so state 0 = expansion (low correlation) and
    state 1 = recession (high correlation).

    The recession regime is characterized by strong cross-indicator
    correlation — during downturns, macro variables co-move more strongly
    (unemployment rises as industrial production falls simultaneously,
    VIX spikes as credit spreads widen, etc.).

    Sorting by average off-diagonal absolute correlation of the
    regime-conditional covariance matrix Sigma_k is robust to PCA sign
    flips (unlike PC1 mean sorting) and to sample-size variation
    (unlike observation-count sorting).

    If correlations are identical across regimes (degenerate case), falls
    back to sorting by average diagonal variance — higher variance = recession.
    """
    scores = []
    for k in range(result.K):
        cov   = result.Sigma_list[k]
        score = _avg_correlation(cov)
        if score < 1e-10:
            # Fallback: sort by average variance
            score = float(np.diag(cov).mean())
        scores.append(score)

    # Ascending: lower correlation = expansion (state 0)
    order = np.argsort(scores)

    if np.array_equal(order, np.arange(result.K)):
        return result   # already correctly ordered

    result.B_list           = [result.B_list[i] for i in order]
    result.Sigma_list       = [result.Sigma_list[i] for i in order]
    result.P                = result.P[np.ix_(order, order)]
    result.pi               = result.pi[order]
    result.xi_filtered      = result.xi_filtered[order, :]
    result.xi_smoothed      = result.xi_smoothed[order, :]
    result.eff_sample_sizes = result.eff_sample_sizes[order]
    return result


# =============================================================================
# MODULE 9 — Result Container
# =============================================================================

@dataclass
class MSVARResult:
    """Container for MSMH-VAR estimation results."""

    # Core parameters
    B_list: list                  # [K] arrays of shape (n, np+1)
    Sigma_list: list              # [K] arrays of shape (n, n)
    P: np.ndarray                 # (K, K) transition matrix
    pi: np.ndarray                # (K,) ergodic distribution

    # Probabilities
    xi_filtered: np.ndarray       # (K, T)
    xi_smoothed: np.ndarray       # (K, T)

    # Fit diagnostics
    log_likelihood: float
    log_lik_history: list
    n_iter: int
    eff_sample_sizes: np.ndarray  # (K,) effective obs per regime

    # Dimensions
    K: int
    p: int
    n: int
    T: int
    data: np.ndarray

    def regime_sequence(self, use_smoothed: bool = True) -> np.ndarray:
        """Most probable regime sequence (Viterbi-style argmax)."""
        probs = self.xi_smoothed if use_smoothed else self.xi_filtered
        return np.argmax(probs, axis=0)

    @property
    def expected_durations(self) -> np.ndarray:
        """Expected duration in each regime: 1 / (1 - p_kk)."""
        return 1.0 / (1.0 - np.diag(self.P).clip(0, 1 - 1e-10))

    @property
    def n_params(self) -> int:
        """Total number of free parameters."""
        var_params = self.K * self.n * (self.n * self.p + 1)      # B matrices
        cov_params = self.K * self.n * (self.n + 1) // 2           # lower-tri Sigma
        trans_params = self.K * (self.K - 1)                       # K rows, each sums to 1
        return var_params + cov_params + trans_params

    @property
    def aic(self) -> float:
        return -2 * self.log_likelihood + 2 * self.n_params

    @property
    def bic(self) -> float:
        return -2 * self.log_likelihood + self.n_params * np.log(self.T)

    def summary(self):
        """Print a summary of the estimation results."""
        print("=" * 60)
        print(f"MSMH-VAR({self.K}, {self.p})  |  n={self.n}, T={self.T}")
        print("=" * 60)
        print(f"Log-likelihood : {self.log_likelihood:.4f}")
        print(f"AIC            : {self.aic:.4f}")
        print(f"BIC            : {self.bic:.4f}")
        print(f"# parameters   : {self.n_params}")
        print(f"EM iterations  : {self.n_iter}")
        print()
        print("Transition Matrix P (row i -> col j):")
        print(np.round(self.P, 4))
        print()
        print("Expected Regime Durations (periods):")
        for k, d in enumerate(self.expected_durations):
            print(f"  Regime {k}: {d:.1f}")
        print()
        print("Effective Sample Sizes:")
        for k, eff in enumerate(self.eff_sample_sizes):
            print(f"  Regime {k}: {eff:.1f} obs ({100*eff/self.T:.1f}%)")
        print()
        for k in range(self.K):
            print(f"--- Regime {k} VAR Intercept (mu) ---")
            print(np.round(self.B_list[k][:, 0], 6))
            print(f"--- Regime {k} Covariance Sigma ---")
            print(np.round(self.Sigma_list[k], 6))
            print()


# =============================================================================
# MODULE 10 — Simulator (for testing and validation)
# =============================================================================

def simulate_msvar(T: int, K: int, p: int,
                   B_list: list, Sigma_list: list, P: np.ndarray,
                   pi: Optional[np.ndarray] = None,
                   random_state: Optional[int] = None) -> tuple:
    """
    Simulate data from an MSMH-VAR(K, p) model.

    Parameters
    ----------
    T          : int         — number of time steps to simulate
    K          : int         — number of regimes
    p          : int         — VAR lag order
    B_list     : list of (n, np+1) — VAR params per regime
    Sigma_list : list of (n, n)    — covariances per regime
    P          : (K, K)            — transition matrix
    pi         : (K,)              — initial regime distribution
    random_state: int

    Returns
    -------
    data   : (T, n) simulated time series
    states : (T,)  true regime sequence
    """
    rng = np.random.default_rng(random_state)
    n = B_list[0].shape[0]

    if pi is None:
        pi = ergodic_distribution(P)

    # Simulate regime sequence
    states = np.zeros(T, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        states[t] = rng.choice(K, p=P[states[t - 1]])

    # Simulate data
    data = np.zeros((T + p, n))
    # Initialize first p observations from unconditional distribution
    data[:p] = rng.multivariate_normal(np.zeros(n), Sigma_list[0], size=p)

    for t in range(p, T + p):
        k = states[t - p]
        B = B_list[k]
        x = np.concatenate([[1.0]] + [data[t - lag] for lag in range(1, p + 1)])
        eps = rng.multivariate_normal(np.zeros(n), Sigma_list[k])
        data[t] = B @ x + eps

    return data[p:], states


# =============================================================================
# QUICK VALIDATION — run if executed directly
# =============================================================================

if __name__ == "__main__":
    print("Running MSMH-VAR self-test...")
    np.random.seed(42)

    # --- True parameters ---
    K_true, p_true, n_true = 2, 1, 3
    B0 = np.array([[0.02,  0.10,  0.05, -0.02],
                   [0.01, -0.05,  0.20,  0.01],
                   [0.01,  0.03, -0.03,  0.15]])
    B1 = np.array([[-0.03, 0.05, -0.10, 0.03],
                   [-0.02, 0.10,  0.05, -0.05],
                   [-0.01, -0.02, 0.08, 0.12]])
    Sigma0 = np.array([[0.0004, 0.0001, 0.0001],
                       [0.0001, 0.0002, 0.0000],
                       [0.0001, 0.0000, 0.0003]])
    Sigma1 = Sigma0 * 4    # high-vol regime
    P_true = np.array([[0.95, 0.05],
                       [0.10, 0.90]])

    # --- Simulate ---
    data, true_states = simulate_msvar(
        T=500, K=K_true, p=p_true,
        B_list=[B0, B1], Sigma_list=[Sigma0, Sigma1],
        P=P_true, random_state=42
    )
    print(f"Simulated data shape: {data.shape}")
    print(f"True regime fractions: {np.bincount(true_states) / len(true_states)}")

    # --- Fit ---
    result = em_fit(data, K=K_true, p=p_true,
                    n_restarts=3, max_iter=200, tol=1e-6,
                    random_state=0, verbose=True)

    result.summary()

    # --- Accuracy check ---
    # build_regressors drops the first p observations
    inferred = result.regime_sequence()
    true_trimmed = true_states[p_true:]   # align with usable observations
    # Account for label switching
    acc1 = np.mean(inferred == true_trimmed)
    acc2 = np.mean(inferred == (1 - true_trimmed))
    acc = max(acc1, acc2)
    print(f"Regime classification accuracy: {acc:.3f}")
