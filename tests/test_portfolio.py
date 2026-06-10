"""Tests for src/portfolio.py"""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.portfolio import _solve_mvo, optimize_regimes


def _moments():
    """Synthetic regime moments for 3 assets."""
    mu0    = np.array([0.01, 0.007, 0.004])
    Sigma0 = np.array([[0.0014, -0.0001, -0.0001],
                       [-0.0001, 0.0008,  0.0001],
                       [-0.0001, 0.0001,  0.0013]])
    mu1    = np.array([-0.007, 0.017, 0.007])
    Sigma1 = np.array([[0.003,  0.0001, -0.0001],
                       [0.0001, 0.0028, -0.0001],
                       [-0.0001,-0.0001, 0.0027]])
    return {
        0: {'mu': mu0, 'sigma': Sigma0, 'mu_sample': mu0,
            'mu_bs_phi': 0.0, 'sigma_raw': Sigma0,
            'lw_shrinkage': 0.0, 'n_obs': 100},
        1: {'mu': mu1, 'sigma': Sigma1, 'mu_sample': mu1,
            'mu_bs_phi': 0.0, 'sigma_raw': Sigma1,
            'lw_shrinkage': 0.0, 'n_obs': 20},
    }


def test_min_variance_weights_sum_to_one():
    m = _moments()[0]
    w = _solve_mvo(m['mu'], m['sigma'], objective='min_variance')
    assert np.isclose(w.sum(), 1.0, atol=1e-5)


def test_min_variance_long_only():
    m = _moments()[0]
    w = _solve_mvo(m['mu'], m['sigma'], objective='min_variance', long_only=True)
    assert (w >= -1e-6).all()


def test_max_sharpe_weights_sum_to_one():
    m = _moments()[0]
    w = _solve_mvo(m['mu'], m['sigma'], objective='max_sharpe')
    assert np.isclose(w.sum(), 1.0, atol=1e-5)


def test_optimize_regimes_returns_both_objectives():
    """optimize_regimes now exposes min_variance and max_sharpe only
    (max_return was removed as it was never reported)."""
    m = _moments()
    results = optimize_regimes(m, ['Equity', 'Bonds', 'Gold'])
    for k in [0, 1]:
        for obj in ('min_variance', 'max_sharpe'):
            assert obj in results[k], f"Missing {obj} for regime {k}"
            w = results[k][obj].weights
            assert np.isclose(w.sum(), 1.0, atol=1e-5), \
                f"Weights don't sum to 1 for regime {k}, {obj}"
            assert (w >= -1e-6).all(), \
                f"Negative weights for regime {k}, {obj}"


def test_bayes_stein_phi_bounds():
    """Bayes-Stein shrinkage intensity must lie in [0, 1]."""
    from src.portfolio import bayes_stein_shrinkage
    mu_k      = np.array([0.01, 0.005, 0.003])
    sigma_k   = np.diag([0.002, 0.001, 0.0015])
    mu_global = np.array([0.008, 0.006, 0.004])
    for n_obs in [5, 20, 50, 200]:
        _, phi = bayes_stein_shrinkage(mu_k, sigma_k, n_obs, mu_global)
        assert 0.0 <= phi <= 1.0, f"phi={phi} out of bounds for n_obs={n_obs}"


def test_bayes_stein_shrinks_toward_global():
    """With few observations, shrunk mean should be closer to global mean."""
    from src.portfolio import bayes_stein_shrinkage
    mu_k      = np.array([0.02, 0.0, 0.0])
    mu_global = np.array([0.005, 0.005, 0.005])
    sigma_k   = np.eye(3) * 0.001
    mu_bs_few,  _ = bayes_stein_shrinkage(mu_k, sigma_k, 10,  mu_global)
    mu_bs_many, _ = bayes_stein_shrinkage(mu_k, sigma_k, 500, mu_global)
    dist_few  = np.linalg.norm(mu_bs_few  - mu_global)
    dist_many = np.linalg.norm(mu_bs_many - mu_global)
    assert dist_few < dist_many, \
        "Fewer observations should shrink more toward global mean"