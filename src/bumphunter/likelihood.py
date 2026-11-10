"""Analytical bin expectations and Poisson negative log-likelihood (NLL)."""

import numpy as np
from scipy.stats import norm


def expected_yields(
    bin_edges: np.ndarray,
    mu: float,
    s_nom: float,
    b_tot: float,
    lam: float,
    m_h: float = 125.0,
    sigma: float = 1.5,
) -> np.ndarray:
    """Compute expected counts per bin: nu_i = mu * s_nom * p_{s,i} + b_tot * p_{b,i}."""
    m_min, m_max = bin_edges[0], bin_edges[-1]
    a = bin_edges[:-1]
    b = bin_edges[1:]

    # Normalized truncated exponential CDF difference per bin
    b_norm = np.exp(-lam * m_min) - np.exp(-lam * m_max)
    p_b = (np.exp(-lam * a) - np.exp(-lam * b)) / b_norm

    # Normalized truncated Gaussian CDF difference per bin
    s_norm = norm.cdf(m_max, loc=m_h, scale=sigma) - norm.cdf(m_min, loc=m_h, scale=sigma)
    p_s = (norm.cdf(b, loc=m_h, scale=sigma) - norm.cdf(a, loc=m_h, scale=sigma)) / s_norm

    return mu * s_nom * p_s + b_tot * p_b


def poisson_nll(counts: np.ndarray, expected: np.ndarray) -> float:
    """Compute Poisson negative log-likelihood (up to additive constants): sum(nu_i - n_i * ln(nu_i))."""
    counts = np.asarray(counts, dtype=float)
    nu = np.maximum(np.asarray(expected, dtype=float), 1e-12)
    return float(np.sum(nu - counts * np.log(nu)))


if __name__ == "__main__":
    edges = np.linspace(100.0, 160.0, 61)

    nu_bg_only = expected_yields(edges, mu=0.0, s_nom=350.0, b_tot=20_000.0, lam=0.03)
    nu_with_higgs = expected_yields(edges, mu=1.0, s_nom=350.0, b_tot=20_000.0, lam=0.03)

    print("=== Task 2: Expected Yields & Poisson NLL (likelihood.py) ===")
    print(f"1. Background-only expectation (mu=0.0): {nu_bg_only.sum():.1f} total events")
    print(f"2. Signal + Background (mu=1.0):         {nu_with_higgs.sum():.1f} total events")
    print(
        f"3. Expected count in 125 GeV bin (mu=0 vs mu=1): "
        f"{nu_bg_only[25]:.1f} vs {nu_with_higgs[25]:.1f} (+{nu_with_higgs[25] - nu_bg_only[25]:.1f} signal)"
    )

    n_obs = np.array([150.0])
    nll_h0 = poisson_nll(n_obs, np.array([100.0]))
    nll_h1 = poisson_nll(n_obs, np.array([150.0]))
    print("\n=== Single-Bin Benchmark (Observed n = 150) ===")
    print(f"   Null hypothesis NLL(nu=100):          {nll_h0:.2f}")
    print(f"   Best-fit hypothesis NLL(nu=150):      {nll_h1:.2f}")
    print(f"   Log-likelihood penalty (NLL_0-NLL_1): {nll_h0 - nll_h1:.2f}")
