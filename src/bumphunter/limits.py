"""Mass scan with Look-Elsewhere Effect (LEE) correction and 95% CL upper limits."""

from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bumphunter.discovery import (
    discovery_significance,
    fit_signal_plus_background,
)
from bumphunter.likelihood import expected_yields, poisson_nll


def mass_scan(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    s_nom: float,
    mass_grid: np.ndarray,
    sigma: float = 1.5,
) -> dict:
    """Scan across mass_grid and compute local and Look-Elsewhere-corrected global significance."""
    mass_grid = np.asarray(mass_grid, dtype=float)
    z_list = []
    p_list = []

    for m_h in mass_grid:
        res = discovery_significance(counts, bin_edges, s_nom=s_nom, m_h=float(m_h), sigma=sigma)
        z_list.append(res["z_sig"])
        p_list.append(res["p_value"])

    z_array = np.asarray(z_list, dtype=float)
    p_array = np.asarray(p_list, dtype=float)

    best_idx = int(np.argmax(z_array))
    best_m_h = float(mass_grid[best_idx])
    max_z_local = float(z_array[best_idx])
    min_p_local = float(p_array[best_idx])

    n_trials = len(mass_grid)
    p_global = float(1.0 - (1.0 - min_p_local) ** n_trials)
    z_global = float(max(0.0, norm.isf(p_global)))

    return {
        "masses": mass_grid,
        "z_local": z_array,
        "best_m_h": best_m_h,
        "max_z_local": max_z_local,
        "p_local": min_p_local,
        "p_global": p_global,
        "z_global": z_global,
    }


def upper_limit_95(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    s_nom: float,
    m_h: float = 125.0,
    sigma: float = 1.5,
) -> dict:
    """Compute one-sided 95% confidence level upper limit on signal strength mu."""
    fit1 = fit_signal_plus_background(counts, bin_edges, s_nom=s_nom, m_h=m_h, sigma=sigma)
    mu_hat = fit1["mu_hat"]
    b_hat = fit1["b_hat"]
    lam_hat = fit1["lam_hat"]

    # Finite-difference Hessian of NLL with respect to mu
    eps = 1e-3
    nll_plus = poisson_nll(
        counts,
        expected_yields(bin_edges, mu=mu_hat + eps, s_nom=s_nom, b_tot=b_hat, lam=lam_hat, m_h=m_h, sigma=sigma),
    )
    nll_center = poisson_nll(
        counts,
        expected_yields(bin_edges, mu=mu_hat, s_nom=s_nom, b_tot=b_hat, lam=lam_hat, m_h=m_h, sigma=sigma),
    )
    nll_minus = poisson_nll(
        counts,
        expected_yields(bin_edges, mu=mu_hat - eps, s_nom=s_nom, b_tot=b_hat, lam=lam_hat, m_h=m_h, sigma=sigma),
    )

    d2_nll = max(1e-12, (nll_plus - 2.0 * nll_center + nll_minus) / (eps ** 2))
    sigma_mu = float(1.0 / np.sqrt(d2_nll))
    mu_95 = float(max(0.0, mu_hat) + 1.64485 * sigma_mu)

    return {
        "mu_hat": float(mu_hat),
        "sigma_mu": sigma_mu,
        "mu_95": mu_95,
    }


if __name__ == "__main__":
    from bumphunter.data import make_binned_data, sample_background, sample_signal

    rng = np.random.default_rng(42)
    m_range = (100.0, 160.0)
    mass_grid = np.linspace(110.0, 150.0, 41)

    print("=== Task 4: Mass Scan & 95% CL Upper Limits (limits.py) ===")

    bg = sample_background(20_000, lam=0.03, m_range=m_range, rng=rng)
    sig = sample_signal(180, m_h=125.0, sigma=1.5, m_range=m_range, rng=rng)
    counts_sb, edges, _ = make_binned_data(np.concatenate([bg, sig]), m_range=m_range, n_bins=60)

    scan = mass_scan(counts_sb, edges, s_nom=350.0, mass_grid=mass_grid, sigma=1.5)
    print("\nExperiment 1: Mass Scan across 41 mass hypotheses (110 GeV -> 150 GeV)")
    print(f"   Best-fit mass hypothesis (best_m_h): {scan['best_m_h']:.1f} GeV (True: 125.0 GeV)")
    print(f"   Local significance (max_z_local):    {scan['max_z_local']:.2f} sigma")
    print(f"   Global significance (z_global):      {scan['z_global']:.2f} sigma (Look-Elsewhere corrected)")
    print(f"   Local p-value  (p_local):            {scan['p_local']:.3e}")
    print(f"   Global p-value (p_global):           {scan['p_global']:.3e}")

    lim_20k = upper_limit_95(counts_sb, edges, s_nom=350.0, m_h=140.0, sigma=1.5)
    print("\nExperiment 2: 95% CL Upper Limit at m_H = 140 GeV (No signal, N_b = 20,000)")
    print(f"   Fitted signal strength (mu_hat): {lim_20k['mu_hat']:.3f}")
    print(f"   Uncertainty (sigma_mu):          {lim_20k['sigma_mu']:.3f}")
    print(f"   95% CL upper limit (mu_95):      {lim_20k['mu_95']:.3f}")

    bg_big = sample_background(200_000, lam=0.03, m_range=m_range, rng=rng)
    counts_big, _, _ = make_binned_data(bg_big, m_range=m_range, n_bins=60)
    lim_200k = upper_limit_95(counts_big, edges, s_nom=3500.0, m_h=140.0, sigma=1.5)
    print("\nExperiment 3: 95% CL Upper Limit with 10x Luminosity (N_b = 200,000)")
    print(f"   Uncertainty (sigma_mu):          {lim_200k['sigma_mu']:.3f} (down from {lim_20k['sigma_mu']:.3f})")
    print(f"   95% CL upper limit (mu_95):      {lim_200k['mu_95']:.3f} (tighter exclusion bound)")
