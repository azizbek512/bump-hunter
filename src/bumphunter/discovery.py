"""Profile likelihood ratio (q0), discovery significance (Z), and p-value calculation."""

from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bumphunter.likelihood import expected_yields, poisson_nll


def fit_background_only(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    s_nom: float,
    m_h: float = 125.0,
    sigma: float = 1.5,
) -> dict:
    """Fit null hypothesis H0 (mu = 0 fixed) by profiling nuisance parameters (b_tot, lam)."""
    b_init = float(np.sum(counts))

    def obj(params):
        b_tot, lam = params
        nu = expected_yields(bin_edges, mu=0.0, s_nom=s_nom, b_tot=b_tot, lam=lam, m_h=m_h, sigma=sigma)
        return poisson_nll(counts, nu)

    res = minimize(
        obj,
        x0=[b_init, 0.03],
        bounds=[(1.0, b_init * 5.0), (1e-4, 0.5)],
        method="L-BFGS-B",
    )
    b_hat, lam_hat = res.x
    return {"mu_hat": 0.0, "b_hat": float(b_hat), "lam_hat": float(lam_hat), "nll": float(res.fun)}


def fit_signal_plus_background(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    s_nom: float,
    m_h: float = 125.0,
    sigma: float = 1.5,
) -> dict:
    """Fit alternative hypothesis H1 by jointly optimizing signal strength mu and background parameters."""
    b_init = float(np.sum(counts))

    def obj(params):
        mu, b_tot, lam = params
        nu = expected_yields(bin_edges, mu=mu, s_nom=s_nom, b_tot=b_tot, lam=lam, m_h=m_h, sigma=sigma)
        return poisson_nll(counts, nu)

    res = minimize(
        obj,
        x0=[1.0, b_init, 0.03],
        bounds=[(-5.0, 10.0), (1.0, b_init * 5.0), (1e-4, 0.5)],
        method="L-BFGS-B",
    )
    mu_hat, b_hat, lam_hat = res.x
    return {"mu_hat": float(mu_hat), "b_hat": float(b_hat), "lam_hat": float(lam_hat), "nll": float(res.fun)}


def discovery_significance(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    s_nom: float,
    m_h: float = 125.0,
    sigma: float = 1.5,
) -> dict:
    """Compute one-sided profile likelihood test statistic q0, asymptotic significance Z, and p-value."""
    fit0 = fit_background_only(counts, bin_edges, s_nom=s_nom, m_h=m_h, sigma=sigma)
    fit1 = fit_signal_plus_background(counts, bin_edges, s_nom=s_nom, m_h=m_h, sigma=sigma)

    nll_0 = fit0["nll"]
    nll_1 = fit1["nll"]
    mu_hat = fit1["mu_hat"]

    if mu_hat <= 0.0:
        q0 = 0.0
    else:
        q0 = max(0.0, 2.0 * (nll_0 - nll_1))

    z_sig = float(np.sqrt(q0))
    p_value = float(norm.sf(z_sig))

    return {
        "mu_hat": mu_hat,
        "q0": float(q0),
        "z_sig": z_sig,
        "p_value": p_value,
        "nll_0": nll_0,
        "nll_1": nll_1,
    }


if __name__ == "__main__":
    from bumphunter.data import make_binned_data, sample_background, sample_signal

    rng = np.random.default_rng(42)
    m_range = (100.0, 160.0)

    print("=== Task 3: Profile Likelihood Discovery Engine (discovery.py) ===")

    bg = sample_background(20_000, lam=0.03, m_range=m_range, rng=rng)
    sig = sample_signal(350, m_h=125.0, sigma=1.5, m_range=m_range, rng=rng)
    counts_sb, edges, _ = make_binned_data(np.concatenate([bg, sig]), m_range=m_range, n_bins=60)
    res_sb = discovery_significance(counts_sb, edges, s_nom=350.0, m_h=125.0, sigma=1.5)

    print("\nExperiment 1: Signal Present (20,000 background + 350 Higgs events)")
    print(f"   Fitted signal strength (mu_hat): {res_sb['mu_hat']:.3f}  (SM benchmark = 1.0)")
    print(f"   Background-only NLL (nll_0):     {res_sb['nll_0']:.2f}")
    print(f"   Signal+Background NLL (nll_1):   {res_sb['nll_1']:.2f}")
    print(f"   Test statistic q0 = 2*dNLL:      {res_sb['q0']:.2f}")
    print(f"   Discovery significance Z:        {res_sb['z_sig']:.2f} sigma  (>= 5.0 sigma -> DISCOVERY)")
    print(f"   Local p-value:                   {res_sb['p_value']:.2e}")

    bg_only = sample_background(20_000, lam=0.03, m_range=m_range, rng=np.random.default_rng(7))
    counts_b, _, _ = make_binned_data(bg_only, m_range=m_range, n_bins=60)
    res_b = discovery_significance(counts_b, edges, s_nom=350.0, m_h=125.0, sigma=1.5)

    print("\nExperiment 2: Background Only (20,000 background, 0 signal)")
    print(f"   Fitted signal strength (mu_hat): {res_b['mu_hat']:.3f}")
    print(f"   Test statistic (q0):             {res_b['q0']:.2f}")
    print(f"   Discovery significance (Z):      {res_b['z_sig']:.2f} sigma  (< 3.0 sigma -> Noise)")
    print(f"   Local p-value:                   {res_b['p_value']:.4f}")
