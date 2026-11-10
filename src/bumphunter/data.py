"""Monte Carlo event generator for LHC H -> gamma gamma invariant mass spectra."""

import numpy as np


def sample_background(
    n_b: int,
    lam: float,
    m_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample non-resonant continuum background on [m_min, m_max] via inverse transform sampling."""
    m_min, m_max = m_range
    u = rng.uniform(0.0, 1.0, size=n_b)
    norm = 1.0 - np.exp(-lam * (m_max - m_min))
    return m_min - np.log(1.0 - u * norm) / lam


def sample_signal(
    n_s: int,
    m_h: float,
    sigma: float,
    m_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Gaussian Higgs signal centered at m_h with detector resolution sigma."""
    m_min, m_max = m_range
    masses = rng.normal(loc=m_h, scale=sigma, size=n_s)
    return np.clip(masses, m_min, m_max)


def make_binned_data(
    masses: np.ndarray,
    m_range: tuple[float, float],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin invariant mass events into a detector histogram: (counts, bin_edges, bin_centers)."""
    counts, bin_edges = np.histogram(masses, bins=n_bins, range=m_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return counts, bin_edges, bin_centers


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    m_range = (100.0, 160.0)

    bg = sample_background(n_b=20_000, lam=0.03, m_range=m_range, rng=rng)
    sig = sample_signal(n_s=350, m_h=125.0, sigma=1.5, m_range=m_range, rng=rng)
    all_masses = np.concatenate([bg, sig])
    counts, edges, _ = make_binned_data(all_masses, m_range=m_range, n_bins=60)

    print("=== Task 1: Collider Event Generation (data.py) ===")
    print(f"1. Continuum background: {len(bg)} events | mean mass = {bg.mean():.2f} GeV")
    print(f"2. Higgs signal:         {len(sig)} events | mean mass = {sig.mean():.2f} GeV, std = {sig.std():.2f} GeV")
    print(f"3. Binned histogram:     {len(counts)} bins (1 GeV width), total = {counts.sum()} events")
    print(f"   Bins around 125 GeV (123-127 GeV): {counts[23:27]}")
