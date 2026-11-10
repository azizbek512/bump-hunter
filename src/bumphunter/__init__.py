"""bumphunter — LHC diphoton resonance search and profile likelihood statistical engine."""

from .data import make_binned_data, sample_background, sample_signal
from .discovery import (
    discovery_significance,
    fit_background_only,
    fit_signal_plus_background,
)
from .likelihood import expected_yields, poisson_nll
from .limits import mass_scan, upper_limit_95

__all__ = [
    "sample_background",
    "sample_signal",
    "make_binned_data",
    "expected_yields",
    "poisson_nll",
    "fit_background_only",
    "fit_signal_plus_background",
    "discovery_significance",
    "mass_scan",
    "upper_limit_95",
]
