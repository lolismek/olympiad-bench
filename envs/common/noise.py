"""Minimal instrument-noise layer. Only what 2018 E2 needs in v0."""

from __future__ import annotations

import numpy as np


def noisy_scale_reading(
    true_value_gf: float,
    rng: np.random.Generator,
    sigma_gf: float = 0.01,
) -> float:
    """Digital scale, Gaussian additive noise at the last-displayed-digit scale."""
    return float(true_value_gf + rng.normal(0.0, sigma_gf))


def noisy_length_reading(
    true_value_m: float,
    rng: np.random.Generator,
    half_division_m: float = 2.5e-4,   # ±0.25 mm rectangular = ruler/measuring tape
) -> float:
    """Ruler / measuring tape: rectangular error bounded by half smallest division."""
    return float(true_value_m + rng.uniform(-half_division_m, +half_division_m))


def noisy_time_reading(
    true_time_s: float,
    rng: np.random.Generator,
    sigma_s: float = 0.1,
) -> float:
    """Stopwatch: human-reaction-time-dominated Gaussian."""
    return float(true_time_s + rng.normal(0.0, sigma_s))
