"""Generic ODE + analytical-solution engine for mechanical/viscoelastic problems.

Two families are supported for v0:

- `shm_damped_driven` (placeholder, unused in 2018 E2): m·ẍ + b·ẋ + k·x = F(t).
- `generalized_kelvin_constant_strain`: closed-form sum of exponentials,
      F(t) = F0 + sum_k F_k · exp(-t/tau_k)
  Under constant strain the Kelvin ODE system has an exact solution, so we
  bypass solve_ivp. We expose `simulate_generalized_kelvin` for the env.

The `solve_ivp` wrapper (`integrate_linear_2nd_order`) is kept here for reuse
when a future mechanics task needs it (pendulum, SHM, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class KelvinParams:
    F0_gf: float
    components: tuple[tuple[float, float], ...]   # ((F_gf, tau_s), ...)


def simulate_generalized_kelvin(
    params: KelvinParams,
    t_grid_s: np.ndarray,
) -> np.ndarray:
    """Evaluate F(t) = F0 + sum F_k exp(-t/tau_k) on the given time grid (seconds)."""
    F = np.full_like(t_grid_s, params.F0_gf, dtype=float)
    for F_k, tau_k in params.components:
        F = F + F_k * np.exp(-t_grid_s / tau_k)
    return F


def integrate_linear_2nd_order(
    m: float,
    b: float,
    k: float,
    forcing,            # callable t -> F(t)
    y0: tuple[float, float],
    t_eval: np.ndarray,
) -> np.ndarray:
    """solve_ivp wrapper for m·ẍ + b·ẋ + k·x = F(t).

    Returns an (N, 2) array of (x, ẋ) at t_eval. Unused by 2018 E2 but kept so
    the same engine module serves a future SHM / pendulum problem.
    """
    def rhs(t, y):
        x, v = y
        return [v, (forcing(t) - b * v - k * x) / m]

    sol = solve_ivp(
        rhs,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=list(y0),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.y.T
