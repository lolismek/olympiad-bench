"""Simulated environment for IPhO 2018 E2 — viscoelastic polymer thread.

Agent-visible tool API:

    measure_length(target: str)          -> {"length_m": float, "sigma_m": float, "unit": "m"}
    read_mass()                          -> {"mass_gf": float, "sigma_gf": float, "unit": "gf"}
    stretch_thread()                     -> {"ok": True, "t_started_s": 0.0}
    wait(duration_s: float)              -> {"clock_s": float}
    read_scale()                         -> {"t_s": float, "P_gf": float, "unit": "gf"}
    submit(answer: dict)                 -> {"accepted": True}

Budget: 200 tool calls. Exceeding budget terminates the episode with score 0.

Two truth modes, controlled by `Env(randomize=...)`:

  • `randomize=False` (default): truth values come from `spec.yaml` verbatim.
    Identical to the originally authored IPhO 2018 E2 problem — use this for
    reproduction of the published setup.
  • `randomize=True`: truth values are perturbed deterministically per seed
    inside a bounded, physically-sound envelope. Use this to test whether a
    model is actually fitting the relaxation curve vs. recalling the
    published τ/E values from training data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from envs.common.noise import (
    noisy_length_reading,
    noisy_scale_reading,
    noisy_time_reading,
)
from envs.common.spec_schema import Spec
from envs.physics.mechanics_ode import KelvinParams, simulate_generalized_kelvin

TASK_DIR = Path(__file__).resolve().parent
SPEC_PATH = TASK_DIR / "spec.yaml"
DEFAULT_BUDGET = 200

# Physical constants used when recomputing derived truths (β, E_k).
_G_SI = 9.80
_THREAD_D_M = 4.80e-4
_S_SI = math.pi * (_THREAD_D_M / 2.0) ** 2

# Solvability envelope for randomized draws. Tightened so that every draw is
# a physically reasonable TPU polymer and the published fit method converges.
_BOUNDS: dict[str, tuple[float, float]] = {
    # Multiplicative factor ranges for each truth before constraint enforcement.
    "F0_gf":   (0.80, 1.20),
    "F1_gf":   (0.70, 1.30),
    "F2_gf":   (0.70, 1.30),
    "tau1_s":  (0.80, 1.25),
    "tau2_s":  (0.75, 1.30),
    "tau3_s":  (0.80, 1.25),
    "epsilon": (0.85, 1.15),
    "l0_m":    (0.95, 1.05),
}
# Mode separation: τ₁/τ₂ and τ₂/τ₃ must remain at least this large to keep
# the three-exponential fit well-conditioned.
_MIN_TAU_RATIO_12 = 4.0
_MIN_TAU_RATIO_23 = 2.5
# Strain envelope: below 0.05 the fit is dominated by ruler noise on ε, above
# 0.35 we drift outside TPU's elastic regime.
_EPS_MIN, _EPS_MAX = 0.05, 0.35


@dataclass
class EnvState:
    spec: Spec
    rng: np.random.Generator
    truth: dict[str, float]
    F3_gf: float
    budget_used: int = 0
    budget_total: int = DEFAULT_BUDGET
    stretched: bool = False
    clock_s: float = 0.0
    t_stretch_s: float = 0.0
    submitted: bool = False
    submitted_answer: dict[str, Any] | None = None
    log: list[dict[str, Any]] = field(default_factory=list)


# --- Truth construction -----------------------------------------------------

def _calibrate_F3(F0: float, F1: float, F2: float,
                  tau1: float, tau2: float, tau3: float,
                  F_at_10_gf: float = 45.41) -> float:
    """Choose F₃ so the simulator reproduces F(10 s) = F_at_10 (official datum).

    Even under randomization we anchor F₃ so that the shortest mode carries a
    physically-meaningful amplitude rather than floating free.
    """
    residual = F_at_10_gf - F0 - F1 * math.exp(-10.0 / tau1) - F2 * math.exp(-10.0 / tau2)
    return max(residual / math.exp(-10.0 / tau3), 0.0)


def _derive(truth: dict[str, float]) -> dict[str, float]:
    """Fill in l_m, β, E₀/E₁/E₂ from perturbed primary truths."""
    eps = truth["epsilon"]
    truth["l_m"] = truth["l0_m"] * (1.0 + eps)
    beta = _G_SI / (eps * _S_SI) * 1.0e-3  # gf → SI factor folded in
    truth["beta_gf_inv_SI"] = beta
    truth["E0_SI"] = beta * truth["F0_gf"]
    truth["E1_SI"] = beta * truth["F1_gf"]
    truth["E2_SI"] = beta * truth["F2_gf"]
    return truth


def _spec_truth(spec: Spec) -> dict[str, float]:
    """Truth dict verbatim from spec.yaml (original fixed problem)."""
    return {p.name: float(p.truth_value) for p in spec.physics_model.parameters}


def _draw_truth(
    spec: Spec,
    rng: np.random.Generator,
    strength: float = 1.0,
    max_tries: int = 100,
) -> dict[str, float]:
    """Deterministically draw a perturbed truth from spec.yaml baseline.

    `strength` ∈ [0, 1] linearly interpolates between no perturbation (0.0)
    and the full `_BOUNDS` range (1.0). Independent multiplicative factors
    are drawn for each primary parameter; then constraints are enforced
    (mode separation, strain envelope, `F(0) < P₀`, positive amplitudes).
    Rejection-sample on constraint violation.
    """
    base = _spec_truth(spec)
    strength = max(0.0, min(1.0, float(strength)))

    for _ in range(max_tries):
        out = dict(base)
        # 1. Independent multiplicative draws on primary parameters.
        for key, (lo, hi) in _BOUNDS.items():
            # Interpolate bounds toward 1.0 as strength decreases.
            lo_s = 1.0 + (lo - 1.0) * strength
            hi_s = 1.0 + (hi - 1.0) * strength
            out[key] *= float(rng.uniform(lo_s, hi_s))

        # 2. Enforce mode-separation: fix τ₁ upward if too close to τ₂, etc.
        if out["tau1_s"] / out["tau2_s"] < _MIN_TAU_RATIO_12:
            out["tau1_s"] = out["tau2_s"] * _MIN_TAU_RATIO_12 * float(rng.uniform(1.0, 1.15))
        if out["tau2_s"] / out["tau3_s"] < _MIN_TAU_RATIO_23:
            out["tau2_s"] = out["tau3_s"] * _MIN_TAU_RATIO_23 * float(rng.uniform(1.0, 1.15))

        # 3. Strain envelope.
        if not (_EPS_MIN <= out["epsilon"] <= _EPS_MAX):
            continue

        # 4. Recompute derived truths and F₃ from primary perturbations.
        _derive(out)
        F3 = _calibrate_F3(
            out["F0_gf"], out["F1_gf"], out["F2_gf"],
            out["tau1_s"], out["tau2_s"], out["tau3_s"],
        )

        # 5. Physical check: F(0) = F₀+F₁+F₂+F₃ must sit well below P₀ so the
        #    scale never goes negative. Margin of 10 gf protects against
        #    noise pushing the reading below zero.
        F_at_zero = out["F0_gf"] + out["F1_gf"] + out["F2_gf"] + F3
        if F_at_zero >= out["P0_gf"] - 10.0:
            continue
        # Positive-amplitude check (catches pathological F₃ = 0 runs).
        if F3 <= 0.1 or out["F0_gf"] <= 1.0 or out["F1_gf"] <= 0.1 or out["F2_gf"] <= 0.1:
            continue

        out["F3_gf_internal"] = F3
        return out

    raise RuntimeError(
        f"_draw_truth: no valid draw within {max_tries} tries "
        "(bounds may be too loose for the problem)."
    )


def _kelvin_from_truth(truth: dict[str, float], F3_gf: float) -> KelvinParams:
    return KelvinParams(
        F0_gf=truth["F0_gf"],
        components=(
            (truth["F1_gf"], truth["tau1_s"]),
            (truth["F2_gf"], truth["tau2_s"]),
            (F3_gf,          truth["tau3_s"]),
        ),
    )


def _true_P_gf(state: EnvState, t_since_stretch_s: float) -> float:
    if t_since_stretch_s < 0:
        return state.truth["P0_gf"]
    kelvin = _kelvin_from_truth(state.truth, state.F3_gf)
    F_gf = float(simulate_generalized_kelvin(kelvin, np.array([t_since_stretch_s]))[0])
    return state.truth["P0_gf"] - F_gf


# --- Env --------------------------------------------------------------------

class Env:
    """Tool-call-dispatched environment. Mutating methods increment budget_used."""

    def __init__(
        self,
        seed: int = 0,
        spec_path: Path | None = None,
        budget: int = DEFAULT_BUDGET,
        randomize: bool = False,
        randomize_strength: float = 1.0,
    ):
        spec = Spec.load(spec_path or SPEC_PATH)
        # Noise rng and truth-draw rng are independent so changing one mode
        # doesn't shift the other's realizations.
        rng = np.random.default_rng(seed)
        if randomize:
            truth_rng = np.random.default_rng(np.uint64(seed) + np.uint64(10**9))
            truth = _draw_truth(spec, truth_rng, strength=randomize_strength)
            F3 = truth.pop("F3_gf_internal")
        else:
            truth = _spec_truth(spec)
            F3 = _calibrate_F3(
                truth["F0_gf"], truth["F1_gf"], truth["F2_gf"],
                truth["tau1_s"], truth["tau2_s"], truth["tau3_s"],
            )
        self.state = EnvState(
            spec=spec, rng=rng, truth=truth, F3_gf=F3, budget_total=budget
        )

    # ---- budget / logging helpers -----------------------------------------
    def _charge(self, tool: str, **kw: Any) -> None:
        self.state.budget_used += 1
        if self.state.budget_used > self.state.budget_total:
            raise RuntimeError("Tool-call budget exhausted.")
        self.state.log.append({"tool": tool, "clock_s": self.state.clock_s, **kw})

    # ---- tools ------------------------------------------------------------
    def measure_length(self, target: str) -> dict[str, Any]:
        self._charge("measure_length", target=target)
        if target == "thread_unstretched":
            truth = self.state.truth["l0_m"]
        elif target == "thread_stretched":
            if not self.state.stretched:
                raise ValueError("Thread not yet stretched.")
            truth = self.state.truth["l_m"]
        else:
            raise ValueError(f"Unknown length target: {target!r}. "
                             "Use 'thread_unstretched' or 'thread_stretched'.")
        value = noisy_length_reading(truth, self.state.rng)
        return {"length_m": round(value, 5), "sigma_m": 2.0e-4, "unit": "m"}

    def read_mass(self) -> dict[str, Any]:
        self._charge("read_mass")
        truth = self.state.truth["P0_gf"]
        value = noisy_scale_reading(truth, self.state.rng, sigma_gf=0.03)
        return {"mass_gf": round(value, 2), "sigma_gf": 0.03, "unit": "gf"}

    def stretch_thread(self) -> dict[str, Any]:
        self._charge("stretch_thread")
        if self.state.stretched:
            raise RuntimeError("Thread already stretched (one-shot).")
        self.state.stretched = True
        self.state.t_stretch_s = self.state.clock_s
        return {"ok": True, "t_started_s": 0.0}

    def wait(self, duration_s: float) -> dict[str, Any]:
        self._charge("wait", duration_s=duration_s)
        if duration_s <= 0:
            raise ValueError("duration_s must be positive.")
        self.state.clock_s += float(duration_s)
        return {"clock_s": self.state.clock_s - self.state.t_stretch_s if self.state.stretched else self.state.clock_s}

    def read_scale(self) -> dict[str, Any]:
        self._charge("read_scale")
        if not self.state.stretched:
            truth_P = self.state.truth["P0_gf"]
            t_reported = 0.0
        else:
            t_elapsed = self.state.clock_s - self.state.t_stretch_s
            truth_P = _true_P_gf(self.state, t_elapsed)
            t_reported = noisy_time_reading(t_elapsed, self.state.rng)
        P_noisy = noisy_scale_reading(truth_P, self.state.rng)
        return {"t_s": round(t_reported, 2), "P_gf": round(P_noisy, 2), "unit": "gf"}

    def submit(self, answer: dict[str, Any]) -> dict[str, Any]:
        self._charge("submit")
        if self.state.submitted:
            raise RuntimeError("Already submitted.")
        self.state.submitted = True
        self.state.submitted_answer = dict(answer)
        return {"accepted": True}
