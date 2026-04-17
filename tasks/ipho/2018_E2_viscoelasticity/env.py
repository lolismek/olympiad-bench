"""Simulated environment for IPhO 2018 E2 — viscoelastic polymer thread.

Agent-visible tool API:

    measure_length(target: str)          -> {"length_m": float, "sigma_m": float, "unit": "m"}
    read_mass()                          -> {"mass_gf": float, "sigma_gf": float, "unit": "gf"}
    stretch_thread()                     -> {"ok": True, "t_started_s": 0.0}
    wait(duration_s: float)              -> {"clock_s": float}
    read_scale()                         -> {"t_s": float, "P_gf": float, "unit": "gf"}
    submit(answer: dict)                 -> {"accepted": True}

Budget: 200 tool calls. Exceeding budget terminates the episode with score 0.
"""

from __future__ import annotations

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


@dataclass
class EnvState:
    spec: Spec
    rng: np.random.Generator
    budget_used: int = 0
    budget_total: int = DEFAULT_BUDGET
    stretched: bool = False
    clock_s: float = 0.0            # simulated wall-clock
    t_stretch_s: float = 0.0        # real time (sim) when stretch happened; measurements use (clock - this)
    submitted: bool = False
    submitted_answer: dict[str, Any] | None = None
    log: list[dict[str, Any]] = field(default_factory=list)


def _truth_kelvin(spec: Spec) -> KelvinParams:
    F0 = spec.param("F0_gf").truth_value
    F1 = spec.param("F1_gf").truth_value
    t1 = spec.param("tau1_s").truth_value
    F2 = spec.param("F2_gf").truth_value
    t2 = spec.param("tau2_s").truth_value
    t3 = spec.param("tau3_s").truth_value
    # The third-component amplitude F3 isn't published. Calibrate it so the
    # engine reproduces the official short-time datum F(10 s) = 45.41 gf.
    import math
    residual_at_10 = 45.41 - F0 - F1 * math.exp(-10.0 / t1) - F2 * math.exp(-10.0 / t2)
    F3 = max(residual_at_10 / math.exp(-10.0 / t3), 0.0)
    return KelvinParams(F0_gf=F0, components=((F1, t1), (F2, t2), (F3, t3)))


def _true_P_gf(spec: Spec, t_since_stretch_s: float) -> float:
    """Simulated scale reading at a given elapsed time after stretching."""
    P0 = spec.param("P0_gf").truth_value
    if t_since_stretch_s < 0:
        return P0
    kelvin = _truth_kelvin(spec)
    F_gf = float(simulate_generalized_kelvin(kelvin, np.array([t_since_stretch_s]))[0])
    return P0 - F_gf


class Env:
    """Tool-call-dispatched environment. Mutating methods increment budget_used."""

    def __init__(self, seed: int = 0, spec_path: Path | None = None, budget: int = DEFAULT_BUDGET):
        spec = Spec.load(spec_path or SPEC_PATH)
        self.state = EnvState(
            spec=spec,
            rng=np.random.default_rng(seed),
            budget_total=budget,
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
            truth = self.state.spec.param("l0_m").truth_value
        elif target == "thread_stretched":
            if not self.state.stretched:
                raise ValueError("Thread not yet stretched.")
            truth = self.state.spec.param("l_m").truth_value
        else:
            raise ValueError(f"Unknown length target: {target!r}. "
                             "Use 'thread_unstretched' or 'thread_stretched'.")
        value = noisy_length_reading(truth, self.state.rng)
        return {"length_m": round(value, 5), "sigma_m": 2.0e-4, "unit": "m"}

    def read_mass(self) -> dict[str, Any]:
        self._charge("read_mass")
        truth = self.state.spec.param("P0_gf").truth_value
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
        spec = self.state.spec
        if not self.state.stretched:
            truth_P = spec.param("P0_gf").truth_value
            t_reported = 0.0
        else:
            t_elapsed = self.state.clock_s - self.state.t_stretch_s
            truth_P = _true_P_gf(spec, t_elapsed)
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
