"""Gate B replay: run the reference contestant across many seeds.

Pass criteria (minimal v0):

  • ≥95% of seeds recover every rubric-scored parameter within the tolerance
    encoded in spec.yaml (relative, multiplicative for τ/E; absolute for ε).
  • Directly measured values (ℓ₀, ℓ, P₀) are within their reported σ
    ≈68% of the time (one-sigma coverage sanity check).
  • A deliberately wrong reference (all answers ×10) fails the tolerance
    check, proving the harness isn't trivially passable.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import sys
TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.common.load_task import load_task_env  # noqa: E402
from envs.common.spec_schema import Spec  # noqa: E402

# Load once — reused across parametrized cases.
_env_mod = load_task_env(TASK_DIR)
_spec = Spec.load(TASK_DIR / "spec.yaml")
_solve_mod = load_task_env.__globals__["load_module"](
    TASK_DIR / "reference_solution.py", "_task_ref"
)

# Parameters that live in env.py's tool outputs but still warrant a sigma check.
DIRECTLY_MEASURED = {"l0_m", "P0_gf", "l_m"}

# Multiplicative tolerance for τ/E fits; the problem's rubric grants points
# within [truth/1.33, truth*1.33] for the dissipation spectrum.
RELATIVE_TOL = 1.33
# ε reported to 3 decimal places → absolute tolerance 0.01 is generous.
EPSILON_ABS_TOL = 0.01

N_SEEDS = 50


def _run_one(seed: int) -> dict:
    env = _env_mod.Env(seed=seed)
    out = _solve_mod.solve(env)
    return out.answer


@pytest.mark.parametrize("seed", list(range(N_SEEDS)))
def test_reference_replays_cleanly(seed: int):
    ans = _run_one(seed)
    assert "tau1_s" in ans and "E0_SI" in ans, ans


def _truth(name: str) -> float:
    return _spec.param(name).truth_value


def _within_ratio(val: float, truth: float, k: float = RELATIVE_TOL) -> bool:
    if truth == 0:
        return abs(val) < k
    return (truth / k) <= val <= (truth * k)


def test_reference_recovers_rubric_parameters():
    """≥95% pass rate on each τ/E parameter across 50 seeds."""
    pass_counts = {
        name: 0
        for name in ["tau1_s", "tau2_s", "tau3_s", "E0_SI", "E1_SI", "E2_SI"]
    }
    for seed in range(N_SEEDS):
        ans = _run_one(seed)
        for name in pass_counts:
            val = ans[name][0]
            if _within_ratio(val, _truth(name)):
                pass_counts[name] += 1

    for name, count in pass_counts.items():
        rate = count / N_SEEDS
        assert rate >= 0.95, f"{name}: only {rate:.0%} of seeds within ±33% tol"


def test_reference_sigma_coverage_for_direct_measurements():
    """ℓ₀, ℓ, P₀ should be within reported σ on ≈68% of seeds."""
    covers = {name: 0 for name in DIRECTLY_MEASURED}
    for seed in range(N_SEEDS):
        ans = _run_one(seed)
        for name in DIRECTLY_MEASURED:
            val, sigma = ans[name][0], ans[name][1]
            if abs(val - _truth(name)) <= sigma:
                covers[name] += 1

    for name, count in covers.items():
        rate = count / N_SEEDS
        # Generous band around Gaussian 68% — ruler is uniform, so ~58% expected.
        assert 0.45 <= rate <= 0.99, (
            f"{name}: σ-coverage {rate:.0%} implausible "
            f"(expected ~58–68%)."
        )


def test_epsilon_within_absolute_tolerance():
    for seed in range(N_SEEDS):
        ans = _run_one(seed)
        eps = ans["epsilon"][0]
        assert abs(eps - _truth("epsilon")) <= EPSILON_ABS_TOL, (
            f"seed {seed}: ε={eps} vs truth {_truth('epsilon')}"
        )


def test_wrong_reference_fails_tolerance():
    """Sanity: a 10× deranged answer must fail the tolerance check.

    Proves the rubric isn't trivially passable — catches the case where a
    future refactor silently makes `_within_ratio` permissive.
    """
    for seed in range(3):  # handful of seeds is enough; this is a sanity check
        ans = _run_one(seed)
        bad_val = ans["tau1_s"][0] * 10.0
        assert not _within_ratio(bad_val, _truth("tau1_s")), (
            "10× wrong answer still within tolerance — rubric is broken."
        )
