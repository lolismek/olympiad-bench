"""Randomized-truth replay: run the reference across 50 seeds with per-seed
perturbed truth values and verify recovery against each seed's own truth.

Purpose: this is the memorization-defeat regression test. If a future change
to env.py (e.g., bounds tightening, new constraints) breaks physical
solvability, or if reference_solution.py starts leaning on the canonical
1546/177/50-second truth, this test trips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.common.load_task import load_task_env  # noqa: E402

_env_mod = load_task_env(TASK_DIR)
_solve_mod = load_task_env.__globals__["load_module"](
    TASK_DIR / "reference_solution.py", "_task_ref_rand"
)

RELATIVE_TOL = 1.33
EPSILON_ABS_TOL = 0.01
N_SEEDS = 50
# Recoverable param coverage required — mirrors the fixed-truth test.
MIN_PASS_RATE = 0.95
RUBRIC_PARAMS = ("tau1_s", "tau2_s", "tau3_s", "E0_SI", "E1_SI", "E2_SI")


def _within_ratio(val: float, truth: float, k: float = RELATIVE_TOL) -> bool:
    if truth == 0:
        return abs(val) < k
    return (truth / k) <= val <= (truth * k)


def _run_one(seed: int) -> tuple[dict, dict]:
    env = _env_mod.Env(seed=seed, randomize=True, randomize_strength=1.0)
    truth = dict(env.state.truth)
    out = _solve_mod.solve(env)
    return out.answer, truth


def test_randomized_truth_varies_across_seeds():
    """Guard against accidentally reverting to the fixed-truth baseline."""
    truths = []
    for seed in range(8):
        env = _env_mod.Env(seed=seed, randomize=True)
        truths.append(env.state.truth["tau1_s"])
    # At least 5 distinct τ₁ values across 8 seeds — otherwise randomization is
    # broken (e.g., RNG seeded identically, or bounds collapsed to zero width).
    assert len(set(truths)) >= 5, f"τ₁ draws collapsed: {truths}"


def test_randomize_false_preserves_canonical_truth():
    """Sanity: randomize=False must return the published IPhO truth verbatim."""
    env = _env_mod.Env(seed=0, randomize=False)
    # These are the canonical spec.yaml values; drift here means the
    # original-problem mode silently changed.
    assert env.state.truth["tau1_s"] == pytest.approx(1546.0, rel=1e-6)
    assert env.state.truth["tau2_s"] == pytest.approx(177.0, rel=1e-6)
    assert env.state.truth["tau3_s"] == pytest.approx(50.0, rel=1e-6)
    assert env.state.truth["P0_gf"] == pytest.approx(81.11, rel=1e-6)


def test_reference_recovers_randomized_rubric_parameters():
    """≥95% pass rate on each τ/E parameter across 50 randomized seeds,
    each scored against its own per-seed truth."""
    pass_counts = {name: 0 for name in RUBRIC_PARAMS}
    for seed in range(N_SEEDS):
        ans, truth = _run_one(seed)
        for name in RUBRIC_PARAMS:
            if _within_ratio(ans[name][0], truth[name]):
                pass_counts[name] += 1

    for name, count in pass_counts.items():
        rate = count / N_SEEDS
        assert rate >= MIN_PASS_RATE, (
            f"{name}: only {rate:.0%} of randomized seeds within ±33% "
            f"of per-seed truth — env bounds may be too loose or the "
            f"reference fit assumes the canonical truth."
        )


def test_reference_epsilon_on_randomized_seeds():
    for seed in range(N_SEEDS):
        ans, truth = _run_one(seed)
        eps = ans["epsilon"][0]
        assert abs(eps - truth["epsilon"]) <= EPSILON_ABS_TOL, (
            f"seed {seed}: ε={eps} vs per-seed truth {truth['epsilon']}"
        )


def test_randomize_strength_zero_matches_baseline():
    """strength=0.0 must collapse the perturbation to identity."""
    env0 = _env_mod.Env(seed=0, randomize=True, randomize_strength=0.0)
    baseline = _env_mod.Env(seed=0, randomize=False)
    for k in ("tau1_s", "tau2_s", "tau3_s", "P0_gf", "epsilon"):
        assert env0.state.truth[k] == pytest.approx(baseline.state.truth[k], rel=1e-9)
