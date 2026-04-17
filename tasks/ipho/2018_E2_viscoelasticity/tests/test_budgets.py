"""Budget-separation and graceful-exhaustion tests.

Regression guard for the bug that killed the 2026-04-17 Sonnet run: the old
unified 200-call budget was shared between experimental and scratchpad tools,
and exhaustion raised instead of returning an error, killing the sample
before it could submit.
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


def _new_env(**kw):
    return _env_mod.Env(seed=0, **kw)


def test_calc_does_not_consume_experimental_budget():
    env = _new_env(budget=3, scratch_budget=50)
    # Burn all 3 experimental calls on calc/linreg — they shouldn't count.
    for _ in range(10):
        env.calc("1 + 1")
    assert env.state.budget_used == 0
    assert env.state.scratch_used == 10
    # Experimental calls still available.
    r = env.read_mass()
    assert "mass_gf" in r


def test_exp_budget_exhaustion_returns_error_dict_not_raises():
    env = _new_env(budget=2)
    env.read_mass()
    env.read_mass()
    # Third call: budget is full; should return error, NOT raise.
    r = env.read_mass()
    assert "error" in r
    assert "budget" in r["error"].lower()
    # Env didn't die — subsequent scratchpad + submit still work.
    assert env.calc("2 + 2")["value"] == 4.0
    assert env.submit({"dummy": [1.0]}) == {"accepted": True}


def test_scratch_budget_exhaustion_returns_error_dict():
    env = _new_env(scratch_budget=2)
    env.calc("1")
    env.linreg([0.0, 1.0], [0.0, 1.0])
    r = env.calc("3 + 4")
    assert "error" in r
    # Experimental tools still function.
    assert "mass_gf" in env.read_mass()


def test_submit_is_free_and_works_after_exp_budget_exhausted():
    env = _new_env(budget=1)
    env.read_mass()               # consume the sole experimental slot
    r = env.read_mass()
    assert "error" in r           # confirm exhaustion
    # submit should still succeed.
    out = env.submit({"l0_m": [0.437, 2e-4]})
    assert out == {"accepted": True}
    assert env.state.submitted
    assert env.state.submitted_answer == {"l0_m": [0.437, 2e-4]}


def test_default_budgets_are_what_docs_claim():
    env = _new_env()
    assert env.state.budget_total == 200
    assert env.state.scratch_total == 500


def test_budget_error_includes_usage_fields():
    env = _new_env(budget=1)
    env.read_mass()
    r = env.read_mass()
    assert r.get("budget_used") == 1
    assert r.get("budget_total") == 1
