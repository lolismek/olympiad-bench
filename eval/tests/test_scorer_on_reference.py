"""Scorer sanity test: feed the reference solution's answer + env call log
into the Inspect scorer and verify the rubric awards near-full credit.

Also checks that a deliberately-wrong answer (all zeros) scores ~0 and that
an empty call log fails the procedural "data_table_has_enough_samples" row.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.common.load_task import load_task_env  # noqa: E402
from envs.common.spec_schema import Spec  # noqa: E402
from eval.scorers import rubric_scorer  # noqa: E402

TASK_DIR = REPO_ROOT / "tasks" / "ipho" / "2018_E2_viscoelasticity"
_env_mod = load_task_env(TASK_DIR)
_spec = Spec.load(TASK_DIR / "spec.yaml")
_solve_mod = load_task_env.__globals__["load_module"](
    TASK_DIR / "reference_solution.py", "_task_ref"
)


class _StubStore:
    def __init__(self, mapping):
        self._m = mapping

    def get(self, key, default=None):
        return self._m.get(key, default)


class _StubState:
    def __init__(self, store_mapping):
        self.store = _StubStore(store_mapping)
        self.output = type("O", (), {"completion": ""})()


async def _score(answer, call_log):
    scorer_ = rubric_scorer(_spec)
    state = _StubState({"submitted_answer": answer, "call_log": call_log})
    score = await scorer_(state, target=None)
    return score


def test_reference_scores_highly():
    env = _env_mod.Env(seed=0)
    out = _solve_mod.solve(env)
    # Build a call log matching what the Inspect tools would produce by
    # counting invocations that the env actually recorded internally.
    call_log = list(env.state.log)
    score = asyncio.run(_score(out.answer, call_log))
    # >=85% of rubric weight earned on a typical seed.
    assert score.value >= 0.85, f"reference scored only {score.value:.2f}"
    details = json.loads(score.explanation)
    assert details["earned_rubric_points"] >= 0.85 * details["max_rubric_points"]


def test_zero_answer_fails():
    zero = {p.name: [0.0, 0.0] for p in _spec.physics_model.parameters}
    score = asyncio.run(_score(zero, []))
    assert score.value <= 0.1, f"all-zero answer scored {score.value:.2f}"


def test_empty_call_log_fails_data_table_row():
    # Reference-quality numeric answer + empty log → procedural rows drop.
    env = _env_mod.Env(seed=0)
    out = _solve_mod.solve(env)
    score = asyncio.run(_score(out.answer, []))
    details = json.loads(score.explanation)
    data_row = next(
        r for r in details["rows"]
        if r["criterion"] == "data_table_has_enough_samples"
    )
    assert data_row["awarded"] == 0.0
