"""Run baselines against IPhO 2018 E2.

Usage
-----

Reference-solution oracle (no model API calls; establishes the score ceiling):

    python -m eval.run_baselines --oracle --seeds 10

Real model via Inspect AI (requires ANTHROPIC_API_KEY / OPENAI_API_KEY /
GOOGLE_API_KEY or AWS Bedrock credentials):

    inspect eval eval/tasks/ipho_2018_E2.py@ipho_2018_E2 \\
        --model anthropic/claude-opus-4-5-20250929 \\
        -T seeds=5

    inspect eval eval/tasks/ipho_2018_E2.py@ipho_2018_E2 \\
        --model openai/gpt-4o \\
        -T seeds=5

    inspect eval eval/tasks/ipho_2018_E2.py@ipho_2018_E2 \\
        --model google/gemini-2.5-pro \\
        -T seeds=5

The oracle run does not go through Inspect; it invokes the reference
contestant directly on the Env and then feeds its answer + tool-call log
through the same scorer that Inspect uses, so the numbers are comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from envs.common.load_task import load_task_env
from envs.common.spec_schema import Spec
from eval.scorers import rubric_scorer

TASK_DIR = REPO_ROOT / "tasks" / "ipho" / "2018_E2_viscoelasticity"
_env_mod = load_task_env(TASK_DIR)
_spec = Spec.load(TASK_DIR / "spec.yaml")
_solve_mod = load_task_env.__globals__["load_module"](
    TASK_DIR / "reference_solution.py", "_task_ref"
)


class _StubStore:
    def __init__(self, m):
        self._m = m

    def get(self, k, default=None):
        return self._m.get(k, default)


class _StubState:
    def __init__(self, m):
        self.store = _StubStore(m)
        self.output = type("O", (), {"completion": ""})()


async def _score_one(answer, log):
    score = rubric_scorer(_spec)
    state = _StubState({"submitted_answer": answer, "call_log": log})
    return await score(state, target=None)


def run_oracle(seeds: int) -> None:
    rows = []
    for s in range(seeds):
        env = _env_mod.Env(seed=s)
        out = _solve_mod.solve(env)
        score = asyncio.run(_score_one(out.answer, list(env.state.log)))
        details = json.loads(score.explanation)
        rows.append(
            {
                "seed": s,
                "fraction": round(score.value, 3),
                "points": round(details["earned_rubric_points"], 2),
                "max": details["max_rubric_points"],
            }
        )
        print(
            f"seed {s:3d}   fraction={score.value:.3f}   "
            f"{details['earned_rubric_points']:.2f}/{details['max_rubric_points']:.1f} pts"
        )

    fractions = [r["fraction"] for r in rows]
    mean = sum(fractions) / len(fractions)
    mn, mx = min(fractions), max(fractions)
    print(f"\noracle across {seeds} seeds: mean={mean:.3f}   min={mn:.3f}   max={mx:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true", help="Run reference-solution baseline.")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    if args.oracle:
        run_oracle(args.seeds)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
