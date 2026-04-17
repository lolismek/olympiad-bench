"""Inspect AI Task wrapper for IPhO 2018 E2 — viscoelastic polymer thread.

Layout (v0, one task, N parallel seeds as Samples):

  • Each `Sample` carries a unique `seed`.
  • A pre-solver initializes `Env(seed)` into `store()` under key "env".
  • Tools wrap env methods and also mirror the call into store["call_log"]
    so the scorer can audit whether the agent actually took measurements.
  • `submit` stores the answer dict into `store["submitted_answer"]`.
  • The scorer reads `store["submitted_answer"]` and `store["call_log"]`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import Generate, TaskState, generate, solver, use_tools
from inspect_ai.tool import tool
from inspect_ai.util import store

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.common.load_task import load_task_env  # noqa: E402
from envs.common.spec_schema import Spec  # noqa: E402
from eval.scorers import rubric_scorer  # noqa: E402

TASK_DIR = REPO_ROOT / "tasks" / "ipho" / "2018_E2_viscoelasticity"
_env_mod = load_task_env(TASK_DIR)
_spec = Spec.load(TASK_DIR / "spec.yaml")
_problem_md = (TASK_DIR / "problem.md").read_text()

N_SEEDS_DEFAULT = 3
MAX_TOOL_CALLS = 220  # 200-call env budget + a few scoring reads


# --- Env access helper ------------------------------------------------------

def _env():
    env = store().get("env")
    if env is None:
        raise RuntimeError(
            "Env not initialized — init_env solver must run before tools."
        )
    return env


def _log(tool_name: str, **kw: Any) -> None:
    log = store().get("call_log") or []
    log.append({"tool": tool_name, **kw})
    store().set("call_log", log)


# --- Tools (thin wrappers around Env) ---------------------------------------

@tool
def measure_length():
    async def execute(target: str) -> dict[str, Any]:
        """Measure a length on the setup.

        Args:
            target: one of 'thread_unstretched' or 'thread_stretched'.
        """
        r = _env().measure_length(target)
        _log("measure_length", target=target, result=r)
        return r

    return execute


@tool
def read_mass():
    async def execute() -> dict[str, Any]:
        """Read the digital scale while the thread hangs free (full weight)."""
        r = _env().read_mass()
        _log("read_mass", result=r)
        return r

    return execute


@tool
def stretch_thread():
    async def execute() -> dict[str, Any]:
        """Hook the stretched-thread fixture (one-shot; starts the clock)."""
        r = _env().stretch_thread()
        _log("stretch_thread", result=r)
        return r

    return execute


@tool
def wait_seconds():
    async def execute(duration_s: float) -> dict[str, Any]:
        """Wait the given number of seconds of simulated wall-clock time.

        Args:
            duration_s: seconds to advance (must be positive).
        """
        r = _env().wait(duration_s)
        _log("wait", duration_s=duration_s, result=r)
        return r

    return execute


@tool
def read_scale():
    async def execute() -> dict[str, Any]:
        """Read the scale now and report (t_since_stretch_s, P_gf)."""
        r = _env().read_scale()
        _log("read_scale", result=r)
        return r

    return execute


@tool
def submit():
    async def execute(answer: dict[str, Any]) -> dict[str, Any]:
        """Submit the final answer dictionary (ends the episode).

        Args:
            answer: mapping from parameter name → [value, sigma] or [value].
        """
        r = _env().submit(answer)
        _log("submit", answer_keys=list(answer.keys()))
        store().set("submitted_answer", dict(answer))
        return r

    return execute


# --- Solver: initialize the env for the sample -----------------------------

@solver
def init_env():
    async def solve(state: TaskState, generate_: Generate) -> TaskState:
        seed = state.metadata.get("seed", 0)
        env = _env_mod.Env(seed=int(seed))
        store().set("env", env)
        store().set("call_log", [])
        return state

    return solve


# --- Task entrypoint -------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an International Physics Olympiad contestant solving an "
    "experimental problem. You have access to a simulated laboratory via "
    "tools. You have a 200-tool-call budget and a single opportunity to "
    "submit your final answer via the `submit` tool. Think carefully about "
    "when to read the scale, how many measurements to take at long times "
    "(slow relaxation), and what schedule of `wait_seconds` to use. Report "
    "every quantity with its uncertainty."
)


@task
def ipho_2018_E2(seeds: int = N_SEEDS_DEFAULT) -> Task:
    samples = [
        Sample(
            id=f"seed-{s}",
            input=_problem_md,
            metadata={"seed": s},
        )
        for s in range(seeds)
    ]

    return Task(
        dataset=samples,
        solver=[
            init_env(),
            use_tools(
                [
                    measure_length(),
                    read_mass(),
                    stretch_thread(),
                    wait_seconds(),
                    read_scale(),
                    submit(),
                ]
            ),
            generate(tool_calls="loop"),
        ],
        scorer=rubric_scorer(_spec),
        message_limit=MAX_TOOL_CALLS,
        config=GenerateConfig(system_message=SYSTEM_PROMPT),
    )
