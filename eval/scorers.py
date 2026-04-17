"""Rubric-based scorer for the IPhO physics benchmark (v0).

The spec's `scoring.rubric` lists rows by criterion + parameter. For
parameter-based rows we check the submitted value against
`truth_value ± tolerance_full` (relative, multiplicative for the physics
parameters). For procedural rows (e.g. "length reported with sigma") we
check lightweight evidence in the store (call log, submitted answer
shape). Unrecognized procedural rows award 0 and are flagged in the
explanation so the rubric can be extended as we iterate.
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from envs.common.spec_schema import Spec


def _coerce_value(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _value_and_sigma(v: Any) -> tuple[float | None, float | None]:
    if isinstance(v, (list, tuple)):
        if len(v) >= 2:
            return _coerce_value(v[0]), _coerce_value(v[1])
        if len(v) == 1:
            return _coerce_value(v[0]), None
    return _coerce_value(v), None


def _within_relative(val: float, truth: float, rel_tol: float) -> bool:
    if truth == 0:
        return abs(val) <= rel_tol
    return abs(val - truth) <= rel_tol * abs(truth)


def _score_parameter_row(answer, spec, row):
    name = row.parameter
    truth = spec.param(name).truth_value
    rel_tol = spec.param(name).tolerance_full
    submitted = _coerce_value(answer.get(name))
    if submitted is None:
        return 0.0, {"status": "missing"}
    passed = _within_relative(submitted, truth, rel_tol)
    return (row.points if passed else 0.0), {
        "submitted": submitted,
        "truth": truth,
        "rel_tol": rel_tol,
        "passed": passed,
    }


def _score_procedural_row(answer, spec, row, call_log):
    crit = row.criterion

    if crit.endswith("_reported_with_sigma"):
        key_map = {
            "A1_length_reported_with_sigma": "l0_m",
            "A2_weight_reported_with_sigma": "P0_gf",
            "A4_stretched_length_reported_with_sigma": "l_m",
        }
        key = key_map.get(crit)
        if key is None:
            return 0.0, {"status": "unmapped_procedural", "criterion": crit}
        _, sigma = _value_and_sigma(answer.get(key))
        passed = sigma is not None and sigma > 0
        return (row.points if passed else 0.0), {"key": key, "passed": passed}

    if crit == "A3_timeseries_captured":
        n = sum(1 for c in call_log if c.get("tool") == "read_scale")
        passed = n >= 5
        return (row.points if passed else 0.0), {"read_scale_calls": n, "passed": passed}

    if crit == "data_table_has_enough_samples":
        n = sum(1 for c in call_log if c.get("tool") == "read_scale")
        threshold = row.min_samples or 30
        passed = n >= threshold
        return (row.points if passed else 0.0), {
            "read_scale_calls": n,
            "required": threshold,
            "passed": passed,
        }

    return 0.0, {"status": "unrecognized", "criterion": crit}


@scorer(metrics=[mean()])
def rubric_scorer(spec: Spec) -> Scorer:

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.store.get("submitted_answer") or {}
        call_log = state.store.get("call_log") or []

        earned = 0.0
        max_total = 0.0
        rows_out = []
        for row in spec.scoring.rubric:
            if row.parameter is not None:
                pts, expl = _score_parameter_row(answer, spec, row)
            else:
                pts, expl = _score_procedural_row(answer, spec, row, call_log)
            earned += pts
            max_total += row.points
            rows_out.append(
                {
                    "criterion": row.criterion,
                    "parameter": row.parameter,
                    "max": row.points,
                    "awarded": pts,
                    **expl,
                }
            )

        normalized = earned / max_total if max_total > 0 else 0.0
        return Score(
            value=normalized,
            answer=json.dumps(answer)[:500],
            explanation=json.dumps(
                {
                    "earned_rubric_points": earned,
                    "max_rubric_points": max_total,
                    "simulated_points_max": spec.meta.simulated_points,
                    "rows": rows_out,
                },
                default=str,
            ),
        )

    return score
