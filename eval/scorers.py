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
import re
from typing import Any

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from envs.common.spec_schema import Spec


# Common stylistic aliases contestants use for our canonical parameter names.
# Keys are canonical (matching spec.yaml); values are alternate spellings.
# We canonicalize the submission before checking so minor deviations (e.g.
# `tau_1`, `τ_1`, `ell_0`) don't silently tank the score.
_ALIAS_MAP: dict[str, list[str]] = {
    "l0_m": ["ell_0", "ell0", "l0", "L0", "ell_unstretched"],
    "l_m": ["ell", "L", "l", "ell_stretched"],
    "P0_gf": ["P_0", "P0", "p_0", "p0", "mass"],
    "epsilon": ["eps", "strain", "ε"],
    "beta_gf_inv_SI": ["beta", "β"],
    "tau1_s": ["tau_1", "tau1", "τ_1", "τ1"],
    "tau2_s": ["tau_2", "tau2", "τ_2", "τ2"],
    "tau3_s": ["tau_3", "tau3", "τ_3", "τ3"],
    "E0_SI": ["E_0", "E0"],
    "E1_SI": ["E_1", "E1"],
    "E2_SI": ["E_2", "E2"],
}


def _normalize_key(k: str) -> str:
    """Lowercase, strip non-alphanumerics, drop common suffixes."""
    k = k.strip().lower()
    k = re.sub(r"[^a-z0-9]", "", k)
    for suffix in ("si", "s", "m", "gf", "unit"):
        if k.endswith(suffix) and len(k) > len(suffix):
            k2 = k[: -len(suffix)]
            if k2:
                return k2
    return k


def _canonicalize_answer(answer: dict[str, Any], canonical_keys: list[str]) -> dict[str, Any]:
    """Map submitted keys onto canonical spec keys via alias table + fuzzy match."""
    out: dict[str, Any] = {}

    # First copy any exact canonical matches straight through.
    for canon in canonical_keys:
        if canon in answer:
            out[canon] = answer[canon]

    # Build a lookup from normalized form → canonical.
    norm_to_canon: dict[str, str] = {}
    for canon in canonical_keys:
        norm_to_canon[_normalize_key(canon)] = canon
        for alias in _ALIAS_MAP.get(canon, []):
            norm_to_canon[_normalize_key(alias)] = canon

    # Any remaining submitted key: see if its normalized form hits an alias.
    for k, v in answer.items():
        if k in out:
            continue
        canon = norm_to_canon.get(_normalize_key(k))
        if canon and canon not in out:
            out[canon] = v
    return out


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

    canonical_keys = [p.name for p in spec.physics_model.parameters]

    async def score(state: TaskState, target: Target) -> Score:
        raw_answer = state.store.get("submitted_answer") or {}
        answer = _canonicalize_answer(raw_answer, canonical_keys)
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
