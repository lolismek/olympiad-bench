# SCHEMA_NOTES

Running log of `spec.yaml` schema changes encountered while hand-authoring real IPhO problems. Append-only — each entry dated.

## v0.1 — 2026-04-17

Initial draft from the broad plan (Appendix, Stage 2). Untested against a real problem.

## v0.2 — 2026-04-17 — after hand-authoring IPhO 2023 E1 problem.md (aborted)

Tried 2023 E1 (spring-mass oscillator) and hit an assumption failure: the real problem is electromechanical, not SHM. Scope diverged too far from the plan's framing. Pivoted target to **IPhO 2018 E2 (viscoelastic polymer thread)** and did not commit a schema change for 2023 E1. Lesson recorded for future Gate A: broad-plan apparatus summaries can mislead; always read the problem PDF before committing a schema.

## v0.3 — 2026-04-17 — after hand-authoring IPhO 2018 E2

Changes made to the v0.1 draft while filling 2018 E2:

1. **`meta.official_points` and `meta.simulated_points` split.** The broad plan assumed we cover a whole problem. Real first-pass scope often omits parts that need a different simulator backend (here: laser-diffraction for diameter, constant-stress E measurement). Added `scope_notes` string to make the deferral explicit.
2. **`physics_model.family` added.** The draft used a free-text `governing_equation` only. With multiple problem types coming, we need a tag so `mechanics_ode.py` can dispatch. Current tag: `generalized_kelvin_constant_strain`. Future tags: `shm_damped_driven`, `rc_circuit`, etc.
3. **`physics_model.relation_beta` added as an optional field.** 2018 E2 needs an auxiliary relation to map between measured gf-force and SI stress. Kept it loose (free-text, optional) because it's problem-specific and I don't want to formalise it until a second example needs it.
4. **`parameters.truth_sigma` added as optional.** Some parameters are measurements (`ℓ₀ = 43.7 ± 0.2 cm`) where the official uncertainty is meaningful; others are fit outcomes where only the value is official. Both now fit the schema.
5. **`parameters.source`** added (`official` vs `derived`). Required because the 2018 marking scheme publishes values but not tolerances — tolerances are a broad-plan convention (±10–30%). Flagging derived ones avoids hiding that.
6. **Rubric-item `parameter` cross-reference.** The draft let each rubric item stand alone as free text. Added optional `parameter` pointing to a `physics_model.parameters.name`, so the scorer can look up the truth value + tolerance without string-parsing the criterion name.
7. **Rubric-item `min_samples`.** Needed for time-series parts like A.3 ("≥30 scale reads during 45 min"). Draft had no way to express sample-count criteria.
8. **`parts.expected_data.timeseries`** — the draft's `expected_data` covered `table` and `plot`; added `timeseries` with `{cols, min_samples, duration_s}` for the A.3 stress-relaxation recording.
9. **`parts.answer` made optional.** Descriptive parts (D.2, D.5) have no numerical answer; the draft required one.
10. **Descriptive/judge-scored parts deliberately excluded from `parts[]`** in v0 rather than included with `points: 0`. Will revisit when we have a judge scorer. Listed as a comment inside `spec.yaml` so the omission is auditable.

What *didn't* need changing (surprising positives):
- Multi-part dependencies (D.11 needs F₀ and τ₁ from D.8/D.9) fit naturally — no conditional-dependency machinery required at the spec level; the env's tool outputs carry the dependencies implicitly.
- Qualitative parts (D.5 "sketch purely elastic") are scorable in v0 as present/absent via a rubric criterion flag; don't need schema changes.
- Partial credit via `tolerance_half` is natural (already in the draft). The marking scheme for this problem is simpler than expected — mostly full-credit-or-zero, with a handful of multi-line questions that would merit partial credit.

Known deferrals (Gate A accepted these and moved on):
- **Figure-aware parts.** A.1 asks the contestant to add 5 mm for each screw; B.1 asks for a sketch of the optics. We're deferring B entirely; for A.1, `measure_length(target='thread_unstretched')` returns the correct value directly and the screw-head adjustment happens inside the env — the agent doesn't have to know about it. Not a faithful-contract win, but an acceptable v0 simplification.
- **Units conversions.** β encodes gf↔SI. Kept as a free `constants:` block for now; a formal units system (`pint`) is premature.
- **Graph submissions.** The plan's non-goals call these out; no schema provision yet.

Schema file: `envs/common/spec_schema.py`. Loads cleanly against the 2018 E2 `spec.yaml` (verified).
