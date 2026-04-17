# olympiad-bench

v0 prototype benchmarking LLM agents on IPhO experimental-round problems inside simulated environments.

Current scope: one problem, **IPhO 2018 E2 — Viscoelasticity of a polymer thread**, used to validate three load-bearing assumptions (spec schema, physics engine, tool API) before committing to broader coverage.

## Layout

- `BROAD_IPHO_PLAN.md` — long-term multi-problem, multi-backend vision.
- `tasks/ipho/2018_E2_viscoelasticity/` — problem.md, spec.yaml, env.py, reference_solution.py, tests/.
- `envs/physics/mechanics_ode.py` — reusable 2nd-order ODE + generalized-Kelvin engine.
- `envs/common/` — spec schema (Pydantic v2), noise layer, task loader.
- `eval/tasks/ipho_2018_E2.py` — Inspect AI task (tools + solver + scorer wiring).
- `eval/scorers.py` — rubric scorer that consumes `spec.yaml`.
- `eval/run_baselines.py` — oracle baseline runner (reference solution as ceiling).

## Gate status

- **Gate A** — spec.yaml authored from official IPhO solution; schema validated.
- **Gate B** — engine, env, reference solution, 50-seed replay test all green (`pytest tasks/`).
- **Gate C** — Inspect AI task + scorer wired; scorer sanity-tested; oracle ceiling = 1.0 / 1.0.
- **Gate D** — pending first real-model baseline run.

## Running

Reference-solution ceiling (no API keys needed):

    python -m eval.run_baselines --oracle --seeds 10

Real-model baseline (requires credentials; each run is 1 × `seeds` × up to 200 tool calls):

    inspect eval eval/tasks/ipho_2018_E2.py@ipho_2018_E2 \
        --model anthropic/claude-opus-4-5-20250929 -T seeds=5

    inspect eval eval/tasks/ipho_2018_E2.py@ipho_2018_E2 \
        --model openai/gpt-4o -T seeds=5

    inspect eval eval/tasks/ipho_2018_E2.py@ipho_2018_E2 \
        --model google/gemini-2.5-pro -T seeds=5

## Testing

    pytest tasks/ipho/2018_E2_viscoelasticity/tests/ -q   # 50-seed replay
    pytest eval/tests/ -q                                  # scorer sanity
