# olympiad-bench

v0 prototype benchmarking LLM agents on IPhO experimental-round problems inside simulated environments.

Current scope: one problem, **IPhO 2018 E2 — Viscoelasticity of a polymer thread**, used to validate three load-bearing assumptions (spec schema, physics engine, tool API) before committing to broader coverage.

- `BROAD_IPHO_PLAN.md` — long-term multi-problem, multi-backend vision.
- `tasks/ipho/2018_E2_viscoelasticity/` — the one task.
- `envs/physics/mechanics_ode.py` — reusable 2nd-order ODE engine.
- `eval/tasks/ipho_2018_E2.py` — Inspect AI task definition.
