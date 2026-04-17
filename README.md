# olympiad-bench

v0 prototype benchmarking LLM agents on IPhO experimental-round problems inside simulated environments.

Current scope: one problem, **IPhO 2018 E2 — Viscoelasticity of a polymer thread**, used to validate three load-bearing assumptions (spec schema, physics engine, tool API) before committing to broader coverage.

## Layout

- `BROAD_IPHO_PLAN.md` — long-term multi-problem, multi-backend vision.
- `tasks/ipho/2018_E2_viscoelasticity/` — problem.md, spec.yaml, env.py, reference_solution.py, tests/.
- `envs/physics/mechanics_ode.py` — reusable 2nd-order ODE + generalized-Kelvin engine.
- `envs/common/` — spec schema (Pydantic v2), noise layer, task loader.
- `eval/tasks/ipho_2018_E2.py` — Inspect AI task (tools + solver + scorer wiring).
- `eval/scorers.py` — rubric scorer that consumes `spec.yaml` + per-seed truth.
- `eval/run_baselines.py` — oracle baseline runner (reference solution as ceiling).
- `eval/run_model.sh` — thin wrapper around `inspect eval` with trace logging.

## Gate status

- **Gate A** — spec.yaml authored from official IPhO solution; schema validated.
- **Gate B** — engine, env, reference solution, 50-seed replay test all green (`pytest tasks/`).
- **Gate C** — Inspect AI task + scorer wired; scorer sanity-tested; oracle ceiling = 1.0 / 1.0.
- **Gate C.5** — truth-randomization mode added to defeat training-data memorization. Reference recovers ≥95% of randomized τ/E rows on 50-seed replay.
- **Gate D** — pending first real-model baseline run at scale.

---

## What's built

### 1. Task authoring (Gate A)

`spec.yaml` encodes the problem machine-readably: physics parameters with truth values and per-parameter tolerances, rubric rows with point weights, apparatus with noise characteristics. `problem.md` is the contestant-facing prose. The Pydantic v2 schema in `envs/common/spec_schema.py` validates both on load.

### 2. Simulation (Gate B)

The environment exposes the lab through 6 tools:

| Tool | Purpose |
|---|---|
| `measure_length(target)` | Ruler reading of ℓ₀ (unstretched) or ℓ (stretched), ±0.5 mm uniform |
| `read_mass()` | Scale reading of total weight P₀, gram-force |
| `stretch_thread()` | One-shot: hooks the fixture and starts the relaxation clock |
| `wait_seconds(duration)` | Advances simulated wall clock |
| `read_scale()` | Time-stamped (t, P(t)) reading once thread is stretched |
| `submit(answer)` | Ends the episode with the final parameter dict |

`envs/physics/mechanics_ode.py` provides the generalized-Kelvin forward model: F(t) = F₀ + Σₖ Fₖ·exp(-t/τₖ). `envs/common/noise.py` wraps each tool return in realistic noise (ruler uniform ±0.5 mm, scale Gaussian σ = 0.01 gf, stopwatch jitter σ = 0.05 s). Budget: 200 tool calls per episode.

### 3. Reference solution (Gate B)

`reference_solution.py` is a deterministic contestant that (a) measures ℓ₀/P₀, (b) stretches, (c) samples `read_scale` on a geometric schedule over ~45 min with triple-averaging, (d) fits the 3-exponential decay in four stages: long-time curve_fit → subtract → mid-time log-linear → subtract → short-time log-linear → joint 7-parameter nonlinear refinement with bounds. It recovers all τ/E within the rubric tolerances on 50 independent seeds. **This defines the oracle ceiling** — no LLM should beat it under the same budget.

### 4. Scoring (Gate C)

`eval/scorers.py` implements a rubric-grade scorer consumed by Inspect. For each rubric row it either (a) checks a submitted parameter against truth within its relative tolerance, or (b) checks a procedural condition against the tool call log (e.g. "≥ 5 timeseries reads", "σ reported with ℓ₀"). It also canonicalizes submitted keys via an alias table so stylistic variations (`τ_1`, `tau1`, `tau_1`, `tau1_s`) all map to the same canonical spec parameter.

### 5. Inspect AI pipeline (Gate C)

`eval/tasks/ipho_2018_E2.py` wires the env, tools, scorer, and per-seed `Sample` into a single Inspect `Task`. Each sample gets its own `Env` instance (initialized by an `init_env` solver) populated into `store()` alongside the call log and truth dict. `parallel_tool_calls=False` prevents race conditions on the simulated clock.

`eval/run_model.sh` wraps `inspect eval` with automatic trace-file rotation so a `tail -f logs/trace.jsonl` gives a live JSONL feed of every tool call during a run.

### 6. Truth-randomization mode (Gate C.5)

**Problem:** On fixed truth (the canonical IPhO 2018 values: τ₁ = 1546 s, τ₂ = 177 s, τ₃ = 50 s, etc.), Sonnet 4.6 scored 1.000 — suspiciously clean. Likely explanation: the official solution and associated numbers are in training data.

**Fix:** `Env(randomize=True)` deterministically perturbs each seed's truth inside a physically-sound envelope before the episode starts. Independent multiplicative factors are drawn per primary parameter (`F₀/F₁/F₂`, `τ₁/τ₂/τ₃`, `ε`, `ℓ₀`); derived parameters (β, E₀/E₁/E₂, ℓ) are recomputed; `F₃` is re-calibrated to anchor F(10 s). Rejection sampling enforces:

- **Mode separation**: τ₁/τ₂ ≥ 4.0, τ₂/τ₃ ≥ 2.5 (keeps the 3-exponential fit identifiable).
- **Strain envelope**: ε ∈ [0.05, 0.35] (stays in TPU's elastic regime).
- **Physical margin**: F(0) ≤ P₀ − 10 gf (scale never reads negative).
- **Positive amplitudes**: no pathological F₃ = 0 runs.

`randomize=False` is the default — the original problem is preserved exactly. `randomize_strength ∈ [0, 1]` linearly interpolates between no perturbation and full bounds.

The scorer reads per-seed truth from `store["truth"]` (injected by `init_env`), so every submission is graded against its own seed's actual values. Noise RNG and truth-draw RNG are independent, so changing randomization mode doesn't shift noise realizations.

### 7. Tests (62 total, all green)

- **Fixed-truth replay** (54): reference solution solves every rubric row on 50 seeds; σ coverage bands for directly measured quantities; a deliberately-wrong reference fails the tolerance check.
- **Randomized-truth replay** (5): truth values genuinely vary across seeds; `randomize=False` preserves canonical truth; `strength=0` collapses to identity; reference recovers ≥95% of each τ/E row against per-seed truth on 50 randomized seeds.
- **Scorer sanity** (3): reference ≥85%; zero-answer ≤10%; empty call-log fails procedural row.

---

## Baseline experiment: Sonnet 4.6 on randomized IPhO 2018 E2

**Command:**

```bash
eval/run_model.sh anthropic/claude-sonnet-4-6 5 true 1.0
```

**Per-seed results:**

| seed | score | notable misses |
|------|------:|---|
| seed-0 | 0.924 | E₂ = 1.40× truth (tolerance ±30%) |
| seed-1 | **1.000** | — |
| seed-2 | **1.000** | — |
| seed-3 | 0.803 | τ₂ = 2.02×, E₂ = 0.46×, τ₃ = 1.80× |
| seed-4 | 0.848 | τ₂ = 0.54×, E₂ = 2.27× |

**Aggregate:** mean = 0.915, stdev = 0.089, min/max = 0.803 / 1.000.

**Comparison to fixed-truth baseline:** on `randomize=false` with the canonical IPhO truth, Sonnet scored 1.000. Randomization drops the mean by ~8.5 points — a modest but nonzero memorization signal.

**Per-seed truth draws (showing genuine variation):**

| seed | τ₁ (s) | τ₂ (s) | τ₃ (s) | E₀ (Pa) | E₁ (Pa) | E₂ (Pa) | ε |
|--|--|--|--|--|--|--|--|
| 0 | 1341 | 186 | 52 | 1.34e7 | 5.6e5 | 4.6e5 | 0.159 |
| 1 | 1290 | 171 | 60 | 1.23e7 | 8.1e5 | 5.2e5 | 0.171 |
| 2 | 1562 | 206 | 44 | 1.34e7 | 6.8e5 | 5.9e5 | 0.162 |
| 3 | 1527 | 188 | 49 | 1.14e7 | 8.2e5 | 4.6e5 | 0.187 |
| 4 | 1393 | 215 | 47 | 1.37e7 | 6.9e5 | 6.5e5 | 0.144 |

(canonical truth: τ₁ = 1546, τ₂ = 177, τ₃ = 50, E₀ = 1.31e7, E₁ = 7.4e5, E₂ = 4.5e5, ε = 0.167)

### Analysis

**Where Sonnet succeeds.** τ₁ (slow mode) is recovered robustly across all seeds (0.88–0.98× truth). All procedural rows pass on every seed — the model reports uncertainties, captures enough timeseries samples, and maintains the expected structure. The easy mechanical parts are solid.

**Where it fails.** Every failure is concentrated in **τ₂/E₂ (intermediate mode)**. The 3-exponential fit is ill-conditioned in the middle: τ₁ and τ₃ each have a time window where they dominate the signal, but τ₂ is always blended with at least one of the others. Small fit errors in the slow mode propagate into the mid-mode amplitude, often by a factor of 2. The reference solution works around this with a joint nonlinear refinement stage; Sonnet produces what looks like sequential log-linear fits and gets stuck in local minima.

**Fairness check.** Per-seed truth varies meaningfully (τ₁ spans 1290–1562, τ₂ spans 171–215). The scorer correctly applies per-seed truth via `store["truth"]`. Tolerances come from `spec.yaml` (τ₂: ±25%, E₂: ±30%). No budget exhaustion, no missed procedural rows, no truncations. The test is grading the model's actual physics ability, not its memory.

---

## Future directions

### Near-term

- **Gate D (immediate):** run Opus 4.7 and GPT-5 / Gemini 2.5 Pro under the same randomized setup for a multi-model comparison. Hypothesis: gap between fixed and randomized will widen for smaller / older models (more reliance on memorized IPhO numbers).
- **Scale seeds to N = 20–50** for tighter confidence intervals on the mean.
- **Variance audit:** run the same seed twice with identical randomization and check that a *temperature 0* model lands within the rubric tolerances of itself — establishes a within-model noise floor for interpreting cross-model gaps.
- **Two-tier scoring** to mirror the actual IPhO rubric: a lenient `tolerance_full` band gives full points; a stricter `tolerance_half` band gives half points. Currently we only use the full band, which is coarser than the original rubric.

### Medium-term

- **More problems.** `BROAD_IPHO_PLAN.md` describes the vision. Each new problem needs Gate A (spec.yaml) + Gate B (env.py + reference + replay test). The envs/physics/ library will grow: optics (Snell, Fraunhofer), thermodynamics (Newton cooling, PV cycles), E&M (RC charging, Hall effect). The env/scorer/tool-calling skeleton transfers.
- **Restore deferred parts.** IPhO 2018 E2 has parts B/C/E that we deferred for v0 (laser diffraction to measure thread diameter; changing thread length; constant-stress elastic verification). Each adds a fresh tool and rubric rows.
- **Richer noise/fault models.** Real labs have instrument drift, systematic offsets, occasional spurious readings. Currently our noise is purely Gaussian/uniform and memoryless.
- **Scaffolded vs. raw-tool comparison.** Run the same models with and without access to Python (code_interpreter tool). Fitting three exponentials by hand is painful; with a scratchpad the mid-mode miss probably disappears.

### Long-term / research

- **Generalized memorization defeat.** Truth randomization only works because we have a physical envelope to stay inside. For problems with discrete answers (e.g. "which mirror is concave"), we need a different memorization-defeat strategy — scenario permutations, relabeled apparatus, etc.
- **Partial-credit reasoning.** Rubric currently awards points only for final-answer correctness. Contestants earn points for *method* even when the arithmetic goes wrong. Could score the model's chain-of-thought and tool-call strategy against a reference workflow.
- **Budget sensitivity curves.** Plot score vs. tool-call budget across 50/100/200/500 calls — shows whether models can *use* more compute, or if their strategy plateaus early.
- **Cross-problem transfer.** Does a model that aces thermodynamics also ace E&M? Correlating per-problem scores across models gives us a "physics ability" factor independent of memorization.

---

## Running

**Oracle reference ceiling (no API keys):**

```bash
python -m eval.run_baselines --oracle --seeds 10
```

**Real-model baseline with fixed truth:**

```bash
eval/run_model.sh anthropic/claude-sonnet-4-6 5                  # seeds=5, randomize=false
```

**Real-model baseline with randomized truth (memorization-defeat):**

```bash
eval/run_model.sh anthropic/claude-sonnet-4-6 5 true             # strength=1.0
eval/run_model.sh anthropic/claude-opus-4-7 5 true 0.5           # half-strength perturbation
```

**Live inspection during a run:**

```bash
tail -f logs/trace.jsonl                  # one JSON line per tool call
inspect view --log-dir logs/              # rich web UI, refreshes ~10s
```

## Testing

```bash
pytest tasks/ipho/2018_E2_viscoelasticity/tests/ -q   # 59 tests: replay (fixed + randomized)
pytest eval/tests/ -q                                  # 3 tests: scorer sanity
```
