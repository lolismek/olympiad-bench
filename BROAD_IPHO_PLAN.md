# IPhO-Sim — Simulated-Environment Benchmark for IPhO Experimental Round

## Context

**Goal.** Build a benchmark where LLM agents attempt the **IPhO (International Physics Olympiad) experimental round** inside faithful simulated environments, with the same information/apparatus/budget a real contestant has. v1 covers IPhO only; this plan deliberately narrows from earlier multi-olympiad scope.

**Why IPhO first.** The experimental round is the canonical olympiad practical: multi-hour, apparatus-based, requires designing measurements, handling noise, propagating uncertainty, and submitting a structured data-plus-final-answer package. It's under-benchmarked (OlympiadBench/HiPhO/PHYBench are *theory* only), public past problems exist 2015–2024, and the physics domains involved (mechanics, optics, circuits, thermal, modern) map cleanly to mature Python simulators.

**Scope of this plan.**
1. A rigorous specification of what a real contestant has during the 5-hour experimental round — this becomes the contract our environment must match.
2. A repeatable **pipeline** to turn any past IPhO experimental problem into an agent-evaluable simulated environment with minimal manual work.
3. A concrete first-task implementation strategy.

---

## Part 1 — IPhO Experimental Round: the contract to match

Sources: [IPhO Statutes](https://www.ipho-new.org/statutes-syllabus/), [IPhO 2021 Lithuania Statutes](https://www.ipho2021.lt/en/statutes/), [IPhO 2024 Iran](https://www.ipho2024.ir/), [Casio IPhO 2023](https://www.casio.com/intl/scientific-calculators/ipho2023/), [Kevin Zhou handouts](https://knzhou.github.io/handouts/Expt.pdf), [IPhO problem archive](https://ipho.olimpicos.net/), [Physics with Stefan archive](https://physicswithstefan.com/ipho-papers/).

### Round format

| Parameter | Value | Source |
|---|---|---|
| Duration | **5 hours** (single session) | IPhO statutes |
| Problems | 1–2 experimental problems | statutes |
| Points | 20 / 50 total (theory = 30) | statutes |
| Rest | ≥1 full day between theory and experimental | statutes |
| Language | English master; delegation leaders translate overnight | statutes |
| Calculator | Non-graphical, ≤3-line display; host-provided (Casio fx-82CW in 2023, fx-82ES Plus 2nd in 2024); memory cleared | 2023/2024 host instructions |
| Formula sheet | **None mandated** — host may optionally provide | statutes |
| Supervision | Invigilators; no external communication | statutes |

### What sits on the contestant's desk (canonical)

1. **Problem booklet** — English + translated versions.
2. **Pre-printed answer sheets** — tables with columns for raw measurements, units, uncertainty; dedicated boxes for final numerical answers (value ± uncertainty, explicit units); space for intermediate work.
3. **Graph paper** — gridded, usually linear; log paper when the problem needs it (pre-provided).
4. **Apparatus kit** — problem-specific (see inventory below).
5. **Approved calculator** — as above.
6. **Stationery** — ruler, pencil, eraser, protractor (inferred from problem designs, not universally documented).
7. **Safety equipment** when the apparatus warrants (goggles for lasers, etc.).
8. **NO** formula sheet, NO textbooks, NO internet.

### Answer format (verified across years)

- Measurement tables: raw data, repeated trials (usually **N ≥ 5**), with column headers and units.
- Computed-quantity rows: intermediate derived values.
- Graphs: on provided paper, with **error bars**, axis labels, units, linear-fit lines where applicable.
- **Final answer box**: value ± uncertainty in correct units, significant figures matched to uncertainty.
- Brief written justification: method, assumptions, dominant error sources. Equations/sketches preferred over prose.

### Uncertainty / error-analysis expectations

- Standard deviation as primary uncertainty for repeated trials (σ with 1–2 sig figs; measured value rounded to match).
- Standard error-propagation formulas (quadrature for independent sources).
- Identification of dominant systematic vs. random errors.
- Marking schemes allocate points *separately* for data, graphs, uncertainty reporting, and final answer correctness — so our scoring must too.

### Apparatus inventory, 2015–2024

| Year | Host | Experiment | Domain | Key apparatus | Sim fit |
|---|---|---|---|---|---|
| 2024 | Iran | E1 Heat conduction in copper rod | Thermal | heaters, thermocouples, ice bath, stopwatch, ruler | 1D heat equation (finite-diff) |
| 2024 | Iran | E2 Diffraction from phase steps | Optics | laser, phase-step sample, screen, ruler | POPPY / wave optics |
| 2023 | Japan | E1 Mass measurement (spring oscillator) | Mechanics | spring, masses, motion sensor | ODE solver (SHM) |
| 2023 | Japan | E2 Birefringence thickness | Optics | quartz plate, polarizer, analyzer, light source | Analytical + POPPY |
| 2022 | CH | E1 Planetary/atmospheric (virtual) | Geophysics | pure simulator | already digital |
| 2022 | CH | E2 Cylindrical diode I-V | Semiconductor | LED/diode, PSU, V/A meters | Shockley-eq solver |
| 2021 | LT | E1 Non-ideal capacitors | EM | capacitors, multimeter, temp bath | lumped-element + RC |
| 2021 | LT | E2 LED temperature effects | Modern | LED, PSU, thermometer, photodetector | I-V + T lookup |
| 2019 | IL | E1 Optics (diffraction grating + disk reflections) | Optics | laser, grating, disk, screen | ray + diffraction |
| 2019 | IL | E2 Thermal & electrical properties of metals | Thermal/EM | samples, multimeter, heater | Wiedemann-Franz + conduction |
| 2018 | PT | E1 Paper transistor | Modern | paper, carbon paint, electrodes, multimeter | I-V with field-effect model |
| 2018 | PT | E2 Viscoelasticity of polymer thread | Mechanics | thread, masses, ruler, timer | Kelvin-Voigt ODE |
| 2017 | ID | E1 Optics of salt solution | Optics | solution, prism/cuvette, light source | Snell's law + concentration |
| 2017 | ID | E2 Earthquake/volcano sensing | Geophysics | accelerometer, shaking table | accel-response ODE |
| 2016 | CH | E1 Electrical conductivity in 2D | EM | thin-film sample, 4-point probe | 2D resistivity model |
| 2016 | CH | E2 Jumping beads | Mechanics | beads, string, vibration gen | forced oscillation |
| 2015 | IN | E1/E2 (titles unverified) | — | — | — |

**Implication for sim stack**: ~70% of the last decade is reproducible with `scipy` ODE solvers + `numpy` + `POPPY` (optics) + `PySpice` (circuits) + a small custom 1D heat solver. No new simulator research required.

### The three PDFs to process per problem

For every year we ingest, we need three documents:
1. **Problem PDF** (statement + apparatus list + measurement requests)
2. **Sample solution PDF** (reference computations, "correct" numeric values)
3. **Marking scheme** (point allocation — often embedded in the solution PDF)

All three are publicly archived on ipho.olimpicos.net, physicswithstefan.com, and host-country sites, but are binary-compressed PDFs — extraction requires local tooling (`pdfplumber` for born-digital, `pytesseract` or `rapidocr` for scans). This is a pipeline step, not a research blocker.

---

## Part 2 — The environment design: what the agent gets

The environment mirrors the real-desk contract as closely as possible.

### Agent input (what it sees)

```
problem.md            # statement + figures (markdown + embedded PNGs)
apparatus_sheet.yaml  # list of instruments with nominal precision
answer_template.yaml  # schema the agent must fill (tables, final-answer boxes)
```

### Agent tool API (what it can call)

```python
# Measurement — all measurements go through this interface so noise is uniform
measure(instrument_id: str, params: dict, repeats: int = 1) -> MeasurementResult
# returns {"value": float | list, "unit": str, "nominal_precision": float}

# Apparatus state (for problems where settings change: heater power, voltage, etc.)
set_apparatus(instrument_id: str, setting: dict) -> None

# Observation (image-returning, for optics/microscopy-like problems)
observe(apparatus_id: str, params: dict) -> Image

# Calculator — equivalent to Casio fx-82 (restricted numerical eval, no symbolic)
calc(expression: str) -> float

# Scratchpad (persisted within a task; mimics contestant's margin notes)
note(text: str) -> None

# Final submission (terminal)
submit(answer: dict) -> None  # validated against answer_template.yaml
```

**Budget**: fixed number of `measure` calls per task, chosen to exceed what a well-prepared contestant would use (e.g., ~200 for a typical problem). Exceeding budget ends the task. This proxies the 5-hour wall clock.

### What's explicitly forbidden (matches real round)

- No formula sheet / textbook / web access (enforced by sandbox, no tools beyond those listed).
- No symbolic math tool — the Casio fx-82 only does arithmetic, trig, log/exp; we expose the same via `calc`.
- No code execution beyond `calc` (a contestant doesn't get a Python REPL).
- No ability to re-read prior measurements except via the agent's own memory/notes (contestants keep their own tables).

This last point is important: **the contract is "answer based on what you measured and noted"**. If the agent wants to preserve state, it must write it to `note`.

### Noise model per instrument (defaults, overridable per problem)

| Instrument | Default noise | Notes |
|---|---|---|
| Ruler / caliper | Rectangular ±half-smallest-division | e.g., ruler ±0.5 mm |
| Stopwatch (human) | σ = 0.1–0.2 s | reaction-time dominated |
| Stopwatch (electronic) | σ = 0.01 s | crystal-limited |
| Digital multimeter | max(0.5% reading, 1 count) | per spec sheet |
| Thermocouple | σ = 0.5–1 K | depends on type |
| Oscilloscope | time ±1% full scale, voltage ±3% | generic |
| CCD / photodetector | photon-shot + read noise (Gaussian) | `photutils`-style |
| Analog meter | σ = ½ smallest division | rectangular or triangular |

Per-problem overrides: whenever the problem explicitly states a tolerance ("ruler with 0.5 mm graduation"), that overrides the default.

---

## Part 3 — The pipeline: past problem → working environment

Nine stages. Stages 1–3 are automated + human-validated; 4–5 are implementation; 6–9 are validation and release.

### Stage 1 — Ingestion
- Script: `scripts/ingest.py --year 2023 --problem E1 --source ipho_olimpicos`
- Downloads problem PDF, solution PDF, marking scheme PDF.
- Extracts text: `pdfplumber` for born-digital (2015+), `pytesseract` fallback for scans.
- Extracts figures as PNGs.
- Stores everything under `tasks/ipho/2023_E1/raw/`.

### Stage 2 — LLM-assisted structured extraction
- Input: raw text + figures.
- Output: `tasks/ipho/2023_E1/spec.yaml` filled to this schema:

```yaml
meta:
  year: 2023
  host: Japan
  problem_id: E1
  domain: [mechanics]
  title: Mass measurement with spring oscillator

apparatus:
  - id: spring
    spec: "spring constant k ≈ 20 N/m (to be determined)"
    nominal_precision: n/a
  - id: motion_sensor
    spec: "position sampling, 50 Hz"
    nominal_precision: {position: 0.5 mm, time: 0.02 s}
  - id: masses
    spec: "set of calibrated masses 50–500 g"
    nominal_precision: {mass: 0.1 g}

parts:
  - part_id: "1.1"
    request: "Measure oscillation period T for 5 masses."
    expected_data: {table: {cols: [m_g, T_s, sigma_T_s], rows: 5}}
    points: 4
  - part_id: "1.2"
    request: "Plot T^2 vs m; extract k from slope."
    expected_data: {plot: {x: m_g, y: T_sq_s2, fit: linear}, answer: {k_N_per_m: [value, sigma]}}
    points: 6
  # ...

physics_model:
  governing_equation: "T = 2*pi*sqrt((m + m0)/k)"
  parameters:
    - name: k
      truth_value: 19.47    # from official solution
      tolerance_full: 0.05  # 5% from marking scheme
      tolerance_half:  0.10
    - name: m0   # effective spring mass
      truth_value: 0.023
      tolerance_full: 0.2

scoring:
  total_points: 20
  rubric:
    - criterion: raw_data_table_complete
      points: 3
    - criterion: graph_has_error_bars
      points: 2
    - criterion: k_within_full_tolerance
      points: 6
    - criterion: uncertainty_on_k_within_2x_truth
      points: 3
    - criterion: ...
```

- Human editor (the benchmark author) reviews and corrects within 10–20 minutes per task. This is the only significant manual step per task.

### Stage 3 — Physics-model validation
- Script: `scripts/validate_model.py --task 2023_E1`
- Loads `physics_model.governing_equation`, plugs in official parameter values, checks that the model reproduces the solution's predicted measurements within rounding.
- If mismatch, the spec is wrong — flag for human review before proceeding.

### Stage 4 — Simulator backend selection and implementation
- Domain tag routes to one of:
  - `envs/physics/mechanics_ode.py` — scipy `solve_ivp`, used for springs, pendulums, viscoelastic models.
  - `envs/physics/mechanics_contact.py` — PyBullet, used when contact/collision matters.
  - `envs/physics/heat_1d.py` — finite-difference 1D heat equation.
  - `envs/physics/circuits.py` — PySpice wrapper.
  - `envs/physics/optics_wave.py` — POPPY for diffraction/interference.
  - `envs/physics/optics_ray.py` — small custom ray tracer for lenses/prisms.
  - `envs/physics/semiconductor.py` — Shockley equation + thermal dependencies.
- Per-task `env.py` inherits the backend, declares apparatus → simulator mapping, exposes the standard tool API (Part 2).

### Stage 5 — Noise layer
- `envs/common/noise.py` provides `apply_noise(value, instrument, rng)` using the defaults + per-task overrides from `spec.yaml`.
- Every `measure()` call routes through this layer. Seed is fixed per evaluation run; parameters (truth values) are per-run-perturbable for contamination protection.

### Stage 6 — Reference solution replay
- Script: `scripts/replay_reference.py --task 2023_E1`
- Programmatically executes the canonical solution (measurements the official solution prescribes) against the env, computes answers, verifies within marking-scheme tolerance. This is a required passing test before the task is released.

### Stage 7 — Contamination protection
- Parameter perturbation: each `spec.yaml` parameter marked `perturbable: true` is drawn from a range (e.g., `k ∈ [15, 25]`) per evaluation run.
- Ground-truth regenerated on-the-fly from the physics model.
- Eval reports both `as-published` and `perturbed` scores — divergence is a contamination signal.

### Stage 8 — Scoring harness
- `eval/scorers.py` implements four independent scorers aligned with IPhO marking:
  1. **Correctness**: final numeric answers within marking-scheme tolerance bands.
  2. **Uncertainty calibration**: reported σ within a configurable factor (default 2×) of the true propagated σ.
  3. **Data quality**: N ≥ required trials, unit correctness, table completeness.
  4. **Graph / process**: LLM-as-judge on rubric items that are hard to check structurally; sampled for human audit.
- Scores aggregate to a 20-point total matching IPhO.

### Stage 9 — Integration & release
- Inspect AI `Task` definition in `eval/tasks/ipho_<year>_<id>.py`.
- Smoke test: `inspect eval --task tasks/ipho/2023_E1 --model gpt-4o` returns non-error score.
- Human-medalist anchor: 1–2 former IPhO competitors attempt each task under the same constraints; target >70% for difficulty calibration.

### Per-task effort estimate

| Stage | Time |
|---|---|
| 1 Ingestion | 5 min (automated) |
| 2 Extraction + human review | 30 min |
| 3 Model validation | 10 min |
| 4 Env implementation (new backend) | 2–8 h first time, <1 h on reuse |
| 5 Noise layer | reuse |
| 6 Reference replay | 30 min |
| 7 Perturbation setup | 30 min |
| 8 Scoring wiring | 20 min |
| 9 Inspect integration | 20 min |
| **Per task (steady state, reusable backend)** | **~3 h** |
| **First task on new backend** | **1–2 days** |

So ~10 tasks per week after backends are built, roughly a month of real work to get to 20 tasks spanning 5–6 backends.

---

## Part 4 — Concrete first step

**Target: IPhO 2018 E2 — Viscoelasticity of polymer thread.**

Why this one:
- Pure mechanics, no special apparatus beyond masses, thread, ruler, timer.
- Physics is a Kelvin-Voigt ODE (standard scipy).
- Born-digital PDF on physicswithstefan.com.
- Noise model is simple (ruler + stopwatch).
- Ground truth is a numeric parameter (relaxation time τ, stiffness).
- Scopes the full pipeline end-to-end in ~1 week.

**Week 1 deliverables**:
1. Repo scaffolding (layout below).
2. Ingestion script handling this one PDF.
3. `spec.yaml` filled and human-verified.
4. `envs/physics/mechanics_ode.py` backend implemented.
5. Task-specific `env.py` wiring the backend to the problem's apparatus.
6. Noise layer for ruler + stopwatch.
7. Inspect AI task definition.
8. Reference-solution replay passes.
9. Three frontier-model baseline scores (GPT-4o, Claude Opus 4.7, Gemini 2.5 Pro).

Once that loop works end-to-end, the next 5 tasks (2023 E1 spring, 2021 E2 LED, 2024 E1 heat, 2019 E1 diffraction, 2022 E2 diode) can be stamped out faster — each introduces one new backend or reuses existing ones.

---

## Proposed repo layout

```
olympiad-bench/
  RESOURCES.md                    # (existing)
  README.md
  pyproject.toml
  envs/
    common/
      noise.py                    # instrument noise models
      budget.py                   # tool-call accounting
      calculator.py               # Casio fx-82-equivalent
      answer_schema.py            # pydantic models for submissions
    physics/
      mechanics_ode.py
      mechanics_contact.py
      heat_1d.py
      circuits.py
      optics_wave.py
      optics_ray.py
      semiconductor.py
  tasks/
    ipho/
      2018_E2_viscoelasticity/
        raw/ (ingested PDFs, figures)
        problem.md
        spec.yaml
        env.py
        reference_solution.py     # canonical solution for replay test
        tests/
  eval/
    scorers.py
    judge_prompts.py
    tasks/                        # Inspect AI Task definitions per problem
  scripts/
    ingest.py
    extract_spec.py               # LLM-assisted
    validate_model.py
    replay_reference.py
    run_baselines.py
  paper/                          # drafts, figures
```

---

## Critical files referenced

- `/Users/alexjerpelea/olympiad-bench/RESOURCES.md` — olympiad directory
- External: [IPhO statutes](https://www.ipho-new.org/statutes-syllabus/), [ipho.olimpicos.net archive](https://ipho.olimpicos.net/), [physicswithstefan.com archive](https://physicswithstefan.com/ipho-papers/), [Kevin Zhou experimental handouts](https://knzhou.github.io/handouts/Expt.pdf)
- Tooling: Inspect AI, scipy, numpy, pdfplumber, POPPY, PySpice, `uncertainties`, pydantic

---

## Risks & open questions

1. **PDF extraction for older problems** — pre-2015 problems often scanned, not born-digital. Mitigation: v1 only targets 2015+; OCR later.
2. **Official marking-scheme access** — some years publish only sample solutions, not detailed marking schemes. Mitigation: derive tolerances from the solution's stated answers + ±5%/±10% bands mirroring typical IPhO conventions; flag derived vs. official in metadata.
3. **"Hands-on" problems that don't simulate well** — e.g., problems requiring tactile alignment, surface-tension judgment, complex real-world optics. Mitigation: skip these (we have ~15 simulatable problems across 2015–2024, which is enough for v1).
4. **Contamination** — IPhO problems and solutions are in training data. Mitigation: parameter perturbation (Stage 7) + report delta between published and perturbed scores.
5. **Agent having physics library access** — a real contestant has their memorized physics knowledge. We must NOT give the agent scipy/numpy for solving the problem — only our exposed tools. Mitigation: Inspect AI sandboxing with strict tool allowlist.
6. **Calculator fidelity** — Python eval is more powerful than Casio fx-82. Mitigation: restrict `calc` to a parsed AST with only arithmetic + math-module functions, no control flow or symbolic ops.
7. **Human-medalist recruitment** — needed for difficulty calibration. Mitigation: Columbia physics club / IPhO alumni networks; 2–3 people × 3 tasks each is sufficient anchor.

---

## Verification (how we know it works)

- `pytest tasks/ipho/2018_E2_viscoelasticity/tests/` — unit tests for env + noise + scoring.
- `python scripts/replay_reference.py --task 2018_E2_viscoelasticity` — canonical solution achieves full marks.
- `python scripts/run_baselines.py --task 2018_E2_viscoelasticity --models gpt-4o,claude-opus-4-7,gemini-2.5-pro` — 3 frontier models get non-trivial, differentiable scores.
- Human IPhO medalist on the same task under the same interface scores >70% (difficulty anchor).
- Parameter-perturbation run: same model on `as-published` vs `perturbed` params — score drop quantified.

---

## Why this shape is right

- **Faithful contract**: agent gets exactly what a contestant gets — apparatus, tool-call budget, calculator, no formula sheet, no external knowledge. Comparable to human performance in a defensible way.
- **Pipeline over one-offs**: the first task is slow; subsequent tasks are ~3 hours each because backends are reusable across problems sharing physics domains.
- **Uncertainty-as-first-class**: scoring the reported σ — not just the final value — is the novel axis, matches how IPhO markers actually grade, and isn't in any existing benchmark.
- **Contamination-resistant by construction**: every parameter is perturbable because the physics model is a symbolic function, not a static dataset.
- **Extensible**: when v1 ships, APhO (same format, disjoint problem set) is ~2 weeks of additional work since the backends are shared.
