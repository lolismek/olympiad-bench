# Viscoelasticity of a polymer thread (IPhO 2018, Experiment 2)

**v0 prototype scope note.** This simulated task covers Parts A (stress-relaxation
measurements) and D (data analysis) from the original 10-point problem, for a total
of **7.6 points**. Part B (laser-diffraction measurement of thread diameter), Part C
(changing to a shorter thread), and Part E (elastic-modulus verification under
constant stress) are deferred: the thread diameter `d` is provided as a given
constant, and only one thread is used.

---

## Introduction

When a solid is subject to an external force it deforms. Stress `σ = F/S` (force per
cross-sectional area) and strain `ε = (ℓ − ℓ₀)/ℓ₀` (relative length change) capture
the deformation. For purely elastic behaviour `σ = E·ε` with `E` the Young modulus.

For real polymers, stress and strain obey a richer relationship because molecular
rearrangements produce an additional *viscous* component. A common phenomenological
model is the **standard linear solid** (SLS), in which a purely elastic spring `E₀`
sits in parallel with a Kelvin–Voigt element (spring `E₁` in series with dashpot
`η₁`). The total stress is

    σ = E₀·ε + τ₁·(E₀ + E₁)·dε/dt − τ₁·dσ/dt,       τ₁ = η₁/E₁.

The model extends to multiple Kelvin–Voigt elements in parallel. **Under constant
strain** (`dε/dt = 0`) the solution is a sum of decaying exponentials:

    σ(t) = ε · (E₀ + Σ_k  E_k · exp(−t/τ_k)).       (★)

Equivalently, if the thread carries a time-dependent tension `F(t)` while hanging
from a fixed-length support with the mass sitting on the scale (so the strain is
frozen), `F(t)` has the same exponential-sum form.

---

## Apparatus (simulated subset)

| # | Item | Simulated / given |
|---|---|---|
| 1 | Upper support + stretching system | implicit — `stretch()` tool |
| 2 | Mass-set (hollow cylinder + holding screw) | implicit — one fixed mass |
| 3 | Long TPU thread | simulated — measurements via tools |
| 6 | Digital scale (readability 0.01 gf, noise ≈0.01 gf) | `read_scale()` tool |
| 8 | Stopwatch | implicit — scale reads are time-stamped |
| 9 / 10 | Ruler / measuring tape (±0.5 mm resolution) | `measure_length(target)` tool |
| — | Thread nominal diameter `d` | **given as constant** (Part B deferred) |

The set-up: the mass-set rests on the digital scale. The thread is attached to the
mass-set and to a holding screw. Before measurement starts the thread is
*unstretched*. At `t = 0` the upper end of the thread is placed on the support,
stretching it vertically with constant strain while the mass continues to rest on
the scale. The scale then reads `P(t) = P₀ − F(t)/g` in gram-force units, where
`F(t)` is the tension pulling the mass up.

---

## Constants supplied

- `g = 9.80 m/s²`
- thread nominal diameter `d = 0.480 mm` (provided — Part B deferred)
- 1 gf = 9.80 × 10⁻³ N

---

## Questions

### Part A — Stress-relaxation measurements (1.9 pts)

- **A.1 (0.3 pt).** Measure the unstretched thread length `ℓ₀` (between screw
  heads, adding 5 mm for each screw). Report `ℓ₀ ± σ_ℓ₀`.
- **A.2 (0.3 pt).** Measure the mass-set weight `P₀` in gf. Report `P₀ ± σ_P₀`.
- **A.3 (1.0 pt).** Stretch the thread and simultaneously start the clock; record
  scale readings `P(t)` for about **45 minutes**.
- **A.4 (0.3 pt).** Measure the stretched thread length `ℓ` after the run. Report
  `ℓ ± σ_ℓ`.

### Part D — Data analysis (5.7 pts)

- **D.1 (0.3 pt).** Compute `F(t) = P₀ − P(t)` in gf for all data.
- **D.2 (0.4 pt).** Describe `F(t)` qualitatively (monotonic decay with long tail).
- **D.3 (0.3 pt).** Compute the strain `ε = (ℓ − ℓ₀)/ℓ₀` and its uncertainty.
- **D.4 (0.3 pt).** Compute `β` such that `σ/ε = β · F` (F in gf, σ in SI).
  `β = g / (ε · S)` with `S = π(d/2)²`.
- **D.5 (0.4 pt).** Sketch what `F(t)` would look like for a purely elastic thread
  (a constant).
- **D.6 (0.5 pt).** Compute `dF/dt` numerically at long times (`t > 1000 s`).
- **D.7 (0.3 pt).** Write `dF/dt` for a single viscoelastic process: `dF/dt =
  −(F₁/τ₁)·exp(−t/τ₁)`.
- **D.8 (1.0 pt).** Fit `ln(−dF/dt) = ln(F₁/τ₁) − t/τ₁` at long times → extract
  `τ₁` and `E₁` in SI.
- **D.9 (0.3 pt).** Compute `E₀` from the long-time baseline `F₀`.
- **D.10 (0.3 pt).** Subtract the elastic baseline and the longest exponential:
  `y(t) = F(t) − F₀ − F₁·exp(−t/τ₁)`.
- **D.11 (1.0 pt).** Fit the second exponential to `y(t)` → `τ₂`, `E₂`.
- **D.12 (0.3 pt).** Identify time window `[t_i, t_f]` where the third component
  dominates.
- **D.13 (0.3 pt).** Estimate `τ₃` from that window.

Final deliverables the scorer evaluates: `E₀`, `E₁`, `τ₁`, `E₂`, `τ₂`, `τ₃` (SI
units). Uncertainties are only required for `ℓ₀`, `P₀`, `ℓ`, `ε`.

---

## Submission format

Call the `submit` tool **once** with a single dictionary argument. The scorer
looks for the exact keys below (both a value and — where noted — a 1-σ
uncertainty, as a two-element list `[value, sigma]`):

| Key | Units | Form | Notes |
|---|---|---|---|
| `l0_m` | metres | `[value, sigma]` | A.1 unstretched length |
| `P0_gf` | gram-force | `[value, sigma]` | A.2 total weight |
| `l_m` | metres | `[value, sigma]` | A.4 stretched length |
| `epsilon` | dimensionless | `[value, sigma]` | D.3 strain |
| `beta_gf_inv_SI` | Pa / gf | `[value]` | D.4 stress-over-force coefficient |
| `tau1_s` | seconds | `[value]` | D.8 slowest relaxation time |
| `E1_SI` | N/m² (Pa) | `[value]` | D.8 modulus of slowest Kelvin mode |
| `E0_SI` | N/m² (Pa) | `[value]` | D.9 elastic baseline modulus |
| `tau2_s` | seconds | `[value]` | D.11 intermediate relaxation time |
| `E2_SI` | N/m² (Pa) | `[value]` | D.11 modulus of intermediate mode |
| `tau3_s` | seconds | `[value]` | D.13 fastest relaxation time |

Index convention: `τ₁ > τ₂ > τ₃` (the subscript is ordered by *decreasing*
relaxation time, matching the official IPhO solution). Please submit all SI
moduli in N/m² (Pa), **not** kPa or MPa.

Example (values illustrative, not correct):

```python
submit(answer={
    "l0_m": [0.437, 2e-4],
    "P0_gf": [81.11, 0.03],
    "l_m": [0.510, 2e-4],
    "epsilon": [0.167, 0.001],
    "beta_gf_inv_SI": [3.24e5],
    "tau1_s": [1550.0],
    "E1_SI": [7.4e5],
    "E0_SI": [1.31e7],
    "tau2_s": [180.0],
    "E2_SI": [4.5e5],
    "tau3_s": [50.0],
})
```
