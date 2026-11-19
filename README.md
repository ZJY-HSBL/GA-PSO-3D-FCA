# GA-PSO-3D-FCA

**Auditable Python reproduction of** *基于GA–PSO分层优化的三维模糊元胞自动机扩散模型研究* (*Three-dimensional fuzzy cellular automata diffusion model based on GA–PSO hierarchical optimization*).

The reference PDF describes a conservative 3-D fuzzy cellular automaton (FCA) for a point-source diffusion field. A genetic algorithm (GA) searches **18 discrete structures** (6/18/26 neighborhoods × 3 fuzzy response modes × inverse-distance powers 1/2); a particle swarm (PSO) independently calibrates six continuous parameters for each chosen structure. This repository implements the forward model, GA, PSO, hierarchical GA→PSO, baselines, paper protocol, metrics, plots, and conservation tests.

> **Reproducibility status:** This is an **independent implementation**, not the original authors' source code. Several numerical definitions necessary to match exact reported values are missing in the paper. We explicitly document our choices instead of presenting calculated outputs as exact numerical replication. The `paper` mode reproduces the stated **experiment design**, but the paper's published metrics cannot be guaranteed.

## Quick start

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -e '.[test]'
pytest -q
# Fast verification (10³ optimization, 16³ validation, one seed):
ga-pso-fca --mode smoke --out results/runs/smoke
# Paper grid resolution, but reduced search budget for a quicker numerical check:
ga-pso-fca --mode verification --out results/runs/verification
# Article's 30³→50³ protocol with 30×30 GA, 30×30 PSO and 20 independent runs:
ga-pso-fca --mode paper --out results/runs/paper
```

**Warning:** `paper` mode is computationally demanding: every GA candidate triggers a complete PSO search, and every particle evaluation runs a multi-step 3D simulation. Expect substantial runtime; use `smoke` first. Default outputs under `results/runs/` are excluded from Git to prevent committing multi-run bulk data.

## Paper protocol versus implemented quantities

| Setting | Value |
|---|---|
| Domain | unit cube, center-cell point source, normalized mass M0=1 |
| Boundary | zero flux / no transfer outside the domain |
| Reference | 3D constant-coefficient Gaussian Green's function, discretized at cell centers and renormalized |
| Cases | isotropic D=(0.0016, 0.0016, 0.0016); anisotropic D=(0.0022, 0.0015, 0.0009) |
| Time | Δt=0.02, 50 total steps |
| Calibration | 30³ grid, time steps 10,20,30 |
| Held-out evaluation | 50³ grid, time steps 40,50 |
| Sensitivity | 30³, 40³ and 50³ grid sizes |
| GA | population 30, 30 generations, crossover .8, mutation .05 |
| PSO | 30 particles, 30 iterations, inertia .7, c1=c2=1.5 |
| Repetition | 20 independent random seeds |
| Params | Dx,Dy,Dz∈[.0006,.0028], λs∈[.70,1.15], mf∈[.35,.65], wf∈[.30,.80] |

Five variants: `CA` (explicit conservative Fickian 6-neighbor baseline), `FCA` (fixed structure/params), `GA-FCA` (structure search with fixed params), `PSO-FCA` (parameter calibration with fixed structure), `GA-PSO-FCA` (inner PSO reinitialized per GA structure). GA caches evaluated structures **within each independent run**; the cache contains results, not shared fitted parameters across distinct structures.

## Numerical details that the reference PDF does **not** fully specify

The source provides general fuzzy mapping Φ and a triangular membership formula, but **not the explicit consequents or complete numerical rule table**, the local input normalization, the formula mapping λs/D/Δt/grid length to λ0, the weights/scales used in the composite fitness, baseline initial parameter assignments, or the exact GA tournament/elitism details. The manuscript also alternates between **nine triangular membership endpoints** and **two optimized membership parameters** (mf,wf), without a unique reconstruction. A bitwise or metric-exact reproduction is therefore underdetermined from the supplied PDF alone.

**Declared defaults (implemented in `src/gapso_fca/core.py`):**

- Normalized gradient: positive neighbor difference divided by the current field peak, clipped to [0,1]. The three triangular fuzzy sets have centers (0,mf,1) and width wf as described in code.
- Rule consequents used for weighted-average defuzzification: neutral `(0.9,1.0,1.1)`, high-gradient-enhancing `(0.6,1.0,1.6)`, and high-gradient-suppressing `(1.4,1.0,0.6)`.
- Direction weight: `(Dx ox² + Dy oy² + Dz oz²) / distance² × distance^-p`.
- Stable total outgoing fraction: `lambda0=clip(2 dt lambda_s (Dx+Dy+Dz) / h², 0, .95)`. Actual destination fractions use the **paper's per-source P/sum(P)** normalization, and update all inflows and outflows simultaneously.
- Dimensionless objective: `mean(relative MSE)+0.1×absolute mass imbalance+0.02×relative spatial roughness`. These weights are **implementation choices**, not extracted from the article.
- Basic CA: conservative, explicit six-face Fickian exchange using the condition's stated diffusion coefficients, unlike the FCA's fraction-normalized directed flows. This ensures a meaningful physically grounded baseline, but is not guaranteed to be the paper's CA code.

These choices may materially change the resulting accuracy and rank ordering. **Do not present the experimental plots as reproduced copies of the paper's Figures 2–8.** They are fresh graphs of our run. The reference's tabulated validation RMSE/R² are shown below only as **targets to compare against**, not our experimental data.

| Article (table/section 3.3) | CA RMSE | FCA RMSE | GA-FCA RMSE | PSO-FCA RMSE | GA-PSO-FCA RMSE |
|---|---:|---:|---:|---:|---:|
| Isotropic | 2.373e-5 | 4.186e-5 | 3.267e-5 | 3.955e-5 | 3.055e-5 |
| Anisotropic | 2.823e-5 | 5.068e-5 | 4.043e-5 | 4.895e-5 | 3.786e-5 |

The original article states that ordinary CA outperforms all four fuzzy models on these simple constant-coefficient fields. An implementation claiming the hierarchical method always dominates ordinary CA would **misstate the source**.

## Actually executed smoke experiment

The example results below were **computed by this code**, with the *reduced* 10³→16³ protocol (not the official 30³→50³ study), seed=0, and very small GA/PSO budgets. The values are illustrative **integration-test measurements only**, not scientific comparators to the source's numbers.

| Condition | Model | Validation RMSE | R² |
|---|---|---:|---:|
| Isotropic | CA | 0.00285187 | 0.95107 |
| Isotropic | FCA | 0.00261637 | 0.95878 |
| Isotropic | GA-FCA | 0.00262060 | 0.95872 |
| Isotropic | PSO-FCA | 0.000840064 | 0.99518 |
| Isotropic | GA-PSO-FCA | 0.00114754 | 0.99121 |
| Anisotropic | CA | 0.00245869 | 0.96269 |
| Anisotropic | FCA | 0.00262410 | 0.95749 |
| Anisotropic | GA-FCA | 0.00266597 | 0.95598 |
| Anisotropic | PSO-FCA | 0.00112979 | 0.99204 |
| Anisotropic | GA-PSO-FCA | 0.00133189 | 0.98906 |

Run `ga-pso-fca --mode smoke` to regenerate the precise CSV, JSON, and all PNG images from scratch. Figures from the executed demo (actual model output):

**Isotropic diffusion: slices and model comparison**

![Isotropic diffusion XY/XZ comparisons](assets/isotropic_slices.png)
![Isotropic metrics](assets/isotropic_metrics.png)

**Anisotropic diffusion: 3D state and convergence**

![Anisotropic diffusion 3D state](assets/anisotropic_3d.png)
![Anisotropic convergence](assets/anisotropic_convergence.png)

## Repository layout

```text
src/gapso_fca/core.py         Conservative CA/FCA updates, Gaussian reference, fitness, metrics
src/gapso_fca/optimize.py     GA discrete search, independent PSO calibration, caching
src/gapso_fca/experiment.py   CLI, data collection, grid sensitivity, figures, CSV/JSON
experiments/paper_protocol.json Machine-readable article parameter settings
tests/test_core.py           Numerical consistency and reproducibility tests
assets/                      Figures from executed smoke run
results/smoke_summary.json    Raw summary from the smoke run
```

The main outputs include `individual_results.csv`, `summary.json`, `run_config.json`, `grid_sensitivity.csv`, grid-sensitivity PNG, selections JSON, and isotropic/anisotropic slices, 3D states, metrics and convergence PNGs. All plotted measurements originate from the solver; the paper's expected metrics are never injected into evaluation or plotted as measured data.

## Tests and provenance

`pytest -q` currently passes **13 tests**, including conservation and non-negativity in all 6/18/26 neighborhood structures and all 3 rule modes, CA symmetry, anisotropic Gaussian behavior, identity metrics, and deterministic optimization. Code and plots are independently created for this reproduction; the supplied PDF is not redistributed in the repository.

**Future work for exact replication:** obtain the original authors' rule tables, λ0 mapping, raw scripts, random seeds, fitted parameters, and all fitness term scales. Re-run paper mode and compare the archived numerical outputs against Figures 2–8 and Tables 3–5 with differences disclosed.
