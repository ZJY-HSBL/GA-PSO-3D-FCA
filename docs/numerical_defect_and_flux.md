# First-stage numerical analysis: a near-equilibrium counterexample and conservative flux

**Status:** independently implemented research prototype. No novelty claim, no formal convergence theorem, no empirical claim of superiority over a finite-volume method. The original manuscript's numerical mapping and fuzzy rule consequents were not fully specified; the counterexample applies to its **stated per-source normalized transfer structure** and to our disclosed implementation of that structure.

## Why the original normalized transfer is problematic

The paper-style per-source transfer fraction can be written

\[
\lambda_{i\to j}=\lambda_0\frac{P_{i\to j}}{\sum_{k\in N_i}P_{i\to k}},\qquad
q_{i\to j}=u_i\lambda_{i\to j}.
\]

Consider a uniform state `u=1` on a zero-flux cubic grid, with one center cell changed to `1-epsilon`. Each of its six face neighbors has exactly **one** downhill neighbor. As long as its fuzzy response is positive, that face's per-source fraction is exactly `lambda_0`, independent of `epsilon`. The center gains approximately `6*lambda_0` even as `epsilon -> 0`, so the discrete maximum principle can be violated, despite total mass conservation. This establishes a lack of **near-equilibrium gradient consistency** for this rule. It does *not* by itself prove that the entire paper's reported simulations are invalid.

## Alternative: symmetric conservative face flux

On a cubic unit domain with `h=1/n`, diagonal nonnegative directional diffusivities, and each face `i--j`, define

\[
K_{ij}=\frac{2D_iD_j}{D_i+D_j},\quad
z_{ij}=\frac{|u_i-u_j|}{\max(u_i,u_j)}\text{ (0 when both vanish)},\quad
G_{ij}=1-\alpha+\alpha\frac{\Phi(z_{ij})}{\Phi(0)},
\]

where `0<=alpha<=1`, `Phi` is the fixed positive fuzzy response used by the original baseline, and `Phi(0)>0`. The single signed flux is

\[
F_{ij}=\Delta t_s\frac{K_{ij}}{h^2}G_{ij}(u_i-u_j),\qquad F_{ji}=-F_{ij}.
\]

For every active cell: `u_i(new)=u_i - sum_j F_ij`. Inactive cells and the external boundary have zero face conductance; material interfaces use a harmonic mean. Full-tensor cross-derivative diffusion is **not supported** by this six-face scheme.

**Proven at the level of one explicit substep:** Set `a_ij=dt_s*K_ij*G_ij/h²` and choose `dt_s` so `sum_j a_ij <= 0.95` at each cell. Then `u_i(new)=(1-sum_j a_ij)u_i + sum_j a_ij u_j` is a convex combination of old values. Therefore for nonnegative inputs the update is nonnegative and cannot leave the initial min/max interval. Since every undirected face is updated with equal and opposite amounts, total mass is conserved up to floating-point summation. `G_ij -> 1` as `u_i-u_j -> 0` at fixed nonzero local state, so flux vanishes with the gradient. `alpha=0` exactly recovers the ordinary six-neighbor explicit Fickian method where its CFL condition permits a single step. These statements do **not** establish global discretization convergence, full-tensor consistency, or superiority to established schemes.

## Executed diagnostic protocol

Run `python -m gapso_fca.research --n 25 --steps 20 --out results/conservative_flux`. Files generated: `defect.csv`, `validation.csv`, `summary.json`, `gradient_consistency.png`, `reference_rmse.png`, and `field_comparison.png`. All measurements are calculated; nothing is filled from the article's reported metrics.

**Observed defect (n=9, dt=0.02, original structure 6-neighbor/R1/r^-1):** for epsilon `1e-10`, the maximum change in the paper-style normalized implementation was `0.08584704`; the new conservative flux change was `1.5554e-12`. The old update also exceeded the initial maximum while the new one did not.

**Small-grid analytical-field check (n=25, 20 steps, dt=0.02):**

| Case | CA RMSE | Original implemented FCA RMSE | Conservative neutral RMSE | Conservative fuzzy RMSE |
|---|---:|---:|---:|---:|
| Isotropic | 4.2523e-4 | 3.0086e-4 | 4.2523e-4 | 2.3281e-4 |
| Anisotropic | 4.3176e-4 | 3.0519e-4 | 4.3176e-4 | 2.2550e-4 |

These are *exploratory*, single-protocol results under an analytically normalized infinite-domain Gaussian reference and an impulsive discrete initial condition. They are **not** the paper's published grid/budget, an independent fine-grid reference, or a complete method validation; the ordering may change with resolution, seeds, initial data, and boundary conditions.

## Required next-stage research (not yet completed)

1. Verify order of accuracy using manufactured solutions, mesh/time refinement, and a genuinely independent reference solver.
2. Implement and benchmark a strong heterogeneous FVM baseline, including interfaces and geometries.
3. Formalize the new fuzzy law and assess sensitivity/identifiability and generalization to unseen materials.
4. Integrate the new model into a clearly constrained optimization scheme, checking full enumeration versus GA over the original 18 rules.
5. Run repeated seeds, performance/memory profiling and uncertainty reporting before revising publication-level conclusions.

## Smooth-field numerical consistency experiment (2026-11-19)

A second *independent-of-point-source* experiment uses an exact homogeneous-Neumann eigenmode of the linear diffusion equation:

\[
u(x,y,z,t)=1+0.2\cos(\pi x)\cos(\pi y)\cos(\pi z)\exp(-3\pi^2Dt),\quad D=0.0016.
\]

The experiment uses cell centers, final time `T=0.2`, `n=12,16,24,32` and `dt` proportional to `h²` (up to integer step counts). The executed results are in [smooth_neumann.csv](../results/conservative_flux/smooth_neumann.csv), and the experiment code is [convergence.py](../src/gapso_fca/convergence.py).

- Neutral conservative flux (`strength=0`) observed order over the last refinement: **2.022**.
- Original first-stage fuzzy multiplier based on `z` (`response_power=1`) observed last-step order: **0.467**; this four-grid test does not show second-order behavior.
- Squared-relative-gradient response (`response_power=2`) observed last-step order: **2.024**, with a monotonically decreasing error on the tested grids.

The squared response is a *candidate consistency-preserving fuzzy law*, not an established novel discretization: for smooth positive fields `z=O(h)` and hence `Phi(z²)/Phi(0)-1=O(h²)` locally. Additional consistency and stability proofs are required for variable/vanishing states and nonsmooth interfaces. This small numerical grid study cannot establish universal second-order convergence.

Re-run: `python -m gapso_fca.convergence --out results/conservative_flux`.
