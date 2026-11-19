# Prior-art screen for the manuscript revision (2026-11-19)

This is a preliminary literature screen, **not** a systematic novelty review. It is meant to prevent overclaiming the first-stage conservative-flux implementation. The strongest new fact currently established by code is a reproducible numerical defect in the paper-style per-source normalization, not a novel finite-volume formulation.

## Closest existing numerical methods

1. J. Droniou, "Finite volume schemes for diffusion equations: introduction to and review of modern methods," *Mathematical Models and Methods in Applied Sciences* 24(8), 1575–1619 (2014), DOI [10.1142/S0218202514400041](https://doi.org/10.1142/S0218202514400041), free [preprint](https://arxiv.org/abs/1407.1567). Reviews two-point and multi-point flux methods and the established notions of coercivity, convergence, and minimum/maximum principles. **Implication:** symmetric conservative face flux, harmonic interfaces and maximum-principle tests cannot be presented as novel on their own.

2. "Cell-centered nonlinear finite-volume methods for the heterogeneous anisotropic diffusion problem," *Journal of Computational Physics* (2017), [publisher link](https://www.sciencedirect.com/science/article/pii/S0021999116305964). Describes nonlinear finite-volume approaches designed to maintain positivity or discrete maximum principles under heterogeneous anisotropic diffusion. **Implication:** our diagonal six-face solver must be compared against published conservative and monotone FVM methods, not only a simplified CA.

3. "An explicit numerical scheme to efficiently simulate molecular diffusion in environments with dynamically changing barriers" (2015), [DOI 10.1080/13873954.2015.1033823](https://doi.org/10.1080/13873954.2015.1033823). Uses a cellular-automata interpretation of molecular diffusion with excluded-volume/dynamic-barrier geometry. **Implication:** adding obstacles or dynamic permeability to a CA is also not, by itself, a distinct contribution.

## Current contribution statement: research hypothesis, not established claim

- The source article's per-source normalization can produce nonvanishing transfer when an isolated concentration gradient tends to zero. The accompanying experiment in this repo directly demonstrates this in our disclosed implementation.
- The proposed alternative uses a symmetric face flux with a response power parameter. In one smooth Neumann eigenmode test, the quadratic-gradient response retains approximately second-order observed convergence, while the linear-gradient response does not show such a trend over the tested grids.
- **Not yet demonstrated:** novelty relative to nonlinear FVM, full material-interface convergence, superior performance in genuinely heterogeneous or non-Fickian physical models, or robustness under full-tensor anisotropy.

## Minimum conditions for a future paper

1. Search for flux-consistent fuzzy/gradient-weighted nonlinear FVM and CA algorithms, and write down actual formula-level distinctions.
2. Prove or counterexample-test stability, positivity, conservation, and consistency for each supported class of diffusion coefficients, including degeneracy.
3. Compare with accurate FVM/FDM and independent reference solvers on preregistered smooth/heterogeneous experiments.
4. Demonstrate an application-specific uncertainty or nonlinear mechanism requiring the fuzzy response, rather than claiming generic finite-volume improvements.
5. Report all baselines and negative findings. Do not infer innovation purely from using GA, PSO, or an enlarged rule library.
