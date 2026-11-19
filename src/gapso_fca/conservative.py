"""Conservative face-flux extension (research prototype, not the original paper).

Diagonal diffusion tensor only. Cell-centered cubic finite-volume grid with
zero-normal-flux outer boundaries and optional impermeable inactive cells.

For every face i--j, the same signed flux is subtracted from i and added to j:

    flux_ij = delta_t * D_harmonic(i,j) / h**2 * G_ij(u) * (u_i-u_j).

G_ij is a positive bounded fuzzy conductance multiplier, symmetric in the
endpoints. It is normalized so G_ij -> 1 as |u_i-u_j| -> 0. Setting
strength=0 recovers ordinary six-neighbor Fickian CA exactly (when its CFL
condition permits a single step). An adaptive explicit time substep ensures
the sum of outgoing conductance weights per cell stays below cfl_limit.

This is an independently specified alternative method; it is not claimed to
be present in, nor a verbatim reproduction of, the reference manuscript.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import Params, Structure, fuzzy_gain


@dataclass(frozen=True)
class FluxDiagnostics:
    substeps: int
    maximum_courant: float
    mass_error: float
    maximum_principle_violation: float


def _diffusion_array(diffusion: Sequence[float] | np.ndarray, shape: tuple[int, int, int]):
    d = np.asarray(diffusion, dtype=np.float64)
    if d.shape == (3,):
        d = np.broadcast_to(d[:, None, None, None], (3,) + shape)
    if d.shape != (3,) + shape or not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError('diffusion must be a nonnegative (Dx,Dy,Dz) vector or (3,n,n,n) array')
    return d


def _face_geometry(d: np.ndarray, mask: np.ndarray):
    """Precompute positive harmonic face conductivities and paired slices."""
    faces = []
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        l, r = tuple(left), tuple(right)
        a, b = d[axis][l], d[axis][r]
        k = np.divide(2 * a * b, a + b, out=np.zeros_like(a), where=(a+b)>0)
        k = np.where(mask[l] & mask[r], k, 0.0)
        faces.append((l, r, k))
    return faces


def conservative_fca_step(
    u: np.ndarray,
    diffusion: Sequence[float] | np.ndarray,
    dt: float,
    *,
    rule: int = 0,
    middle: float = .50,
    width: float = .55,
    strength: float = 1.0,
    mask: np.ndarray | None = None,
    cfl_limit: float = .95,
    max_substeps: int = 4096,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, FluxDiagnostics]:
    """Advance one conservative, positivity-preserving diffusion macro-step.

    Assumes equally sized cubic cells covering a unit cube; u is the
    per-cell concentration/state. Inactive cells are impermeable and must
    initially contain zero. Only diagonal directional diffusivities are
    represented; general full-tensor anisotropy requires a different stencil.

    For a face weight a_ij = dt_sub * k_ij * G_ij / h**2 >= 0,
    the update is u_i'=(1-sum_j a_ij)u_i+sum_j a_ij*u_j. Our adaptive
    restriction sum_j a_ij<=cfl_limit<1 preserves the discrete maximum
    principle, positivity, and pairwise mass conservation (to roundoff).
    """
    state = np.asarray(u, dtype=np.float64)
    if state.ndim != 3 or min(state.shape) < 2 or len(set(state.shape)) != 1:
        raise ValueError('u must be a cubic three-dimensional array, n >= 2')
    if not np.isfinite(dt) or dt <= 0 or not np.all(np.isfinite(state)) or np.any(state < 0):
        raise ValueError('dt must be positive and state finite and nonnegative')
    if not (0 <= strength <= 1) or not (0 < cfl_limit < 1) or max_substeps < 1:
        raise ValueError('strength/cfl_limit/max_substeps outside admissible range')
    if rule not in (0, 1, 2) or not (0 < middle < 1) or width <= 0:
        raise ValueError('invalid fuzzy configuration')
    active = np.ones_like(state, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if active.shape != state.shape or np.any(state[~active] != 0):
        raise ValueError('mask must match state; inactive cells must be empty')
    d = _diffusion_array(diffusion, state.shape)
    faces = _face_geometry(d, active)
    n = state.shape[0]
    inv_h2 = float(n*n)
    structure = Structure(neighborhood=6, rule=rule, distance_power=1)
    params = Params(middle=middle, width=width)
    zero_response = float(fuzzy_gain(np.zeros((1, 1, 1)), structure, params)[0, 0, 0])
    result = state.copy()
    start_min, start_max = float(state.min()), float(state.max())
    initial_mass = float(state.sum())
    remaining = float(dt)
    steps = 0
    worst_courant = 0.0
    while remaining > dt*1e-13:
        rates = np.zeros_like(result)
        edges = []
        for l, r, harmonic in faces:
            if strength == 0:
                g = 1.0
            else:
                left, right = result[l], result[r]
                relative = np.divide(np.abs(left-right), np.maximum(left, right),
                                     out=np.zeros_like(left), where=np.maximum(left, right)>0)
                g = (1.0-strength) + strength * (
                    fuzzy_gain(relative, structure, params) / zero_response)
            conductivity = harmonic * inv_h2 * g
            rates[l] += conductivity
            rates[r] += conductivity
            edges.append((l, r, conductivity))
        max_rate = float(rates.max())
        sub_dt = min(remaining, cfl_limit / max_rate) if max_rate else remaining
        worst_courant = max(worst_courant, sub_dt * max_rate)
        next_state = result.copy()
        for l, r, conductivity in edges:
            flow = sub_dt * conductivity * (result[l] - result[r])
            next_state[l] -= flow
            next_state[r] += flow
        result = next_state
        remaining -= sub_dt
        steps += 1
        if steps > max_substeps:
            raise RuntimeError('required substeps exceed max_substeps; reduce dt')
    violation = max(0.0, start_min-float(result.min()), float(result.max())-start_max)
    diag = FluxDiagnostics(steps, worst_courant, abs(initial_mass-float(result.sum())), violation)
    return (result, diag) if return_diagnostics else result


def conservative_simulate(
    initial: np.ndarray,
    diffusion: Sequence[float] | np.ndarray,
    dt: float,
    steps: int,
    *,
    snapshots: Sequence[int] = (),
    **kwargs,
) -> dict[int, np.ndarray]:
    """Simulate from an explicitly supplied initial field, returning snapshots."""
    if steps < 0 or any(t < 0 or t > steps for t in snapshots):
        raise ValueError('invalid requested step')
    state = np.asarray(initial, dtype=np.float64).copy()
    saved = {0: state.copy()} if 0 in snapshots else {}
    for t in range(1, steps + 1):
        state = conservative_fca_step(state, diffusion, dt, **kwargs)
        if t in snapshots:
            saved[t] = state.copy()
    return saved
