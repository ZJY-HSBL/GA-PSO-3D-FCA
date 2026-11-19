"""Mass-conservative 3D CA/FCA kernels and synthetic diffusion reference fields.

The paper specifies the high-to-low exchange and the normalized outgoing
fractions, but does not disclose its actual fuzzy rule consequents, input
normalization, or formula mapping lambda_s to lambda_0. Those reproducibility
decisions are explicitly recorded in ASSUMPTIONS below.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import numpy as np

ASSUMPTIONS = {
    "fuzzy_inputs": "nonnegative neighbor difference divided by the current spatial peak; clipping to [0,1]",
    "membership": "three clipped triangular basis functions centered at 0, m_f and 1; width w_f controls shoulders",
    "rule_consequents": "R1=(0.9,1.0,1.1); R2=(0.6,1.0,1.6); R3=(1.4,1.0,0.6)",
    "directional_weight": "(Dx*ox^2+Dy*oy^2+Dz*oz^2)/distance^2 times distance^(-p)",
    "lambda_mapping": "clip(2*dt*lambda_s*(Dx+Dy+Dz)/h^2,0,0.95)",
    "fitness_weights": "1.0 relative-MSE + 0.1 absolute mass deviation + 0.02 relative smoothness",
    "ordinary_ca": "explicit conservative finite-volume six-neighbor Fickian exchange with given physical D",
    "source_location": "one full unit of mass in cell (n//2,n//2,n//2), including even grids",
}

@dataclass(frozen=True)
class Structure:
    neighborhood: int = 6
    rule: int = 0
    distance_power: int = 1

    def __post_init__(self):
        if self.neighborhood not in (6, 18, 26) or self.rule not in (0, 1, 2) or self.distance_power not in (1, 2):
            raise ValueError('Invalid structure')

    def key(self):
        return (self.neighborhood, self.rule, self.distance_power)


@dataclass(frozen=True)
class Params:
    dx: float = 0.0016
    dy: float = 0.0016
    dz: float = 0.0016
    scale: float = 0.92
    middle: float = 0.50
    width: float = 0.55

    @classmethod
    def from_array(cls, arr):
        return cls(*[float(x) for x in arr])

    def as_array(self):
        return np.array([self.dx, self.dy, self.dz, self.scale, self.middle, self.width], dtype=float)


BOUNDS = np.array([
    [0.0006, 0.0028], [0.0006, 0.0028], [0.0006, 0.0028],
    [0.70, 1.15], [0.35, 0.65], [0.30, 0.80],
], dtype=float)


def point_source(n: int) -> np.ndarray:
    arr = np.zeros((n, n, n), dtype=np.float64)
    arr[n // 2, n // 2, n // 2] = 1.0
    return arr


@lru_cache(maxsize=24)
def offset_slices(n: int, neighborhood: int):
    edges = []
    for ox, oy, oz in product((-1, 0, 1), repeat=3):
        r2 = ox*ox + oy*oy + oz*oz
        if r2 == 0 or (neighborhood == 6 and r2 != 1) or (neighborhood == 18 and r2 > 2):
            continue
        offset = (ox, oy, oz)
        src = tuple(slice(max(0, -v), n-max(0, v)) for v in offset)
        dst = tuple(slice(max(0, v), n-max(0, -v)) for v in offset)
        edges.append((offset, r2 ** 0.5, src, dst))
    return tuple(edges)


def reference_field(n: int, step: int, dt: float, diffusion: tuple[float, float, float]):
    """Grid-center sample of the infinite-space Green function, renormalized to M0=1."""
    if step < 1:
        return point_source(n)
    t = step * dt
    axes = (np.arange(n) - n // 2) / n
    q = sum((a.reshape(tuple(n if k == dim else 1 for k in range(3))) ** 2) /
            (4 * d * t) for dim, (a, d) in enumerate(zip((axes,)*3, diffusion)))
    # Omit the common prefactor: normalization cancels it exactly.
    g = np.exp(-q)
    return g / np.sum(g)


def triangular(z: np.ndarray, a: float, b: float, c: float):
    """Triangle with endpoints and peak; peaks can be outside [0,1] for shoulder sets."""
    return np.maximum(0., np.minimum((z-a)/max(b-a, 1e-12), (c-z)/max(c-b, 1e-12)))


def fuzzy_gain(z: np.ndarray, structure: Structure, params: Params) -> np.ndarray:
    m, w = params.middle, params.width
    low = triangular(z, -w, 0., m)
    mid = triangular(z, m-w, m, m+w)
    high = triangular(z, m, 1., 1.+w)
    # Numerical guards prevent a zero firing strength outside all three triangles.
    mu = np.stack((low, mid, high), axis=0)
    vals = np.array(((0.9, 1., 1.1), (0.6, 1., 1.6), (1.4, 1., 0.6))[structure.rule])
    return np.where(mu.sum(axis=0) > 1e-12,
                    np.sum(mu * vals[:, None, None, None], axis=0) /
                    np.maximum(mu.sum(axis=0), 1e-12), 1.0)


def ca_step(u: np.ndarray, diffusion: tuple[float, float, float], dt: float):
    """Conservative 6-neighbor explicit Fickian CA (zero normal flux at boundaries)."""
    n = u.shape[0]
    h2 = (1. / n) ** 2
    if 2*dt*sum(diffusion)/h2 > 1.0:
        raise ValueError('Explicit CA stability bound exceeded')
    out = u.copy()
    for ax, d in enumerate(diffusion):
        left = [slice(None)]*3
        right = [slice(None)]*3
        left[ax] = slice(None, -1)
        right[ax] = slice(1, None)
        l, r = tuple(left), tuple(right)
        flux = (dt*d/h2)*(u[l]-u[r])
        out[l] -= flux
        out[r] += flux
    return out


def fca_step(u: np.ndarray, structure: Structure, params: Params, dt: float):
    """Two-pass directed, high-to-low, per-source normalization; exactly conservative."""
    n = u.shape[0]
    edges = offset_slices(n, structure.neighborhood)
    peak = max(float(u.max()), 1e-15)
    d = np.array([params.dx, params.dy, params.dz])
    denom = np.zeros_like(u)
    # One normalization per source cell, over all *activated* outgoing directions.
    for vec, distance, src, dst in edges:
        v = np.array(vec, dtype=float)
        weight = float(np.dot(d, v*v)/(distance*distance) / distance**structure.distance_power)
        delta = np.maximum(u[src] - u[dst], 0.0)
        response = delta * fuzzy_gain(delta/peak, structure, params) * weight
        denom[src] += response
    transfer_fraction = min(0.95, max(0., 2.*dt*params.scale*sum(d)*n*n))
    result = u.copy()
    for vec, distance, src, dst in edges:
        v = np.array(vec, dtype=float)
        weight = float(np.dot(d, v*v)/(distance*distance) / distance**structure.distance_power)
        delta = np.maximum(u[src] - u[dst], 0.0)
        response = delta * fuzzy_gain(delta/peak, structure, params) * weight
        flow = transfer_fraction*u[src]*response/np.maximum(denom[src], 1e-30)
        result[src] -= flow
        result[dst] += flow
    return result


def simulate(n: int, steps: int, dt: float, method='FCA',
             structure: Structure | None = None, params: Params | None = None,
             true_diffusion=(.0016, .0016, .0016), snapshots=()):
    """Return snapshots by step (e.g., 10,20,30,40,50); immutable input state."""
    structure = structure or Structure()
    params = params or Params()
    u = point_source(n)
    saved = {0: u.copy()} if 0 in snapshots else {}
    for step in range(1, steps+1):
        if method == 'CA':
            u = ca_step(u, true_diffusion, dt)
        else:
            u = fca_step(u, structure, params, dt)
        if step in snapshots:
            saved[step] = u.copy()
    return saved


def metrics(pred: np.ndarray, reference: np.ndarray):
    err = pred-reference
    ss = np.sum(err**2)
    denominator = np.sum((reference-reference.mean())**2)
    return dict(rmse=float(np.sqrt(np.mean(err**2))), mae=float(np.mean(abs(err))),
                r2=float(1-ss/denominator) if denominator>0 else float('nan'),
                mass_error=float(abs(1.0-pred.sum())))


def fitness(structure: Structure, params: Params, n: int, dt: float,
            calibration: tuple[int,...], references: dict[int,np.ndarray]):
    saved = simulate(n, max(calibration), dt, 'FCA', structure, params, snapshots=calibration)
    rel_mse, rel_smooth = [], []
    for t in calibration:
        actual, target = saved[t], references[t]
        scale = max(float(np.mean(target**2)), 1e-22)
        rel_mse.append(float(np.mean((actual-target)**2)/scale))
        smooth = sum(np.mean(np.diff(actual, axis=ax)**2) for ax in range(3))/3.0
        rel_smooth.append(float(smooth/scale))
    return (float(np.mean(rel_mse)) + 0.1*abs(1-saved[max(calibration)].sum()) +
            0.02*float(np.mean(rel_smooth)))
