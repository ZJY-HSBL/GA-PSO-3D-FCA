"""Reproducible first-stage numerical defect and conservative-flux study.

Run: python -m gapso_fca.research --out results/conservative_flux

All quantitative results are computed from the numerical solvers, never
copied from the paper. This study is diagnostic, NOT a claim of novelty or
of superiority over well-established finite-volume discretizations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from .core import Params, Structure, ca_step, fca_step, metrics, point_source, reference_field
from .conservative import conservative_fca_step


def run(out: str | Path, n: int = 25, steps: int = 20, dt: float = .02):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    epsilons = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    defect_rows = []
    for eps in epsilons:
        u = np.ones((9, 9, 9), dtype=float)
        u[4, 4, 4] -= eps
        old = fca_step(u, Structure(), Params(), dt)
        new = conservative_fca_step(u, (.0016,)*3, dt)
        defect_rows.append({
            'epsilon': eps,
            'original_change_linf': float(np.max(abs(old-u))),
            'conservative_change_linf': float(np.max(abs(new-u))),
            'original_maximum_principle_violation': max(0., float(old.max()-u.max()), float(u.min()-old.min())),
            'conservative_maximum_principle_violation': max(0., float(new.max()-u.max()), float(u.min()-new.min())),
            'original_mass_error': float(abs(old.sum()-u.sum())),
            'conservative_mass_error': float(abs(new.sum()-u.sum())),
        })
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(epsilons, [r['original_change_linf'] for r in defect_rows], 'o-', label='Paper-style normalized FCA')
    ax.loglog(epsilons, [r['conservative_change_linf'] for r in defect_rows], 's-', label='Proposed edge-flux FCA')
    ax.set(xlabel='Single-cell concentration difference (epsilon)', ylabel='Maximum one-step state change',
           title='Near-equilibrium response: vanishing vs non-vanishing flux')
    ax.grid(True, which='both', alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'gradient_consistency.png', dpi=170)
    plt.close(fig)

    cases = {
        'isotropic': (.0016, .0016, .0016),
        'anisotropic': (.0022, .0015, .0009),
    }
    validation = []
    plot_slices = {}
    for case, d in cases.items():
        states = {key: point_source(n) for key in ('CA', 'original_FCA', 'flux_neutral', 'flux_fuzzy')}
        for _ in range(steps):
            states['CA'] = ca_step(states['CA'], d, dt)
            states['original_FCA'] = fca_step(states['original_FCA'], Structure(), Params(*d), dt)
            states['flux_neutral'] = conservative_fca_step(states['flux_neutral'], d, dt, strength=0)
            states['flux_fuzzy'] = conservative_fca_step(states['flux_fuzzy'], d, dt, strength=1)
        ref = reference_field(n, steps, dt, d)
        plot_slices[case] = {**states, 'reference': ref}
        for name, arr in states.items():
            validation.append({'case': case, 'model': name, **metrics(arr, ref),
                               'minimum': float(arr.min()), 'maximum': float(arr.max())})
    with (output/'validation.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(validation[0]))
        writer.writeheader()
        writer.writerows(validation)
    with (output/'defect.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(defect_rows[0]))
        writer.writeheader()
        writer.writerows(defect_rows)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    ordering = ('reference', 'CA', 'original_FCA', 'flux_neutral', 'flux_fuzzy')
    for i, (case, fields) in enumerate(plot_slices.items()):
        vmax = float(fields['reference'].max())
        for j, model in enumerate(ordering):
            axes[i,j].imshow(fields[model][:, :, n//2], origin='lower', vmin=0, vmax=vmax, cmap='viridis')
            axes[i,j].set_title(f'{case}: {model}', fontsize=9)
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
    fig.suptitle(f'3D point-source diffusion XY sections, n={n}, t={steps*dt:g}')
    fig.tight_layout()
    fig.savefig(output/'field_comparison.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for i, case in enumerate(cases):
        rows = [v for v in validation if v['case'] == case]
        axes[i].bar(range(len(rows)), [v['rmse'] for v in rows])
        axes[i].set_xticks(range(len(rows)), [r['model'] for r in rows], rotation=38, ha='right', fontsize=8)
        axes[i].set_title(case)
        axes[i].set_ylabel('RMSE vs normalized analytical point-source field')
    fig.tight_layout()
    fig.savefig(output/'reference_rmse.png', dpi=160)
    plt.close(fig)

    result = {
        'status': 'diagnostic_first_stage_only',
        'source': 'independently executed numerical solvers; no manual target metrics',
        'configuration': {'grid': n, 'steps': steps, 'dt': dt, 'fuzzy_structure': [6, 0, 1],
                          'fuzzy_params': {'middle': .5, 'width': .55, 'strength': 1.0}},
        'defect': defect_rows,
        'validation': validation,
        'interpretation': 'The proposed method recovers CA at strength=0, and has vanishing flux for vanishing gradients. This does not establish improved accuracy over CA, FDM or FVM.',
    }
    (output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/conservative_flux')
    parser.add_argument('--n', type=int, default=25)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--dt', type=float, default=.02)
    args = parser.parse_args()
    result = run(args.out, args.n, args.steps, args.dt)
    for row in result['validation']:
        print(f"{row['case']:12s} {row['model']:14s} RMSE={row['rmse']:.8g} R2={row['r2']:.5f}")
    print(f"Wrote computed results and three plots to {args.out}")


if __name__ == '__main__':
    main()
