"""Smooth Neumann eigenmode grid convergence benchmark, independent of point sources.

The *linear* diffusion PDE has exact solution u=1+A*cos(pi*x)*cos(pi*y)*cos(pi*z)*exp(-3*pi^2*D*t)
with homogeneous Neumann outer boundaries. The quadratic-gradient fuzzy response
is a research candidate to retain the same smooth-field asymptotic order.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from .conservative import conservative_fca_step


def run(out: str | Path, grids=(12, 16, 24, 32), diffusion=.0016, t_final=.2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    methods = {'neutral': (0.0, 1.0), 'linear_gradient_fuzzy': (1.0, 1.0),
               'quadratic_gradient_fuzzy': (1.0, 2.0)}
    rows = []
    for name, (strength, power) in methods.items():
        previous_n = None
        previous_error = None
        for n in grids:
            n = int(n)
            x = (np.arange(n)+.5)/n
            one = np.cos(np.pi*x)
            shape = one[:,None,None]*one[None,:,None]*one[None,None,:]
            u = 1.0 + .2*shape
            steps = math.ceil(t_final*n*n/5.0)
            dt = t_final/steps
            for _ in range(steps):
                u = conservative_fca_step(u, (diffusion,)*3, dt,
                    strength=strength, response_power=power)
            exact = 1.0 + .2*np.exp(-3*np.pi*np.pi*diffusion*t_final)*shape
            error = float(np.sqrt(np.mean((u-exact)**2)))
            order = (math.log(previous_error/error)/math.log(n/previous_n)
                     if previous_n is not None else None)
            rows.append({'model': name, 'n': n, 'steps': steps,
                         'dt': dt, 'rmse': error, 'observed_order': order,
                         'mass_error': float(abs(u.sum()-n**3))})
            previous_n, previous_error = n, error
    with (out/'smooth_neumann.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in methods:
        group = [x for x in rows if x['model'] == name]
        ax.loglog([x['n'] for x in group], [x['rmse'] for x in group], 'o-', label=name)
    ax.set(xlabel='Cells per dimension (n)', ylabel='RMSE vs exact Neumann eigenmode',
           title='Smooth-field grid convergence (dt proportional to h²)')
    ax.grid(True, which='both', alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'smooth_neumann.png', dpi=170)
    plt.close(fig)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/conservative_flux')
    args=parser.parse_args()
    for row in run(args.out):
        print(row['model'],row['n'],f"RMSE={row['rmse']:.8g}",f"order={row['observed_order']}")


if __name__=='__main__':
    main()
