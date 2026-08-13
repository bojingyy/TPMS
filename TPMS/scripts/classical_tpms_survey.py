"""Reproduce the flavor of Figs. 11-16: for each classical TPMS type
(P, D, G, IWP, FRD, N), sweep shell thickness to cover a range of relative
densities, homogenize each, and plot the resulting E-nu space and SA/V
ratios -- the same kind of comparison the paper uses to benchmark its
TPMS-like family against, restricted here to the classical surfaces
themselves (our neural TPMS-like family is covered separately by
scripts/build_shell_lattice.py + scripts/run_homogenization.py, since each
sample there needs a full neural-surface training run).

Usage:
    python scripts/classical_tpms_survey.py --out out/classical_survey.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tpms import classical_tpms as C
from tpms import homogenization as H
from tpms import metrics as M


def survey_one(name, thicknesses, resolution, period):
    rows = []
    coords = np.linspace(0, period, resolution, endpoint=False)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    phi = C.SURFACES[name](X, Y, Z)
    for t in thicknesses:
        density = (np.abs(phi) < t).astype(float)
        rd = density.mean()
        if rd < 0.03 or rd > 0.8 or not H.is_connected(density):
            continue
        try:
            CH = H.homogenize(density, E0=1.0, nu0=0.3, cell_size=period, maxiter=3000)
        except RuntimeError:
            continue
        props = H.elastic_properties(CH)
        rows.append(dict(name=name, thickness=t, rd=rd, **props))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=30)
    parser.add_argument('--n_thickness', type=int, default=10)
    parser.add_argument('--out', default=os.path.join(
        os.path.dirname(__file__), '..', 'out', 'classical_survey.png'))
    args = parser.parse_args()

    period = 2 * np.pi
    thicknesses = np.linspace(0.08, 0.6, args.n_thickness)

    all_rows = []
    for name in tqdm(C.SURFACES, desc='surfaces'):
        all_rows += survey_one(name, thicknesses, args.resolution, period)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'P': '#e69138', 'D': '#6aa84f', 'G': '#3d85c6',
              'IWP': '#c27ba0', 'FRD': '#f1c232', 'N': '#8e7cc3'}
    for name in C.SURFACES:
        rows = [r for r in all_rows if r['name'] == name]
        if not rows:
            continue
        E = [r['E'] for r in rows]
        nu = [r['nu'] for r in rows]
        rd = [r['rd'] for r in rows]
        axes[0].plot(nu, E, 'o-', color=colors[name], label=name, markersize=4)
        axes[1].plot(rd, E, 'o-', color=colors[name], label=name, markersize=4)

    axes[0].set_xlabel("Poisson's ratio"); axes[0].set_ylabel("Young's modulus (E0=1)")
    axes[0].set_title('E-nu space (classical TPMS)'); axes[0].legend(fontsize=8)
    axes[1].set_xlabel('relative density'); axes[1].set_ylabel("Young's modulus (E0=1)")
    axes[1].set_title('E vs relative density'); axes[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    print("\nSample at target relative density ~0.3 (paper Fig. 12/14 comparison point):")
    for name in C.SURFACES:
        rows = [r for r in all_rows if r['name'] == name]
        if not rows:
            continue
        closest = min(rows, key=lambda r: abs(r['rd'] - 0.3))
        print(f"  {name:4s} rd={closest['rd']:.3f}  E={closest['E']:.4f}  "
              f"nu={closest['nu']:.4f}  Z={closest['Z']:.3f}")


if __name__ == '__main__':
    main()
