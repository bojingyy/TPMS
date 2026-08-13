"""Reproduce Fig. 6-style visualizations: geometric variations of a
boundary curve for a fixed topology, and Fig. 9-style full unit-cell
synthesis via reflection/translation.

Usage:
    python scripts/visualize_boundary.py --topo 1 --out out/boundary_topo1.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from tpms import boundary as B
from tpms import topology as T


def plot_cube(ax):
    lines = [[T.VERTICES[v1], T.VERTICES[v2]] for v1, v2 in T.EDGES]
    ax.add_collection3d(Line3DCollection(lines, colors='black', linewidths=0.6))


def plot_curves(ax, curves, color='#3b82c4'):
    for c in curves:
        c_closed = np.concatenate([c, c[:1]], axis=0)
        ax.plot(c_closed[:, 0], c_closed[:, 1], c_closed[:, 2], color=color, linewidth=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topo', type=int, default=1, help='1-indexed topology id from enumerate_topologies')
    parser.add_argument('--out', default=None)
    parser.add_argument('--n_variations', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    reps, _ = T.enumerate_topologies(include_empty_face_state=False)
    topo = reps[args.topo - 1]
    print(f"topo_{args.topo}: {len(topo.active_edges())} nodes, "
          f"{len(topo.arcs)} arcs, {topo.n_loops()} loop(s)")

    rng = np.random.default_rng(args.seed)
    fig = plt.figure(figsize=(4 * args.n_variations, 4))
    for i in range(args.n_variations):
        ax = fig.add_subplot(1, args.n_variations, i + 1, projection='3d')
        plot_cube(ax)
        if i == 0:
            params = B.default_params(topo)
            title = "default"
        else:
            params = B.random_params(topo, rng)
            title = f"variation {i}"
        curves = B.build_boundary_curves(topo, params, n_samples=64)
        plot_curves(ax, curves)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(f"topo_{args.topo}: {title}", fontsize=9)
        ax.view_init(elev=20, azim=-60)

    fig.tight_layout()
    out = args.out or os.path.join(os.path.dirname(__file__), '..', 'out',
                                    f'boundary_topo{args.topo}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
