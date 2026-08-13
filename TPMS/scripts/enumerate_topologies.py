"""Reproduce Fig. 5: enumerate and visualize the 20 unique 1/8-cube boundary
curve topologies.

Usage:
    python scripts/enumerate_topologies.py [--out out/topologies.png]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from tpms import topology as T


def edge_midpoint(edge_id):
    v1, v2 = T.EDGES[edge_id]
    p1 = np.array(T.VERTICES[v1], dtype=float)
    p2 = np.array(T.VERTICES[v2], dtype=float)
    return (p1 + p2) / 2


def plot_topology(ax, topo, title):
    # cube wireframe
    lines = []
    for v1, v2 in T.EDGES:
        p1, p2 = T.VERTICES[v1], T.VERTICES[v2]
        lines.append([p1, p2])
    ax.add_collection3d(Line3DCollection(lines, colors='black', linewidths=0.6))

    # boundary curve arcs, drawn as straight chords between cube-edge
    # midpoints (matches the schematic style of Fig. 5/6).
    arc_lines = []
    for arc in topo.arcs:
        a, b = tuple(arc)
        arc_lines.append([edge_midpoint(a), edge_midpoint(b)])
    ax.add_collection3d(Line3DCollection(arc_lines, colors='#3b82c4', linewidths=2.0))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    ax.view_init(elev=20, azim=-60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(
        os.path.dirname(__file__), '..', 'out', 'topologies.png'))
    args = parser.parse_args()

    reps, n_valid = T.enumerate_topologies(include_empty_face_state=False)
    print(f"Raw valid topologies after closed-curve degree filter: {n_valid} "
          f"(paper: 256)")
    print(f"Unique topologies after cube-symmetry dedup: {len(reps)} "
          f"(paper: 20)")

    n = len(reps)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    for i, topo in enumerate(reps):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        plot_topology(ax, topo, f"topo_{i+1}  ({topo.n_loops()} loop(s), "
                                 f"{len(topo.arcs)} arcs)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
