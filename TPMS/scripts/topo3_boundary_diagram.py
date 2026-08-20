"""topo_3 boundary-curve diagram, reproducing the paper's Fig. 6 construction.

Pipeline (all from this repo's modules, which implement the paper directly):
  tpms/topology.py  -- enumerate the 20 unique 1/8-cube topologies (Sec. 2.2)
  tpms/spline.py    -- cubic Hermite curve, paper Eq. 2
  tpms/boundary.py  -- per-node position/derivative parameters (Sec. 2.2
                       "Boundary geometric variation"), constrained exactly as
                       the paper states: node position stays on its cube edge,
                       derivatives stay tangent to the cube face.

The construction, panel by panel:
  1. Topology graph: each node is a boundary/cube-edge crossing, each arc a
     face crossing, drawn as straight chords (paper Fig. 5 schematic style).
  2. The same chords (grey, dashed) with the Hermite tangent vectors at every
     node -- each perpendicular to that node's cube edge and tangent to the
     face carrying the arc. These are what bend each straight chord into a
     bulging arc; the resulting curve is drawn in blue over them.
  3-4. Geometric variations, from sampling node positions/derivatives in the
     paper's own Table A.1 ranges.

topo_3 is one of only three 1/8-cube topologies (with topo_1 and topo_6) that
the paper notes is triply periodic on its own (Sec. 2.2), and it is the
topology behind the FRD-like shell-lattice family (topo_3_rrr, Fig. 10d).

Usage:
    python scripts/topo3_boundary_diagram.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from tpms import boundary as B
from tpms import topology as T

TOPO_INDEX = 3  # 1-indexed, matches out/topologies.png and paper Fig. 5

CURVE_C = '#2f7fc1'
CHORD_C = '#b0b0b0'
NODE_C = '#c4453b'
VEC_C = '#2e9e4f'


def plot_cube(ax):
    lines = [[T.VERTICES[a], T.VERTICES[b]] for a, b in T.EDGES]
    ax.add_collection3d(Line3DCollection(lines, colors='black', linewidths=0.7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-60)


def draw_nodes(ax, topo, params):
    for e in topo.active_edges():
        p = B.node_position(e, params[e].s)
        ax.scatter(*p, color=NODE_C, s=30, depthshade=False, zorder=5)


def draw_chords(ax, topo, params, **kw):
    """Straight chords node->node: the topology graph before any curving."""
    segs = []
    for loop in B.find_loops(topo):
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            segs.append([B.node_position(a, params[a].s),
                         B.node_position(b, params[b].s)])
    ax.add_collection3d(Line3DCollection(segs, **kw))


def draw_curve(ax, topo, params, lw=2.6):
    for c in B.build_boundary_curves(topo, params, n_samples=96):
        cc = np.concatenate([c, c[:1]], axis=0)
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], color=CURVE_C, linewidth=lw, zorder=4)


def draw_tangents(ax, topo, params, scale=0.30):
    """The Hermite tangent vectors (Eq. 2 d1/d2) at every node, i.e. the
    vectors that bend each straight chord into its arc."""
    for loop in B.find_loops(topo):
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            face = T.shared_face(a, b)
            pa = B.node_position(a, params[a].s)
            pb = B.node_position(b, params[b].s)
            # emergent tangent at a (points into the face), incident tangent at
            # b (still along direction of travel, so it points back out through
            # b's cube edge) -- matching build_boundary_curves.
            d1 = B.node_face_direction(a, face) * params[a].d_out
            d2 = -B.node_face_direction(b, face) * params[b].d_in
            for p, d in ((pa, d1), (pb, -d2)):
                v = d * scale
                ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], color=VEC_C,
                          linewidth=1.5, arrow_length_ratio=0.3, zorder=6)


def main():
    reps, n_valid = T.enumerate_topologies(include_empty_face_state=False)
    assert n_valid == 256 and len(reps) == 20, (
        f"topology enumeration mismatch: {n_valid}/{len(reps)}, paper: 256/20")
    topo = reps[TOPO_INDEX - 1]
    n_nodes = len(topo.active_edges())
    print(f"topo_{TOPO_INDEX}: {n_nodes} nodes, {len(topo.arcs)} arcs, "
          f"{topo.n_loops()} loop(s); d_circular={B.D_CIRCULAR:.4f}")

    default_params = B.default_params(topo)
    rng = np.random.default_rng(3)

    fig = plt.figure(figsize=(17, 4.6))

    # (a) topology graph
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    plot_cube(ax)
    draw_chords(ax, topo, default_params, colors=CURVE_C, linewidths=2.2)
    draw_nodes(ax, topo, default_params)
    ax.set_title(f"(a) topo_{TOPO_INDEX} topology graph\n"
                 f"{n_nodes} nodes, {len(topo.arcs)} arcs, {topo.n_loops()} closed loop",
                 fontsize=9.5)

    # (b) construction: chords + tangent vectors + resulting curve
    ax = fig.add_subplot(1, 4, 2, projection='3d')
    plot_cube(ax)
    draw_chords(ax, topo, default_params, colors=CHORD_C, linewidths=1.2,
                linestyles='dashed')
    draw_tangents(ax, topo, default_params)
    draw_curve(ax, topo, default_params)
    draw_nodes(ax, topo, default_params)
    ax.set_title("(b) Hermite construction (Eq. 2)\n"
                 "tangent vectors bend each chord into an arc", fontsize=9.5)

    # (c),(d) geometric variations
    for i in range(2):
        ax = fig.add_subplot(1, 4, 3 + i, projection='3d')
        plot_cube(ax)
        params = B.random_params(topo, rng)
        draw_curve(ax, topo, params)
        draw_nodes(ax, topo, params)
        ax.set_title(f"({'cd'[i]}) geometric variation {i + 1}\n"
                     "node positions + derivatives varied", fontsize=9.5)

    fig.suptitle(
        "topo_3 boundary curve construction (paper Sec. 2.2 \"Boundary geometric variation\", Eq. 2, Fig. 6)\n"
        "Node positions constrained to their cube edge; derivatives constrained tangent to the cube face. "
        "Parameter ranges from Table A.1.",
        fontsize=11, y=1.04)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), '..', 'out',
                       'topo3_boundary_diagram.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
