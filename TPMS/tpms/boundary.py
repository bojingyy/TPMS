"""Boundary curve construction from a topology + geometric (spline) parameters
(paper Sec. 2.2 "Boundary geometric variation", Eq. 2, Fig. 6, Fig. A.22) and
synthesis of the full triply-periodic unit cell from the 1/8-cube boundary via
per-axis reflection/translation symmetry (topo_id_x1x2x3 naming, Sec. 2.2).

Each active node (a cube edge crossed by the boundary curve) carries 3 scalar
parameters, matching the paper's parameter count (Fig. 9, Fig. A.22, Table
A.1: N position + 2N derivative parameters for N nodes):
    s      -- position offset along the cube edge from its midpoint
    d_in   -- incident derivative magnitude (curve arriving at this node)
    d_out  -- emergent derivative magnitude (curve leaving this node)
Derivative *directions* are fixed by convention to be perpendicular to the
cube edge, tangent to the face, pointing into the face interior -- the
standard convention for smoothly filleted corner curves and consistent with
the bulging patch shapes in Fig. 6/8 of the paper.
"""
import itertools
from dataclasses import dataclass

import numpy as np

from . import spline, topology as T


@dataclass
class NodeParams:
    s: float = 0.0
    d_in: float = 1.5
    d_out: float = 1.5


def node_position(edge_id, s):
    p = T.edge_midpoint(edge_id).copy()
    p[T.edge_axis(edge_id)] += s
    return p


def node_face_direction(edge_id, face_idx):
    """Unit vector, tangent to `face_idx`, perpendicular to the cube edge
    `edge_id`, pointing into the face interior."""
    axis_p = T.edge_axis(edge_id)
    face_axis, _face_val = T.FACES[face_idx]
    q = [a for a in range(3) if a not in (axis_p, face_axis)][0]
    fixed = T.edge_fixed_coords(edge_id)
    sign = 1.0 if fixed[q] == 0 else -1.0
    d = np.zeros(3)
    d[q] = sign
    return d


def find_loops(topo: T.Topology):
    """Decompose a topology's arcs into ordered cyclic node sequences."""
    adj = {e: [] for e in topo.active_edges()}
    for arc in topo.arcs:
        a, b = tuple(arc)
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    loops = []
    for start in adj:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        cur = start
        nxt = adj[cur][0]
        while True:
            loop.append(nxt)
            visited.add(nxt)
            neighbors = adj[nxt]
            new_next = neighbors[1] if neighbors[0] == cur else neighbors[0]
            cur, nxt = nxt, new_next
            if nxt == start:
                break
        loops.append(loop)
    return loops


def random_params(topo: T.Topology, rng=None, s_range=(-0.45, 0.45),
                   d_range=(0.2, 0.6)):
    rng = rng or np.random.default_rng()
    params = {}
    for e in topo.active_edges():
        params[e] = NodeParams(
            s=rng.uniform(*s_range),
            d_in=rng.uniform(*d_range),
            d_out=rng.uniform(*d_range),
        )
    return params


def default_params(topo: T.Topology, s=0.0, d=0.4):
    return {e: NodeParams(s=s, d_in=d, d_out=d) for e in topo.active_edges()}


def build_boundary_curves(topo: T.Topology, params, n_samples=64):
    """Return a list of (K, 3) arrays, one closed polyline per boundary
    loop, each lying in the 1/8-cube domain [0,1]^3."""
    loops = find_loops(topo)
    curves = []
    for loop in loops:
        k = len(loop)
        segs = []
        for i in range(k):
            a, b = loop[i], loop[(i + 1) % k]
            face_idx = T.shared_face(a, b)
            pa, pb = params[a], params[b]
            p1 = node_position(a, pa.s)
            p2 = node_position(b, pb.s)
            d1 = node_face_direction(a, face_idx) * pa.d_out
            d2 = node_face_direction(b, face_idx) * pb.d_in
            t = np.linspace(0, 1, n_samples, endpoint=False)
            segs.append(spline.hermite_eval(p1, d1, p2, d2, t))
        curves.append(np.concatenate(segs, axis=0))
    return curves


# ---------------------------------------------------------------------------
# Full unit-cell synthesis via per-axis reflection ('r') / translation ('t').
# ---------------------------------------------------------------------------

def octant_transform(axis_signs, sym_code):
    """axis_signs: tuple of 0/1 per axis (1 = mirrored/negative octant).
    sym_code: tuple of 'r'/'t' per axis. Maps points in [0,1]^3 to the
    corresponding octant of [-1,1]^3."""
    def transform(pts):
        pts = np.array(pts, dtype=float, copy=True)
        for axis in range(3):
            if axis_signs[axis]:
                if sym_code[axis] == 'r':
                    pts[..., axis] = -pts[..., axis]
                else:
                    pts[..., axis] = pts[..., axis] - 1.0
        return pts
    return transform


def synthesize_unit_cell(curves, sym_code):
    """Replicate 1/8-cube boundary curves (in [0,1]^3) into all 8 octants of
    [-1,1]^3 according to the per-axis reflection/translation code."""
    out = {}
    for axis_signs in itertools.product([0, 1], repeat=3):
        transform = octant_transform(axis_signs, sym_code)
        out[axis_signs] = [transform(c) for c in curves]
    return out
