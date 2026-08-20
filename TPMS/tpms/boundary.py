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
cube edge and tangent to the face containing the arc -- exactly the paper's
stated constraint ("the derivatives are varied while being constrained to stay
on the cube faces", Sec. 2.2). Because Eq. 2's d1 = P'(0) and d2 = P'(1) are
both tangents *in the direction of travel*, the emergent derivative at the
start node points into the face interior while the incident derivative at the
end node points back out through that node's cube edge; see the sign flip in
`build_boundary_curves`. With this convention an arc between two mid-edge
nodes sharing a cube corner is an exact quarter circle at d = `D_CIRCULAR`,
reproducing the strongly-bulging fillet arcs of the paper's Fig. 6.
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


# Paper Table A.1 sets the 1/8 design space as a 2x2x2 mm cube, with node
# position offsets in [-0.45, 0.45] mm and derivative magnitudes in [1.0, 2.0]
# mm. Our 1/8 cube is the unit cube [0,1]^3, so those ranges scale by 1/2.
S_RANGE = (-0.225, 0.225)
D_RANGE = (0.5, 1.0)

# A cubic Hermite arc joining two points at distance r from a shared cube
# corner reproduces the quarter circle of radius r when the tangent magnitude
# is 3 * (4/3)*(sqrt(2)-1)*r = 4*(sqrt(2)-1)*r ~= 1.657*r (the standard cubic-
# Bezier circle approximation, converted from Bezier control-point offset to
# Hermite tangent by the factor 3). For default mid-edge nodes r = 0.5, giving
# ~0.828 -- the roundest arc this parameterization can make, and comfortably
# inside the paper's own D_RANGE above.
D_CIRCULAR = 4.0 * (np.sqrt(2.0) - 1.0) * 0.5


def random_params(topo: T.Topology, rng=None, s_range=S_RANGE, d_range=D_RANGE):
    rng = rng or np.random.default_rng()
    params = {}
    for e in topo.active_edges():
        params[e] = NodeParams(
            s=rng.uniform(*s_range),
            d_in=rng.uniform(*d_range),
            d_out=rng.uniform(*d_range),
        )
    return params


def default_params(topo: T.Topology, s=0.0, d=None):
    """Default geometry: nodes at cube-edge midpoints, tangent magnitudes set
    to the value that makes each arc an exact quarter circle."""
    if d is None:
        d = D_CIRCULAR
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
            # Hermite tangents are both "direction of travel" along the curve
            # (Eq. 2: d1 = P'(0), d2 = P'(1)). Leaving node `a` the curve heads
            # *into* the face interior; arriving at node `b` it is still heading
            # in its direction of travel, i.e. *out* through b's cube edge --
            # hence the sign flip on d2. Using +into-face for d2 as well makes
            # P'(1) point back the way the curve came, producing an S-hook at
            # every arrival node and flattening the arc (see module docstring).
            d1 = node_face_direction(a, face_idx) * pa.d_out
            d2 = -node_face_direction(b, face_idx) * pb.d_in
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
