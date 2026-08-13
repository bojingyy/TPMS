"""1/8-cube boundary topology enumeration (paper Sec. 2.2, Figs. 3-5).

A boundary curve on the 1/8 cube is represented as a graph: each of the 12
cube edges is a potential *node* (a point where the boundary curve crosses
that cube edge), and each of the 6 cube faces contributes *arcs* connecting
pairs of its 4 bounding edges (the curve segment crossing that face).

Per face there are 8 combinatorially distinct, non-self-crossing ways to
connect its 4 bounding edges (Fig. 3):
  - 4 ways connecting one *adjacent* pair of edges with a single arc,
  - 2 ways connecting the *opposite* pair of edges with a single arc,
  - 2 ways connecting *all four* edges with two non-crossing arcs.
A face may also touch no boundary curve at all ("empty"); we model this as a
9th pattern so faces away from the curve enumerate correctly, then discard
the fully-empty topology (no boundary at all) as degenerate.

Enumerating all combinations of per-face patterns over the 6 faces and
keeping only those where every cube edge is used by 0 or 2 of its adjacent
faces (so the curve has no loose ends -> is automatically a union of closed
loops) reproduces the topology search of Sec. 2.2. Two topologies related by
one of the cube's 48 symmetries (24 rotations x 2 for reflection) are
considered the same topology; deduplicating yields the unique topology list
(paper reports 20; see `enumerate_topologies` docstring for how our count
compares).
"""
import itertools
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Cube combinatorics: 8 vertices, 12 edges, 6 faces.
# ---------------------------------------------------------------------------

VERTICES = list(itertools.product([0, 1], repeat=3))
VERTEX_ID = {v: i for i, v in enumerate(VERTICES)}


def _build_edges():
    edges = []
    seen = set()
    for v in VERTICES:
        for axis in range(3):
            w = list(v)
            w[axis] = 1 - w[axis]
            w = tuple(w)
            e = tuple(sorted((VERTEX_ID[v], VERTEX_ID[w])))
            if e not in seen:
                seen.add(e)
                edges.append(e)
    return edges


EDGES = _build_edges()  # 12 edges, each (vertex_id_a, vertex_id_b), a < b
EDGE_ID = {e: i for i, e in enumerate(EDGES)}
N_EDGES = len(EDGES)
assert N_EDGES == 12

FACES = [(axis, value) for axis in range(3) for value in (0, 1)]  # 6 faces
FACE_NAMES = ['-X', '+X', '-Y', '+Y', '-Z', '+Z']


def _face_edges_cyclic(axis, value):
    """4 global edge ids bounding this face, in cyclic (boundary) order."""
    other_axes = [a for a in range(3) if a != axis]
    corner_order = [(0, 0), (1, 0), (1, 1), (0, 1)]

    def make_vertex(c):
        v = [0, 0, 0]
        v[axis] = value
        v[other_axes[0]] = c[0]
        v[other_axes[1]] = c[1]
        return tuple(v)

    corners = [make_vertex(c) for c in corner_order]
    edges = []
    for k in range(4):
        v1, v2 = corners[k], corners[(k + 1) % 4]
        e = tuple(sorted((VERTEX_ID[v1], VERTEX_ID[v2])))
        edges.append(EDGE_ID[e])
    return edges


FACE_EDGES = [_face_edges_cyclic(axis, value) for axis, value in FACES]


def edge_axis(edge_id):
    """Axis (0,1,2) that this cube edge is parallel to."""
    v1, v2 = EDGES[edge_id]
    p1, p2 = VERTICES[v1], VERTICES[v2]
    for axis in range(3):
        if p1[axis] != p2[axis]:
            return axis
    raise ValueError("degenerate edge")


def edge_fixed_coords(edge_id):
    """The two fixed (non-varying) coordinates of this edge as {axis: value}."""
    v1, v2 = EDGES[edge_id]
    p1 = VERTICES[v1]
    axis = edge_axis(edge_id)
    return {a: p1[a] for a in range(3) if a != axis}


def edge_midpoint(edge_id):
    v1, v2 = EDGES[edge_id]
    p1 = np.array(VERTICES[v1], dtype=float)
    p2 = np.array(VERTICES[v2], dtype=float)
    return (p1 + p2) / 2


def shared_face(edge_a, edge_b):
    """Index into FACES/FACE_EDGES of the unique face bordered by both
    edges (raises if they don't share exactly one face)."""
    candidates = [i for i, fe in enumerate(FACE_EDGES)
                  if edge_a in fe and edge_b in fe]
    if len(candidates) != 1:
        raise ValueError(
            f"edges {edge_a},{edge_b} share {len(candidates)} faces, expected 1")
    return candidates[0]

# Per-face local connection patterns (Fig. 3): chords among local slots 0..3
# (slots are the face's 4 bounding edges in cyclic order).
LOCAL_PATTERNS = {
    'empty': [],
    'adj0': [(0, 1)],
    'adj1': [(1, 2)],
    'adj2': [(2, 3)],
    'adj3': [(3, 0)],
    'opp0': [(0, 2)],
    'opp1': [(1, 3)],
    'full0': [(0, 1), (2, 3)],
    'full1': [(1, 2), (3, 0)],
}
PATTERN_NAMES = list(LOCAL_PATTERNS.keys())          # 9, includes 'empty'
NONEMPTY_PATTERN_NAMES = [p for p in PATTERN_NAMES if p != 'empty']  # 8

# ---------------------------------------------------------------------------
# The 48 cube symmetries (signed permutations of the 3 axes).
# ---------------------------------------------------------------------------


def _build_symmetries():
    """Return a list of edge-id permutations, one per cube symmetry."""
    perms = []
    for axis_perm in itertools.permutations(range(3)):
        for signs in itertools.product([0, 1], repeat=3):
            def transform(v, axis_perm=axis_perm, signs=signs):
                w = [0, 0, 0]
                for src_axis in range(3):
                    dst_axis = axis_perm.index(src_axis)
                    c = v[src_axis]
                    if signs[src_axis]:
                        c = 1 - c
                    w[dst_axis] = c
                return tuple(w)

            vertex_perm = [VERTEX_ID[transform(v)] for v in VERTICES]
            edge_perm = np.empty(N_EDGES, dtype=int)
            for eid, (a, b) in enumerate(EDGES):
                na, nb = vertex_perm[a], vertex_perm[b]
                new_e = tuple(sorted((na, nb)))
                edge_perm[eid] = EDGE_ID[new_e]
            perms.append(edge_perm)
    return perms


CUBE_SYMMETRIES = _build_symmetries()  # 48 permutations of edge ids
assert len(CUBE_SYMMETRIES) == 48


# ---------------------------------------------------------------------------
# Topology representation and enumeration.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Topology:
    """A boundary-curve topology on the 1/8 cube.

    `face_patterns`: name of the connection pattern used on each of the 6
    faces (order matches `FACES`/`FACE_NAMES`).
    `arcs`: frozenset of frozenset({edge_id_a, edge_id_b}) - the curve
    segments, each spanning one face.
    """
    face_patterns: tuple
    arcs: frozenset

    def active_edges(self):
        return sorted({e for arc in self.arcs for e in arc})

    def n_loops(self):
        """Number of disjoint closed loops in the boundary curve."""
        adj = {e: [] for e in self.active_edges()}
        for arc in self.arcs:
            a, b = tuple(arc)
            adj[a].append(b)
            adj[b].append(a)
        seen = set()
        loops = 0
        for e in adj:
            if e in seen:
                continue
            loops += 1
            stack = [e]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(adj[cur])
        return loops


def _arcs_for_combo(pattern_names):
    """Build the global arc set for a choice of per-face pattern names."""
    arcs = []
    degree = np.zeros(N_EDGES, dtype=int)
    for face_idx, pname in enumerate(pattern_names):
        slots = FACE_EDGES[face_idx]
        for a, b in LOCAL_PATTERNS[pname]:
            ea, eb = slots[a], slots[b]
            arcs.append(frozenset({ea, eb}))
            degree[ea] += 1
            degree[eb] += 1
    return arcs, degree


def _canonical_form(arcs):
    """Canonical representative of a topology's arc set under the 48 cube
    symmetries, used to deduplicate equivalent topologies."""
    best = None
    for perm in CUBE_SYMMETRIES:
        mapped = frozenset(
            frozenset({int(perm[a]), int(perm[b])})
            for arc in arcs for a, b in [tuple(arc)]
        )
        key = tuple(sorted(tuple(sorted(arc)) for arc in mapped))
        if best is None or key < best:
            best = key
    return best


def enumerate_topologies(include_empty_face_state=True, require_single_loop=False):
    """Enumerate unique 1/8-cube boundary topologies (Sec. 2.2).

    With `include_empty_face_state=True` each face may additionally touch no
    curve at all (9 states/face); this is needed to reach sparse topologies
    such as the paper's topo_13/topo_15 (a single line through the cube).
    The paper's own count (8 states/face, i.e. every face is assumed to be
    touched, giving 8**6 raw combinations before filtering) is available via
    `include_empty_face_state=False`, matching their stated "256 after the
    degree filter". Our implementation is a faithful re-derivation of the
    method (closed-curve degree filter + cube-symmetry dedup); the exact
    integer counts are reported by `scripts/enumerate_topologies.py` and may
    differ slightly from the paper's 20 depending on this empty-face
    modeling choice, which the paper's text does not fully disambiguate.
    """
    names = PATTERN_NAMES if include_empty_face_state else NONEMPTY_PATTERN_NAMES

    valid = []
    for combo in itertools.product(names, repeat=6):
        arcs, degree = _arcs_for_combo(combo)
        if np.any((degree != 0) & (degree != 2)):
            continue
        if not arcs:
            continue  # degenerate: no boundary at all
        valid.append((combo, frozenset(arcs)))

    canon_to_rep = {}
    for combo, arcs in valid:
        topo = Topology(face_patterns=combo, arcs=arcs)
        if require_single_loop and topo.n_loops() != 1:
            continue
        canon = _canonical_form(arcs)
        if canon not in canon_to_rep:
            canon_to_rep[canon] = topo

    reps = list(canon_to_rep.values())
    # Stable, deterministic ordering: by number of arcs, then canonical key.
    reps.sort(key=lambda t: (len(t.arcs), _canonical_form(t.arcs)))
    return reps, len(valid)
