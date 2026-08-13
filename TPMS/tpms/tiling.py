"""Synthesize the full triply-periodic unit cell from a 1/8-cube shell mesh
via per-axis reflection/translation symmetry, and tile it into an N x N x N
lattice (paper Sec. 2.2 "topo_id_x1x2x3" synthesis; Figs. 9, 17, 18).
"""
import itertools

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import boundary as B


def _open_boundary_vertices(faces, n_verts):
    """Vertex indices that touch an edge used by exactly one triangle (i.e.
    the rim of an open patch)."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    return np.unique(boundary_edges)


def weld_vertices(verts, faces, tol):
    """Merge vertices within `tol` of each other, remapping faces and
    dropping any that degenerate. Needed because reflected/translated
    octant copies of a neural-network-extracted patch only meet
    approximately (network smoothing + finite training), not to
    floating-point precision, so a naive exact-match weld leaves gaps.

    Only *open-boundary* vertices (the rim of each octant patch) are
    considered as weld candidates: with a tolerance comparable to the local
    mesh edge length, matching against *all* vertices would transitively
    chain together nearby vertices along the interior of a smooth surface
    and collapse the whole mesh, not just the seams.
    """
    n = len(verts)
    candidates = _open_boundary_vertices(faces, n)
    if len(candidates) == 0:
        return verts, faces

    sub = verts[candidates]
    tree = cKDTree(sub)
    pairs = np.array(sorted(tree.query_pairs(r=tol)), dtype=np.int64)

    inverse_local = np.arange(len(candidates))
    if len(pairs) > 0:
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                            shape=(len(candidates), len(candidates)))
        _n_clusters, inverse_local = connected_components(
            graph, directed=False, connection='weak')

    # Map each candidate vertex to its cluster representative (first member).
    remap = np.arange(n)
    cluster_rep = {}
    for local_idx, cluster_id in enumerate(inverse_local):
        global_idx = candidates[local_idx]
        if cluster_id not in cluster_rep:
            cluster_rep[cluster_id] = global_idx
        remap[global_idx] = cluster_rep[cluster_id]

    # Average position within each merged cluster, keep others unchanged.
    new_verts = verts.copy()
    for cluster_id in np.unique(inverse_local):
        members = candidates[inverse_local == cluster_id]
        new_verts[cluster_rep[cluster_id]] = verts[members].mean(axis=0)

    used = np.unique(remap[faces.reshape(-1)])
    old_to_new = -np.ones(n, dtype=np.int64)
    old_to_new[used] = np.arange(len(used))

    new_faces = old_to_new[remap[faces]]
    degenerate = ((new_faces[:, 0] == new_faces[:, 1])
                  | (new_faces[:, 1] == new_faces[:, 2])
                  | (new_faces[:, 0] == new_faces[:, 2]))
    new_faces = new_faces[~degenerate]
    return new_verts[used], new_faces


def synthesize_unit_cell_mesh(verts01, faces, sym_code):
    """verts01: (V,3) in [0,1]^3 (the 1/8-cube). Returns a merged (not yet
    vertex-welded) mesh spanning [-1,1]^3 (all 8 octants)."""
    all_verts, all_faces = [], []
    offset = 0
    for axis_signs in itertools.product([0, 1], repeat=3):
        transform = B.octant_transform(axis_signs, sym_code)
        v = transform(verts01)
        det = 1
        for axis in range(3):
            if axis_signs[axis] and sym_code[axis] == 'r':
                det *= -1
        f = faces[:, ::-1].copy() if det < 0 else faces.copy()
        all_verts.append(v)
        all_faces.append(f + offset)
        offset += len(v)
    return np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0)


def tile_lattice(verts, faces, n=(1, 1, 1), cell_size=2.0):
    """Translate copies of a unit-cell mesh into an n[0] x n[1] x n[2]
    periodic lattice."""
    all_v, all_f = [], []
    offset = 0
    for i, j, k in itertools.product(range(n[0]), range(n[1]), range(n[2])):
        shift = np.array([i, j, k], dtype=float) * cell_size
        all_v.append(verts + shift)
        all_f.append(faces + offset)
        offset += len(verts)
    return np.concatenate(all_v, axis=0), np.concatenate(all_f, axis=0)
