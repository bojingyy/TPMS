"""Reproduces the key validation checks from README.md "Validation results".
Run with: python -m pytest tests/ -v  (or just `python tests/test_validation.py`)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import trimesh

from tpms import homogenization as H
from tpms import metrics as M
from tpms import topology as T


def test_topology_enumeration_matches_paper():
    reps, n_valid = T.enumerate_topologies(include_empty_face_state=False)
    assert n_valid == 256, f"expected 256 raw topologies, got {n_valid}"
    assert len(reps) == 20, f"expected 20 unique topologies, got {len(reps)}"


def test_topology_loops_are_closed():
    reps, _ = T.enumerate_topologies(include_empty_face_state=False)
    for topo in reps:
        degree = {}
        for arc in topo.arcs:
            for e in arc:
                degree[e] = degree.get(e, 0) + 1
        assert all(d == 2 for d in degree.values())


def test_homogenization_recovers_base_material_on_solid_cube():
    density = np.ones((8, 8, 8))
    CH = H.homogenize(density, E0=1.0, nu0=0.3, cell_size=2.0)
    props = H.elastic_properties(CH)
    assert abs(props['E'] - 1.0) < 1e-8
    assert abs(props['nu'] - 0.3) < 1e-8
    assert abs(props['Z'] - 1.0) < 1e-8  # isotropic


def test_homogenization_matches_paper_p_surface_ballpark():
    from tpms import classical_tpms as C
    res = 24
    period = 2 * np.pi
    coords = np.linspace(0, period, res, endpoint=False)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    phi = C.phi_P(X, Y, Z)
    density = (np.abs(phi) < 0.34).astype(float)  # ~30% relative density
    CH = H.homogenize(density, E0=1.0, nu0=0.3, cell_size=period)
    props = H.elastic_properties(CH)
    # Paper Fig. 12a/14a: P surface at rd~0.3 has E in [0.08, 0.16], nu in
    # [0.20, 0.35], Zener in [2, 3.5]. Loose bounds here since this test
    # uses a coarser grid than the paper's 60^3.
    assert 0.03 < props['E'] < 0.25
    assert 0.15 < props['nu'] < 0.45
    assert 1.2 < props['Z'] < 5.0


def test_mean_curvature_and_sav_on_sphere():
    R = 2.0
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=R)
    Hc = M.discrete_mean_curvature(mesh)
    assert abs(Hc.mean() - 1.0 / R) < 0.02
    sav = M.sa_to_v_ratio(mesh)
    assert abs(sav - 3.0 / R) < 0.02


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
