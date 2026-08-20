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


def test_boundary_arc_is_exact_quarter_circle():
    """Eq. 2's d1=P'(0) and d2=P'(1) are both tangents in the direction of
    travel. Regression guard for a sign error where d2 pointed back into the
    face the curve came from, putting an S-hook at each arrival node and
    flattening every arc toward its chord.

    With the correct convention an arc between two mid-edge nodes sharing a
    cube corner is the exact quarter circle through them at d = D_CIRCULAR.
    """
    from tpms import boundary as B
    from tpms import spline

    # z=0 face, corner (0,0,0): nodes at (0.5,0,0) and (0,0.5,0), both at
    # radius 0.5 from the corner.
    p1 = np.array([0.5, 0.0, 0.0])
    p2 = np.array([0.0, 0.5, 0.0])
    m = B.D_CIRCULAR
    d1 = np.array([0.0, 1.0, 0.0]) * m      # leaves p1 into the face
    d2 = np.array([-1.0, 0.0, 0.0]) * m     # arrives at p2 still travelling -x
    c = spline.hermite_eval(p1, d1, p2, d2, np.linspace(0, 1, 400))
    radius = np.linalg.norm(c[:, :2], axis=1)
    assert np.abs(radius - 0.5).max() < 1e-3, (
        f"arc deviates from the quarter circle by "
        f"{np.abs(radius - 0.5).max():.4f}")


def test_boundary_curves_are_closed_in_bounds_and_non_backtracking():
    """Every topology's boundary curve must close up, stay inside the 1/8
    cube, and have no arc that reverses against its own chord (the paper
    assumes boundaries free from self-intersection, Sec. 2.2)."""
    from tpms import boundary as B
    from tpms import spline

    reps, _ = T.enumerate_topologies(include_empty_face_state=False)
    rng = np.random.default_rng(0)
    for topo in reps:
        for params in (B.default_params(topo), B.random_params(topo, rng)):
            for curve in B.build_boundary_curves(topo, params, n_samples=128):
                steps = np.linalg.norm(
                    np.diff(np.vstack([curve, curve[:1]]), axis=0), axis=1)
                # closed: the wrap-around step is just another sampling step
                assert steps.max() < 3 * np.median(steps), "curve is not closed"
                assert curve.min() > -1e-9 and curve.max() < 1 + 1e-9, \
                    "curve leaves the 1/8 cube"

            for loop in B.find_loops(topo):
                for i in range(len(loop)):
                    a, b = loop[i], loop[(i + 1) % len(loop)]
                    face = T.shared_face(a, b)
                    pa = B.node_position(a, params[a].s)
                    pb = B.node_position(b, params[b].s)
                    da = B.node_face_direction(a, face) * params[a].d_out
                    db = -B.node_face_direction(b, face) * params[b].d_in
                    _, C1, C2, C3 = spline.hermite_coeffs(pa, da, pb, db)
                    t = np.linspace(0, 1, 200)[:, None]
                    deriv = C1 + 2 * C2 * t + 3 * C3 * t ** 2
                    chord = pb - pa
                    proj = deriv @ (chord / np.linalg.norm(chord))
                    assert proj.min() >= 0, \
                        f"arc {a}->{b} backtracks against its chord"


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
