"""End-to-end pipeline (paper Fig. 1): pick a 1/8-cube boundary topology,
generate its Hermite-spline geometry, train the neural minimal-surface
solver with C1 boundary-continuity constraints, extract + synthesize the
full periodic unit cell, and (optionally) homogenize it.

Usage:
    python scripts/build_shell_lattice.py --topo 1 --sym rrr \
        --n_iterations 30000 --out out/topo1_rrr

Note on runtime: `--n_iterations` trades training time for how well the
surface reaches its true boundary curve (see TPMS/README.md "Fidelity
notes"); 30-60k iterations (a few minutes on a modern GPU) gives a
recognizable shell-lattice patch, DeepCurrents' own default of 100k gives a
tighter fit.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from tpms import boundary as B
from tpms import homogenization as H
from tpms import topology as T
from tpms.mesh_extract import extract_shell_mesh
from tpms.shell import build_shell_lattice


def voxelize_for_homogenization(mesh, resolution):
    """Binary occupancy grid over the mesh's bounding lattice cell via
    trimesh's containment test (used only for the optional --homogenize
    step; mesh should be reasonably closed for this to be meaningful)."""
    import trimesh
    lo = mesh.bounds[0]
    extent = mesh.bounds[1] - mesh.bounds[0]
    coords = [lo[a] + (np.arange(resolution) + 0.5) / resolution * extent[a]
              for a in range(3)]
    X, Y, Z = np.meshgrid(*coords, indexing='ij')
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    inside = mesh.contains(pts)
    return inside.reshape(resolution, resolution, resolution).astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topo', type=int, default=1, help='1-indexed topology (1-20)')
    parser.add_argument('--sym', type=str, default='rrr', help="3 chars from {r,t}, per-axis symmetry")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_iterations', type=int, default=30000)
    parser.add_argument('--resolution', type=int, default=140, help='marching-cubes grid resolution')
    parser.add_argument('--threshold_frac', type=float, default=0.05)
    parser.add_argument('--weld_tol', type=float, default=None)
    parser.add_argument('--n_lattice', type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument('--homogenize', action='store_true')
    parser.add_argument('--homog_resolution', type=int, default=40)
    parser.add_argument('--out', default=os.path.join(
        os.path.dirname(__file__), '..', 'out', 'shell_lattice'))
    args = parser.parse_args()

    assert len(args.sym) == 3 and all(c in 'rt' for c in args.sym)
    sym_code = tuple(args.sym)

    reps, _ = T.enumerate_topologies(include_empty_face_state=False)
    topo = reps[args.topo - 1]
    print(f"topo_{args.topo}: {len(topo.active_edges())} nodes, "
          f"{len(topo.arcs)} arcs, {topo.n_loops()} loop(s), sym={args.sym}")

    rng = np.random.default_rng(args.seed)
    params = B.random_params(topo, rng)

    def progress(rec):
        print(rec)

    mesh, model, history = build_shell_lattice(
        topo, params, sym_code,
        train_kwargs=dict(n_iterations=args.n_iterations, seed=args.seed,
                           log_every=max(1, args.n_iterations // 10),
                           progress_cb=progress),
        resolution=args.resolution, threshold_frac=args.threshold_frac,
        n_lattice=tuple(args.n_lattice), weld_tol=args.weld_tol)

    os.makedirs(args.out, exist_ok=True)
    mesh.export(os.path.join(args.out, 'mesh.ply'))
    torch.save(model.state_dict(), os.path.join(args.out, 'model.pt'))
    with open(os.path.join(args.out, 'params.json'), 'w') as f:
        json.dump({str(k): vars(v) for k, v in params.items()}, f, indent=2)

    print(f"mesh: verts={len(mesh.vertices)} faces={len(mesh.faces)} "
          f"watertight={mesh.is_watertight} n_components={mesh.body_count}")
    print(f"Saved to {args.out}")

    if args.homogenize:
        density = voxelize_for_homogenization(mesh, args.homog_resolution)
        print(f"voxel relative density: {density.mean():.4f}")
        CH = H.homogenize(density, E0=1.0, nu0=0.3,
                           cell_size=mesh.bounds[1, 0] - mesh.bounds[0, 0])
        props = H.elastic_properties(CH)
        print("homogenized properties:", {k: round(float(v), 5) for k, v in props.items()})


if __name__ == '__main__':
    main()
