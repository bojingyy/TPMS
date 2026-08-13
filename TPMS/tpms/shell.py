"""End-to-end shell-lattice construction (paper Sec. 2.4): topology + spline
params + symmetry code -> trained minimal surface -> extracted shell mesh ->
synthesized unit cell -> periodic lattice, as a single convenience pipeline.
"""
import trimesh

from . import mesh_extract, tiling
from . import train_minimal


def build_shell_lattice(topo, params, sym_code, model=None, train_kwargs=None,
                         resolution=100, threshold_frac=0.05,
                         n_lattice=(1, 1, 1), weld_tol=None):
    """Returns (trimesh.Trimesh, model, history).

    `weld_tol` merges vertices across octant/lattice seams within this
    distance (in the [-1,1]^3 unit-cell units); the network only reaches its
    true boundary curve approximately (see README "Fidelity notes"), so a
    tolerance-based weld (default: 3x the marching-cubes grid spacing) is
    needed to stitch neighboring patches into a connected shell, unlike an
    exact-match weld which would leave visible gaps.
    """
    history = None
    if model is None:
        train_kwargs = dict(train_kwargs or {})
        model, history = train_minimal.train(topo, params, sym_code, **train_kwargs)

    verts_u, faces = mesh_extract.extract_shell_mesh(
        model, resolution=resolution, threshold_frac=threshold_frac)
    verts01 = (verts_u + 1.0) / 2.0

    cell_v, cell_f = tiling.synthesize_unit_cell_mesh(verts01, faces, sym_code)
    lat_v, lat_f = tiling.tile_lattice(cell_v, cell_f, n=n_lattice, cell_size=2.0)

    if weld_tol is None:
        weld_tol = 3.0 * (2.0 / (resolution - 1))
    lat_v, lat_f = tiling.weld_vertices(lat_v, lat_f, tol=weld_tol)

    mesh = trimesh.Trimesh(vertices=lat_v, faces=lat_f, process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh, model, history
