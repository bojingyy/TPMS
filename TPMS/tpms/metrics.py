"""Geometric metrics used in Sec. 3.1: surface-area-to-volume ratio (Fig.
16) and discrete mean curvature (Fig. 15, via the standard cotangent-formula
discrete Laplace-Beltrami operator, as in Meyer et al. 2003 / the
`libigl`-style approach the paper cites [36] Peyré's toolbox)."""
import numpy as np


def surface_area(mesh):
    return mesh.area


def enclosed_volume(mesh):
    """Absolute enclosed volume; requires (approximately) watertight mesh."""
    return abs(mesh.volume)


def sa_to_v_ratio(mesh):
    return surface_area(mesh) / enclosed_volume(mesh)


def discrete_mean_curvature(mesh):
    """Per-vertex mean curvature via the discrete Laplace-Beltrami operator
    with cotangent weights: H_i = |L(x)_i| / 2, signed by agreement with the
    vertex normal (Meyer, Desbrun, Schroder, Barr 2003). Vectorized over all
    3 (vertex-triple, opposite-angle) combinations per face at once."""
    verts = mesh.vertices
    faces = mesh.faces
    n = len(verts)

    L = np.zeros((n, 3))
    mixed_area = np.zeros(n)
    face_area = mesh.area_faces

    for i in range(3):
        a = faces[:, i]
        b = faces[:, (i + 1) % 3]
        c = faces[:, (i + 2) % 3]
        u = verts[a] - verts[c]
        v = verts[b] - verts[c]
        cos_c = np.einsum('ij,ij->i', u, v) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
        cos_c = np.clip(cos_c, -1 + 1e-9, 1 - 1e-9)
        cot_c = cos_c / np.sqrt(1 - cos_c ** 2)
        contrib_a = cot_c[:, None] * (verts[b] - verts[a])
        contrib_b = cot_c[:, None] * (verts[a] - verts[b])
        np.add.at(L, a, contrib_a)
        np.add.at(L, b, contrib_b)
        np.add.at(mixed_area, faces[:, i], face_area / 3.0)

    mixed_area = np.clip(mixed_area, 1e-12, None)
    Hvec = -L / (2 * mixed_area[:, None])  # sign convention: convex (outward
    # bulge, e.g. a sphere with outward normals) => positive H
    H = 0.5 * np.linalg.norm(Hvec, axis=1)
    normals = mesh.vertex_normals
    sign = np.sign(np.einsum('ij,ij->i', Hvec, normals))
    sign[sign == 0] = 1
    return H * sign
