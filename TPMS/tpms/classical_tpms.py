"""Classical TPMS implicit (level-set) equations, for the comparisons in
Sec. 3 (Figs. 11-16). P, D, G are given explicitly in the paper (Eq. 1);
I-WP, F-RD and Neovius are the standard formulas from the TPMS literature
(Schoen 1970 / von Schnering & Nesper 1991), used by the paper (Fig. 11) as
the five classical "outer boundary topologies" compared against.

All surfaces are 2*pi-periodic in x, y, z.
"""
import numpy as np
from skimage import measure


def phi_P(x, y, z):
    return np.cos(x) + np.cos(y) + np.cos(z)


def phi_D(x, y, z):
    return (np.cos(x) * np.cos(y) * np.cos(z)
            - np.sin(x) * np.sin(y) * np.sin(z))


def phi_G(x, y, z):
    return (np.sin(x) * np.cos(y) + np.sin(z) * np.cos(x)
            + np.sin(y) * np.cos(z))


def phi_IWP(x, y, z):
    return (2 * (np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z)
                 + np.cos(z) * np.cos(x))
            - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)))


def phi_FRD(x, y, z):
    return (4 * np.cos(x) * np.cos(y) * np.cos(z)
            - (np.cos(2 * x) * np.cos(2 * y) + np.cos(2 * y) * np.cos(2 * z)
               + np.cos(2 * z) * np.cos(2 * x)))


def phi_N(x, y, z):
    return 3 * (np.cos(x) + np.cos(y) + np.cos(z)) + 4 * np.cos(x) * np.cos(y) * np.cos(z)


SURFACES = {
    'P': phi_P,
    'D': phi_D,
    'G': phi_G,
    'IWP': phi_IWP,
    'FRD': phi_FRD,
    'N': phi_N,
}


def sample_grid(phi, n_cells=1, resolution=60, period=2 * np.pi):
    """Evaluate `phi` on a regular grid covering `n_cells` periods per axis.
    Returns (values, spacing) where values has shape (R,R,R)."""
    L = n_cells * period
    coords = np.linspace(0, L, resolution * n_cells, endpoint=False)
    spacing = coords[1] - coords[0]
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    values = phi(X, Y, Z)
    return values, spacing


def extract_mesh(name_or_fn, c=0.0, n_cells=1, resolution=60, period=2 * np.pi):
    """Marching-cubes mesh of {phi = c}, tiled over `n_cells` periods/axis.
    Returns (vertices, faces) with vertices in physical (radian) units."""
    phi = SURFACES[name_or_fn] if isinstance(name_or_fn, str) else name_or_fn
    values, spacing = sample_grid(phi, n_cells=n_cells, resolution=resolution,
                                   period=period)
    verts, faces, _normals, _values = measure.marching_cubes(
        values, level=c, spacing=(spacing, spacing, spacing))
    return verts, faces
