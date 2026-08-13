"""Numerical homogenization of a periodic voxel microstructure via 3D
periodic finite elements (paper Sec. 3, Eq. 6-10; ref. [29] Liu et al.,
ref. [40] Liu & Tovar "top3d"). Computes the homogenized elastic stiffness
matrix C_H of a unit cell from a binary (or fractional) density grid, then
Young's modulus, Poisson's ratio, shear modulus and Zener anisotropy ratio.

Method: 8-node trilinear hexahedral elements, one per voxel, periodic
boundary conditions built directly into the DOF numbering (node indices wrap
via modular arithmetic, following Andreassen & Andreasen 2014's 2D scheme,
extended here to 3D), solved for 6 independent unit macroscopic strain load
cases (Voigt order 11,22,33,23,13,12, matching paper Eq. 6).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Voigt order: (11, 22, 33, 23, 13, 12)
UNIT_STRAINS = np.eye(6)

# 2-point Gauss quadrature abscissae for [-1,1], exact for the (bi/tri)linear
# shape-function derivatives appearing in B^T C B.
_GP = np.array([-1, 1]) / np.sqrt(3)


def _shape_derivs_natural(xi, eta, zeta):
    """dN_i/d(xi,eta,zeta) for the 8-node trilinear hex, node order matching
    corners (+-1,+-1,+-1) via `signs` below."""
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    dN = np.empty((8, 3))
    for i, (sx, sy, sz) in enumerate(signs):
        dN[i, 0] = 0.125 * sx * (1 + sy * eta) * (1 + sz * zeta)
        dN[i, 1] = 0.125 * sy * (1 + sx * xi) * (1 + sz * zeta)
        dN[i, 2] = 0.125 * sz * (1 + sx * xi) * (1 + sy * eta)
    return dN


def _isotropic_C(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.array([
        [lam + 2 * mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 * mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 * mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu],
    ])
    return C


def _B_matrix(dN_xyz):
    """dN_xyz: (8,3) physical-space shape function gradients -> (6,24) B."""
    B = np.zeros((6, 24))
    for i in range(8):
        dx, dy, dz = dN_xyz[i]
        B[0, 3 * i + 0] = dx
        B[1, 3 * i + 1] = dy
        B[2, 3 * i + 2] = dz
        B[3, 3 * i + 1] = dz; B[3, 3 * i + 2] = dy
        B[4, 3 * i + 0] = dz; B[4, 3 * i + 2] = dx
        B[5, 3 * i + 0] = dy; B[5, 3 * i + 1] = dx
    return B


def element_matrices(h, nu):
    """Precompute the per-Gauss-point B matrices, quadrature weight*detJ,
    and the unit-modulus (E=1) element stiffness Ke and load matrix Fe
    (24x6, one column per unit macroscopic strain case)."""
    C_unit = _isotropic_C(1.0, nu)
    detJ = (h / 2) ** 3
    Bs, wJs = [], []
    for xi in _GP:
        for eta in _GP:
            for zeta in _GP:
                dN_nat = _shape_derivs_natural(xi, eta, zeta)
                dN_xyz = dN_nat * (2.0 / h)  # constant Jacobian for a cube
                Bs.append(_B_matrix(dN_xyz))
                wJs.append(detJ)  # gauss weight = 1 per point (2-pt rule)
    Bs = np.stack(Bs)      # (8, 6, 24)
    wJs = np.array(wJs)    # (8,)

    # Bs: (gauss=8, strain=6, dof=24); contract strain indices through C_unit.
    Ke = np.einsum('g,gsd,st,gtc->dc', wJs, Bs, C_unit, Bs)  # (24,24)
    Fe = np.einsum('g,gsd,st,tp->dp', wJs, Bs, C_unit, UNIT_STRAINS)  # (24,6)
    return Bs, wJs, Ke, Fe, C_unit


_CORNERS = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                      (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])


def _all_element_dof(nx, ny, nz):
    """(n_el, 24) global DOF indices for every element at once (node order
    matching `_shape_derivs_natural`'s corner sign convention), fully
    vectorized -- a Python loop over voxels is the bottleneck at the paper's
    target 60^3 resolution."""
    ex, ey, ez = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                              indexing='ij')
    ex = ex.ravel(); ey = ey.ravel(); ez = ez.ravel()  # (n_el,)
    # corner node coords, shape (n_el, 8)
    cx = (ex[:, None] + _CORNERS[None, :, 0]) % nx
    cy = (ey[:, None] + _CORNERS[None, :, 1]) % ny
    cz = (ez[:, None] + _CORNERS[None, :, 2]) % nz
    nid = cx * (ny * nz) + cy * nz + cz  # (n_el, 8)
    dof = np.empty((len(ex), 24), dtype=np.int64)
    dof[:, 0::3] = 3 * nid
    dof[:, 1::3] = 3 * nid + 1
    dof[:, 2::3] = 3 * nid + 2
    return dof


def is_connected(density, threshold=0.5, dominant_frac=0.95):
    """Cheap (non-periodic, so conservative) connectivity check. Isolated
    material islands surrounded by void leave near-zero-stiffness rigid-body
    modes beyond the one we pin, which makes the CG solve below stall for
    thousands of iterations -- worth filtering out before calling
    `homogenize` when scanning many random samples (e.g. GAN dataset
    generation)."""
    from scipy import ndimage
    solid = density > threshold
    if solid.sum() == 0:
        return False
    labeled, n = ndimage.label(solid, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return True
    sizes = ndimage.sum(solid, labeled, range(1, n + 1))
    return sizes.max() / solid.sum() > dominant_frac


def homogenize(density, E0=1.0, nu0=0.3, Emin_ratio=1e-4, cell_size=2.0,
               pin_node=0, maxiter=5000):
    """density: (nx,ny,nz) array in [0,1] (voxel material fraction).
    cell_size: physical side length of the periodic unit cell.
    Returns the 6x6 homogenized stiffness matrix C_H (Voigt order
    11,22,33,23,13,12, matching paper Eq. 6)."""
    nx, ny, nz = density.shape
    h = cell_size / nx
    n_nodes = nx * ny * nz
    n_dof = 3 * n_nodes

    Bs, wJs, Ke_unit, Fe_unit, C_unit = element_matrices(h, nu0)

    E_voxel = (Emin_ratio + density.ravel(order='C') * (1 - Emin_ratio)) * E0
    n_el = nx * ny * nz
    edof = _all_element_dof(nx, ny, nz)  # (n_el, 24), fully vectorized

    # K assembly: rows/cols are outer product of edof with itself per element.
    rows = np.repeat(edof, 24, axis=1).reshape(n_el, 24, 24)
    cols = np.repeat(edof[:, None, :], 24, axis=1)
    vals = E_voxel[:, None, None] * Ke_unit[None, :, :]
    K = sp.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())),
                       shape=(n_dof, n_dof)).tocsr()

    # F assembly: (n_el, 24, 6) scattered into (n_dof, 6).
    Fvals = E_voxel[:, None, None] * Fe_unit[None, :, :]
    F = np.zeros((n_dof, 6))
    np.add.at(F, edof, Fvals)

    # Pin rigid-body translation (periodic-only BC leaves a null space).
    pin_dofs = np.array([3 * pin_node, 3 * pin_node + 1, 3 * pin_node + 2])
    free = np.setdiff1d(np.arange(n_dof), pin_dofs)

    K_ff = K[free][:, free].tocsr()
    F_f = F[free]

    # K_ff is SPD; sparse LU (via spla.factorized) has catastrophic fill-in
    # on these periodic 3D meshes even at modest resolution, so we use a
    # Jacobi-preconditioned CG solve instead, which scales to the paper's
    # 60^3 grids.
    diag = K_ff.diagonal()
    M = spla.LinearOperator(K_ff.shape, matvec=lambda x: x / diag)
    chi = np.zeros((n_dof, 6))
    for p in range(6):
        x, info = spla.cg(K_ff, F_f[:, p], M=M, rtol=1e-8, maxiter=maxiter)
        if info != 0:
            raise RuntimeError(f"CG did not converge for load case {p} (info={info})")
        chi[free, p] = x

    # Homogenized stiffness via element-wise energy formula.
    chi_e = chi[edof]  # (n_el, 24, 6)
    return _assemble_CH(density, E0, Emin_ratio, Bs, wJs, C_unit, chi_e, cell_size)


def _assemble_CH(density, E0, Emin_ratio, Bs, wJs, C_unit, chi_e, cell_size):
    nx, ny, nz = density.shape
    n_el = nx * ny * nz
    E_voxel = (Emin_ratio + density.ravel(order='C') * (1 - Emin_ratio)) * E0

    # eps_fluct[e, g, :, p] = Bs[g] @ chi_e[e, :, p]
    eps_fluct = np.einsum('gab,ebp->egap', Bs, chi_e)  # (n_el, 8, 6, 6)
    eps_total = UNIT_STRAINS[None, None, :, :] - eps_fluct  # broadcast (n_el,8,6,6)

    # C_H_pq = (1/|Y|) sum_e E_voxel[e] * sum_g wJ[g] * eps_total[e,g,:,p]^T C_unit eps_total[e,g,:,q]
    Ce = np.einsum('egap,ab,egbq->egpq', eps_total, C_unit, eps_total)
    contrib = np.einsum('e,g,egpq->pq', E_voxel, wJs, Ce)
    volume = cell_size ** 3
    return contrib / volume


def elastic_properties(C_H):
    """Young's modulus, Poisson's ratio, shear modulus, Zener ratio from a
    (near-)cubic-symmetric homogenized stiffness matrix (paper Eq. 7-10)."""
    C11 = (C_H[0, 0] + C_H[1, 1] + C_H[2, 2]) / 3
    C12 = (C_H[0, 1] + C_H[0, 2] + C_H[1, 2]) / 3
    C44 = (C_H[3, 3] + C_H[4, 4] + C_H[5, 5]) / 3
    E = (C11 ** 2 + C11 * C12 - 2 * C12 ** 2) / (C11 + C12)
    nu = C12 / (C11 + C12)
    G = C44
    Z = 2 * C44 / (C11 - C12)
    return dict(E=E, nu=nu, G=G, Z=Z, C11=C11, C12=C12, C44=C44)
