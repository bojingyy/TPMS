"""Mesh extraction from a trained `SurfaceModel` (paper Sec. 2.3 end / Sec.
2.4, Fig. 7-8: "the discrete mesh model ... extracted based on the ...
marching cube algorithm").

Note on why we threshold |df| rather than f itself: f_theta is trained only
through its *gradient* (Eq. 3: minimize ||df + alpha_Gamma||), so f is a
multi-valued-potential-like quantity (comparable to a winding number /
solid-angle function) that is only defined up to an additive constant and
drifts slowly away from the surface -- it does not have a single clean global
zero level set. What *does* robustly localize the surface is |df|
(equivalently |current|, since alpha's contribution is smooth and small): it
spikes sharply exactly where the network is trying to fit the jump
discontinuity of the true (surface-crossing) potential, and is small
everywhere else. We verified this empirically (see scripts/validate_minimal_surface.py):
for a flat disk spanning a circle, |df| is 5-100x larger for points on the
disk than for points just outside it. So we run marching cubes on the
|df| field, which directly yields a closed shell hugging the true minimal
surface -- conveniently already giving the shell of Sec. 2.4 a nonzero base
thickness (controlled by `threshold_frac` and network smoothness), which can
be refined further by `tpms.shell`.
"""
import numpy as np
import torch
from skimage import measure


@torch.no_grad()
def _eval_only_f(model, pts, device, batch=1 << 16):
    out = []
    for chunk in torch.split(pts, batch):
        out.append(model(chunk.to(device), only_f=True).cpu())
    return torch.cat(out, dim=0)


def _eval_grad_norm(model, pts, device, batch=1 << 14):
    out = []
    for chunk in torch.split(pts, batch):
        chunk = chunk.to(device)
        chunk.requires_grad_(True)
        res = model(chunk)
        out.append(res['df'].norm(dim=-1).detach().cpu())
    return torch.cat(out, dim=0)


def extract_shell_mesh(model, resolution=100, domain=(-1.0, 1.0),
                        threshold_frac=0.05, device=None):
    """Marching-cubes extraction of the |df| ridge (see module docstring).

    threshold_frac: isosurface level as a fraction of the field's max value
    (lower => thicker shell capturing more of the decay tail, higher =>
    thinner shell hugging the ridge peak).

    Returns (vertices, faces) with vertices in `domain` units.
    """
    device = device or next(model.parameters()).device
    lo, hi = domain
    coords = torch.linspace(lo, hi, resolution)
    spacing = (coords[1] - coords[0]).item()
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    grad_norm = _eval_grad_norm(model, pts, device).numpy().reshape(
        resolution, resolution, resolution)

    level = threshold_frac * grad_norm.max()
    if not (grad_norm.min() < level < grad_norm.max()):
        raise RuntimeError(
            f"threshold {level} outside data range "
            f"[{grad_norm.min()}, {grad_norm.max()}]; adjust threshold_frac")

    verts, faces, _normals, _values = measure.marching_cubes(
        grad_norm, level=level, spacing=(spacing, spacing, spacing))
    verts = verts + lo
    return verts, faces
