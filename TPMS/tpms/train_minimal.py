"""Training loop for the neural minimal-surface solver with boundary C1
continuity constraints (paper Eq. 3, 4, 5).

The 1/8-cube domain [0,1]^3 is rescaled to [-1,1]^3 for training (matching
DeepCurrents' convention); its own two end-faces per axis, u_axis = -1 and
u_axis = +1, are exactly the faces subject to the per-axis symmetry
constraint (reflection 'r' or translation 't') used to synthesize the full
unit cell (see tpms.boundary.synthesize_unit_cell).

Modeling choice (paper does not fully spell out which face each constraint
applies to -- see TPMS/README.md "Fidelity notes"): for a reflection axis we
apply the orthogonality constraint (Eq. 5) at *both* end-faces of that axis;
for a translation axis we apply the periodic gradient-matching constraint
(Eq. 4) *between* the two end-faces of that axis, and since which of the +/-
sign cases in Eq. 4 applies depends on global orientation bookkeeping we
don't track, we take the min of the two (the network can satisfy whichever
is geometrically consistent).
"""
import numpy as np
import torch

from . import boundary as B
from . import topology as T
from .surface_model import DOMAIN, SurfaceModel, resample_closed_polyline


def to_training_domain(curve_01):
    """[0,1]^3 -> [-1,1]^3."""
    return 2.0 * curve_01 - 1.0


def build_bdry_tensor(topo, params, n_per_loop=200, device='cpu'):
    curves = B.build_boundary_curves(topo, params, n_samples=64)
    curves = [to_training_domain(c) for c in curves]
    resampled = [resample_closed_polyline(c, n_per_loop) for c in curves]
    bdry = torch.tensor(np.stack(resampled, axis=0), dtype=torch.float32,
                         device=device)
    return bdry


def sample_face_points(n, axis, value, device):
    pts = torch.empty(n, 3, device=device).uniform_(*DOMAIN)
    pts[:, axis] = value
    return pts


def continuity_losses(model, sym_code, n_bdry_samples, device):
    """Eq. 4 (translation) / Eq. 5 (reflection) losses, evaluated on the
    training domain's own end-faces per axis."""
    e_r = torch.zeros((), device=device)
    e_t = torch.zeros((), device=device)
    n_r, n_t = 0, 0

    for axis in range(3):
        if sym_code[axis] == 'r':
            for value in (-1.0, 1.0):
                p = sample_face_points(n_bdry_samples, axis, value, device)
                p.requires_grad_(True)
                out = model(p)
                e_r = e_r + out['df'][:, axis].abs().mean()
                n_r += 1
        else:  # 't'
            p_common = torch.empty(n_bdry_samples, 3, device=device).uniform_(*DOMAIN)
            p_lo = p_common.clone(); p_lo[:, axis] = -1.0
            p_hi = p_common.clone(); p_hi[:, axis] = 1.0
            p_lo.requires_grad_(True)
            p_hi.requires_grad_(True)
            out_lo = model(p_lo)
            out_hi = model(p_hi)
            diff = (out_lo['df'] - out_hi['df']).norm(dim=-1)
            summ = (out_lo['df'] + out_hi['df']).norm(dim=-1)
            e_t = e_t + torch.minimum(diff, summ).mean()
            n_t += 1

    if n_r > 0:
        e_r = e_r / n_r
    if n_t > 0:
        e_t = e_t / n_t
    return e_r, e_t


def sample_near_boundary(bdry, n, jitter_std, device):
    """Extra training points near the true boundary curve(s), jittered off
    the curve by a small amount. Uniform-only sampling (as in DeepCurrents)
    under-trains the network exactly where the field has the sharpest
    features (see README "Fidelity notes" on why |df| doesn't fully reach
    the curve without this); mixing in curve-proximal samples each
    iteration measurably tightens the fit there without changing the loss
    itself (Eq. 3 is still evaluated identically at every sampled point)."""
    L, N, _ = bdry.shape
    li = torch.randint(0, L, (n,), device=device)
    ni = torch.randint(0, N, (n,), device=device)
    pts = bdry[li, ni] + torch.randn(n, 3, device=device) * jitter_std
    return pts.clamp(*DOMAIN)


def train(topo, params, sym_code, n_iterations=100000, n_samples=2**12,
          n_bdry_samples=512, n_curve_samples=512, curve_jitter=0.05,
          lr=5e-4, lambda_r=1.0, lambda_t=1.0,
          rff_sigma=2, seed=1, device=None, log_every=1000,
          progress_cb=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    bdry = build_bdry_tensor(topo, params, device=device)
    model = SurfaceModel(bdry=bdry, rff_sigma=rff_sigma).to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.6)

    history = []
    for it in range(n_iterations):
        opt.zero_grad()
        x_uniform = torch.empty(n_samples, 3, device=device).uniform_(*DOMAIN)
        if n_curve_samples > 0:
            x_curve = sample_near_boundary(bdry, n_curve_samples, curve_jitter, device)
            x = torch.cat([x_uniform, x_curve], dim=0)
        else:
            x = x_uniform
        x.requires_grad_(True)
        out = model(x)
        loss_main = out['current'].norm(p=2, dim=-1).mean()

        e_r, e_t = continuity_losses(model, sym_code, n_bdry_samples, device)
        loss = loss_main + lambda_r * e_r + lambda_t * e_t

        loss.backward()
        opt.step()
        if it % 10000 == 0 and it > 0:
            sched.step()

        if it % log_every == 0:
            rec = dict(it=it, loss=loss.item(), main=loss_main.item(),
                       e_r=float(e_r), e_t=float(e_t))
            history.append(rec)
            if progress_cb:
                progress_cb(rec)

    return model, history
