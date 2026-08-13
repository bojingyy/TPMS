"""Inverse-design conditional GAN (paper Appendix, Fig. A.21, Eq. A.1):
learns a mapping from target material properties (E, nu) to the geometric
parameters of a TPMS-like shell-lattice unit cell.

Architecture follows Fig. A.21 exactly (FC + LeakyReLU stacks); `param_dim`
generalizes their fixed 5 (p1,p2,p3,d,t) -- which is specific to the reduced,
fully cubic-symmetric parameterization of topo_1_rrr/topo_18_rrr/topo_19_rrr
that Table A.1 uses -- to whatever parameter-vector length a given topology
+ symmetry choice actually has in this codebase (see README "Fidelity
notes": we did not re-derive that specific cubic-symmetry parameter
reduction, so our own generated datasets have more free parameters per
node/topology than their 5).
"""
import numpy as np
import torch
import torch.nn as nn


def _mlp(dims, final_activation=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LeakyReLU(0.2))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, param_dim, cond_dim=2, noise_dim=3):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = _mlp([cond_dim + noise_dim, 128, 256, 512, 1024, param_dim])

    def forward(self, y, z=None):
        if z is None:
            z = torch.randn(y.shape[0], self.noise_dim, device=y.device)
        return self.net(torch.cat([y, z], dim=-1))


class Discriminator(nn.Module):
    def __init__(self, param_dim, cond_dim=2):
        super().__init__()
        self.net = _mlp([cond_dim + param_dim, 1024, 512, 256, 128, 1],
                         final_activation=nn.Sigmoid())

    def forward(self, x, y):
        return self.net(torch.cat([y, x], dim=-1))


class AuxRegressor(nn.Module):
    def __init__(self, param_dim, cond_dim=2):
        super().__init__()
        self.net = _mlp([param_dim, 1024, 512, 256, 128, cond_dim])

    def forward(self, x):
        return self.net(x)


def train_cgan(x_data, y_data, param_dim, cond_dim=2, noise_dim=3,
                n_iterations=10000, batch_size=32, lr=2e-4, lam=20.0,
                device=None, log_every=1000, progress_cb=None):
    """x_data: (N, param_dim) geometric params. y_data: (N, cond_dim) (E,nu),
    both already normalized to a comparable scale (see README)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    x_data = torch.as_tensor(x_data, dtype=torch.float32, device=device)
    y_data = torch.as_tensor(y_data, dtype=torch.float32, device=device)
    n = x_data.shape[0]

    G = Generator(param_dim, cond_dim, noise_dim).to(device)
    D = Discriminator(param_dim, cond_dim).to(device)
    R = AuxRegressor(param_dim, cond_dim).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_r = torch.optim.Adam(R.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    history = []
    for it in range(n_iterations):
        idx = torch.randint(0, n, (batch_size,), device=device)
        x_real, y_real = x_data[idx], y_data[idx]

        # --- Discriminator step ---
        with torch.no_grad():
            x_fake = G(y_real)
        d_real = D(x_real, y_real)
        d_fake = D(x_fake, y_real)
        loss_d = bce(d_real, torch.ones_like(d_real)) + \
            bce(d_fake, torch.zeros_like(d_fake))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        # --- Auxiliary regressor step (real data, Eq. A.1's E_real term) ---
        y_pred_real = R(x_real)
        loss_r = (y_real - y_pred_real).abs().mean()
        opt_r.zero_grad(); loss_r.backward(); opt_r.step()

        # --- Generator step ---
        x_fake = G(y_real)
        d_fake_g = D(x_fake, y_real)
        y_pred_fake = R(x_fake)
        loss_g_adv = bce(d_fake_g, torch.ones_like(d_fake_g))
        loss_g_aux = (y_real - y_pred_fake).abs().mean()
        loss_g = loss_g_adv + lam * loss_g_aux
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if it % log_every == 0:
            rec = dict(it=it, loss_d=loss_d.item(), loss_g=loss_g.item(),
                       loss_r=loss_r.item(), loss_g_aux=loss_g_aux.item())
            history.append(rec)
            if progress_cb:
                progress_cb(rec)

    return G, D, R, history


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum()
    return 1 - ss_res / ss_tot
