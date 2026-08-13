"""Generate a (geometric params) <-> (E, nu) dataset for the inverse-design
GAN (paper Appendix). Uses the classical Schwarz-P family (params: level-set
value c, shell thickness t) as a fast stand-in so the full inverse-design
pipeline can be demonstrated end-to-end within this session's time budget --
each sample only costs one voxelization + homogenization solve (~0.1-1s at
this script's grid resolution), unlike a genuine TPMS-like sample from
tpms.train_minimal, which needs a full neural-surface training run (minutes)
per sample the way the paper's own 5136-sample dataset was built. See
TPMS/README.md "Fidelity notes" for how to swap in the real generator.

Usage:
    python scripts/generate_gan_dataset.py --n_samples 300 --out out/gan_dataset.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm

from tpms import classical_tpms as C
from tpms import homogenization as H


def sample_one(c, thickness, resolution, period):
    coords = np.linspace(0, period, resolution, endpoint=False)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    phi = C.phi_P(X, Y, Z)
    density = (np.abs(phi - c) < thickness).astype(float)
    rd = density.mean()
    if rd < 0.02 or rd > 0.85:
        return None
    if not H.is_connected(density):
        return None
    try:
        CH = H.homogenize(density, E0=1.0, nu0=0.3, cell_size=period, maxiter=3000)
    except RuntimeError:
        return None
    props = H.elastic_properties(CH)
    return rd, props


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=300)
    parser.add_argument('--resolution', type=int, default=24)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=os.path.join(
        os.path.dirname(__file__), '..', 'out', 'gan_dataset.npz'))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    period = 2 * np.pi

    params, ys = [], []
    pbar = tqdm(total=args.n_samples)
    while len(params) < args.n_samples:
        c = rng.uniform(-1.3, 1.3)
        thickness = rng.uniform(0.15, 0.55)
        result = sample_one(c, thickness, args.resolution, period)
        if result is None:
            continue
        rd, props = result
        if not (np.isfinite(props['E']) and np.isfinite(props['nu'])
                and 0 < props['E'] < 2 and 0 < props['nu'] < 0.5):
            continue
        params.append([c, thickness])
        ys.append([props['E'], props['nu']])
        pbar.update(1)
    pbar.close()

    params = np.array(params)
    ys = np.array(ys)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, params=params, y=ys,
             param_names=['c', 'thickness'], y_names=['E', 'nu'])
    print(f"Saved {len(params)} samples to {args.out}")
    print(f"E range [{ys[:,0].min():.4f}, {ys[:,0].max():.4f}]  "
          f"nu range [{ys[:,1].min():.4f}, {ys[:,1].max():.4f}]")


if __name__ == '__main__':
    main()
