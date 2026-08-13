"""Train the inverse-design conditional GAN (paper Appendix, Fig. A.21) on a
dataset produced by generate_gan_dataset.py, and evaluate the property
prediction error via the R^2 score (paper Eq. A.2, Fig. A.23).

Usage:
    python scripts/train_inverse_gan.py --data out/gan_dataset.npz --out out/gan
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
import torch

from tpms import inverse_design as ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join(
        os.path.dirname(__file__), '..', 'out', 'gan_dataset.npz'))
    parser.add_argument('--out', default=os.path.join(
        os.path.dirname(__file__), '..', 'out', 'gan'))
    parser.add_argument('--n_iterations', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.data)
    params, y = data['params'], data['y']
    param_names, y_names = list(data['param_names']), list(data['y_names'])

    # Normalize both spaces to ~[0,1] for stable GAN training.
    p_lo, p_hi = params.min(0), params.max(0)
    y_lo, y_hi = y.min(0), y.max(0)
    params_n = (params - p_lo) / (p_hi - p_lo)
    y_n = (y - y_lo) / (y_hi - y_lo)

    rng = np.random.default_rng(args.seed)
    n = len(params)
    idx = rng.permutation(n)
    n_test = int(n * args.test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    G, D, R, history = ID.train_cgan(
        params_n[train_idx], y_n[train_idx], param_dim=params.shape[1],
        cond_dim=y.shape[1], n_iterations=args.n_iterations,
        batch_size=args.batch_size, log_every=max(1, args.n_iterations // 10))
    for r in history:
        print(r)

    device = next(G.parameters()).device
    G.eval(); R.eval()
    with torch.no_grad():
        y_test = torch.tensor(y_n[test_idx], dtype=torch.float32, device=device)
        x_gen = G(y_test)
        y_gen_pred = R(x_gen).cpu().numpy()  # R's estimate of the generated sample's property
    y_test_np = y_n[test_idx]
    r2 = ID.r2_score(y_test_np, y_gen_pred)
    print(f"R^2 (generated-sample property vs target, normalized space): {r2:.4f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'R': R.state_dict(),
                'p_lo': p_lo, 'p_hi': p_hi, 'y_lo': y_lo, 'y_hi': y_hi,
                'param_names': param_names, 'y_names': y_names},
               os.path.join(args.out, 'model.pt'))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot([h['it'] for h in history], [h['loss_d'] for h in history], label='D')
    axes[0].plot([h['it'] for h in history], [h['loss_g'] for h in history], label='G')
    axes[0].plot([h['it'] for h in history], [h['loss_r'] for h in history], label='R')
    axes[0].legend(); axes[0].set_xlabel('iter'); axes[0].set_title('cGAN losses')

    axes[1].scatter(y_test_np[:, 0], y_gen_pred[:, 0], s=10)
    lims = [0, 1]
    axes[1].plot(lims, lims, 'k--', linewidth=1)
    axes[1].set_xlabel(f'target {y_names[0]} (norm.)')
    axes[1].set_ylabel(f'R(G(y)) predicted {y_names[0]} (norm.)')
    axes[1].set_title(f'R^2 = {r2:.3f}')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'training_summary.png'), dpi=150)
    print(f"Saved model + summary to {args.out}")


if __name__ == '__main__':
    main()
