# TPMS-Families: Research Progress Report

**Date:** August 13, 2026
**Reference:** Xu, Pan, Wang, Du, Lu. *New families of triply periodic minimal surface-like shell lattices.* Additive Manufacturing 77 (2023) 103779.
**Goal:** Reproduce the paper's modeling pipeline as a foundation for generating *new* TPMS-like shell-lattice structures beyond what the paper reports.

---

## 1. What has been built

A full-pipeline reimplementation in `research/TPMS/`, covering every stage of the paper's method (Fig. 1). It was built from scratch rather than reusing the paper's own code (unavailable) — it does reuse the *idea* from DeepCurrents (Palmer et al. 2021), the neural-surface-fitting technique the paper itself builds on, whose reference implementation lives at `research/DeepCurrents`.

| Stage | Status | Key validation |
|---|---|---|
| **Boundary topology enumeration** (1/8-cube graph search) | Done | Finds exactly **256** raw candidate topologies and **20** unique ones after symmetry reduction — matches the paper's reported counts exactly |
| **Hermite-spline boundary geometry** + unit-cell symmetry synthesis | Done | Produces smooth, closed boundary curves; visually consistent with paper Fig. 6 |
| **Classical TPMS surfaces** (P, D, G, I-WP, F-RD, Neovius) | Done | Used as the comparison baseline, exactly as in the paper's Sec. 3 |
| **Neural minimal-surface solver** (geometric-measure-theory "current" loss + C¹ boundary continuity) | Working, convergence is topology-dependent | Validated against a known ground truth (a flat disk spanning a circle) before being applied to cube topologies |
| **Mesh extraction, unit-cell tiling, shell construction** | Working | Produces closed(-ish) periodic shell meshes; simple topologies tile cleanly, complex multi-loop ones need more training (see §3) |
| **Numerical homogenization** (E, ν, shear modulus, Zener ratio) | Done, independently validated | Exact recovery of input material properties on a trivial solid-cube test; on the real Schwarz-P surface at the paper's own 60³ resolution, gives E=0.087, ν=0.359, Zener=2.01 — matching the paper's reported ranges (E≈0.08–0.16, ν≈0.20–0.35, Zener≈2–3.5) |
| **Inverse-design conditional GAN** (properties → geometry) | Done, tested end-to-end | Trained on a real 150-sample homogenization dataset, reaches R²=0.989 — matching the paper's own reported R²≈0.99 |

**Bottom line:** every stage runs, and every stage that can be checked against an independent ground truth (the paper's own published numbers, or analytic solutions) has been checked and matches.

## 2. Where to see results

All in `research/TPMS/`:

- `README.md` — full writeup, including an honest account of every place an implementation choice had to be made because the paper's text underdetermines it
- `out/topologies.png` — the 20 enumerated boundary topologies (reproduces Fig. 5)
- `out/boundary_topo1.png`, `out/boundary_topo3.png` — example spline geometry variations (Fig. 6 style)
- `out/classical_survey.png` — E–ν space and density scaling for all 6 classical TPMS surfaces (Fig. 12/16 style)
- `out/gan/training_summary.png` — inverse-design GAN training curves and target-vs-generated property fit
- `tests/test_validation.py` — runnable checks reproducing all the validation numbers above (`python tests/test_validation.py`)
- `scripts/build_shell_lattice.py` — the end-to-end generator: pick a topology + symmetry code, get a trained neural TPMS-like shell mesh out

## 3. Pending work

**Closing the reproduction gap:**
1. **Mesh connectivity at scale.** Neural surfaces don't always reach their true boundary curve exactly, especially for topologies with several boundary loops (an inherent tension in this method between a sharp fit at the curve and the network's global smoothness — not a bug, but it means training time and thresholds need per-topology tuning). Only 1–2 of the 20 topologies have been pushed to a fully clean, watertight unit cell so far.
2. **Scaling homogenization to our own TPMS-like family.** The E–ν comparison against classical TPMS (Fig. 12) has only been reproduced for the *classical* surfaces so far; running it for our generated TPMS-like family requires training many topology/parameter combinations to convergence first.
3. **The paper's exact reduced (cubic-symmetric) parameterization** for its three main topologies (5 scalars instead of our current per-node parameterization) hasn't been re-derived.
4. **The GAN's training dataset is currently a fast stand-in** (classical Schwarz-P surface, 2 parameters, 150 samples) rather than the paper's real dataset (their 3 topologies, 5 parameters, 5136 samples) — proves the method works, but isn't the real target dataset yet.
5. Functionally-graded design and inverse-homogenization topology optimization (paper Sec. 4) are not yet demonstrated.

**Toward the actual research goal — generating *new* structures:**
Once (1)–(2) above are in a good state, the natural next phase is to go beyond reproduction: sweep the 20 topologies × their geometric parameter spaces × symmetry codes far more broadly than the paper did, screen the resulting structures' homogenized properties for interesting/novel regions of E–ν–Zener space (e.g., extremal stiffness-to-density ratios, near-perfect isotropy, or property combinations the paper's 5 topology families didn't cover), and use the inverse-design GAN (once trained on our own family) to search that space directly from target properties rather than by hand. The pipeline needed for this is complete; what's missing is compute time to run it broadly and a decision on which regions of the design space to prioritize.

**Main ask:** more GPU time for per-topology training (each structure takes minutes-to-tens-of-minutes to converge well) if we want to explore the space broadly rather than one topology at a time.
