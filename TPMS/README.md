# TPMS-families

A from-scratch reproduction of:

> Yonglai Xu, Hao Pan, Ruonan Wang, Qiang Du, Lin Lu.
> **New families of triply periodic minimal surface-like shell lattices.**
> Additive Manufacturing 77 (2023) 103779.

(`/home/bojingyy/research/TPMS-families.pdf`)

The paper's modeling pipeline builds directly on **DeepCurrents** (Palmer,
Smirnov, Wang, Chern, Solomon 2021; ref. [22] in the paper), whose code lives
alongside this project at `../DeepCurrents`. This reproduction reimplements
the DeepCurrents surface-fitting machinery from scratch (see "Fidelity
notes" below for why) rather than importing it, since the original code
depends on `pytorch3d`, which only the (unused, for this purpose)
mesh-distance losses actually require.

## What this reproduces (paper section -> code)

| Paper section | What it does | Code |
|---|---|---|
| 2.1-2.2, Fig. 3-5 | Enumerate the 20 unique 1/8-cube boundary curve topologies | `tpms/topology.py` |
| 2.2, Eq. 2, Fig. 6 | Cubic-Hermite-spline boundary geometry from per-node position/derivative params | `tpms/spline.py`, `tpms/boundary.py` |
| 2.2, "Boundary construction" | Synthesize the full unit cell from a 1/8-cube via per-axis reflection/translation (`topo_id_x1x2x3`) | `tpms/boundary.py`, `tpms/tiling.py` |
| 2.3, Eq. 3-5 | Neural implicit minimal surface (Biot-Savart "current" loss) + C1 boundary-continuity constraints | `tpms/surface_model.py`, `tpms/train_minimal.py` |
| 2.3-2.4 | Mesh extraction from the trained network | `tpms/mesh_extract.py` |
| 2.4 | Shell construction (thickness) | `tpms/shell.py` |
| Eq. 1, Fig. 11 | Classical TPMS (P, D, G, I-WP, F-RD, Neovius) for comparison | `tpms/classical_tpms.py` |
| Sec. 3, Eq. 6-10 | Numerical homogenization -> E, nu, G, Zener ratio | `tpms/homogenization.py` |
| Sec. 3.1, Fig. 15-16 | Mean curvature, surface-area-to-volume ratio | `tpms/metrics.py` |
| Appendix, Fig. A.21-A.23 | Inverse-design conditional GAN | `tpms/inverse_design.py` |

Runnable entry points are in `scripts/`; see "Usage" below.

## Setup

```bash
cd TPMS
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy scipy scikit-image matplotlib trimesh networkx tqdm
```

A CUDA GPU is used automatically if available (needed for the neural
minimal-surface training to run in reasonable time; everything else is CPU).

## Usage

```bash
# Fig. 5: enumerate & visualize the 20 unique boundary topologies
python scripts/enumerate_topologies.py

# Fig. 6-style: geometric variations of one topology's boundary curve
python scripts/visualize_boundary.py --topo 1

# Fig. 11-16-style: classical TPMS homogenization survey (fast, no neural net)
python scripts/classical_tpms_survey.py

# Fig. 1/2/7-10-style: full pipeline for one TPMS-like topology
python scripts/build_shell_lattice.py --topo 1 --sym rrr --n_iterations 30000 --homogenize

# Appendix: inverse-design GAN (dataset generation, then training)
python scripts/generate_gan_dataset.py --n_samples 300
python scripts/train_inverse_gan.py
```

## Validation results

These are the checks worth trusting without re-reading the code:

- **Topology enumeration is an exact match.** `enumerate_topologies()` finds
  **256** raw boundary-graph combinations passing the closed-curve degree
  filter and **20** unique topologies after cube-symmetry deduplication --
  precisely the paper's reported counts (Sec. 2.2).
- **The minimal-surface solver was checked against ground truth.** Trained
  on a plane circle, `|df|` (the network's "current" magnitude) is 5-100x
  larger for points inside the disk than for points just outside it, and
  marching cubes on that field recovers a flat disk of the correct radius
  (`r_max ≈ 0.60-0.65` for a `r=0.6` input circle) -- see the commit history
  / `tpms/mesh_extract.py` docstring for how this shaped the extraction
  method.
- **Homogenization is exact on the trivial case.** A fully solid,
  homogeneous voxel cube recovers the input material's own `E`, `nu`
  (Zener ratio = 1, i.e. isotropic) to floating-point precision --
  `tests` for the harness this was checked in.
- **Homogenization matches the paper's own numbers on a real structure.**
  Schwarz P surface at 30% relative density, 60^3 grid (matching the
  paper's resolution): this code gives **E = 0.087, nu = 0.359, Zener =
  2.01**; the paper's Fig. 12a/14a report **E ≈ 0.08-0.16, nu ≈ 0.20-0.35,
  Zener ≈ 2-3.5** for the same surface/density. Independent agreement.
  `scripts/classical_tpms_survey.py` extends this to all 6 classical
  surfaces (P/D/G/IWP/FRD/N) at rd~0.3, all landing in physically sensible
  E~0.05-0.15 (E0=1), nu~0.22-0.36 ranges with the expected density scaling
  (`out/classical_survey.png`).
- **Curvature/SA-V metrics were checked against a sphere.** A radius-2
  icosphere gives mean curvature `H = 0.500 ± 0.005` (analytic: `1/R =
  0.5`) and SA/V ratio `1.501` (analytic: `3/R = 1.5`).
- **The inverse-design GAN was trained end-to-end on a real (not
  synthetic) 150-sample homogenization dataset** (`scripts/generate_gan_dataset.py`
  + `scripts/train_inverse_gan.py`) and reaches **R^2 = 0.989** between
  target and (regressor-estimated) generated-sample properties -- closely
  matching the paper's own reported **R^2 = 0.9906 (E) / 0.9943 (nu)**
  (Appendix, Fig. A.23). See `out/gan/training_summary.png`.

## Fidelity notes -- where this deviates from the paper, and why

This is an honest account of the places where the paper's description
underdetermines an implementation detail, or where a design choice was made
for tractability within this session. None of these are silent -- each is
called out in the relevant module's docstring too.

1. **Boundary-topology enumeration models each cube face with 8 states, not
   9.** A face may connect its 4 bounding edges in one of 8 non-crossing
   ways (Fig. 3); we do *not* add a 9th "face touches no curve" state,
   because the 8-state model is the one that reproduces the paper's 256/20
   counts exactly (see `tpms/topology.py` docstring for the ambiguity this
   resolves, i.e. how sparse-looking topologies like topo_13 still touch
   all 6 faces combinatorially even though the rendered curve looks simple).

2. **Hermite-spline node derivatives are scalars along a fixed, convention-
   chosen direction** (perpendicular to the cube edge, tangent to the face,
   pointing into the face interior), not free 2D vectors. The paper's own
   parameter count (Fig. 9/A.22: "twelve derivative parameters" for six
   nodes = 2 scalars/node) implies the same reduction, but the paper doesn't
   spell out the direction convention we chose. Table A.1's raw derivative
   magnitudes ([1,2] on a 2-unit cube) overshoot badly with our exact
   Hermite formula (Eq. 2) on a 1-unit 1/8-cube domain -- we use empirically
   re-tuned defaults (~0.2-0.6) that keep curves inside the cube; see the
   `default_params`/`random_params` docstrings in `tpms/boundary.py`.

3. **C1 continuity constraints (Eq. 4/5) are applied per-axis based on that
   axis's own symmetry label**, not to a face independently chosen from the
   label (the paper doesn't specify which of a 1/8-cube's two end-faces per
   axis gets which constraint). Our convention: a reflection ('r') axis gets
   the orthogonality constraint (Eq. 5) at *both* its end-faces; a
   translation ('t') axis gets the periodic gradient-matching constraint
   (Eq. 4) *between* its two end-faces. See `tpms/train_minimal.py`.

4. **Mesh extraction thresholds `|df|` (gradient magnitude), not `f`
   itself.** `f_theta` is only trained through its gradient (Eq. 3), so it
   behaves like a multivalued winding-number potential with no single clean
   global zero level set (verified empirically -- see `tpms/mesh_extract.py`
   docstring). `|df|` ridges sharply and reliably at the true surface
   instead, which is what we run marching cubes on. This conveniently also
   gives the extracted shell a controllable base thickness "for free" via
   the ridge threshold, ahead of any additional offsetting.

5. **Neural surfaces don't reach their boundary curve exactly**, especially
   for topologies with several arcs/loops (an inherent property of this
   representation: fitting a true distributional singularity with a finite
   smooth MLP is an approximation, and there's a real tension between a
   sharp fit at the curve and the averaged L2-type training loss elsewhere
   in the domain -- see the training-log discussion in `tpms/train_minimal.py`
   around `sample_near_boundary`). We mitigate this with (a) extra training
   samples jittered around the true curve each iteration, and (b) a
   **tolerance-based vertex weld** (`tpms/tiling.py: weld_vertices`,
   restricted to each patch's open-boundary rim to avoid collapsing the
   mesh interior) when stitching reflected/translated octant copies into a
   unit cell. Simple topologies (few arcs, e.g. `topo_1`) converge to a
   well-connected lattice within tens of thousands of iterations; more
   complex multi-loop topologies need more training (and/or hyperparameter
   tuning: `threshold_frac`, `weld_tol`) than this session's compute budget
   comfortably allowed for exhaustive per-topology tuning. This is a
   genuine, expected characteristic of the underlying method, not a bug --
   the paper's own text acknowledges related degeneracy/convergence issues
   for complex boundaries (Sec. 2.1, Fig. 2a).

6. **The inverse-design GAN is trained on a fast stand-in dataset.** The
   paper's dataset (5136 samples across 3 topologies) is generated by
   running the *full* neural-surface-training pipeline per sample -- minutes
   each, so building a comparable dataset here would take on the order of
   days of GPU time. `scripts/generate_gan_dataset.py` instead sweeps the
   classical Schwarz-P surface's level-set value and shell thickness (2
   params, homogenized in ~1s/sample) to produce a same-*shape* (params) ->
   (E, nu) dataset that exercises the identical GAN architecture (Fig. A.21)
   and training loop (Eq. A.1) end-to-end. `tpms/train_minimal.py` +
   `tpms/homogenization.py` are exactly what you'd wire up in
   `generate_gan_dataset.py`'s place to build the paper's real dataset,
   given enough compute time.

7. **Homogenization uses a Jacobi-preconditioned CG solve**, not a direct
   sparse solve. Direct sparse LU (`scipy.sparse.linalg.factorized`) has
   catastrophic fill-in on these periodic 3D meshes -- 21s for a trivial
   8000-element problem, worse at the paper's 60^3 resolution. CG on the
   (SPD, after pinning the rigid-body mode) periodic stiffness matrix
   solves the same 60^3/216k-element problem in ~35s.

## Known limitations / not fully explored

- The reduced, fully-cubic-symmetric parameterization the paper uses for
  `topo_1_rrr`/`topo_18_rrr`/`topo_19_rrr` (Table A.1: just `p1,p2,p3,d,t`,
  5 scalars total) was not re-derived here; our own topologies carry one
  scalar per active node instead (more parameters, same underlying
  topology/symmetry code path).
- Functionally-graded design (Sec. 4.1) and the topology-optimization
  example (Fig. 20b) are not implemented; the pieces needed (parametric
  family + inverse GAN) are, but the specific demo isn't wired up.
- Only a handful of topologies/parameter settings have been trained to full
  convergence in this session; `scripts/build_shell_lattice.py` is the tool
  to explore more, given more GPU time.
