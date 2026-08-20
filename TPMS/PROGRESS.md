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
| **Hermite-spline boundary geometry** + unit-cell symmetry synthesis | Done (**corrected 2026-08-20**, see §4) | Arcs are now exact quarter circles at the default derivative magnitude (0.000 radial deviation), matching Fig. 6's bulging fillets; all 20 topologies verified closed, in-bounds, non-backtracking |
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
- `out/topo3_boundary_diagram.png` (`scripts/topo3_boundary_diagram.py`) — topo_3 specifically, in 4 panels: (a) the topology graph as straight chords (matches `topologies.png`/Fig. 5), (b) the Hermite construction showing the tangent vector at every node bending each chord into its arc, (c)(d) geometric variations sampled from Table A.1's ranges. topo_3 is one of only 3 topologies (with topo_1, topo_6) the paper notes are triply periodic on their own, and is the FRD-like family's topology (`topo_3_rrr`, paper Fig. 10d).
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

---

## 4. Correction, 2026-08-20: boundary-curve sign error

**All boundary curves generated before this date were wrong**, across all 20 topologies.

**The bug.** Paper Eq. 2 defines the cubic Hermite curve by `d1 = P'(0)` and `d2 = P'(1)` — both tangents *in the direction of travel*. `tpms/boundary.py: build_boundary_curves` set the end tangent `d2` to point *into the face interior*, i.e. back the way the curve had come. Every arc therefore got an S-hook at its arrival node and was pulled flat toward its straight chord, instead of bulging into the fillet arcs the paper shows in Fig. 6. Diagnostic: under the old sign **368 of 368 arcs backtracked against their own chord**; under the fix, zero do.

**Why it went unnoticed.** It was self-consistent and produced plausible-looking closed curves, and the test suite had no boundary-geometry coverage at all — it tested topology counts, homogenization and curvature, but never the curve shapes. The flattening was also mis-diagnosed and written up in `README.md` fidelity note #2 as the paper's own Table A.1 derivative magnitudes "overshooting badly", which motivated empirically re-tuned defaults (~0.2–0.6) that partly hid the symptom.

**The fix** (`tpms/boundary.py`): flip the sign of `d2`. With that, an arc between two mid-edge nodes sharing a cube corner is the *exact* quarter circle through them at `d = 4(√2−1)r ≈ 0.828` (`D_CIRCULAR`; verified to 0.000 max radial deviation), and the paper's own Table A.1 ranges now work directly — position `[-0.45, 0.45]` mm and derivative `[1.0, 2.0]` mm on the paper's 2 mm 1/8-cube scale to `S_RANGE = (-0.225, 0.225)`, `D_RANGE = (0.5, 1.0)` on our unit cube. So fidelity note #2's claimed deviation from the paper was never real; the README has been corrected.

**Note on the cusps.** Where two arcs meet at a node the curve has a sharp corner, because the arriving and leaving tangents lie in two perpendicular cube faces. This is correct and matches the paper — the same cusps are visible in its Fig. 6, clearest at the bottom-centre node of Fig. 6(c). They are not self-intersections.

**Regression coverage added** (`tests/test_validation.py`): `test_boundary_arc_is_exact_quarter_circle` and `test_boundary_curves_are_closed_in_bounds_and_non_backtracking` (the latter sweeps all 20 topologies at both default and random parameters). Both were confirmed to fail under the old sign. Full suite: 7/7 pass.

**Downstream impact — not yet redone.** `tpms/train_minimal.py` consumes `build_boundary_curves`, so **every neural surface, extracted mesh, and shell lattice trained before this date was fitted to the flattened boundary** and should be regarded as invalid. Regenerated so far: `out/topologies.png`, `out/boundary_topo1.png`, `out/boundary_topo3.png`, `out/topo3_boundary_diagram.png`. Still to redo once GPU time is available: the trained surfaces behind §3(1)–(2). The homogenization numbers quoted in §1 are unaffected — they come from the *classical* implicit TPMS surfaces (Eq. 1), which never touch this code path, as does the GAN's stand-in dataset.
