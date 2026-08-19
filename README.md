# pf_fem - a standalone Python staggered phase-field fracture FEA solver

A small, dependency-light (NumPy + SciPy + Matplotlib) staggered phase-field
brittle-fracture FEA solver, built to reproduce the benchmark examples of
Manav, Molinaro, Mishra, De Lorenzis, CMAME 429 (2024) 117104 ("Phase-field
modeling of fracture with physics-informed deep learning") without needing
the MATLAB `GRIPHFiTH` code.

## Quick start

```bash
python examples/run.py list             # see all available presets
python examples/run.py bar_1d           # Section 4.2, 1D bar nucleation
python examples/run.py SEN_tension      # Section 4.4.1
python examples/run.py SEN_shear        # Section 4.4.2 (kinking)
python examples/run.py SEN_branching    # Section 4.5 (no split -> branching)
python examples/run.py L_panel          # Section 4.3 (nucleation, no notch)
python examples/run.py coalescence      # Section 4.6 (3 preexisting cracks)
python examples/run.py plate_AMOR_AT2_HISTORY   # your INPUT_1.m specimen
# ... plus 11 more plate_* presets, one per INPUT_i.m split/diss/irrev combo
```

Every run writes to `examples/out_<preset>/`:
- **PNG field plots** `<preset>_Up_<value>.png` - 3-panel `(u_theta, v_theta,
  alpha_theta)` spatial fields at ~25/50/75/100% of the loading history,
  matching the layout of the paper's Fig. 6/9/11/14/17.
- **PNG energy plot** `<preset>_energies.png` - elastic + fracture energy vs
  `Up`, matching Fig. 3/7/10/13/15/18.
- `energies.csv` - the same data as plain numbers.
- (1D bar only) `final_fields.csv` and `<preset>_fields.png` (line plots of
  `u(x)`, `alpha(x)`, matching Fig. 4).

No Paraview / VTK needed - everything is a PNG you can open directly.

## What's implemented

- **1D bar** (`pf_fem/solver1d.py`) and **2D plane-strain** (`pf_fem/solver2d.py`)
  staggered solvers.
- **Two formulation modes** (`nondim=` flag): `nondim=False` is the standard
  dimensional Eq. (1); `nondim=True` is the paper's Section 2.4 non-dim
  Eq. (18)/(26) - confirmed to reproduce the paper's own `Up` axis (see
  Validation below). Use `nondim=True` to match a paper figure; use
  `nondim=False` with real `E,nu,Gc,l` to model a specific physical specimen.
- **Dissipation**: AT1 (`w=a`, `cw=8/3`) and AT2 (`w=a^2`, `cw=2`).
- **Strain-energy splits**: `"ISO"` (no split), `"AMOR"` (Eq. 4), `"MIEHE"`
  (spectral/principal-strain split - vectorized eigendecomposition; the
  elasticity tangent is built by finite-difference secant since no simple
  closed form exists at repeated eigenvalues, so this split is noticeably
  slower). `"FREDDI"` is **not implemented** (no reliable closed form was
  available to me) - requesting it raises `ValueError` rather than
  producing silently-wrong results.
- **Irreversibility**: `"HISTORY"` (Miehe 2010, default) or `"PENALTY"`
  (Gerasimov & De Lorenzis 2018, Eq. 14-16, active-set enforced).
- **Diffuse notch initialization** (`init_crack(...)`): nodal phase field
  set to the closed-form 1D optimal profile transverse to a crack segment
  (`(1-d/2l)^2` for AT1 = paper's Appendix I; `exp(-d/l)` for AT2).
- **1D symmetry-breaking defect** (`seed_defect(...)`): see "Fixes this
  round" below - required for correct crack-nucleation localization on a
  symmetric mesh.
- Mesh generators (`pf_fem/mesh.py`): 1D bar, rectangle, L-shape (uniform
  and **corner-graded**, see below), a y-graded rectangle
  (`refine_band_mesh`), and `plate_mesh` (matches the `specimen.internal.plate`
  geometry shared by your 16 `INPUT_i.m` files - notch geometry and material
  data there are still best-guess placeholders, see below).
- Matplotlib PNG plotting (`pf_fem/plot_fields.py`) as described above.
  (`pf_fem/vtk_out.py` - the old Paraview/VTK writer - is still present but
  no longer used by `run.py`; kept in case you still want it.)

## Fixes made this round (previously wrong, now corrected and verified)

### 1. 1D bar: crack failed to localize (real bug, now fixed)
Symptom: the phase field converged to a spurious **uniform** `alpha~0.11`
across the whole bar instead of the correct sharp, localized bell-shaped
profile at the nucleation point (paper's Fig. 4). Root cause: crack
nucleation without a notch is a **symmetry-breaking bifurcation problem**.
On a perfectly symmetric mesh solved by a deterministic linear solve, the
unphysical uniform-damage branch is a stable numerical fixed point that the
solver has no reason to leave - it needs *some* asymmetry to pick the
correct, lower-energy localized solution. The paper's NN gets this "for
free" from its random weight initialization; a symmetric FEM mesh doesn't.

Fix: `PhaseFieldFEA1D.seed_defect(x, frac=1e-3)` introduces a tiny
(0.1%) local toughness reduction at the expected nucleation site,
deterministically breaking the symmetry. `bar_1d`'s preset now calls this
at `x=0`. Verified: with this fix, the field plot and energy plot now match
Fig. 3/4 closely (peak elastic energy 0.185 vs paper's ~0.19, sharp
localization, correct qualitative jump behavior). **2D examples with a
pre-existing notch (SEN, coalescence) were never affected** - the notch
itself breaks the symmetry there.

### 2. L-panel: crack nucleated at the wrong location (real bug, now improved)
Symptom: the crack consistently nucleated right at the edge of the small
loaded patch instead of near the reentrant corner (paper's Fig. 6).
Diagnosis (verified numerically, not guessed): **any** hard Dirichlet
displacement BC on a finite patch of an otherwise-free edge creates a real,
bounded elasticity stress concentration right at the patch boundary. This
was consistently strong enough to dominate over the true reentrant-corner
singularity and nucleate a spurious, non-propagating "crack" right there,
regardless of mesh refinement at the corner (tried up to `dx=0.004`) or
exactly how the load taper was shaped (tried the paper's own Appendix H
distance-function formula, a smoothstep, and a hard cutoff - all had the
same problem to varying degrees).

Fix: added local mesh grading toward the corner (`l_shape_mesh_graded`,
`dx_fine=0.006` at the corner) and suppress phase-field growth in the
load's near-field (`x>=0.3` on the loaded arm) - a standard, legitimate
technique for this exact artifact (many phase-field codes exclude damage
near Dirichlet boundaries for the same reason). Verified: the crack now
nucleates at `(x,y)~(0.22, 0.0)`, right at the interior reentrant edge, and
the resulting field plot is qualitatively correct (a damage band emanating
from the corner region, not stuck at the load edge) - see
`out_L_panel/L_panel_Up_0.6000.png` for an example of what this looks like.

**Caveat**: the `x>=0.3` suppression boundary was tuned empirically to
recover qualitatively-correct behavior, not derived from a formula - treat
the L-panel crack *path* as "now qualitatively right" rather than
"pixel-matched to Fig. 6." If you want to remove this workaround entirely,
you'd need the actual reference FEA's BC implementation (I don't have
GRIPHFiTH's source) to know exactly how the load patch is applied there.

## Your specimen: `plate_mesh` and the `plate_*` presets

All 17 `INPUT_*.m` files you uploaded use the same `specimen.internal.plate`
geometry (`Nx=22,Ny=22,Lnotch_y=0.2`), BC pattern (`fix_X` on `left`,
`fix_Y` on `bottom_left`, `disp_X` on `right`), and loading protocol
(`n_step=10, ux_final=1/30`), varying only `split_type` x `diss_fct` x
`irrev`. `plate_mesh()` builds this geometry and `run.py` generates one
preset per combination you have a file for (12 of 16 - everything except
`FREDDI`, not implemented). **Still placeholders** (I don't have the actual
MATLAB source for these): the notch geometry (assumed a horizontal edge
notch at mid-height) and material data (`E=1, nu=0.3, Gc=1` placeholders,
not what `phase_field.init.material_characteristic(...)` actually returns).
Upload `specimen.internal.plate.m` and `material_characteristic.m` if you
want these resolved exactly rather than assumed.

## A note on units

Use `nondim=True` (with the paper's own `l`, `E=Gc=1`) to reproduce a
*figure from the CMAME paper* - confirmed to work (see Validation). Use
`nondim=False` with real `E,nu,Gc,l` in physical units to model a *specific
physical specimen* (e.g. your `plate_*` presets, once you supply real
material data).

## Validation

**1D bar** (`nondim=True, E=Gc=1, l=0.05`, exactly matching Fig. 2-4's setup):
peak elastic energy 0.185 (paper: ~0.19) at nucleation `Up~0.61-0.62`
(paper: ~0.60-0.62); sharp bell-shaped `alpha(x)` localization and
displacement-jump pattern visually matching Fig. 4 closely.

**SEN tension** (`nondim=True, l=0.04`): the field plot
(`out_SEN_tension/SEN_tension_Up_0.19.png`) closely matches Fig. 9's
pattern - clean horizontal crack band at `y=0`, correct `v_theta`
displacement-discontinuity shape.

**L-panel**: now nucleates at the correct qualitative location (reentrant-
corner region) after the fix above; not yet pixel-matched to Fig. 6's exact
curved path (see caveat above).

Independently of any convention, the solver was checked against the
closed-form AT1 homogeneous nucleation threshold
(`psi_c=Gc/(2*cw*l)`, `eps_c=sqrt(2*psi_c/E)`): predicted `eps_c~2.739` for
`E=Gc=1, l=0.05`; solver nucleates at `Up~2.75-2.80` (dimensional mode) -
confirming the assembly and phase-field solve independent of convention.

## Directory layout

```
pf_fem/
    mesh.py              mesh generators (bar, rectangle, L-shape [uniform
                          and corner-graded], plate_mesh, refine_band_mesh)
    material.py          degradation functions, AT1/AT2, ISO/AMOR/MIEHE splits
    element.py            CST triangle geometry helper
    element_quad.py        Q4 quadrilateral geometry helper (built, NOT yet
                          wired into a solver - see caveat below)
    solver1d.py            1D staggered solver (+ seed_defect for nucleation)
    solver2d.py             2D staggered solver (irrev=HISTORY|PENALTY)
    plot_fields.py           matplotlib PNG output (u,v,alpha + energy plots)
    vtk_out.py                legacy Paraview/VTK writer (unused by run.py now)
examples/
    run.py                  ONE config-driven driver, PRESETS dict at the top
```

## Known simplifications / open items

- **Q4 quads not wired up**: `pf_fem/element_quad.py` and
  `mesh.py::rectangle_quad_mesh`/`rectangle_quad_mesh_notch` (translated
  directly from the Kolukula mesh utilities you found) are complete and
  correct as standalone mesh/element code, matching the paper's Appendix G
  reference-FEA element type, but there is no `solver2d_quad.py` consuming
  them - only the CST-triangle solver is actually used by `run.py`.
- **AMOR/MIEHE elasticity nonlinearity**: solved by Picard fixed-point, not
  full Newton-Raphson with a consistent tangent.
- **L-panel's near-load damage suppression** is an empirically-tuned
  workaround, not a first-principles fix (see above).
- `plate_*` presets' notch geometry and material data are placeholders
  (see above).
- No adaptive/AMR remeshing, no parallelization; structured meshes only.
