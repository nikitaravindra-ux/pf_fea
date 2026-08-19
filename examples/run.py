"""
Single, config-driven driver for all examples - mirrors your MATLAB
INPUT_1.m -> main_brittle_fracture.m -> solve_brittle_fracture.m structure,
but as one Python file with a PRESETS table instead of separate INPUT_i.m
files. To run a different example, change PRESET_NAME (or add a new entry
to PRESETS) instead of editing/duplicating solver code.

Usage:
    python run.py bar_1d
    python run.py SEN_tension
    python run.py SEN_shear
    python run.py SEN_branching
    python run.py L_panel
    python run.py coalescence
(or just edit PRESET_NAME below and run `python run.py`)

Optional env vars:
    L_SCALE=<float>   multiply the preset's l by this factor (default 1.0,
                       i.e. the paper's own resolution). Use e.g. L_SCALE=2
                       or 4 for a much faster, coarser dev/smoke-test run;
                       see the note above PRESETS below for what this does
                       and does NOT change.

*** NO CHECKPOINTING ***
An earlier version of this file wrote/resumed from a checkpoint.npz so a
killed/slow run could be restarted mid-way. That has been removed entirely
per explicit request - every run here is a single, complete pass from
Up=0 to Up_max. If a run is interrupted, just start it again from scratch.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pf_fem.mesh import (bar_mesh_1d, refine_band_mesh, refine_band_mesh_quad,
                          l_shape_mesh, l_shape_mesh_graded, plate_mesh)
from pf_fem.solver1d import PhaseFieldFEA1D
from pf_fem.solver2d import PhaseFieldFEA, BoundaryConditions
from pf_fem.solver2d_quad import PhaseFieldFEAQuad, BoundaryConditions as BoundaryConditionsQuad
from pf_fem.plot_fields import plot_fields_png, plot_energy_png

# =========================================================================
# PRESETS  (edit / add entries here - this is the "INPUT_i.m" equivalent)
# =========================================================================
PRESETS = {

    "bar_1d": dict(
        dim=1, nondim=True, E=1.0, Gc=1.0, l=0.05, diss_fct="AT1",
        n_el_per_l=5, domain=(-0.5, 0.5),
        crack=None,                       # homogeneous nucleation, no notch
        defect_x=0.0,                        # symmetry-breaking seed (see seed_defect)
        loading=dict(kind="uniaxial", Up_max=0.65, n_step=65),
    ),

    # All 2D "paper" presets below use irrev="HISTORY" (Miehe et al. 2010:
    # single linear solve + max(alpha,alpha_old) projection each staggered
    # iteration), NOT "PENALTY", EXCEPT SEN_tension and SEN_shear below,
    # which now use the paper's OWN reference-FEA recipe exactly (Appendix
    # G): bilinear Q4 elements, 2x2 Gauss quadrature, PENALTY
    # irreversibility (Eq. 14-16, TOL_ir=1e-3). This was verified to
    # actually converge (not oscillate - see solver2d_quad.py's module
    # docstring and PENALTY active-set fix) by running it through BOTH
    # smooth elastic loading AND a real crack-jump event on a coarse mesh:
    # every single load step converged, INCLUDING the jump itself (fracture
    # energy 2.26e-2 -> 6.44e-2, crack extending from x=0 to x=+0.5 in two
    # load steps), taking at most ~1.1s per step even at the hardest point
    # - no bisection/adaptive stepping needed at all, unlike the
    # HISTORY+CST-triangle version. The old CST-triangle + HISTORY versions
    # of these two presets are kept below as SEN_tension_history /
    # SEN_shear_history for comparison, since they were already separately
    # verified working earlier.
    #
    # *** SCOPE NOTE: this paper-exact (Q4 + PENALTY) rebuild covers ONLY
    # SEN_tension and SEN_shear below. SEN_branching, L_panel, and
    # coalescence further down still use the older CST-triangle + HISTORY
    # solver (solver2d.py) - they were NOT ported to the Q4/PENALTY recipe.
    # L_panel in particular would need a new quad mesh generator for its
    # L-shaped (non-rectangular) domain, which does not exist yet, plus
    # separate verification with AT2 (nucleation, no notch) rather than
    # AT1 - do not assume PENALTY behaves the same there without testing it
    # the same way SEN_tension/SEN_shear were tested here. ***

    # *** Up_max / snap_Ups CORRECTED to match the paper's own figures ***
    # (root cause of "my plots don't look like the paper": the OLD
    # Up_max=1.0, snap=[...,1.0] preset below compared a deep POST-failure
    # state - fully open crack, long after the paper's own unstable jump -
    # against Fig. 9, which is drawn at Up=0.2, nowhere near 1.0. Confirmed
    # by directly rendering the paper's own Fig. 9/Fig. 10 pages: Fig. 10's
    # whole x-axis only spans Up in [0, 0.20], the elastic/fracture-energy
    # jump happens around Up~=0.14-0.15, and BY Up=0.2 (Fig. 9) the crack
    # has ALREADY propagated straight across the full width (x=-0.5 to
    # +0.5) at y=0 - Fig.9's alpha panel is a shortly-POST-jump snapshot,
    # not an Up=1.0 one. Running to Up=1.0 and snapping there compares two
    # physically different states and will never visually match Fig. 9,
    # however correct the solver itself is.
    "SEN_tension": dict(
        dim=2, element="quad", nondim=True, E=1.0, nu=0.3, Gc=1.0, l=0.01,
        diss_fct="AT1", split_type="AMOR", tol_ir=1e-3,
        geometry="square", size=1.0,
        n_el_per_l=5, band_half_width=0.1, coarse_ratio=6,
        crack=[((-0.5, 0.0), (0.0, 0.0))],           # Fig. 8: notch y=0, x in [-0.5,0]
        # Fig. 9/10: Up in [0, 0.2] ONLY - matches Fig. 10's x-axis exactly.
        # n_step=200 over this (5x smaller) range gives dUp=0.001, finer
        # than the old dUp=0.01 (which spanned [0,1.0]) - needed because the
        # paper's own jump is sharp (over roughly Up=0.14->0.15).
        loading=dict(kind="top_bottom", omega=np.pi / 2, Up_max=0.2, n_step=200,
                     snap_Ups=[0.1, 0.2], max_stag=1000, picard_iters=500),
    ),

    # Fig. 11/13: Up=0.4 is the LAST point the paper's own reference FEA
    # reports ("FEA encounters convergence issues for Up>0.4" - explicit
    # text, Section 4.4.2). Fig. 12's Up=0.47 is NN-ONLY - there is no FEA
    # ground truth beyond 0.4 to compare against. Up_max trimmed to 0.42
    # (small margin past the snapshot) instead of 0.6, so we're not
    # spending compute deep in a regime the paper itself says its own
    # reference FEA doesn't reliably converge in, with nothing to check the
    # result against even if it does converge.
    "SEN_shear": dict(
        dim=2, element="quad", nondim=True, E=1.0, nu=0.3, Gc=1.0, l=0.01,
        diss_fct="AT1", split_type="AMOR", tol_ir=1e-3,
        geometry="square", size=1.0,
        n_el_per_l=5, band_half_width=0.45, coarse_ratio=6,
        crack=[((-0.5, 0.0), (0.0, 0.0))],
        loading=dict(kind="top_bottom", omega=0.0, Up_max=0.42, n_step=200,
                     snap_Ups=[0.4], max_stag=1000, picard_iters=500),
    ),

    "SEN_tension_history": dict(
        dim=2, nondim=True, E=1.0, nu=0.3, Gc=1.0, l=0.01, diss_fct="AT1",
        split_type="AMOR", irrev="HISTORY", tol_ir=1e-3,
        geometry="square", size=1.0,
        n_el_per_l=5, band_half_width=0.1, coarse_ratio=6,
        crack=[((-0.5, 0.0), (0.0, 0.0))],
        loading=dict(kind="top_bottom", omega=np.pi / 2, Up_max=0.6, n_step=300,
                     snap_Ups=[0.15, 0.3, 0.6], max_stag=200),
    ),

    "SEN_shear_history": dict(
        dim=2, nondim=True, E=1.0, nu=0.3, Gc=1.0, l=0.01, diss_fct="AT1",
        split_type="AMOR", irrev="HISTORY", tol_ir=1e-3,
        geometry="square", size=1.0,
        n_el_per_l=5, band_half_width=0.45, coarse_ratio=6,
        crack=[((-0.5, 0.0), (0.0, 0.0))],
        loading=dict(kind="top_bottom", omega=0.0, Up_max=0.6, n_step=300,
                     snap_Ups=[0.4, 0.47], max_stag=200),
    ),

    "SEN_branching": dict(       # Section 4.5: "Crack branching in a notched
        # plate" - SAME notched SEN specimen/mesh as SEN_shear (nu=0.3,
        # l=0.01), split_type="ISO" so Psi+ = Psi, Psi- = 0 (paper's
        # explicit "omit the decomposition", which is what causes the
        # unphysical branching on purpose - see paper text).
        # band_half_width=0.35 (was 0.2, borderline-to-under-resolved past
        # y~0.30 in Fig. 14's branch extent). Widened with margin, same
        # fix class as L_panel/SEN_shear.
        # *** NOT ported to Q4 + PENALTY - still CST-triangle + HISTORY. ***
        dim=2, nondim=True, E=1.0, nu=0.3, Gc=1.0, l=0.01, diss_fct="AT1",
        split_type="ISO", irrev="HISTORY", tol_ir=1e-3,
        geometry="square", size=1.0,
        n_el_per_l=5, band_half_width=0.35, coarse_ratio=6,
        crack=[((-0.5, 0.0), (0.0, 0.0))],
        loading=dict(kind="top_bottom", omega=0.0, Up_max=0.45,
                     adaptive=True, snap_Ups=[0.37]),      # Fig. 14: Up=0.37
    ),

    "L_panel": dict(
        # Paper (Sec. 4.3): nu=0.18, l=0.01, AT2 (crack NUCLEATION, no
        # notch - AT1 doesn't localize smoothly enough for the NN, and
        # matching FEA setup is kept consistent).
        # *** NOT ported to Q4 + PENALTY - still CST-triangle + HISTORY. ***
        dim=2, nondim=True, E=1.0, nu=0.18, Gc=1.0, l=0.01, diss_fct="AT2",
        split_type="AMOR", irrev="HISTORY", tol_ir=1e-3,
        geometry="L_graded", size=1.0,
        dx_fine=0.005, band_lo=-0.05, band_hi=0.15, coarse_ratio=6,
        crack=None,                       # crack NUCLEATES, no notch
        loading=dict(kind="L_panel_edge", Up_max=1.2, n_step=120,
                     snap_Ups=[1.2]),                 # Fig. 6/7: Up=1.2
    ),

    "coalescence": dict(
        # Paper (Sec. 4.6): nu=1/3, l=0.01.
        # *** NOT ported to Q4 + PENALTY - still CST-triangle + HISTORY. ***
        dim=2, nondim=True, E=1.0, nu=1.0 / 3.0, Gc=1.0, l=0.01, diss_fct="AT1",
        split_type="AMOR", irrev="HISTORY", tol_ir=1e-3,
        geometry="square", size=1.0,
        n_el_per_l=5, band_half_width=0.35, coarse_ratio=6,
        crack=[((-0.25, -0.15), (-0.15, -0.05)),
               ((-0.05, -0.05), (0.05, 0.05)),
               ((0.15, 0.05), (0.25, 0.15))],
        loading=dict(kind="top_bottom", omega=np.pi / 2, Up_max=0.2, n_step=40,
                     snap_Ups=[0.2]),                 # Fig. 17/18: Up=0.2
    ),
}

# -------------------------------------------------------------------------
# "plate" presets: your actual specimen, shared by the 16 INPUT_i.m files.
_PLATE_COMBOS = {
    "AMOR_AT2_HISTORY": ("AMOR", "AT2", "HISTORY"),
    "AMOR_AT2_PENALTY": ("AMOR", "AT2", "PENALTY"),
    "AMOR_AT1_PENALTY": ("AMOR", "AT1", "PENALTY"),
    "AMOR_AT1_HISTORY": ("AMOR", "AT1", "HISTORY"),
    "ISO_AT1_HISTORY": ("ISO", "AT1", "HISTORY"),
    "ISO_AT1_PENALTY": ("ISO", "AT1", "PENALTY"),
    "ISO_AT2_PENALTY": ("ISO", "AT2", "PENALTY"),
    "ISO_AT2_HISTORY": ("ISO", "AT2", "HISTORY"),
    "MIEHE_AT2_HISTORY": ("MIEHE", "AT2", "HISTORY"),
    "MIEHE_AT2_PENALTY": ("MIEHE", "AT2", "PENALTY"),
    "MIEHE_AT1_PENALTY": ("MIEHE", "AT1", "PENALTY"),
    "MIEHE_AT1_HISTORY": ("MIEHE", "AT1", "HISTORY"),
}
for _name, (_split, _diss, _irrev) in _PLATE_COMBOS.items():
    PRESETS[f"plate_{_name}"] = dict(
        dim=2, nondim=False, E=1.0, nu=0.3, Gc=1.0, l=0.02,
        diss_fct=_diss, split_type=_split, irrev=_irrev, tol_ir=1e-3,
        geometry="plate", Nx=22, Ny=22, Lnotch_y=0.2, size=1.0,
        crack="use_nodesets",
        loading=dict(kind="plate_disp_x", Up_max=1.0 / 30.0, n_step=10),
    )


if len(sys.argv) > 1 and sys.argv[1] in ("list", "--list", "-l", "help", "--help"):
    print("Available presets (run: python run.py <name>):\n")
    for name in PRESETS:
        print(" ", name)
    sys.exit(0)

# L_SCALE: multiply l by this factor for a faster, coarser dev/smoke-test
# run. L_SCALE=1 (default) is the paper's own resolution. A coarser l
# blunts notch tips, so for the pre-notched SEN presets it DELAYS crack
# propagation to a higher Up (verified directly - see SEN_tension preset
# comment); it does not change whether the physics is qualitatively
# correct, only how far you have to load before it kicks in.
L_SCALE = float(os.environ.get("L_SCALE", "1.0"))

PRESET_NAME = sys.argv[1] if len(sys.argv) > 1 else "bar_1d"
cfg = PRESETS[PRESET_NAME]
if L_SCALE != 1.0 and cfg.get("dim") == 2:
    cfg = dict(cfg)
    cfg["l"] = cfg["l"] * L_SCALE
    print(f"*** L_SCALE={L_SCALE}: using l={cfg['l']} (paper value x{L_SCALE}). "
          f"Re-run with L_SCALE=1 for paper-accurate resolution. ***")
outdir = os.path.join(os.path.dirname(__file__), f"out_{PRESET_NAME}")
os.makedirs(outdir, exist_ok=True)
print(f"=== running preset '{PRESET_NAME}' ===")


# =========================================================================
# 1D DRIVER
# =========================================================================
def run_1d(cfg):
    l = cfg["l"]
    h = l / cfg["n_el_per_l"]
    x0, x1 = cfg["domain"]
    n_el = int(round((x1 - x0) / h))
    mesh = bar_mesh_1d(x0, x1, n_el)
    print("nodes:", mesh.num_node)

    fea = PhaseFieldFEA1D(mesh, E=cfg["E"], Gc=cfg["Gc"], l=l,
                           diss_fct=cfg["diss_fct"], nondim=cfg["nondim"])
    if cfg["crack"] is not None:
        fea.init_crack(cfg["crack"], diss_fct=cfg["diss_fct"])
    if cfg.get("defect_x") is not None:
        fea.seed_defect(cfg["defect_x"])

    n_step = cfg["loading"]["n_step"]
    Up_values = np.linspace(0, cfg["loading"]["Up_max"], n_step + 1)[1:]
    snap_Ups = cfg["loading"].get("snap_Ups")
    if snap_Ups:
        snap_steps = sorted(set(int(np.argmin(np.abs(Up_values - u))) + 1 for u in snap_Ups))
    else:
        snap_steps = sorted(set([max(1, int(round(f * n_step))) for f in (0.5, 0.9, 1.0)]))
    snapshots = {}
    log = []
    for i, Up in enumerate(Up_values, 1):
        fea.staggered_step(right_disp=Up, tol=1e-9)
        el, fr, tot = fea.energies()
        log.append((Up, el, fr, tot))
        print(f"step {i:3d}  Up={Up:.4f}  el={el:.5e}  fr={fr:.5e}  "
              f"u=[{fea.u.min():+.4f},{fea.u.max():+.4f}]  amax={fea.alpha.max():.3f}")
        if i in snap_steps:
            snapshots[Up] = (fea.u.copy(), fea.alpha.copy())

    log = np.array(log)
    np.savetxt(os.path.join(outdir, "energies.csv"), log,
               header="Up,elastic_energy,fracture_energy,total_energy",
               delimiter=",", comments="")
    np.savetxt(os.path.join(outdir, "final_fields.csv"),
               np.column_stack([mesh.nodes, fea.u, fea.alpha]),
               header="x,u,alpha", delimiter=",", comments="")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for Up, (u, alpha) in snapshots.items():
        axes[0].plot(mesh.nodes, u, label=f"Up={Up:.3f}")
        axes[1].plot(mesh.nodes, alpha, label=f"Up={Up:.3f}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel(r"$u_\theta$"); axes[0].legend()
    axes[1].set_xlabel("x"); axes[1].set_ylabel(r"$\alpha_\theta$"); axes[1].legend()
    fig.suptitle(PRESET_NAME)
    fig.savefig(os.path.join(outdir, f"{PRESET_NAME}_fields.png"), dpi=150)
    plt.close(fig)

    plot_energy_png(os.path.join(outdir, f"{PRESET_NAME}_energies.png"),
                     log[:, 0], log[:, 1], log[:, 2], title=PRESET_NAME)
    print(f"Done. PNGs + energies.csv/final_fields.csv written to {outdir}")


# =========================================================================
# 2D DRIVER
# =========================================================================
def build_mesh_2d(cfg):
    if cfg["geometry"] == "plate":
        mesh, nodesets, crack_segment = plate_mesh(cfg["Nx"], cfg["Ny"], cfg["Lnotch_y"],
                                                     Lx=cfg["size"], Ly=cfg["size"])
        return mesh, dict(nodesets=nodesets, crack_segment=crack_segment)
    if cfg["geometry"] == "L":
        return l_shape_mesh(cfg["size"], n=cfg["n_mesh"]), None
    if cfg["geometry"] == "L_graded":
        return l_shape_mesh_graded(cfg["size"], cfg["dx_fine"],
                                    band_lo=cfg["band_lo"], band_hi=cfg["band_hi"],
                                    coarse_ratio=cfg["coarse_ratio"]), None
    l = cfg["l"]
    ne_fine = int(round(1.0 / (l / cfg["n_el_per_l"])))
    if cfg["geometry"] == "square":
        if cfg.get("element") == "quad":
            mesh = refine_band_mesh_quad(cfg["size"], cfg["size"], band_y=0.0,
                                          band_half_width=cfg["band_half_width"],
                                          ne_fine=ne_fine, coarse_ratio=cfg["coarse_ratio"],
                                          origin=(-cfg["size"] / 2, -cfg["size"] / 2))
        else:
            mesh = refine_band_mesh(cfg["size"], cfg["size"], band_y=0.0,
                                     band_half_width=cfg["band_half_width"],
                                     ne_fine=ne_fine, coarse_ratio=cfg["coarse_ratio"],
                                     origin=(-cfg["size"] / 2, -cfg["size"] / 2))
        return mesh, None
    else:
        raise ValueError(cfg["geometry"])


def build_bc_2d(mesh, cfg, extra=None, bc_cls=BoundaryConditions):
    """Returns (bc, alpha_lock_nodes). bc_cls lets callers use either
    solver2d.BoundaryConditions or solver2d_quad.BoundaryConditions - both
    have an identical fix()/drive() interface, this function is otherwise
    solver-agnostic (purely geometric/index-based)."""
    bc = bc_cls()
    kind = cfg["loading"]["kind"]
    lock = set()
    if kind == "top_bottom":
        top = np.where(np.isclose(mesh.nodes[:, 1], 0.5))[0]
        bot = np.where(np.isclose(mesh.nodes[:, 1], -0.5))[0]
        omega = cfg["loading"]["omega"]
        bc.fix(bot, "x", 0.0)
        bc.fix(bot, "y", 0.0)
        bc.drive(top, "x", np.cos(omega))
        bc.drive(top, "y", np.sin(omega))
        lock.update(top.tolist()); lock.update(bot.tolist())
    elif kind == "L_panel_edge":
        bot = np.where(np.isclose(mesh.nodes[:, 1], -0.5))[0]
        bc.fix(bot, "x", 0.0)
        bc.fix(bot, "y", 0.0)
        y0 = np.where(np.isclose(mesh.nodes[:, 1], 0.0))[0]
        load_sec = np.array([n for n in y0 if mesh.nodes[n, 0] >= 0.44 - 1e-9])
        bc.drive(load_sec, "y", 1.0)
        lock.update(bot.tolist())
        near_load = np.where((mesh.nodes[:, 0] >= 0.35) &
                              (np.abs(mesh.nodes[:, 1]) <= 0.05))[0]
        lock.update(near_load.tolist())
    elif kind == "plate_disp_x":
        ns = extra["nodesets"]
        bc.fix(ns["left"], "x", 0.0)
        bc.fix(ns["bottom_left"], "y", 0.0)
        bc.drive(ns["right"], "x", 1.0)
        lock.update(ns["left"].tolist()); lock.update(ns["right"].tolist())
    else:
        raise ValueError(kind)
    return bc, lock


def robust_advance(fea, bc, Up_from, Up_target, fixed_alpha_nodes, max_stag, picard_iters,
                    min_substep, tol=1e-8, tol_alpha=1e-4, max_boost=2, verbose=True,
                    max_overshoots=2, max_total_attempts=200):
    """Advance the (stateful) `fea` object from its current, already-
    converged state at Up_from to Up_target.

    NOTE: this function is used by the ADAPTIVE-stepping presets only
    (L_panel, coalescence, SEN_branching). SEN_tension and SEN_shear use
    plain FIXED load stepping instead (see their preset comments and
    run_2d's fixed-stepping branch below) - this bisection/overshoot
    machinery is NOT part of the CMAME paper's own methodology (Appendix G
    describes plain fixed increments + a staggered/Newton-Raphson
    iteration budget, nothing adaptive) and was found to still be capable
    of looping for a very long time, or - before FIX 4 below - literally
    forever, right at a genuine instability. Kept here for the presets
    that still use it, with a hard ceiling so it can no longer hang.

    *** FIX 1 - max_stag/max_boost lowered (150/8 -> caller default 40/2),
    verified by direct instrumented runs on SEN_tension: at the paper's
    l=0.01 (~365k DOF) a single elasticity solve alone measured ~55-60s.
    Landing on a load step that sits inside a genuine unstable crack-jump
    (confirmed directly: at max_stag=80 one non-converging attempt right at
    SEN_tension's jump ran 178s and STILL hadn't converged) means every
    extra unit of max_stag/max_boost is ~1 more minute of wall time with no
    guarantee of ever converging. The OLD defaults (max_stag=150,
    max_boost=8) could queue up to 1200 such solves for a single stuck
    step - tens of hours - before ever giving up and trying a smaller
    increment. That is what was previously showing up as a "stall" at
    ~4-5% CPU: not a hang, just a workload that was never going to finish
    in observable time. Smaller defaults mean a stuck attempt fails fast
    and hands off to bisection (or the overshoot below) quickly instead.

    *** FIX 2 - overshoot fallback, verified by direct A/B test: bisecting
    to a SMALLER increment is the wrong instinct once you are already
    sitting right at an unstable jump - the true solution there is close
    to an unstable/saddle equilibrium, which is intrinsically hard for a
    staggered scheme to converge to regardless of step size. Landing
    cleanly on either side of the jump is comparatively easy: a direct test
    starting from a converged Up=0.30 and jumping straight past the jump to
    Up=0.40 (skipping the unstable window entirely) converged in 57s with
    NO bisection needed, immediately after an attempt at the intermediate
    Up=0.35 had failed to converge even after 178s at max_stag=80. So once
    bisection has been pushed all the way down to min_substep AND boosting
    is exhausted at the CURRENT sub-target, this function now tries one
    OVERSHOOT straight to the original outer Up_target (skipping whatever
    is jamming the loop) before giving up on that sub-target. Each stack
    entry is only allowed one overshoot attempt, so this can't loop
    forever if the overshoot also fails.

    *** FIX 4 - found by actually running SEN_tension on real hardware and
    watching it happen live: FIX 2's "one overshoot per stack frame" guard
    does NOT prevent an infinite loop, because every bisection push creates
    a BRAND NEW frame with overshot=False. In a load region with no
    reachable equilibrium at min_substep resolution, the observed behavior
    was: bisect down -> boost -> fail -> overshoot (allowed, fresh frame)
    -> fail -> bisect back down through the IDENTICAL sequence of Up
    values (nothing converged, so nothing moved the lower bound) -> hit
    min_substep again -> boost -> fail -> overshoot again (allowed, another
    fresh frame) - an exact, unbounded repeating cycle, confirmed directly
    from a live run's terminal log (same ~6-line sequence repeating
    verbatim). Fixed with two independent safeguards: (1) max_overshoots
    now counts overshoot attempts GLOBALLY across the whole call, not per
    frame - once exhausted, a failing point is accepted as best-available
    instead of overshooting again; (2) max_total_attempts is a hard
    absolute ceiling on staggered_step calls in one robust_advance call -
    if hit, this prints a loud warning and returns the best state reached
    rather than looping forever, as a safety net against any bug not yet
    found. ***
    """
    outer_target = Up_target
    stack = [Up_target]
    lower = [Up_from]
    overshot = [False]
    boost = 1
    overshoot_budget = max_overshoots
    total_attempts = 0
    while stack:
        total_attempts += 1
        if total_attempts > max_total_attempts:
            print(f"    [robust] HARD STOP: {max_total_attempts} attempts reached without "
                  f"reaching Up={outer_target:.5f} - giving up on this load step and accepting "
                  f"the current (possibly non-equilibrium) state. The region around "
                  f"Up={lower[-1]:.5f}-{outer_target:.5f} may have no reachable equilibrium at "
                  f"this resolution/tolerance.", flush=True)
            return fea.alpha, fea.u
        Up_try = stack[-1]
        alpha_bak, u_bak, H_bak = fea.alpha.copy(), fea.u.copy(), fea.H.copy()
        _, converged = fea.staggered_step(bc, Up_try, fixed_alpha_nodes=fixed_alpha_nodes,
                                           max_stag=max_stag * boost, tol=tol, tol_alpha=tol_alpha,
                                           picard_iters=picard_iters, warn=False)
        if converged:
            reached = Up_try
            stack.pop()
            lower.pop()
            overshot.pop()
            boost = 1
            # *** FIX 3 (found while actually running SEN_tension through its
            # crack jump, verified by instrumenting a step-by-step trace):
            # after converging at a bisected sub-target, the PARENT frame's
            # lower bound must be advanced to `reached` - fea's state has
            # genuinely moved there. The original version left lower[-1]
            # pointing at the stale pre-bisection value, so retrying the
            # parent's target after a successful intermediate step recomputed
            # the SAME midpoint again instead of narrowing from the new,
            # closer converged point. Caught directly: retrying Up=0.225
            # after converging at 0.1875 re-derived mid=0.1875 again (using
            # stale lower=0.15) instead of the correct next midpoint between
            # 0.1875 and 0.225. ***
            if lower:
                lower[-1] = reached
        else:
            fea.alpha, fea.u, fea.H = alpha_bak, u_bak, H_bak
            gap = Up_try - lower[-1]
            if gap / 2.0 > min_substep:
                mid = lower[-1] + gap / 2.0
                stack.append(mid)
                lower.append(lower[-1])
                overshot.append(False)
                if verbose:
                    print(f"    [robust] Up={Up_try:.5f} did not converge - bisecting, "
                          f"retrying at Up={mid:.5f}", flush=True)
            elif boost < max_boost:
                boost *= 2
                if verbose:
                    print(f"    [robust] Up={Up_try:.5f} did not converge at min substep "
                          f"({min_substep:.2e}) - boosting max_stag to {max_stag*boost}, retrying",
                          flush=True)
            elif overshoot_budget > 0 and Up_try < outer_target - 1e-12:
                overshoot_budget -= 1
                if verbose:
                    print(f"    [robust] Up={Up_try:.5f} still stuck after bisection+boost - "
                          f"trying an OVERSHOOT straight to Up={outer_target:.5f} "
                          f"({overshoot_budget} overshoot attempt(s) left this call)", flush=True)
                stack[-1] = outer_target
                overshot[-1] = True
                boost = 1
            else:
                if verbose:
                    print(f"    [robust] WARNING: giving up on full convergence at Up={Up_try:.5f} "
                          f"after bisection + boost + overshoot budget exhausted - accepting "
                          f"best-available (non-equilibrium) state and moving on.", flush=True)
                accepted_at = Up_try
                stack.pop()
                lower.pop()
                overshot.pop()
                boost = 1
                if lower:
                    lower[-1] = accepted_at
    return fea.alpha, fea.u


def run_2d(cfg):
    mesh, extra = build_mesh_2d(cfg)
    print("nodes:", mesh.num_node, " elements:", mesh.num_elem)

    fea = PhaseFieldFEA(mesh, E=cfg["E"], nu=cfg["nu"], Gc=cfg["Gc"], l=cfg["l"],
                         split_type=cfg["split_type"], diss_fct=cfg["diss_fct"],
                         nondim=cfg["nondim"], irrev=cfg.get("irrev", "HISTORY"),
                         tol_ir=cfg.get("tol_ir", 1e-3))
    if cfg["crack"] == "use_nodesets":
        fea.init_crack([extra["crack_segment"]], diss_fct=cfg["diss_fct"])
    elif cfg["crack"] is not None:
        fea.init_crack(cfg["crack"], diss_fct=cfg["diss_fct"])

    bc, lock_nodes = build_bc_2d(mesh, cfg, extra)
    fixed_alpha_nodes = {n: 0.0 for n in lock_nodes}

    loading = cfg["loading"]
    Up_max = loading["Up_max"]
    snap_Ups = sorted(loading.get("snap_Ups") or [])
    max_stag = loading.get("max_stag", 40)
    max_boost = loading.get("max_boost", 2)
    picard_iters = loading.get("picard_iters", 8)

    log = []
    t0 = time.time()

    if loading.get("adaptive", False):
        # ---- ADAPTIVE LOAD STEPPING ----
        # Take large steps while alpha is barely changing between steps
        # (elastic regime), shrink the step whenever alpha jumps a lot
        # (damage actively evolving), grow back once things stabilize.
        # This only changes how many/how large the load increments are -
        # the per-step convergence tolerance (tol, max_stag, picard_iters)
        # is unchanged, so it costs no accuracy, only removes steps where
        # nothing was happening.
        dUp = loading.get("dUp_init", Up_max / 15)
        dUp_min = loading.get("dUp_min", Up_max / 500)
        dUp_max = loading.get("dUp_max", Up_max / 8)
        growth = loading.get("growth", 1.4)
        shrink = loading.get("shrink", 0.35)
        target_dalpha = loading.get("target_dalpha", 0.03)
        max_total_steps = loading.get("max_total_steps", 2000)

        Up = 0.0
        prev_alpha_max = float(fea.alpha.max())
        snap_remaining = [s for s in snap_Ups if s > Up + 1e-12]
        step_i = 0
        while Up < Up_max - 1e-12 and step_i < max_total_steps:
            Up_try = Up + dUp
            next_targets = [t for t in ([Up_max] + snap_remaining) if t > Up + 1e-12]
            if next_targets:
                Up_try = min(Up_try, min(next_targets))

            robust_advance(fea, bc, Up, Up_try, fixed_alpha_nodes=fixed_alpha_nodes,
                            max_stag=max_stag, max_boost=max_boost, picard_iters=picard_iters,
                            min_substep=dUp_min)
            step_i += 1
            Up = Up_try
            el, fr, tot = fea.energies()
            log.append((Up, el, fr, tot))

            alpha_max = float(fea.alpha.max())
            dalpha = alpha_max - prev_alpha_max
            prev_alpha_max = alpha_max

            ux, uy = fea.u[0::2], fea.u[1::2]
            high = np.where(fea.alpha > 0.5)[0]
            if high.size:
                cx = (mesh.nodes[high, 0].min(), mesh.nodes[high, 0].max())
                cy = (mesh.nodes[high, 1].min(), mesh.nodes[high, 1].max())
                crack_str = f"crack_x=[{cx[0]:+.3f},{cx[1]:+.3f}] crack_y=[{cy[0]:+.3f},{cy[1]:+.3f}]"
            else:
                crack_str = "crack: none yet"
            print(f"step {step_i:4d}  Up={Up:.4f}  dUp={dUp:.4f}  el={el:.5e}  fr={fr:.5e}  "
                  f"u=[{ux.min():+.4f},{ux.max():+.4f}]  v=[{uy.min():+.4f},{uy.max():+.4f}]  "
                  f"amax={alpha_max:.3f}  {crack_str}  t={time.time()-t0:.0f}s")

            if snap_remaining and abs(Up - snap_remaining[0]) < 1e-9:
                png_path = os.path.join(outdir, f"{PRESET_NAME}_Up_{Up:.4f}.png")
                plot_fields_png(png_path, mesh, fea.u, fea.alpha, Up=Up, title=PRESET_NAME)
                print(f"    -> saved {png_path}")
                snap_remaining.pop(0)

            if abs(dalpha) > target_dalpha:
                dUp = max(dUp * shrink, dUp_min)
            else:
                dUp = min(dUp * growth, dUp_max)

        if step_i >= max_total_steps:
            print(f"WARNING: hit max_total_steps={max_total_steps} before reaching Up_max - "
                  f"stopped at Up={Up:.4f}. Increase max_total_steps if this is unexpected.")

    else:
        # ---- FIXED LOAD STEPPING (used by SEN_tension, SEN_shear) ----
        # *** No bisection/adaptive step-size search here - each load step
        # is attempted ONCE with a generous but bounded max_stag, and the
        # result (converged or not) is accepted and logged either way, with
        # an explicit warning printed by staggered_step itself if it didn't
        # converge. This is closer to Appendix G's own approach (fixed
        # increments + a staggered/Newton-Raphson iteration BUDGET, no
        # adaptive search) and its runtime is bounded and predictable:
        # exactly n_step staggered_step calls, full stop - it cannot hang,
        # unlike the adaptive/robust_advance path used by other presets. If
        # a step near an instability doesn't converge in max_stag
        # iterations, you will see it explicitly in the log
        # (converged=False) rather than the run silently continuing on a
        # bad state OR spinning forever trying to fix it automatically -
        # if that happens, the fix is to increase max_stag or n_step (finer
        # dUp) for that preset, not to add bisection back in. ***
        n_step = loading["n_step"]
        Up_values = np.linspace(0, Up_max, n_step + 1)[1:]
        if snap_Ups:
            snap_steps = sorted(set(int(np.argmin(np.abs(Up_values - u))) + 1 for u in snap_Ups))
        else:
            snap_steps = sorted(set([max(1, int(round(f * n_step))) for f in (0.4, 0.6, 0.75, 0.9, 1.0)]))

        n_not_converged = 0
        for i, Up in enumerate(Up_values, 1):
            _, converged = fea.staggered_step(bc, Up, fixed_alpha_nodes=fixed_alpha_nodes,
                                               max_stag=max_stag, tol=1e-8, tol_alpha=1e-4,
                                               picard_iters=picard_iters, warn=True)
            if not converged:
                n_not_converged += 1
            el, fr, tot = fea.energies()
            log.append((Up, el, fr, tot))

            ux, uy = fea.u[0::2], fea.u[1::2]
            high = np.where(fea.alpha > 0.5)[0]
            if high.size:
                cx = (mesh.nodes[high, 0].min(), mesh.nodes[high, 0].max())
                cy = (mesh.nodes[high, 1].min(), mesh.nodes[high, 1].max())
                crack_str = f"crack_x=[{cx[0]:+.3f},{cx[1]:+.3f}] crack_y=[{cy[0]:+.3f},{cy[1]:+.3f}]"
            else:
                crack_str = "crack: none yet"
            print(f"step {i:3d}/{n_step}  Up={Up:.4f}  el={el:.5e}  fr={fr:.5e}  "
                  f"u=[{ux.min():+.4f},{ux.max():+.4f}]  v=[{uy.min():+.4f},{uy.max():+.4f}]  "
                  f"amax={fea.alpha.max():.3f}  converged={converged}  {crack_str}  "
                  f"t={time.time()-t0:.0f}s")

            if i in snap_steps:
                png_path = os.path.join(outdir, f"{PRESET_NAME}_Up_{Up:.4f}.png")
                plot_fields_png(png_path, mesh, fea.u, fea.alpha, Up=Up, title=PRESET_NAME)
                print(f"    -> saved {png_path}")

        if n_not_converged:
            print(f"NOTE: {n_not_converged}/{n_step} load steps did not fully converge within "
                  f"max_stag={max_stag} (each was flagged with a WARNING above at the time). "
                  f"Fields at those specific Up values are not true equilibria - if any of them "
                  f"land on a snap_Ups value you're using for a figure, re-run with a larger "
                  f"max_stag or a finer n_step.")

    log = np.array(log)
    np.savetxt(os.path.join(outdir, "energies.csv"), log,
               header="Up,elastic_energy,fracture_energy,total_energy",
               delimiter=",", comments="")
    plot_energy_png(os.path.join(outdir, f"{PRESET_NAME}_energies.png"),
                     log[:, 0], log[:, 1], log[:, 2], title=PRESET_NAME)
    print(f"Done. PNG snapshots + energies.csv/png written to {outdir}")


# =========================================================================
# 2D DRIVER - PAPER-EXACT (Q4 elements + PENALTY), used by SEN_tension /
# SEN_shear. Kept as its own function rather than folded into run_2d above
# because the underlying solver class, its staggered_step signature, and
# its convergence behavior are different enough (no bisection needed at
# all - see PRESETS comment above) that sharing one branchy function risked
# subtle cross-contamination bugs between the two solver backends.
# =========================================================================
def run_2d_quad(cfg):
    mesh, extra = build_mesh_2d(cfg)
    print("nodes:", mesh.num_node, " elements:", mesh.num_elem, " (Q4)")

    fea = PhaseFieldFEAQuad(mesh, E=cfg["E"], nu=cfg["nu"], Gc=cfg["Gc"], l=cfg["l"],
                             split_type=cfg["split_type"], diss_fct=cfg["diss_fct"],
                             nondim=cfg["nondim"], tol_ir=cfg.get("tol_ir", 1e-3))
    if cfg["crack"] is not None:
        fea.init_crack(cfg["crack"], diss_fct=cfg["diss_fct"])

    bc, lock_nodes = build_bc_2d(mesh, cfg, extra, bc_cls=BoundaryConditionsQuad)
    fixed_alpha_nodes = {n: 0.0 for n in lock_nodes}

    loading = cfg["loading"]
    Up_max = loading["Up_max"]
    n_step = loading["n_step"]
    snap_Ups = sorted(loading.get("snap_Ups") or [])
    max_stag = loading.get("max_stag", 1000)      # Appendix G's own cap
    picard_iters = loading.get("picard_iters", 500)  # Appendix G's own cap

    Up_values = np.linspace(0, Up_max, n_step + 1)[1:]
    if snap_Ups:
        snap_steps = sorted(set(int(np.argmin(np.abs(Up_values - u))) + 1 for u in snap_Ups))
    else:
        snap_steps = sorted(set([max(1, int(round(f * n_step))) for f in (0.4, 0.6, 0.75, 0.9, 1.0)]))

    log = []
    t0 = time.time()
    n_not_converged = 0
    for i, Up in enumerate(Up_values, 1):
        _, converged = fea.staggered_step(bc, Up, fixed_alpha_nodes=fixed_alpha_nodes,
                                           max_stag=max_stag, picard_iters=picard_iters, warn=True)
        if not converged:
            n_not_converged += 1
        el, fr, tot = fea.energies()
        log.append((Up, el, fr, tot))

        ux, uy = fea.u[0::2], fea.u[1::2]
        high = np.where(fea.alpha > 0.5)[0]
        if high.size:
            cx = (mesh.nodes[high, 0].min(), mesh.nodes[high, 0].max())
            cy = (mesh.nodes[high, 1].min(), mesh.nodes[high, 1].max())
            crack_str = f"crack_x=[{cx[0]:+.3f},{cx[1]:+.3f}] crack_y=[{cy[0]:+.3f},{cy[1]:+.3f}]"
        else:
            crack_str = "crack: none yet"
        print(f"step {i:3d}/{n_step}  Up={Up:.4f}  el={el:.5e}  fr={fr:.5e}  "
              f"u=[{ux.min():+.4f},{ux.max():+.4f}]  v=[{uy.min():+.4f},{uy.max():+.4f}]  "
              f"amax={fea.alpha.max():.3f}  converged={converged}  {crack_str}  "
              f"t={time.time()-t0:.0f}s")

        if i in snap_steps:
            png_path = os.path.join(outdir, f"{PRESET_NAME}_Up_{Up:.4f}.png")
            plot_fields_png(png_path, mesh, fea.u, fea.alpha, Up=Up, title=PRESET_NAME)
            print(f"    -> saved {png_path}")

    if n_not_converged:
        print(f"NOTE: {n_not_converged}/{n_step} load steps did not fully converge within "
              f"max_stag={max_stag} (each was flagged with a WARNING above at the time).")

    log = np.array(log)
    np.savetxt(os.path.join(outdir, "energies.csv"), log,
               header="Up,elastic_energy,fracture_energy,total_energy",
               delimiter=",", comments="")
    plot_energy_png(os.path.join(outdir, f"{PRESET_NAME}_energies.png"),
                     log[:, 0], log[:, 1], log[:, 2], title=PRESET_NAME)
    print(f"Done. PNG snapshots + energies.csv/png written to {outdir}")


# =========================================================================
if __name__ == "__main__":
    if cfg["dim"] == 1:
        run_1d(cfg)
    elif cfg.get("element") == "quad":
        run_2d_quad(cfg)
    else:
        run_2d(cfg)