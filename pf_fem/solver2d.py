"""
2D staggered phase-field brittle-fracture FEA solver.

Elements       : constant-strain triangles (CST), one integration point.
Elasticity     : plane strain, isotropic or Amor volumetric/deviatoric split.
                 The Amor split introduces a mild bimodular (tension/
                 compression) nonlinearity, solved by fixed point (Picard)
                 iteration on the active-branch flag per element.
Phase field    : AT1 or AT2. Irreversibility: "HISTORY" (default - Miehe
                 et al. 2010 history-field method: driving force frozen at
                 its running max, single linear solve + monotonicity
                 projection) or "PENALTY" (Gerasimov & De Lorenzis 2018,
                 Eq. 14-16 - what the paper's own reference FEA uses per
                 Appendix G, but see __init__'s docstring: this codebase's
                 implementation of it has a confirmed non-convergent
                 active-set oscillation and should be considered
                 unreliable until fixed).
Coupling       : standard staggered (alternate min.) scheme, iterated to a
                 dual (relative-energy AND max nodal |d_alpha|) tolerance
                 each load step - see staggered_step().
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .element import tri_geom
from .material import degradation, diss_params, split_energy_stress, tangent_matrices


class BoundaryConditions:
    """fixed:  dict  dof -> constant value (applied every step, unscaled)
    driven: dict  dof -> unit value (multiplied by the current load factor Up)
    dof index convention: 2*node for u_x , 2*node+1 for u_y
    """

    def __init__(self):
        self.fixed = {}
        self.driven = {}

    def fix(self, nodes, component, value=0.0):
        comp = 0 if component in ("x", 0) else 1
        for n in np.atleast_1d(nodes):
            self.fixed[2 * int(n) + comp] = value

    def drive(self, nodes, component, unit_value=1.0):
        comp = 0 if component in ("x", 0) else 1
        for n in np.atleast_1d(nodes):
            self.driven[2 * int(n) + comp] = unit_value


class PhaseFieldFEA:
    def __init__(self, mesh, E=1.0, nu=0.3, Gc=1.0, l=0.02,
                 split_type="AMOR", diss_fct="AT2", nondim=False,
                 irrev="HISTORY", tol_ir=1e-3):
        """nondim=False (default): standard *dimensional* formulation,
            Eq. (1) of the paper:  Gc/cw * ( w(a)/l + l*|grad a|^2 )
        nondim=True: the paper's Section 2.4 *non-dimensional* formulation,
            Eq. (18)/(26):  (1/cw) * ( w(a) + l^2*|grad a|^2 )   [Gc == 1]
            Use this ONLY if you are also using non-dimensional geometry,
            E, and l as the paper does (see README, "A note on units").
            Gc is ignored (treated as 1) in this mode.
        irrev: "HISTORY" (Miehe et al. 2010 - DEFAULT, see *** BUG NOTICE
            *** below) or "PENALTY" (Gerasimov & De Lorenzis 2018,
            Eq. 14-16 of the paper - this is what Appendix G says the
            paper's *own reference FEA* uses, with TOL_ir=1e-3, but see
            the bug notice: THIS IMPLEMENTATION OF IT IS BROKEN).

        *** BUG NOTICE (found by direct diagnostic, 2025 debugging session) ***
        The "PENALTY" active-set loop in solve_phasefield() does NOT
        converge - confirmed directly by instrumenting it: it perpetually
        flip-flops between two active-sets (e.g. 53 <-> 11557 active DOFs
        on a real test mesh) for as many as 60 inner iterations, because
        the penalty stiffness gamma_ir (~4e5 for TOL_ir=1e-3, AT1) is
        switched fully on/off every iteration with no damping/line-search
        (a textbook semismooth-Newton chattering failure). This was
        confirmed to be the root cause of the SEN_tension mismatch
        reported against the paper's Fig. 9/10: at max_stag=30 the solver
        never got anywhere near a converged staggered state, producing
        wrong u/v/alpha fields and a "staircase" energy curve instead of
        the paper's clean curve. Switching to "HISTORY" (single linear
        solve + max(alpha,alpha_old) projection, exactly as solver1d.py
        already does) converges in ~2 staggered iterations and reproduces
        the paper's Fig. 9/10 field and energy shapes (verified directly).
        HISTORY is therefore now the default. If you need literal
        Gerasimov-penalty behavior, "PENALTY" is left in place for
        reference but should be considered UNRELIABLE until someone adds
        damping/relaxation to its active-set loop.
        tol_ir: irreversibility tolerance, TOL_ir in Eq. (15)-(16). The
            paper uses TOL_ir=5e-3 for the NN/DRM (Section 2.3) but
            TOL_ir=1e-3 for the reference FEA (Appendix G).
        """
        self.mesh = mesh
        self.E = E
        self.nu = nu
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))
        self.Gc = Gc
        self.l = l
        self.split_type = split_type
        self.cw, self.w_const, self.w_coef = diss_params(diss_fct)
        self.nondim = nondim
        self.irrev = irrev
        self.tol_ir = tol_ir
        if diss_fct.upper() == "AT1":
            self.gamma_ir = (27.0 / (64.0 * tol_ir ** 2))
        else:
            self.gamma_ir = (1.0 / tol_ir ** 2 - 1.0)
        # gamma_ir above is the Gc/l - normalized factor from Eq. (15)-(16);
        # multiply by Gc/l (dimensional) or 1/l (nondim, Gc=1) at use-time.

        n = mesh.num_node
        self.alpha = np.zeros(n)
        self.u = np.zeros(2 * n)
        self.H = np.zeros(mesh.num_elem)  # history variable (elementwise)

        # ---- precompute per-element geometry ONCE, as arrays (vectorized
        # assembly - the original per-element Python loop was the main
        # bottleneck preventing use of the paper's actual mesh resolution,
        # e.g. l=0.01; this precompute + the vectorized assemble methods
        # below give the same numerical result ~10-100x faster). ----
        self._geom = [tri_geom(mesh.nodes[el]) for el in mesh.elements]
        n_el = mesh.num_elem
        self._area = np.array([g[0] for g in self._geom])              # (n_el,)
        self._grad = np.array([g[1] for g in self._geom])              # (n_el,3,2)
        self._Bmat = np.array([g[2] for g in self._geom])              # (n_el,3,6)
        el = mesh.elements
        dofs = np.empty((n_el, 6), dtype=int)
        dofs[:, 0::2] = 2 * el
        dofs[:, 1::2] = 2 * el + 1
        self._el_dofs = dofs                                            # (n_el,6)
        self._pf_nodes = el                                              # (n_el,3)
        # phase-field local matrices (mesh-only, independent of alpha/H)
        self._Kdiff_local = self._area[:, None, None] * np.einsum(
            'nia,nja->nij', self._grad, self._grad)                    # (n_el,3,3)
        Mref = np.array([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]]) / 12.0
        self._Mloc_local = self._area[:, None, None] * Mref[None, :, :]  # (n_el,3,3)
        self._area3 = self._area / 3.0                                  # (n_el,)

    # ------------------------------------------------------------------
    def init_crack(self, segments, diss_fct="AT1"):
        """Prescribe an initial notch/crack by assigning the *nodal* phase
        field directly to the known closed-form 1D optimal-profile solution
        transverse to each crack segment (Appendix I of the paper for AT1;
        the analogous exponential profile for AT2). Irreversibility
        (alpha = max(alpha, alpha_old), enforced every phase-field solve)
        then keeps this initial crack from healing - no extra history
        forcing is required.

        segments: list of ((x0,y0),(x1,y1)) crack line segments (each may
        also be a single point tuple for a very short/point notch).
        """
        d = np.full(self.mesh.num_node, np.inf)
        for (p0, p1) in segments:
            p0 = np.asarray(p0, dtype=float)
            p1 = np.asarray(p1, dtype=float)
            di = self._dist_to_segment(self.mesh.nodes, p0, p1)
            d = np.minimum(d, di)
        if diss_fct.upper() == "AT1":
            a0 = np.where(d < 2 * self.l, (1.0 - d / (2 * self.l)) ** 2, 0.0)
        else:  # AT2 optimal profile: alpha = exp(-|y|/l)
            a0 = np.exp(-d / self.l)
            a0 = np.where(a0 > 1e-3, a0, 0.0)
        self.alpha = np.maximum(self.alpha, a0)

    @staticmethod
    def _dist_to_segment(pts, p0, p1):
        v = p1 - p0
        vv = v @ v
        if vv < 1e-14:
            return np.linalg.norm(pts - p0, axis=1)
        t = np.clip(((pts - p0) @ v) / vv, 0.0, 1.0)
        proj = p0[None, :] + t[:, None] * v[None, :]
        return np.linalg.norm(pts - proj, axis=1)

    # ------------------------------------------------------------------
    def _numerical_tangent(self, eps, h=1e-6):
        """Per-element 3x3 D_plus, D_minus via central finite differences of
        split_energy_stress's sigma_plus/sigma_minus - used for splits (MIEHE)
        without a simple closed-form tangent."""
        n = eps.shape[0]
        D_plus = np.zeros((n, 3, 3))
        D_minus = np.zeros((n, 3, 3))
        for k in range(3):
            dE = np.zeros_like(eps); dE[:, k] = h
            _, _, sp_p, sm_p, _ = split_energy_stress(eps + dE, self.lam, self.mu, self.split_type)
            _, _, sp_m, sm_m, _ = split_energy_stress(eps - dE, self.lam, self.mu, self.split_type)
            D_plus[:, :, k] = (sp_p - sp_m) / (2 * h)
            D_minus[:, :, k] = (sm_p - sm_m) / (2 * h)
        return D_plus, D_minus

    def _elasticity_assemble(self, active_tension, eps_prev=None):
        mesh = self.mesh
        alpha_e = self.alpha[mesh.elements].mean(axis=1)
        g_e = degradation(alpha_e)
        if self.split_type.upper() == "MIEHE":
            if eps_prev is None:
                eps_prev = np.zeros((mesh.num_elem, 3))
            D_plus, D_minus = self._numerical_tangent(eps_prev)
        else:
            D_plus, D_minus = tangent_matrices(self.lam, self.mu, self.split_type, active_tension)

        n_dof = 2 * mesh.num_node
        D = g_e[:, None, None] * D_plus + D_minus                        # (n_el,3,3)
        B = self._Bmat                                                    # (n_el,3,6)
        # Ke[n,a,b] = area * sum_k,l B[n,k,a] D[n,k,l] B[n,l,b]
        Ke = self._area[:, None, None] * np.einsum('nka,nkl,nlb->nab', B, D, B)  # (n_el,6,6)

        dofs = self._el_dofs                                              # (n_el,6)
        rows = np.repeat(dofs, 6, axis=1).ravel()
        cols = np.tile(dofs, (1, 6)).ravel()
        K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n_dof, n_dof))
        return K

    def solve_displacement(self, bc: BoundaryConditions, Up, picard_iters=15, tol=1e-8):
        """Solve the (possibly bimodular) elastic problem for the current
        alpha field via Picard iteration: for AMOR, on the tension/
        compression active set; for MIEHE, on the current strain (used to
        build a numerical secant tangent each iteration)."""
        mesh = self.mesh
        n_dof = 2 * mesh.num_node
        active_tension = np.ones(mesh.num_elem, dtype=bool)
        eps_prev = np.zeros((mesh.num_elem, 3))

        fixed_dofs = np.array(sorted(bc.fixed.keys()), dtype=int)
        fixed_vals = np.array([bc.fixed[d] for d in fixed_dofs])
        driven_dofs = np.array(sorted(bc.driven.keys()), dtype=int)
        driven_vals = np.array([bc.driven[d] for d in driven_dofs]) * Up

        presc_dofs = np.concatenate([fixed_dofs, driven_dofs]) if driven_dofs.size else fixed_dofs
        presc_vals = np.concatenate([fixed_vals, driven_vals]) if driven_dofs.size else fixed_vals
        free_mask = np.ones(n_dof, dtype=bool)
        free_mask[presc_dofs] = False
        free_dofs = np.where(free_mask)[0]

        u = self.u.copy()
        u[presc_dofs] = presc_vals

        for it in range(picard_iters):
            K = self._elasticity_assemble(active_tension, eps_prev)
            F = np.zeros(n_dof)
            rhs = F[free_dofs] - K[free_dofs, :][:, presc_dofs] @ presc_vals
            Kff = K[free_dofs, :][:, free_dofs]
            u_free = spla.spsolve(Kff.tocsc(), rhs)
            u_new = u.copy()
            u_new[free_dofs] = u_free

            # update active-tension flags / secant point from the new strain
            eps = self._strain(u_new)
            tr = eps[:, 0] + eps[:, 1]
            new_active = tr >= 0.0
            changed = np.sum(new_active != active_tension)
            strain_change = np.max(np.abs(eps - eps_prev)) if it > 0 else np.inf
            u = u_new
            active_tension = new_active
            eps_prev = eps
            if self.split_type.upper() == "MIEHE":
                if strain_change < tol:
                    break
            elif changed == 0:
                break
        self.u = u
        return u, active_tension

    def _strain(self, u):
        el = self.mesh.elements
        ue = np.empty((el.shape[0], 6))
        ue[:, 0::2] = u[2 * el]
        ue[:, 1::2] = u[2 * el + 1]
        eps = np.einsum('nka,na->nk', self._Bmat, ue)  # (n_el,3)
        return eps

    def update_history(self, active_tension=None):
        eps = self._strain(self.u)
        if active_tension is None:
            tr = eps[:, 0] + eps[:, 1]
            active_tension = tr >= 0.0
        psi_plus, psi_minus, *_ = split_energy_stress(eps, self.lam, self.mu, self.split_type)
        if self.irrev.upper() == "PENALTY":
            # PENALTY method uses the *actual* current driving force (no
            # history freeze) - irreversibility is instead enforced by the
            # penalty term in solve_phasefield.
            self.H = psi_plus
        else:
            self.H = np.maximum(self.H, psi_plus)
        return psi_plus, psi_minus

    # ------------------------------------------------------------------
    def _phasefield_assemble(self):
        mesh = self.mesh
        n = mesh.num_node
        if self.nondim:
            # Eq. (18)/(26): (1/cw) * ( w(a) + l^2 |grad a|^2 ), Gc == 1
            coefA = 1.0 / self.cw
            diff_len = self.l ** 2   # multiplies grad.grad directly (no extra /l)
            inv_l_w = 1.0            # w-term has NO 1/l factor in nondim form
        else:
            # Eq. (1): Gc/cw * ( w(a)/l + l |grad a|^2 )
            coefA = self.Gc / self.cw
            diff_len = self.l
            inv_l_w = 1.0 / self.l

        coef_e = 2.0 * self.H + coefA * self.w_coef * inv_l_w              # (n_el,)
        Ke = (coefA * 2 * diff_len) * self._Kdiff_local \
             + coef_e[:, None, None] * self._Mloc_local                    # (n_el,3,3)
        Fe_const = (2.0 * self.H - coefA * self.w_const * inv_l_w)         # (n_el,)
        Fe = np.repeat(Fe_const[:, None] * self._area3[:, None], 3, axis=1)  # (n_el,3)

        nodes = self._pf_nodes                                              # (n_el,3)
        rows = np.repeat(nodes, 3, axis=1).ravel()
        cols = np.tile(nodes, (1, 3)).ravel()
        K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
        F = np.bincount(nodes.ravel(), weights=Fe.ravel(), minlength=n)
        return K, F

    def _nodal_lumped_weight(self):
        if not hasattr(self, "_lumped_w"):
            w = np.bincount(self._pf_nodes.ravel(),
                             weights=np.repeat(self._area3, 3), minlength=self.mesh.num_node)
            self._lumped_w = w
        return self._lumped_w

    def solve_phasefield(self, fixed_alpha_nodes=None):
        mesh = self.mesh
        n = mesh.num_node
        K, F = self._phasefield_assemble()

        fixed_nodes = np.array(sorted(fixed_alpha_nodes.keys())) if fixed_alpha_nodes else np.array([], dtype=int)
        fixed_vals = np.array([fixed_alpha_nodes[k] for k in fixed_nodes]) if fixed_alpha_nodes else np.array([])

        free_mask = np.ones(n, dtype=bool)
        if fixed_nodes.size:
            free_mask[fixed_nodes] = False
        free = np.where(free_mask)[0]

        alpha_old = self.alpha.copy()

        if self.irrev.upper() == "PENALTY":
            # Eq. (14)-(16): add gamma_ir * <alpha - alpha_old>_-  penalty,
            # enforced by a small active-set (semismooth-Newton-like) loop.
            l_or_1 = 1.0 if self.nondim else self.l
            gamma_ir_abs = self.gamma_ir * (1.0 if self.nondim else self.Gc) / l_or_1
            w = self._nodal_lumped_weight()
            K = K.tolil()
            active = np.zeros(n, dtype=bool)
            for _ in range(8):
                Kp = K.copy()
                Fp = F.copy()
                Fp[active] += gamma_ir_abs * w[active] * alpha_old[active]
                for i in np.where(active)[0]:
                    Kp[i, i] += gamma_ir_abs * w[i]
                Kp = Kp.tocsr()
                alpha = alpha_old.copy()
                if fixed_nodes.size:
                    alpha[fixed_nodes] = fixed_vals
                rhs = Fp[free] - Kp[free, :][:, fixed_nodes] @ fixed_vals if fixed_nodes.size else Fp[free]
                Kff = Kp[free, :][:, free]
                alpha[free] = spla.spsolve(Kff.tocsc(), rhs)
                if np.array_equal(new_active, active):
                    break
                active = new_active
            alpha = np.clip(alpha, 0.0, 1.0)
        else:
            # HISTORY (Miehe et al. 2010): driving force already frozen at
            # its running max in self.H (see update_history), so a single
            # linear solve plus a monotonicity projection is sufficient.
            alpha = alpha_old.copy()
            if fixed_nodes.size:
                alpha[fixed_nodes] = fixed_vals
            rhs = F[free] - K[free, :][:, fixed_nodes] @ fixed_vals if fixed_nodes.size else F[free]
            Kff = K[free, :][:, free]
            alpha[free] = spla.spsolve(Kff.tocsc(), rhs)
            alpha = np.clip(alpha, 0.0, 1.0)
            alpha = np.maximum(alpha, alpha_old)

        self.alpha = alpha
        return alpha


    # ------------------------------------------------------------------
    def energies(self):
        eps = self._strain(self.u)
        alpha_avg = self.alpha[self.mesh.elements].mean(axis=1)
        g_e = degradation(alpha_avg)
        psi_plus, psi_minus, *_ = split_energy_stress(eps, self.lam, self.mu, self.split_type)
        areas = self._area
        el_en = np.sum(areas * (g_e * psi_plus + psi_minus))

        # fracture energy = Gc/cw * int( w(alpha)/l + l|grad alpha|^2 )   [dimensional]
        #                 = (1/cw) * int( w(alpha) + l^2|grad alpha|^2 ) [nondim, Gc=1]
        a_nodes = self.alpha[self.mesh.elements]                      # (n_el,3)
        grad_a_vec = np.einsum('nia,ni->na', self._grad, a_nodes)     # (n_el,2)
        grad_alpha = np.einsum('na,na->n', grad_a_vec, grad_a_vec)    # (n_el,)
        a_avg = a_nodes.mean(axis=1)
        w_alpha = a_avg if self.w_coef == 0 else a_avg ** 2
        if self.nondim:
            frac_en = np.sum(areas * (1.0 / self.cw) * (w_alpha + self.l ** 2 * grad_alpha))
        else:
            frac_en = np.sum(areas * (self.Gc / self.cw) * (w_alpha / self.l + self.l * grad_alpha))
        return el_en, frac_en, el_en + frac_en

    # ------------------------------------------------------------------
    def staggered_step(self, bc: BoundaryConditions, Up, fixed_alpha_nodes=None,
                        max_stag=200, tol=1e-8, tol_alpha=1e-4, picard_iters=15,
                        verbose=False, warn=True):
        """One (pseudo-time) load step: iterate elasticity <-> phase field
        to a converged staggered solution.

        *** FIX: the old version broke out of the loop on a relative-energy
        criterion ALONE. Energy can appear to plateau (relative change below
        `tol`) while alpha is still creeping node-by-node - this is exactly
        what was happening near the SEN_tension critical load: energy
        changing by ~1e-6/iter looks "converged" against a loose tol, but
        alpha (and hence the true equilibrium state) had NOT settled. Now
        BOTH the relative energy change AND the max nodal |alpha - alpha
        from previous iteration| must be below tolerance before accepting
        convergence, and - critically - if max_stag is exhausted WITHOUT
        both criteria met, this prints an explicit WARNING (previously
        silent) and returns converged=False so the caller can react (e.g.
        retry with a smaller Up increment) instead of unknowingly plotting
        a garbage intermediate state. ***

        Returns (total_energy, converged: bool).
        """
        prev_energy = None
        alpha_prev = self.alpha.copy()
        converged = False
        rel, max_dalpha = np.inf, np.inf
        for it in range(max_stag):
            u, active_tension = self.solve_displacement(bc, Up, picard_iters=picard_iters)
            self.update_history(active_tension)
            alpha = self.solve_phasefield(fixed_alpha_nodes)
            _, _, tot_en = self.energies()
            max_dalpha = float(np.max(np.abs(alpha - alpha_prev)))
            if verbose:
                print(f"    stag iter {it+1}: total energy = {tot_en:.6e}  "
                      f"max|d_alpha|={max_dalpha:.3e}")
            if prev_energy is not None:
                rel = abs(tot_en - prev_energy) / max(abs(tot_en), 1e-12)
                if rel < tol and max_dalpha < tol_alpha:
                    prev_energy = tot_en
                    converged = True
                    break
            prev_energy = tot_en
            alpha_prev = alpha.copy()
        if not converged and warn:
            print(f"    WARNING: staggered loop did NOT converge within "
                  f"max_stag={max_stag} at Up={Up:.5f} "
                  f"(rel_energy_change={rel:.2e}, max|d_alpha|={max_dalpha:.2e}). "
                  f"This step's fields are NOT a true equilibrium - raise "
                  f"max_stag and/or shrink the load increment near this Up.")
        return prev_energy, converged