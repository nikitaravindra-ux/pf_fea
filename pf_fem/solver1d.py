"""
1D staggered phase-field brittle-fracture FEA solver (linear elastic bar,
no strain-energy split). Set nondim=True to match Eq. (26) of the paper
exactly (Section 4.2); default is the standard dimensional Eq. (1) form.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .material import degradation, diss_params


class PhaseFieldFEA1D:
    def __init__(self, mesh, E=1.0, Gc=1.0, l=0.05, diss_fct="AT1", nondim=False):
        self.mesh = mesh
        self.E = E
        self.Gc = Gc
        self.l = l
        self.cw, self.w_const, self.w_coef = diss_params(diss_fct)
        self.nondim = nondim

        n = mesh.num_node
        self.alpha = np.zeros(n)
        self.u = np.zeros(n)
        self.H = np.zeros(mesh.num_elem)
        self._defect = np.ones(mesh.num_elem)  # local toughness multiplier (see seed_defect)

        self._h = np.diff(mesh.nodes)  # element lengths

    # ------------------------------------------------------------------
    def seed_defect(self, x_defect, frac=1e-3):
        """Introduce a tiny (frac) local reduction of the fracture toughness
        at x_defect. Phase-field crack NUCLEATION (no pre-existing notch) is
        a symmetry-breaking bifurcation problem: on a perfectly symmetric
        mesh, a deterministic linear solve can get stuck exactly on the
        unphysical *uniform*-damage equilibrium branch instead of
        spontaneously localizing to the correct (lower-energy) profile
        (the paper's NN avoids this "for free" via its random weight
        initialization; a symmetric FEM mesh has no such built-in
        asymmetry). This seeds a deterministic nucleation site, matching
        standard FEM practice for this class of problem.
        """
        centroids = 0.5 * (self.mesh.nodes[self.mesh.elements[:, 0]] +
                            self.mesh.nodes[self.mesh.elements[:, 1]])
        e = np.argmin(np.abs(centroids - x_defect))
        self._defect[e] = 1.0 - frac

    # ------------------------------------------------------------------
    def init_crack(self, x_crack, diss_fct="AT1"):
        """Assign the nodal phase field to the closed-form optimal-profile
        solution around x_crack (see PhaseFieldFEA.init_crack in solver2d.py
        for the rationale)."""
        d = np.abs(self.mesh.nodes - x_crack)
        if diss_fct.upper() == "AT1":
            a0 = np.where(d < 2 * self.l, (1.0 - d / (2 * self.l)) ** 2, 0.0)
        else:
            a0 = np.exp(-d / self.l)
            a0 = np.where(a0 > 1e-3, a0, 0.0)
        self.alpha = np.maximum(self.alpha, a0)

    # ------------------------------------------------------------------
    def solve_displacement(self, left_fixed=True, right_disp=None):
        n = self.mesh.num_node
        alpha_e = self.alpha[self.mesh.elements].mean(axis=1)
        g_e = degradation(alpha_e)
        rows, cols, vals = [], [], []
        for e, el in enumerate(self.mesh.elements):
            h = self._h[e]
            k = g_e[e] * self.E / h
            Ke = k * np.array([[1, -1], [-1, 1]])
            for a in range(2):
                for b in range(2):
                    rows.append(el[a]); cols.append(el[b]); vals.append(Ke[a, b])
        K = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

        fixed = {}
        if left_fixed:
            fixed[0] = 0.0
        if right_disp is not None:
            fixed[n - 1] = right_disp
        fixed_dofs = np.array(sorted(fixed.keys()))
        fixed_vals = np.array([fixed[d] for d in fixed_dofs])
        free = np.array([i for i in range(n) if i not in fixed])

        u = np.zeros(n)
        u[fixed_dofs] = fixed_vals
        rhs = -K[free, :][:, fixed_dofs] @ fixed_vals
        Kff = K[free, :][:, free]
        u[free] = spla.spsolve(Kff.tocsc(), rhs)
        self.u = u
        return u

    def _strain(self):
        du = np.diff(self.u)
        return du / self._h

    def update_history(self):
        strain = self._strain()
        psi = 0.5 * self.E * strain ** 2
        self.H = np.maximum(self.H, psi)
        return psi

    # ------------------------------------------------------------------
    def solve_phasefield(self, fixed_nodes=None):
        n = self.mesh.num_node
        rows, cols, vals = [], [], []
        F = np.zeros(n)
        if self.nondim:
            coefA = 1.0 / self.cw
            diff_len = self.l ** 2
            inv_l_w = 1.0
        else:
            coefA = self.Gc / self.cw
            diff_len = self.l
            inv_l_w = 1.0 / self.l
        for e, el in enumerate(self.mesh.elements):
            h = self._h[e]
            He = self.H[e]
            coefAe = coefA * self._defect[e]
            Kdiff = (coefAe * 2 * diff_len / h) * np.array([[1, -1], [-1, 1]])
            coef_e = 2 * He + coefAe * self.w_coef * inv_l_w
            Mloc = (h / 6.0) * np.array([[2, 1], [1, 2]])
            Ke = Kdiff + coef_e * Mloc
            Fe = (2 * He - coefAe * self.w_const * inv_l_w) * (h / 2.0) * np.ones(2)
            for a in range(2):
                F[el[a]] += Fe[a]
                for b in range(2):
                    rows.append(el[a]); cols.append(el[b]); vals.append(Ke[a, b])
        K = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

        fixed_nodes = fixed_nodes or {}
        fixed_idx = np.array(sorted(fixed_nodes.keys()), dtype=int) if fixed_nodes else np.array([], dtype=int)
        fixed_vals = np.array([fixed_nodes[k] for k in fixed_idx]) if fixed_nodes else np.array([])
        free_mask = np.ones(n, dtype=bool)
        if fixed_idx.size:
            free_mask[fixed_idx] = False
        free = np.where(free_mask)[0]

        alpha = self.alpha.copy()
        if fixed_idx.size:
            alpha[fixed_idx] = fixed_vals
        rhs = F[free] - (K[free, :][:, fixed_idx] @ fixed_vals if fixed_idx.size else 0.0)
        Kff = K[free, :][:, free]
        alpha[free] = spla.spsolve(Kff.tocsc(), rhs)

        alpha = np.clip(alpha, 0.0, 1.0)
        alpha = np.maximum(alpha, self.alpha)  #irreversible damage
        self.alpha = alpha
        return alpha

    # ------------------------------------------------------------------
    def energies(self):
        strain = self._strain()
        alpha_e = self.alpha[self.mesh.elements].mean(axis=1)
        g_e = degradation(alpha_e)
        el_en = np.sum(self._h * 0.5 * g_e * self.E * strain ** 2)

        a_nodes = self.alpha[self.mesh.elements]
        grad_a = np.diff(self.alpha) / self._h
        a_avg = a_nodes.mean(axis=1)
        w_a = a_avg if self.w_coef == 0 else a_avg ** 2
        if self.nondim:
            frac_en = np.sum(self._h * (1.0 / self.cw) * (w_a + self.l ** 2 * grad_a ** 2))
        else:
            frac_en = np.sum(self._h * (self.Gc / self.cw) * (w_a / self.l + self.l * grad_a ** 2))
        return el_en, frac_en, el_en + frac_en

    # ------------------------------------------------------------------
    def staggered_step(self, right_disp, fixed_alpha_nodes=None, left_fixed=True,
                        max_stag=50, tol=1e-8, verbose=False):
        prev = None
        for it in range(max_stag):
            self.solve_displacement(left_fixed=left_fixed, right_disp=right_disp)
            self.update_history()
            self.solve_phasefield(fixed_alpha_nodes)
            _, _, tot = self.energies()
            if verbose:
                print(f"    stag {it+1}: E={tot:.6e}")
            if prev is not None and abs(tot - prev) / max(abs(tot), 1e-12) < tol:
                prev = tot
                break
            prev = tot
        return prev
