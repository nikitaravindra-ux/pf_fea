"""
Phase-field material model.

Implements the AT1 / AT2 degradation & dissipation functions and the
Isotropic / Amor (volumetric-deviatoric) strain-energy splits, following
Section 2.1 of Manav, Molinaro, Mishra, De Lorenzis, CMAME 429 (2024) 117104.
"""
import numpy as np

ETA = 1e-6  # residual stiffness g(1) = eta (avoids full singularity)


def diss_params(diss_fct):
    """Return (cw, w_const, w_coef) such that w'(alpha) = w_const + w_coef*alpha.
    AT1: w(a)=a      -> w'=1            , cw = 8/3
    AT2: w(a)=a**2   -> w'=2a           , cw = 2
    """
    if diss_fct == "AT1":
        return 8.0 / 3.0, 1.0, 0.0
    elif diss_fct == "AT2":
        return 2.0, 0.0, 2.0
    else:
        raise ValueError(f"unknown diss_fct {diss_fct}")


def degradation(alpha):
    """g(alpha) = (1-alpha)^2 + eta ,   g'(alpha) = 2*alpha - 2"""
    g = (1.0 - alpha) ** 2 + ETA
    return g


def macaulay_plus(x):
    return np.maximum(x, 0.0)


def macaulay_minus(x):
    return np.minimum(x, 0.0)


# ---------------------------------------------------------------------
# Strain-energy split (plane strain).  eps given in Voigt notation
# [exx, eyy, gamma_xy] (engineering shear).
# ---------------------------------------------------------------------
def split_energy_stress(eps, lam, mu, split_type):
    """Return psi_plus, psi_minus, sigma_plus, sigma_minus (Voigt [sxx,syy,sxy])
    and the "active" flag per element used to build the tangent (bulk modulus
    branch) for the AMOR split.
    eps: (n,3) array
    """
    exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    tr = exx + eyy

    if split_type.upper() == "ISO":
        # No split: full elastic energy is degraded (Psi- = 0)
        C11 = lam + 2 * mu
        C12 = lam
        C33 = mu
        sxx = C11 * exx + C12 * eyy
        syy = C12 * exx + C11 * eyy
        sxy = C33 * gxy
        psi_plus = 0.5 * (sxx * exx + syy * eyy + sxy * gxy)
        psi_minus = np.zeros_like(psi_plus)
        sigma_plus = np.column_stack([sxx, syy, sxy])
        sigma_minus = np.zeros_like(sigma_plus)
        active_tension = np.ones_like(tr, dtype=bool)
        return psi_plus, psi_minus, sigma_plus, sigma_minus, active_tension

    elif split_type.upper() == "AMOR":
        K = lam + 2.0 / 3.0 * mu
        tr_p = macaulay_plus(tr)
        tr_m = macaulay_minus(tr)
        # deviatoric strain (in-plane relevant components), 3D deviator with ezz=-tr/3
        exx_dev = exx - tr / 3.0
        eyy_dev = eyy - tr / 3.0
        ezz_dev = -tr / 3.0
        # tr(e:e) = exx_dev^2+eyy_dev^2+ezz_dev^2 + 2*(gxy/2)^2
        tr_ee = exx_dev ** 2 + eyy_dev ** 2 + ezz_dev ** 2 + 2.0 * (gxy / 2.0) ** 2

        psi_plus = 0.5 * K * tr_p ** 2 + mu * tr_ee
        psi_minus = 0.5 * K * tr_m ** 2

        sxx_p = K * tr_p + 2 * mu * exx_dev
        syy_p = K * tr_p + 2 * mu * eyy_dev
        sxy_p = 2 * mu * (gxy / 2.0)  # = mu*gxy

        sxx_m = K * tr_m
        syy_m = K * tr_m
        sxy_m = np.zeros_like(sxy_p)

        sigma_plus = np.column_stack([sxx_p, syy_p, sxy_p])
        sigma_minus = np.column_stack([sxx_m, syy_m, sxy_m])
        active_tension = tr >= 0.0
        return psi_plus, psi_minus, sigma_plus, sigma_minus, active_tension

    elif split_type.upper() == "MIEHE":
        # Spectral (principal-strain) split, Miehe et al. 2010: decompose the
        # strain tensor into positive/negative parts via its eigenvalues,
        # then Psi+ uses <eps>+ (only positive eigenvalues kept), Psi- uses
        # <eps>-. Fully vectorized via batched np.linalg.eigh (n,2,2).
        n = eps.shape[0]
        E2 = np.zeros((n, 2, 2)) 
        E2[:, 0, 0] = exx
        E2[:, 1, 1] = eyy
        E2[:, 0, 1] = gxy / 2.0
        E2[:, 1, 0] = gxy / 2.0
        evals, evecs = np.linalg.eigh(E2)          # evals:(n,2), evecs:(n,2,2)
        ep = macaulay_plus(evals)                   # (n,2)
        em = macaulay_minus(evals)
        # Ep = evecs @ diag(ep) @ evecs.T, batched
        Ep = np.einsum('nij,nj,nkj->nik', evecs, ep, evecs)
        Em = np.einsum('nij,nj,nkj->nik', evecs, em, evecs)
        tr_p = macaulay_plus(tr)
        tr_m = macaulay_minus(tr)
        psi_plus = 0.5 * lam * tr_p ** 2 + mu * np.einsum('nij,nij->n', Ep, Ep)
        psi_minus = 0.5 * lam * tr_m ** 2 + mu * np.einsum('nij,nij->n', Em, Em)
        eye2 = np.eye(2)[None, :, :]
        sig_p = lam * tr_p[:, None, None] * eye2 + 2 * mu * Ep
        sig_m = lam * tr_m[:, None, None] * eye2 + 2 * mu * Em
        sigma_plus = np.column_stack([sig_p[:, 0, 0], sig_p[:, 1, 1], sig_p[:, 0, 1]])
        sigma_minus = np.column_stack([sig_m[:, 0, 0], sig_m[:, 1, 1], sig_m[:, 0, 1]])
        active_tension = tr >= 0.0
        return psi_plus, psi_minus, sigma_plus, sigma_minus, active_tension

    else:
        raise ValueError(f"unknown split_type {split_type}")


def tangent_matrices(lam, mu, split_type, active_tension):
    """Return per-element 3x3 Voigt tangent matrices D_plus, D_minus such that
    sigma_plus = D_plus @ eps , sigma_minus = D_minus @ eps  (piecewise-linear
    in the AMOR case, exact linear-elastic in the ISO case).
    active_tension: (n,) bool, only used by AMOR (branch selection for K term).
    """
    n = active_tension.shape[0]
    if split_type.upper() == "ISO":
        C11 = lam + 2 * mu
        C12 = lam
        C33 = mu
        Dp = np.array([[C11, C12, 0.0], [C12, C11, 0.0], [0.0, 0.0, C33]])
        D_plus = np.repeat(Dp[None, :, :], n, axis=0)
        D_minus = np.zeros_like(D_plus)
        return D_plus, D_minus

    elif split_type.upper() == "AMOR":
        K = lam + 2.0 / 3.0 * mu
        # deviatoric projector (in Voigt, engineering shear) acting alone (mu part)
        # sigma_dev = 2*mu*[exx-tr/3, eyy-tr/3, gxy/2]
        Pdev = np.array([[2.0 / 3.0, -1.0 / 3.0, 0.0],
                          [-1.0 / 3.0, 2.0 / 3.0, 0.0],
                          [0.0, 0.0, 0.5]])
        D_dev = 2 * mu * Pdev
        Ivec = np.array([1.0, 1.0, 0.0])
        Kmat = K * np.outer(Ivec, Ivec)  # K * tr * [1,1,0] ,  d(K tr)/deps = K * I⊗I

        D_plus = np.zeros((n, 3, 3))
        D_minus = np.zeros((n, 3, 3))
        for e in range(n):
            if active_tension[e]:
                D_plus[e] = Kmat + D_dev
                D_minus[e] = 0.0
            else:
                D_plus[e] = D_dev
                D_minus[e] = Kmat
        return D_plus, D_minus
    elif split_type.upper() == "MIEHE":
        raise NotImplementedError(
            "MIEHE tangent is computed numerically in "
            "solver2d._elasticity_assemble; tangent_matrices() does not "
            "support split_type='MIEHE' directly.")
    else:
        raise ValueError(f"unknown split_type {split_type}")
