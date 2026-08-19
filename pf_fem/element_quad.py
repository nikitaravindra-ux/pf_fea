"""
Bilinear 4-node quadrilateral (Q4) element, 2x2 Gauss quadrature - matching
the element type used for the reference FEA in Appendix G of the paper
("quadrilateral elements with bilinear shape functions and two Gauss
points per parametric direction").

Node ordering (CCW): 1=bottom-left, 2=bottom-right, 3=top-right, 4=top-left
(matching MeshRectanglularPlate.m's connectivity convention).
"""
import numpy as np

_GP = 1.0 / np.sqrt(3.0)
GAUSS_2x2 = [(-_GP, -_GP, 1.0), (_GP, -_GP, 1.0), (_GP, _GP, 1.0), (-_GP, _GP, 1.0)]


def quad_shape(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                          (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dN_dxi, dN_deta


def quad_gauss_data(coords):
    """coords: (4,2) node coordinates, CCW.
    Returns a list of 4 (weight_times_detJ, N(4,), B(3,8), gradN(2,4)) tuples,
    one per 2x2 Gauss point - cached once per element since geometry is fixed.
    gradN rows are [dN/dx ; dN/dy], used by the scalar phase-field problem.
    """
    data = []
    for xi, eta, w in GAUSS_2x2:
        N, dN_dxi, dN_deta = quad_shape(xi, eta)
        J = np.zeros((2, 2))
        J[0, 0] = dN_dxi @ coords[:, 0]
        J[0, 1] = dN_dxi @ coords[:, 1]
        J[1, 0] = dN_deta @ coords[:, 0]
        J[1, 1] = dN_deta @ coords[:, 1]
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Non-positive Jacobian - check quad node ordering (must be CCW)")
        invJ = np.linalg.inv(J)
        # [dN/dx; dN/dy] = invJ^T @ [dN/dxi; dN/deta]
        gradN = invJ.T @ np.vstack([dN_dxi, dN_deta])  # (2,4)
        dN_dx, dN_dy = gradN[0], gradN[1]

        B = np.zeros((3, 8))
        for a in range(4):
            B[0, 2 * a] = dN_dx[a]
            B[1, 2 * a + 1] = dN_dy[a]
            B[2, 2 * a] = dN_dy[a]
            B[2, 2 * a + 1] = dN_dx[a]
        data.append((w * detJ, N, B, gradN))
    return data
