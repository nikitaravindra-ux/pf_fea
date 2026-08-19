import numpy as np


def tri_geom(coords):
    """coords: (3,2) node coordinates (ccw).
    Returns area, grad (3,2) shape-function gradients [dN_a/dx, dN_a/dy],
    and B (3,6) strain-displacement matrix for Voigt eps=[exx,eyy,gamma_xy].
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * A2
    if area <= 0:
        raise ValueError("Non positive triangle area - check node ordering")
    b = np.array([y2 - y3, y3 - y1, y1 - y2]) / A2
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / A2
    grad = np.column_stack([b, c])  # (3,2)  dN/dx , dN/dy
    B = np.zeros((3, 6))
    for a in range(3):
        B[0, 2 * a] = b[a]
        B[1, 2 * a + 1] = c[a]
        B[2, 2 * a] = c[a]
        B[2, 2 * a + 1] = b[a]
    return area, grad, B
