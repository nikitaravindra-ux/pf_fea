"""
Mesh generation utilities.

Provides:
    - bar_mesh_1d(L, n_el)                     : 1D linear bar mesh
    - rectangle_mesh(Lx, Ly, nx, ny, origin)    : structured triangular mesh
      of a rectangle, CROSSED (4 triangles per quad, via a center node)
    - l_shape_mesh(...)                        : L-shaped panel built by
      removing one quadrant of a rectangular mesh
    - l_shape_mesh_graded(...)                 : L-shaped panel with a
      dx_fine-resolution CORRIDOR band covering the whole expected crack
      path (not just the corner), see fix #2 below

*** FIX #1 - MESH DIRECTIONAL BIAS ***
Every quad-to-triangle split in this file previously used a SINGLE fixed
diagonal (n00-n11) for every quad in the mesh. On a structured grid that is
a real, well-documented source of directional bias for phase-field /
gradient-damage models: a crack whose "true" (continuum) path is not
exactly aligned with that diagonal (or with the axis directions) can get
numerically pinned to a mesh line instead. Confirmed directly by an actual
run: on the old mesh, the L-panel crack stayed locked to the single node
row y=0 for the first ~29 load steps, never curving upward toward the
applied load. Fixed by splitting every quad into 4 triangles around its
own center node ("crossed"/union-jack pattern), which is symmetric under
90-degree rotation and so does not prefer either diagonal.

*** FIX #2 - REFINEMENT ZONE TOO SMALL (root cause of the energy staircase) ***
l_shape_mesh_graded() previously used a single radially-graded axis
(_graded_axis, now removed) shared by x and y, fine only within
`fine_half_width` (e.g. 0.12) of the reentrant corner (0,0) and coarsening
geometrically beyond that in EVERY direction. But the L-panel crack does
not stay near the corner - confirmed directly from a full run: the crack
tip reached crack_x=-0.393 by Up=1.2, while the old fine zone only
extended to +/-0.12. Beyond that radius, element size grew geometrically
(ratio 1.15 per ring) up to 8x the paper's resolution (0.04 vs 0.005) by
the time the tip got there. Every time the propagating crack tip crossed
from one (increasingly large) element into the next, the released energy
jumped by a correspondingly larger discrete amount - this is exactly the
"staircase" visible in the elastic/fracture energy vs. Up plot, and why
the steps visibly grow in size as Up increases (the tip is moving through
progressively coarser rings as it travels further from the corner).

Fixed by refining a CORRIDOR: dx_fine resolution in x across essentially
the entire domain width (the crack can travel the full width of the left
arm), and dx_fine resolution in y only within a band [band_lo, band_hi]
around the corner height y=0 (sized from the actual observed crack
y-range, with margin), coarsening in y only outside that band. This
matches how the paper itself describes its mesh ("regions of the domain
where the crack is expected to propagate are finely discretized... away
from these regions, element size smoothly increases up to 4l") - a
corridor along the whole expected path, not a small patch at one point.
"""
import numpy as np


class Mesh1D:
    def __init__(self, nodes, elements):
        self.nodes = nodes            # (n_node,) x-coordinates
        self.elements = elements      # (n_el, 2) node indices
        self.num_node = nodes.shape[0]
        self.num_elem = elements.shape[0]
        self.dim = 1


class Mesh2D:
    def __init__(self, nodes, elements):
        self.nodes = nodes            # (n_node, 2)
        self.elements = elements      # (n_el, 3) node indices (CST, ccw)
        self.num_node = nodes.shape[0]
        self.num_elem = elements.shape[0]
        self.dim = 2


def bar_mesh_1d(x0, x1, n_el):
    nodes = np.linspace(x0, x1, n_el + 1)
    elements = np.column_stack([np.arange(n_el), np.arange(1, n_el + 1)])
    return Mesh1D(nodes, elements)


def _crossed_quad_grid(xs, ys):
    """Shared helper: build a crossed (4-tri-per-quad, union-jack) CST mesh
    over the tensor grid xs (x-coords) x ys (y-coords). Returns (nodes,
    elements) with corner nodes first (row-major j*nx+i) followed by one
    center node per quad - center nodes are cell midpoints, so they never
    coincide with a domain-boundary or load-edge coordinate and don't
    interfere with np.isclose-based boundary-node lookups elsewhere.
    """
    nx_c, ny_c = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    corner_nodes = np.column_stack([X.ravel(), Y.ravel()])

    def cid(i, j):
        return j * nx_c + i

    centers = []
    elements = []
    n_corner = nx_c * ny_c
    for j in range(ny_c - 1):
        for i in range(nx_c - 1):
            n00, n10 = cid(i, j), cid(i + 1, j)
            n11, n01 = cid(i + 1, j + 1), cid(i, j + 1)
            cx = 0.5 * (xs[i] + xs[i + 1])
            cy = 0.5 * (ys[j] + ys[j + 1])
            c = n_corner + len(centers)
            centers.append([cx, cy])
            elements.append([n00, n10, c])
            elements.append([n10, n11, c])
            elements.append([n11, n01, c])
            elements.append([n01, n00, c])
    nodes = np.vstack([corner_nodes, np.array(centers).reshape(-1, 2)]) if centers else corner_nodes
    elements = np.array(elements, dtype=int)
    return nodes, elements


def _grade_positions(start, end, dy0, direction, max_dy, growth=1.15):
    """1D geometric grading helper: positions from `start` toward `end`
    (direction = +1 or -1), beginning at spacing dy0, growing by `growth`
    each step up to a cap of max_dy. Shared by refine_band_mesh() and
    l_shape_mesh_graded()."""
    ys = [start]
    dy = dy0
    y = start
    while True:
        y = y + direction * dy
        if direction > 0 and y >= end:
            ys.append(end)
            break
        if direction < 0 and y <= end:
            ys.append(end)
            break
        ys.append(y)
        dy = min(dy * growth, max_dy)
    return ys


def rectangle_mesh(Lx, Ly, nx, ny, origin=(0.0, 0.0)):
    """Structured, CROSSED triangular mesh of [ox, ox+Lx] x [oy, oy+Ly]
    (see module docstring, Fix #1).

    nx, ny : number of divisions along x and y.
    Four triangles per quad cell (union-jack pattern via a center node).
    """
    ox, oy = origin
    xs = np.linspace(ox, ox + Lx, nx + 1)
    ys = np.linspace(oy, oy + Ly, ny + 1)
    nodes, elements = _crossed_quad_grid(xs, ys)
    return Mesh2D(nodes, elements)


def l_shape_mesh(L, notch_frac=0.5, n=44):
    """L-shaped panel occupying [-L/2, L/2] x [-L/2, L/2] with the
    quadrant  x>0 & y<0  removed - full-height LEFT arm (x in [-L/2,0]),
    upper-RIGHT arm only (x in [0,L/2], y in [0,L/2]), reentrant corner at
    (0,0). Built on the crossed background mesh from rectangle_mesh.
    """
    mesh = rectangle_mesh(L, L, n, n, origin=(-L / 2, -L / 2))
    cx = mesh.nodes[mesh.elements].mean(axis=1)
    keep = ~((cx[:, 0] > 0) & (cx[:, 1] < 0))
    elements = mesh.elements[keep]
    used = np.unique(elements)
    remap = -np.ones(mesh.num_node, dtype=int)
    remap[used] = np.arange(used.size)
    nodes = mesh.nodes[used]
    elements = remap[elements]
    return Mesh2D(nodes, elements)


def l_shape_mesh_graded(L, dx_fine, band_lo=-0.05, band_hi=0.15, coarse_ratio=6,
                         x_fine_margin=0.05, growth=1.15,
                         load_seg=None, load_dx_fine=None):
    """L-shaped panel like l_shape_mesh(), but with a dx_fine-resolution
    CORRIDOR mesh covering the whole expected crack path (see module
    docstring, Fix #2), instead of only a small patch at the reentrant
    corner. Full-height LEFT arm, upper-RIGHT arm only (bottom-right
    quadrant missing) - reentrant corner at (0,0).

    band_lo, band_hi: y-range (around the corner height y=0) that stays at
        dx_fine resolution across the ENTIRE x-extent of the domain.
        Choose this from the actual crack y-range you observe (with some
        margin) - e.g. if a first run shows the crack reaching y=+0.06,
        band_hi=0.15 gives comfortable headroom.
    x_fine_margin: x stays at dx_fine resolution everywhere except within
        this distance of the domain's left/right edges (which coarsen,
        since the crack never reaches all the way to x=+/-L/2).
    coarse_ratio, growth: outside the band/margin, element size grows
        geometrically (factor `growth` per ring) up to dx_fine*coarse_ratio.
    load_seg, load_dx_fine: accepted for call-signature compatibility with
        the old corner-only version - unused, since x is now dx_fine
        almost everywhere so the loaded edge is already fully resolved.
    """
    half = L / 2

    # x: fine (dx_fine) across [-(half-x_fine_margin), +(half-x_fine_margin)],
    # coarsening only in the last x_fine_margin near each edge.
    x_fine_extent = half - x_fine_margin
    xs_pos = _grade_positions(0.0, x_fine_extent, dx_fine, +1, dx_fine, growth)
    xs_pos = xs_pos + _grade_positions(x_fine_extent, half, dx_fine, +1,
                                        dx_fine * coarse_ratio, growth)[1:]
    xs = np.unique(np.round(np.concatenate([-np.array(xs_pos)[::-1], xs_pos]), 12))

    # y: fine (dx_fine) within [band_lo, band_hi], coarsening outside it.
    ys_mid = list(np.arange(band_lo, band_hi + 1e-12, dx_fine))
    ys_top = _grade_positions(band_hi, half, dx_fine, +1, dx_fine * coarse_ratio, growth)[1:]
    ys_bot = _grade_positions(band_lo, -half, dx_fine, -1, dx_fine * coarse_ratio, growth)[1:]
    ys = np.array(sorted(set(round(v, 12) for v in (ys_bot[::-1] + ys_mid + ys_top))))

    nodes, elements = _crossed_quad_grid(xs, ys)

    cx = nodes[elements].mean(axis=1)
    keep = ~((cx[:, 0] > 0) & (cx[:, 1] < 0))
    elements = elements[keep]
    used = np.unique(elements)
    remap = -np.ones(nodes.shape[0], dtype=int)
    remap[used] = np.arange(used.size)
    nodes = nodes[used]
    elements = remap[elements]
    return Mesh2D(nodes, elements)


def refine_band_mesh_quad(Lx, Ly, band_y, band_half_width, ne_fine, coarse_ratio=6,
                           growth=1.15, origin=(0.0, 0.0)):
    """Structured Q4 (bilinear quad) mesh matching refine_band_mesh's
    corridor grading (fine at dy=Ly/ne_fine inside |y-band_y|<band_half_width,
    growing geometrically away from the band), but built directly as
    quadrilaterals - matching Appendix G's element type - instead of
    crossed triangles. No directional-bias mitigation is needed here the
    way it was for the triangle mesh (mesh.py's Fix #1): a structured quad
    grid has no diagonal to bias toward in the first place.
    """
    ox, oy = origin
    dy_fine = Ly / ne_fine
    y_band_lo = band_y - band_half_width
    y_band_hi = band_y + band_half_width

    ys_mid = list(np.arange(y_band_lo, y_band_hi + 1e-12, dy_fine))
    ys_top = _grade_positions(y_band_hi, oy + Ly, dy_fine, +1, dy_fine * coarse_ratio, growth)[1:]
    ys_bot = _grade_positions(y_band_lo, oy, dy_fine, -1, dy_fine * coarse_ratio, growth)[1:]
    ys = np.array(sorted(set(round(v, 12) for v in (ys_bot[::-1] + ys_mid + ys_top))))

    nx = int(np.round(Lx / dy_fine))
    xs = np.linspace(ox, ox + Lx, nx + 1)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    npx, npy = len(xs), len(ys)

    def nid(i, j):
        return j * npx + i

    elements = []
    for j in range(npy - 1):
        for i in range(npx - 1):
            n1 = nid(i, j); n2 = nid(i + 1, j)
            n3 = nid(i + 1, j + 1); n4 = nid(i, j + 1)
            elements.append([n1, n2, n3, n4])
    return Mesh2DQuad(nodes, np.array(elements, dtype=int))


class Mesh2DQuad:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self.num_node = nodes.shape[0]
        self.num_elem = elements.shape[0]
        self.dim = 2
        self.elem_type = "Q4"


def rectangle_quad_mesh(L, B, Nx, Ny):
    npx, npy = Nx + 1, Ny + 1
    xs = np.linspace(0, L, npx)
    ys = np.linspace(0, B, npy)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i, j):
        return j * npx + i

    elements = []
    for j in range(Ny):
        for i in range(Nx):
            n1 = nid(i, j); n2 = nid(i + 1, j)
            n3 = nid(i + 1, j + 1); n4 = nid(i, j + 1)
            elements.append([n1, n2, n3, n4])
    return Mesh2DQuad(nodes, np.array(elements, dtype=int))


def rectangle_quad_mesh_notch(L, B, Nx, Ny, Lref_x, Lref_y, Nref_x, Nref_y):
    nx = np.concatenate([
        np.linspace(0, 0.5 * (L - Lref_x), Nx + 1),
        np.linspace(0.5 * (L - Lref_x), 0.5 * (L + Lref_x), Nref_x + 1),
        np.linspace(0.5 * (L + Lref_x), L, Nx + 1),
    ])
    ny = np.concatenate([
        np.linspace(0, 0.5 * (B - Lref_y), Ny + 1),
        np.linspace(0.5 * (B - Lref_y), 0.5 * (B + Lref_y), Nref_y + 1),
        np.linspace(0.5 * (B + Lref_y), B, Ny + 1),
    ])
    nx = np.unique(np.round(nx, 12))
    ny = np.unique(np.round(ny, 12))
    npx, npy = len(nx), len(ny)
    X, Y = np.meshgrid(nx, ny, indexing="xy")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i, j):
        return j * npx + i

    elements = []
    for j in range(npy - 1):
        for i in range(npx - 1):
            n1 = nid(i, j); n2 = nid(i + 1, j)
            n3 = nid(i + 1, j + 1); n4 = nid(i, j + 1)
            elements.append([n1, n2, n3, n4])
    return Mesh2DQuad(nodes, np.array(elements, dtype=int))


def plate_mesh(Nx, Ny, Lnotch_y, Lx=1.0, Ly=1.0):
    """*** ASSUMPTION FLAG *** (unchanged): the real
    specimen.internal.plate.m notch geometry/orientation was not available
    to me. This assigns the notch as a horizontal segment from (0, Ly/2)
    to (Lnotch_y*Ly, Ly/2) - the most common SENT convention."""
    mesh = rectangle_mesh(Lx, Ly, Nx, Ny, origin=(0.0, 0.0))
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    tol = 1e-9
    left = np.where(x <= tol)[0]
    right = np.where(x >= Lx - tol)[0]
    bl_idx = np.argmin(x ** 2 + y ** 2)
    bottom_left = np.array([bl_idx])
    notch_len = Lnotch_y * Ly
    y_mid = Ly / 2.0
    h = Ly / Ny
    crack = np.where((np.abs(y - y_mid) < h * 0.51) & (x <= notch_len + tol))[0]
    nodesets = dict(left=left, right=right, bottom_left=bottom_left, crack=crack)
    crack_segment = ((0.0, y_mid), (notch_len, y_mid))
    return mesh, nodesets, crack_segment


def refine_band_mesh(Lx, Ly, band_y, band_half_width, ne_fine, coarse_ratio=4,
                      origin=(0.0, 0.0)):
    """Rectangle mesh with a horizontally graded triangulation: elements of
    size ~Ly/ne_fine inside |y-band_y|<band_half_width, growing
    geometrically away from the band, crossed quad-split (Fix #1)."""
    ox, oy = origin
    dy_fine = Ly / ne_fine
    y_band_lo = band_y - band_half_width
    y_band_hi = band_y + band_half_width

    ys_mid = list(np.arange(y_band_lo, y_band_hi + 1e-12, dy_fine))
    ys_top = _grade_positions(y_band_hi, oy + Ly, dy_fine, +1, dy_fine * coarse_ratio)[1:]
    ys_bot = _grade_positions(y_band_lo, oy, dy_fine, -1, dy_fine * coarse_ratio)[1:]
    ys = sorted(set([round(v, 12) for v in (ys_bot[::-1] + ys_mid + ys_top)]))
    ys = np.array(ys)

    nx = int(np.round(Lx / dy_fine))
    xs = np.linspace(ox, ox + Lx, nx + 1)

    nodes, elements = _crossed_quad_grid(xs, ys)
    return Mesh2D(nodes, elements)