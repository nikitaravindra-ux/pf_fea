"""
Matplotlib plotting of spatial fields (u, v, alpha), matching the 3-panel
layout used throughout the CMAME paper's figures (e.g. Fig. 6, 9, 11, 14,
17: displacement components + phase field, side by side, at a fixed Up).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def plot_fields_png(filename, mesh, u, alpha, Up=None, title=None,
                     u_range=None, v_range=None, alpha_range=(0.0, 1.0)):
    """mesh: Mesh2D (CST triangles) or Mesh2DQuad.
    u: (2*n_node,) flat displacement vector [u0,v0,u1,v1,...]
    alpha: (n_node,) phase field
    Saves a 3-panel figure (u, v, alpha) to `filename` (PNG).
    """
    nodes = mesh.nodes
    ux = u[0::2]
    uy = u[1::2]

    if mesh.elements.shape[1] == 3:
        triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], mesh.elements)
    else:
        # split each quad into 2 triangles for matplotlib's triangulation-based plotting
        quads = mesh.elements
        tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    panels = [
        (ux, r"$u_\theta$", "viridis", u_range),
        (uy, r"$v_\theta$", "viridis", v_range),
        (alpha, r"$\alpha_\theta$", "viridis", alpha_range),
    ]
    for ax, (field, label, cmap, rng) in zip(axes, panels):
        vmin, vmax = (None, None) if rng is None else rng
        tpc = ax.tripcolor(triang, field, shading="gouraud", cmap=cmap,
                            vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(tpc, ax=ax, shrink=0.85)

    suptitle = title or ""
    if Up is not None:
        suptitle += f"  (Up = {Up:.4f})" if suptitle else f"Up = {Up:.4f}"
    if suptitle:
        fig.suptitle(suptitle)

    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_energy_png(filename, Up_values, elastic, fracture, title=None):
    """Elastic + fracture energy vs Up, matching Fig. 3/7/10/13/15/18 style."""
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.plot(Up_values, elastic, label=r"$\mathcal{E}^{el}$")
    ax.plot(Up_values, fracture, label=r"$\mathcal{E}^{d}$")
    ax.set_xlabel(r"$U_p$")
    ax.set_ylabel(r"$\mathcal{E}$")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
