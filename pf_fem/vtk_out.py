import numpy as np


def write_vtk_2d(filename, mesh, point_data=None, cell_data=None, title="phase field FEA"):
    """Write a legacy ASCII VTK file (UNSTRUCTURED_GRID, triangles) readable
    directly by Paraview.
    point_data: dict name -> (n_node,) or (n_node,3) array
    cell_data : dict name -> (n_elem,) or (n_elem,3) array
    """
    n_node = mesh.num_node
    n_el = mesh.num_elem
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n_node} float\n")
        for p in mesh.nodes:
            f.write(f"{p[0]:.8e} {p[1]:.8e} 0.0\n")
        f.write(f"CELLS {n_el} {n_el*4}\n")
        for el in mesh.elements:
            f.write(f"3 {el[0]} {el[1]} {el[2]}\n")
        f.write(f"CELL_TYPES {n_el}\n")
        for _ in range(n_el):
            f.write("5\n")  # VTK_TRIANGLE

        if point_data:
            f.write(f"POINT_DATA {n_node}\n")
            for name, arr in point_data.items():
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                    for v in arr:
                        f.write(f"{v:.8e}\n")
                else:
                    ncomp = arr.shape[1]
                    if ncomp == 2:
                        arr = np.column_stack([arr, np.zeros(arr.shape[0])])
                        ncomp = 3
                    f.write(f"VECTORS {name} float\n")
                    for row in arr:
                        f.write(" ".join(f"{v:.8e}" for v in row) + "\n")

        if cell_data:
            f.write(f"CELL_DATA {n_el}\n")
            for name, arr in cell_data.items():
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                    for v in arr:
                        f.write(f"{v:.8e}\n")
                else:
                    ncomp = arr.shape[1]
                    if ncomp == 2:
                        arr = np.column_stack([arr, np.zeros(arr.shape[0])])
                    f.write(f"VECTORS {name} float\n")
                    for row in arr:
                        f.write(" ".join(f"{v:.8e}" for v in row) + "\n")
