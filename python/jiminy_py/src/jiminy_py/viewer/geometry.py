""" TODO: Write documentation.
"""
from typing import Tuple

import numpy as np

import hppfcl


def extract_vertices_and_faces_from_geometry(
        geom: hppfcl.CollisionGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """Extract vertices and faces from a triangle-based collision geometry

    :param geom: Collision geometry from which to extract data.

    :returns: Tuple containing the vertices and faces as np.ndarray objects
              in this exact order.
    """
    if isinstance(geom, hppfcl.HeightFieldOBBRSS):
        x_grid, y_grid = geom.getXGrid(), geom.getYGrid()
        x_dim, y_dim = len(x_grid), len(y_grid)
        vertices = np.stack((
            np.tile(x_grid, y_dim),
            np.repeat(y_grid, x_dim),
            geom.getHeights().flat
        ), axis=1, dtype=np.float32)

        num_faces = 2 * (x_dim - 1) * (y_dim - 1)
        faces = np.empty((num_faces, 3), dtype=np.uint32)
        tri_index = 0
        for i in range(x_dim - 1):
            for j in range(y_dim - 1):
                k = j * x_dim + i
                faces[tri_index:(tri_index + 2)].flat[:] = (
                    k, k + x_dim + 1, k + 1,
                    k, k + x_dim, k + x_dim + 1)
                tri_index += 2
    else:
        if isinstance(geom, hppfcl.Convex):
            vertices = geom.points()
            num_faces, get_faces = geom.num_polygons, geom.polygons
        elif isinstance(geom, hppfcl.BVHModelBase):
            vertices = geom.vertices().astype(np.float32)
            num_faces, get_faces = geom.num_tris, geom.tri_indices
        else:
            raise ValueError(f"CollisionGeometry '{geom}' is not supported.")

        faces = np.empty((num_faces, 3), dtype=np.uint32)
        for i in range(num_faces):
            tri = get_faces(i)
            for j in range(3):
                faces[i, j] = tri[j]

    return vertices, faces
