# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import os
import numpy as np
import trimesh
import pyvista as pv
from robot.shape.shape_utils import get_scale_and_center
from robot.datasets.vtk_utils import convert_faces_into_file_format
from robot.datasets.data_utils import compute_interval

PROGRAM_PATH = "/"
get_shape_path = (
    lambda shape_name: os.path.join(
        os.path.join(PROGRAM_PATH, "shapes", shape_name, "train", shape_name)
    )
    + ".off"
)


def read_off(fpath):
    shape = trimesh.load(fpath)
    return shape.vertices, shape.faces


def get_shape(shape_name):
    shape_path = get_shape_path(shape_name)
    verts, faces = read_off(shape_path)
    return verts, faces


def normalize_vertice(vertices):
    scale, shift = get_scale_and_center(vertices, percentile=100)
    vertices = (vertices - shift) / scale
    return vertices


def subdivide(vertices, faces, level=2):
    for _ in range(level):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    return vertices, faces


if __name__ == "__main__":
    shape_name = "3d_sphere"
    level = 1
    saving_path = "/playpen-raid1/zyshen/debug/robot/divide_{}_level{}.vtk".format(
        shape_name, level
    )
    verts, faces = get_shape(shape_name)
    verts, faces = subdivide(verts, faces, level=level)
    verts = normalize_vertice(verts).astype(np.float32)
    faces = convert_faces_into_file_format(faces)
    compute_interval(verts)
    data = pv.PolyData(verts, faces)
    data.save(saving_path)
    shape_name = "3d_cube"
    level = 4
    saving_path = "/playpen-raid1/zyshen/debug/robot/divide_{}_level{}.vtk".format(
        shape_name, level
    )
    verts, faces = get_shape(shape_name)
    verts, faces = subdivide(verts, faces, level=level)
    verts = normalize_vertice(verts).astype(np.float32)
    faces = convert_faces_into_file_format(faces)
    compute_interval(verts)
    data = pv.PolyData(verts, faces)
    data.save(saving_path)

    """
    the min interval is 0.0191220190793802
    the min interval is 0.125
    """
