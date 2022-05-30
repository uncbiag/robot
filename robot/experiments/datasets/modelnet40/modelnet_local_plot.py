import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath("../../../.."))
from robot.experiments.datasets.modelnet40.get_one_case import get_one_case
from robot.utils.local_feature_extractor import compute_anisotropic_gamma_from_points
from robot.utils.visualizer import (
    visualize_point_fea_with_arrow,
    visualize_point_overlap,
)

import torch
from robot.utils.shape_visual_utils import make_ellipsoid
from robot.global_variable import shape_type

assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cpu")  # cuda:0  cpu

source, target = get_one_case(device=device)(item=1)
points = source.points
npoints = points.shape[1]
points_np = points.detach().cpu().numpy().squeeze()
aniso_kernel_scale = 0.05
Gamma, principle_weight, eigenvector, mass = compute_anisotropic_gamma_from_points(
    points,
    cov_sigma_scale=0.02,
    aniso_kernel_scale=aniso_kernel_scale,
    leaf_decay=True,
    principle_weight=None,
    eigenvalue_min=0.1,
    iter_twice=True,
    return_details=True,
)
principle_weight_np = principle_weight.squeeze().numpy()
eigenvector_np = eigenvector.squeeze().numpy()
nsample = 10
full_index = np.arange(npoints)
index = np.random.choice(full_index, nsample, replace=False)
spheres_list = [
    make_ellipsoid(
        200,
        radius=principle_weight_np[ind],
        center=points_np[ind],
        rotation=eigenvector_np[ind],
    )
    for ind in index
]
spheres = np.concatenate(spheres_list, 0)
fg_spheres_color = np.array([[0.8, 0.5, 0.2]] * len(spheres))
# visualize_point_fea(points, mass , rgb_on=False)

visualize_point_overlap(
    points,
    spheres,
    points,
    fg_spheres_color,
    "aniso_filter_with_kernel_radius",
)
