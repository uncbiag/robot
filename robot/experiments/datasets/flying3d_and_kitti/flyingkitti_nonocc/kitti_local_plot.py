import os, sys
import numpy as np
from robot.utils.local_feature_extractor import compute_anisotropic_gamma_from_points
from robot.utils.visualizer import (
    visualize_point_fea_with_arrow,
    visualize_point_overlap,
)

sys.path.insert(0, os.path.abspath("../../../.."))
import torch
from robot.datasets.data_utils import get_obj, read_json_into_list
from robot.utils.shape_visual_utils import make_ellipsoid
from robot.global_variable import shape_type

assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cpu")  # cuda:0  cpu
reader_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_reader()"
normalizer_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_normalizer()"
sampler_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_sampler(num_sample=20000)"
use_local_mount = True
remote_mount_transfer = lambda x: x.replace(
    "/playpen-raid1", "/home/zyshen/remote/llr11_mount"
)
path_transfer = (
    (lambda x: remote_mount_transfer(x)) if use_local_mount else (lambda x: x)
)
dataset_json_path = "/playpen-raid1/zyshen/data/flying3d_nonocc_test_on_kitti/test/pair_data.json"  # home/zyshen/remote/llr11_mount
dataset_json_path = path_transfer(dataset_json_path)
pair_name_list, pair_info_list = read_json_into_list(dataset_json_path)
pair_path_list = [
    [pair_info["source"]["data_path"], pair_info["target"]["data_path"]]
    for pair_info in pair_info_list
]
pair_id = 5
pair_path = pair_path_list[pair_id]
pair_path = [path_transfer(path) for path in pair_path]
get_obj_func = get_obj(
    reader_obj, normalizer_obj, sampler_obj, device, expand_bch_dim=True
)
source, source_interval = get_obj_func(pair_path[1])
source_points, source_weights = source["points"], source["weights"]

points_1 = source["points"][:, :10000]

npoints, points = points_1.shape[1], points_1
points_np = points.detach().cpu().numpy().squeeze()
aniso_kernel_scale = 1.0
Gamma, principle_weight, eigenvector, mass = compute_anisotropic_gamma_from_points(
    points,
    cov_sigma_scale=0.5,
    aniso_kernel_scale=aniso_kernel_scale,
    leaf_decay=True,
    principle_weight=None,
    eigenvalue_min=0.1,
    iter_twice=True,
    return_details=True,
)
principle_weight_np = principle_weight.squeeze().numpy()
eigenvector_np = eigenvector.squeeze().numpy()
nsample = 30
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
    point_size=[10, 5],
    rgb_on=True,
)
