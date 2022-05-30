import os, sys
import numpy as np

from robot.experiments.datasets.lung.global_variable import lung_expri_path
from robot.experiments.datasets.lung.lung_data_analysis import get_half_lung
from robot.experiments.datasets.lung.visualizer import lung_plot, camera_pos

sys.path.insert(0, os.path.abspath("../../../.."))
from robot.datasets.data_utils import get_obj, read_json_into_list
from robot.global_variable import *
from robot.utils.shape_visual_utils import make_ellipsoid
from robot.experiments.datasets.lung.lung_feature_extractor import (
    compute_anisotropic_gamma_from_points,
)
from robot.utils.visualizer import (
    visualize_point_fea_with_arrow,
    visualize_point_overlap, default_plot,visualize_point_fea
)

# import pykeops
# pykeops.clean_pykeops()

task_name = "local_aniso_kernel_visualize"
dataset_json_path = os.path.join(ROBOT_PATH,"demos/data/lung_data/lung_dataset_splits/train/pair_data.json")
saving_output_path = os.path.join(lung_expri_path, "output/{}".format(task_name))
path_transfer = lambda x: x.replace('./',ROBOT_PATH+"/")
pair_name_list, pair_info_list = read_json_into_list(dataset_json_path)

pair_path_list = [
    [path_transfer(pair_info["source"]["data_path"]), path_transfer(pair_info["target"]["data_path"]) ]
    for pair_info in pair_info_list
]
pair_id = 0

path_1, path_2 = pair_path_list[pair_id]
saving_path = "/playpen-raid1/zyshen/debug/debugg_point_visual2.vtk"
reader_obj = "lung_dataloader_utils.lung_reader(use_radius=False)"
scale = (
    -1
)  # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
normalizer_obj = "lung_dataloader_utils.lung_normalizer(scale={})".format(
    [100, 100, 100]
)
sampler_obj = "lung_dataloader_utils.lung_sampler( method='combined',scale=0.0003,num_sample=100000,sampled_by_weight=True)"
get_obj_func = get_obj(
    reader_obj, normalizer_obj, sampler_obj, device=torch.device("cpu")
)
source_obj, source_interval = get_obj_func(path_1)
target_obj, target_interval = get_obj_func(path_2)

points_1 = source_obj["points"]
weights_1 = source_obj["weights"]
points_2 = target_obj["points"]

npoints, points = points_1.shape[1], points_1
points_np = points.detach().cpu().numpy().squeeze()
aniso_kernel_scale = 0.02
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
nsample = 700
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
#fg_spheres_color = np.array([[0.8, 0.5, 0.2]] * len(spheres))
fg_spheres_color = np.array([[195, 250, 248]] * len(spheres))
fg_spheres_color = fg_spheres_color
# visualize_point_fea(points, mass , rgb_on=False)

# visualize_point_fea_with_arrow(
#     points_1, mass, eigenvector[:, :, :, 0] * 0.01, rgb_on=False
# )

# camera_pos = [(2.7588283090717782, 6.555762175709003, 1.673781643266848),
#  (0.023061048658378214, -0.0019414722919464111, -0.031303226947784424),
#  (-0.10449633369794017, -0.209208619403546, 0.9722717057545956)]

# visualize_point_overlap(
#     points,
#     spheres,
#     weights_1,
#     fg_spheres_color,
#     "aniso_filter_with_kernel_radius",
#     source_plot_func=lung_plot(color="source"),
#     #source_plot_func=default_plot("viridis"),
#     target_plot_func=default_plot(rgb=True),
#     opacity=(0.8,0.03),
#     camera_pos =camera_pos
# )

visualize_point_fea(
    points,
    weights_1,
    cmap= "Reds",
    camera_pos=camera_pos,
    show=True,
    col_adaptive=False,
    opacity=1.0
)