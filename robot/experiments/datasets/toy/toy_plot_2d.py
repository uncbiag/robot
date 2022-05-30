import os, sys

from robot.shape.point_interpolator import nadwat_kernel_interpolator
from robot.utils.local_feature_extractor import (
    feature_extractor,
    compute_anisotropic_gamma_from_points,
)
from robot.utils.shape_visual_utils import make_ellipsoid

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))

from robot.datasets.data_utils import (
    get_file_name,
    generate_pair_name,
    compute_interval,
)

from robot.global_variable import shape_type
from robot.utils.visualizer import (
    visualize_point_fea,
    visualize_point_fea_with_arrow,
    visualize_point_overlap, default_plot,
)
from robot.experiments.datasets.toy.toy_utils import *
from robot.utils.utils import memory_sort

assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
server_path = "./toy_synth/"  # "/playpen-raid1/"#"/home/zyshen/remote/llr11_mount/"
source_path = server_path + "tree_2d_source.png"
target_path = server_path + "tree_2d_target.png"


def get_points(path, npoint=1000, dtype=torch.FloatTensor):
    A = load_image(path)
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]), np.linspace(0, 1, A.shape[1]))
    grid = list(zip(xg.ravel(), yg.ravel()))
    grid = np.array(grid)
    A = A.transpose().ravel()
    dots = grid[A > 0.9]
    dots = np.concatenate([dots[:, 1:], dots[:, 0:1]], 1)
    # dots += (.5 / A.shape[0]) * np.random.standard_normal(dots.shape)
    rand_state = np.random.RandomState(0)
    index = rand_state.choice(np.arange(len(dots)), npoint, replace=False)
    dots = dots[index]
    dots, _ = memory_sort(dots, eps=0.0001)
    weights = np.ones([npoint, 1]) / npoint
    return torch.from_numpy(dots).type(dtype), torch.from_numpy(weights).type(dtype)


####################  prepare data ###########################
pair_name = generate_pair_name([source_path, target_path])
points, weights = get_points(source_path, npoint=3000)
min_interval = compute_interval(points)
print("the interval is {}".format(min_interval))
points = points[None]
weights = weights[None]
fea_type_list = ["eigenvalue_prod", "eigenvector_main"]
fea_extractor = feature_extractor(fea_type_list, radius=0.01, std_normalize=False)
combined_fea, mass = fea_extractor(points, weights)
pointfea, main_direction = combined_fea[..., :1], combined_fea[..., 1:]
# visualize_point_fea_with_arrow(points, mass, main_direction*0.005 , rgb_on=False)

points = torch.Tensor(points)
points_np = points.detach().cpu().numpy().squeeze()
weights = torch.ones(points.shape[0], points.shape[1], 1)
npoints = points.shape[1]
Gamma, principle_weight, eigenvector, mass = compute_anisotropic_gamma_from_points(
    points,
    cov_sigma_scale=0.01,
    aniso_kernel_scale=0.05,
    leaf_decay=True,
    principle_weight=None,
    eigenvalue_min=0.3,
    iter_twice=True,
    return_details=True,
)
filtered_points = nadwat_kernel_interpolator(scale=0.04, exp_order=2, iso=False)(
    points, points, points, weights, Gamma
)
# visualize_point_fea(points, mass, rgb_on=False)
# visualize_point_fea(filtered_points, mass, rgb_on=False)
principle_weight_np = principle_weight.squeeze().numpy()
eigenvector_np = eigenvector.squeeze().numpy()
nsample = 30
full_index = np.arange(npoints)
index = np.random.choice(full_index, nsample, replace=False)
spheres_list = [
    make_ellipsoid(
        200,
        ndim=2,
        radius=principle_weight_np[ind],
        center=points_np[ind],
        rotation=eigenvector_np[ind],
    )
    for ind in index
]
spheres = np.concatenate(spheres_list, 0)
fg_spheres_color = np.array([[0.8, 0.5, 0.2]] * len(spheres))
# visualize_point_fea(points, mass , rgb_on=False)
# visualize_point_fea_with_arrow(points, mass,eigenvector[...,0]*0.005,rgb_on=False)
visualize_point_overlap(
    points,
    spheres,
    mass,
    fg_spheres_color,
    "aniso_filter_with_kernel_radius",
    source_plot_func=default_plot(cmap="magma",point_size=20),
    target_plot_func=default_plot(cmap="viridis",point_size=10),
)
