import pyvista as pv
import numpy as np
import torch
from robot.utils.visualizer import (
    visualize_point_fea,
    visualize_point_fea_with_arrow,
    visualize_point_overlap,
)
from robot.datasets.data_utils import compute_interval
from robot.utils.shape_visual_utils import make_sphere, make_ellipsoid
from robot.shape.point_interpolator import *
from robot.utils.local_feature_extractor import *

###############################################################################
# Create a dataset to plot


def make_spirial_points(noise=0.1):
    """Helper to make XYZ points"""
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
    z = np.linspace(-2, 2, 1000)
    r = z ** 2 + 1 + np.random.rand(len(z)) * noise
    x = r * np.sin(theta) + np.random.rand(len(z)) * noise
    y = r * np.cos(theta) + np.random.rand(len(z)) * noise
    return np.column_stack((x, y, z))


def make_line_points():
    """Helper to make XYZ points"""
    x = np.linspace(-5, 5, 1000)
    y = np.random.rand(1000) * 0.001
    z = np.random.rand(1000) * 0.001
    return np.column_stack((x, y, z))


def generate_aniso_gamma_on_sphere(
    points, foregroud_bool_index, foreground_sigma, background_sigma
):
    npoints = len(points)
    nfg = sum(foregroud_bool_index)
    fg_gamma = 1 / (np.array(foreground_sigma) ** 2)
    bg_gamma = 1 / (np.array(background_sigma) ** 2)
    Gamma = np.repeat(np.diag(bg_gamma)[None], npoints, axis=0)
    fg_Gamma = np.repeat(np.diag(fg_gamma)[None], nfg, axis=0)
    Gamma[foregroud_bool_index] = fg_Gamma
    Gamma = Gamma.reshape(npoints, -1)
    return Gamma


def generate_eigen_vector_from_main_direction(main_direction):
    pass


# demo1 visualize main direction on a spirial
points = make_spirial_points(noise=0.1)
points = points.astype(np.float32)
compute_interval(points)
points = torch.Tensor(points)[None]
weights = torch.ones(points.shape[0], 1) / points.shape[0]
fea_type_list = ["eigenvalue_prod", "eigenvector_main"]
fea_extractor = feature_extractor(fea_type_list, radius=0.1, std_normalize=False)
combined_fea, mass = fea_extractor(points, weights)
pointfea, main_direction = combined_fea[..., :1], combined_fea[..., 1:]
# visualize_point_fea_with_arrow(points, mass, main_direction*0.1 , rgb_on=False)


# #demo2 visualize anisotropic kernel on sphere
# points_np = make_sphere(radius= (1.,1.,1.))
# points_np = points_np.astype(np.float32)
# npoints = len(points_np)
# bool_index = points_np[:,2].squeeze()>0.8
# compute_interval(points_np)
# fg_sigma = [0.2,0.2,0.1]
# bg_sigma = [0.05,0.05,0.05]
# Gamma = generate_aniso_gamma_on_sphere(points_np,bool_index,[0.2,0.2,0.1],[0.05,0.05,0.05])
# points = torch.Tensor(points_np)[None].contiguous()
# Gamma = torch.Tensor(Gamma)[None].contiguous()
# mass,_,_ = compute_aniso_local_moments(points,Gamma)
# fg_nsample = 3
# bg_nsample = 20
# full_index = np.arange(npoints)
# fg_index = np.random.choice(full_index[bool_index], fg_nsample, replace=False)
# bg_index = np.random.choice(full_index[np.invert(bool_index)], bg_nsample, replace=False)
# fg_spheres_list = [make_sphere(600,radius=fg_sigma, center=points_np[ind]) for ind in fg_index]
# bg_spheres_list = [make_sphere(50,radius=bg_sigma, center=points_np[ind]) for ind in bg_index]
# fg_spheres = np.concatenate(fg_spheres_list,0)
# bg_spheres = np.concatenate(bg_spheres_list,0)
# fg_spheres_color = np.array([[0.8,.5,.2]]*len(fg_spheres))
# bg_spheres_color = np.array([[-1.2,-1,-1.2]]*len(bg_spheres))
# surrounding_spheres = np.concatenate([fg_spheres,bg_spheres],0)
# surrounding_spheres_color = np.concatenate([fg_spheres_color,bg_spheres_color],0)
# #visualize_point_fea(points, mass , rgb_on=False)
# visualize_point_overlap(points, surrounding_spheres,mass,surrounding_spheres_color,"aniso_filter_with_kernel_radius",point_size=[10,5],rgb_on=[False,True])

#


# visualize anisotropic kernel on spirial
points = make_spirial_points(noise=0.2)
points = points.astype(np.float32)
points = torch.Tensor(points)[None]
points_np = points.detach().cpu().numpy().squeeze()
weights = torch.ones(points.shape[0], points.shape[1], 1)
npoints = points.shape[1]
Gamma, principle_weight, eigenvector, mass = compute_anisotropic_gamma_from_points(
    points,
    cov_sigma_scale=0.1,
    aniso_kernel_scale=0.4,
    leaf_decay=True,
    principle_weight=None,
    eigenvalue_min=0.4,
    iter_twice=True,
    return_details=True,
)
filtered_points = nadwat_kernel_interpolator(scale=0.4, exp_order=2, iso=False)(
    points, points, points, weights, Gamma
)
visualize_point_fea(points, mass, rgb_on=False)
visualize_point_fea(filtered_points, mass, rgb_on=False)
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
visualize_point_fea_with_arrow(points, mass, eigenvector[..., 0] * 0.05, rgb_on=False)
visualize_point_overlap(
    points,
    spheres,
    mass,
    fg_spheres_color,
    "aniso_filter_with_kernel_radius",
    point_size=[10, 5],
    rgb_on=[False, True],
)
