"""
uniform mass sampling from a given shape
"""
import os
import numpy as np
import torch

from robot.datasets.data_utils import get_obj
from robot.modules_reg.networks.pointconv_util import index_points_group
from geomloss import SamplesLoss
from pykeops.torch import LazyTensor
from robot.utils.visualizer import visualize_point_pair, visualize_landmark_overlap, default_plot


def NN():
    def compute(pc1, pc2):
        pc_i = LazyTensor(pc1[:,:,None])
        pc_j = LazyTensor(pc2[:,None])
        dist2 = pc_i.sqdist(pc_j)
        K_min, index = dist2.min_argmin(dim=2)
        Kmin_pc3 = index_points_group(pc2,index)
        return Kmin_pc3
    return compute



def wasserstein_barycenter_mapping(input_points, wsm_num,blur=0.01,sparse=False):
    """

    :param input_points: BxNxD
    :param wsm_num: M
    :param blur:
    :return:
    """
    grad_enable_record =torch.is_grad_enabled()
    geomloss  = SamplesLoss(loss='sinkhorn',blur=blur, scaling=0.8,reach=None,debias=False,potentials=True,backend='multiscale',truncate=5 if sparse else None)
    points2 = input_points
    B, n_input, device = input_points.shape[0], input_points.shape[1], input_points.device
    weights2 = torch.ones(B, n_input).to(device)/n_input
    prob = torch.ones(n_input).to(device)
    idx = prob.multinomial(num_samples=wsm_num, replacement=False)
    points1 = points2[:,idx].contiguous().clone()
    weights1 = torch.ones(B, wsm_num).to(device)/wsm_num
    device = points2.device
    sqrt_const2 = torch.tensor(np.sqrt(2),dtype=torch.float32, device=device)
    F_i, G_j = geomloss(weights1, points1, weights2,  points2)
    B, N, M, D = points1.shape[0], points1.shape[1], points2.shape[1], points2.shape[2]
    torch.set_grad_enabled(grad_enable_record)
    a_i, x_i = LazyTensor(weights1.view(B,N, 1, 1)), LazyTensor(points1.view(B,N, 1, -1))
    b_j, y_j = LazyTensor(weights2.view(B,1, M, 1)), LazyTensor(points2.view(B,1, M, -1))
    F_i, G_j = LazyTensor(F_i.view(B,N, 1, 1)), LazyTensor(G_j.view(B,1, M, 1))
    xx_i = x_i / (sqrt_const2 * blur)
    yy_j = y_j / (sqrt_const2 * blur)
    f_i = a_i.log() + F_i / blur ** 2
    g_j = b_j.log() + G_j/ blur ** 2
    C_ij = ((xx_i - yy_j) ** 2).sum(-1)
    log_P_ij = (f_i + g_j - C_ij)
    position_to_map = LazyTensor(points2.view(B,1, M, -1))  # Bx1xMxD
    mapped_position = log_P_ij.sumsoftmaxweight(position_to_map,dim=2)
    nn_interp = NN()
    sampled_points = nn_interp(mapped_position,points2)
    return points1,sampled_points



if __name__ == "__main__":

    ###  1D demo
    # def make_spirial_points(noise=0.0):
    #     """Helper to make XYZ points"""
    #     theta = np.linspace(-4 * np.pi, 4 * np.pi, 10000)
    #     z = np.linspace(-2, 2, 10000)
    #     r = z ** 2 + 1 + np.random.rand(len(z)) * noise
    #     x = r * np.sin(theta) + np.random.rand(len(z)) * noise
    #     y = r * np.cos(theta) + np.random.rand(len(z)) * noise
    #     return np.column_stack((x, y, z))
    #
    # points = make_spirial_points(noise=0.0)
    # points = points.astype(np.float32)
    # points = torch.Tensor(points[None]).cuda()
    # before_sample,sampled_points = wasserstein_barycenter_mapping(points,wsm_num=100,blur=1e-3)
    # visualize_point_pair(points, sampled_points,points, sampled_points,"dense","sparse")
    # visualize_point_pair(before_sample, sampled_points,before_sample, sampled_points,"before_sampled","sparse")

    ### 3D demo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    expri_settings = {
        "bunny": {"data_path": "./data/toy_demo_data/bunny_30k.ply"},
        "dragon": {"data_path": "./data/toy_demo_data/dragon_30k.ply"},
        "armadillo": {"data_path": "./data/toy_demo_data/armadillo_30k.ply"}
    }
    output_path = "./output/ot_fun"
    bunny_path = expri_settings["bunny"]["data_path"]
    dragon_path = expri_settings["dragon"]["data_path"]
    armadillo_path = expri_settings["armadillo"]["data_path"]
    os.makedirs(output_path, exist_ok=True)

    ####################  prepare data ###########################
    reader_obj = "toy_dataset_utils.toy_reader()"
    sampler_obj = "toy_dataset_utils.toy_sampler()"
    normalizer_obj = "toy_dataset_utils.toy_normalizer()"
    get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device)
    bunny_obj, bunny_interval = get_obj_func(bunny_path)
    points = bunny_obj["points"]
    before_sample, sampled_points = wasserstein_barycenter_mapping(points, wsm_num=100, blur=1e-3)
    # visualize_point_pair(points, sampled_points,points, sampled_points,"dense","sparse")
    # visualize_point_pair(before_sample, sampled_points,before_sample, sampled_points,"before_sampled","sparse")
    visualize_landmark_overlap(
        points,
        before_sample,
        points,
        torch.norm(before_sample, 2, 2),
        point_plot_func=default_plot(cmap="magma", rgb=True, point_size=15, render_points_as_spheres=True),
        landmark_plot_func=default_plot(cmap="viridis", rgb=True, point_size=30, render_points_as_spheres=True),
        title= "source",
        opacity=(0.1, 1),
    )
    visualize_landmark_overlap(
        points,
        sampled_points,
        points,
        torch.norm(sampled_points, 2, 2),
        point_plot_func=default_plot(cmap="magma", rgb=True, point_size=15, render_points_as_spheres=True),
        landmark_plot_func=default_plot(cmap="viridis", rgb=True, point_size=30, render_points_as_spheres=True),
        title="source",
        opacity=(0.1, 1),
    )





