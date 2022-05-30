from pykeops.torch.cluster import grid_cluster
import numpy as np
import torch

try:
    from torch_scatter import scatter
except:
    print("torch scatter is not detected, voxel grid sampling is disabled")
from pointnet2.lib.pointnet2_utils import furthest_point_sample
from robot.modules_reg.networks.pointconv_util import index_points_gather

from random import Random
import time


def grid_sampler(scale):
    """
    :param scale: voxelgrid gather the point info inside grids of "scale" size
    :return:
    """

    def sampling(points, weights=None):
        """
        :param points: NxD tensor
        :param weights: Nx1 tensor
        :return:
        """
        if weights == None:
            weights = torch.ones(points.shape[0], 1).to(points.device)
        index = grid_cluster(points, scale).long()
        points = scatter(points * weights, index, dim=0)
        cluster_weights = scatter(weights, index, dim=0)
        points = points / cluster_weights
        return points, cluster_weights, index

    return sampling


def uniform_sampler(num_sample, fixed_random_seed=True, sampled_by_weight=True):
    """
    :param num_sample: float
    :param rand_generator:
    :return:
    """

    def sampling(points, weights=None):
        """
        :param points:  NxD tensor
        :return:
        """
        if fixed_random_seed:
            np.random.seed(0)
        if weights is None:
            weights = torch.ones(points.shape[0], 1).to(points.device)
        npoints = points.shape[0]
        if sampled_by_weight:
            weights_np = weights.squeeze().detach().cpu().numpy()
            try:
                rand_ind = np.random.choice(
                    np.arange(npoints),
                    num_sample,
                    replace=False,
                    p=weights_np / weights_np.sum(),
                )
            except:
                print("failed to sample {} from {} points".format(num_sample, npoints))
                rand_ind = np.random.choice(
                    np.arange(npoints),
                    num_sample,
                    replace=True,
                    p=weights_np / weights_np.sum(),
                )
        else:
            try:
                rand_ind = np.random.choice(
                    np.arange(npoints), num_sample, replace=False
                )
            except:
                #print("failed to sample {} from {} points".format(num_sample, npoints))
                rand_ind = np.random.choice(
                    np.arange(npoints), num_sample, replace=True
                )
        rand_ind.sort()
        points = points[rand_ind]
        rand_ind = torch.from_numpy(rand_ind).to(points.device)
        if weights is not None:
            weights = weights[rand_ind]
            return points, weights, rand_ind
        else:
            return points, rand_ind

    return sampling


def point_grid_sampler(scale):
    """
    :param scale: voxelgrid gather the point info inside grids of "scale" size
    :return:
    """
    grid_point_sampler = grid_sampler(scale)

    def sampling(input_shape):
        from robot.global_variable import Shape

        nbatch = input_shape.nbatch
        sampled_points_list = []
        sampled_weights_list = []
        sampled_pointfea_list = []
        D = input_shape.points.shape[-1]
        device = input_shape.points.device
        for i in range(nbatch):
            points = input_shape.points[i]
            weights = input_shape.weights[i]
            sampled_points, sampled_weights, index = grid_point_sampler(points, weights)
            if input_shape.pointfea is not None:
                sampled_pointfea = scatter(
                    input_shape.pointfea[i] * weights, index, dim=0
                )
                sampled_pointfea = sampled_pointfea / sampled_weights
                sampled_pointfea_list.append(sampled_pointfea)
            sampled_points_list.append(sampled_points)
            sampled_weights_list.append(sampled_weights)
        max_len = max(
            [sampled_points.shape[0] for sampled_points in sampled_points_list]
        )
        cleaned_sampled_points_list, cleaned_sampled_weights_list = [], []
        for sampled_points, sampled_weights in zip(
            sampled_points_list, sampled_weights_list
        ):
            nsample = sampled_points.shape[0]
            if nsample < max_len:
                zeros_cat_points = torch.zeros(max_len - nsample, D, device=device)
                zeros_cat_weights = torch.zeros(max_len - nsample, 1, device=device)
                cleaned_sampled_points_list.append(
                    torch.cat([sampled_points, zeros_cat_points], 0)
                )
                cleaned_sampled_weights_list.append(
                    torch.cat([sampled_weights, zeros_cat_weights], 0)
                )
        sampled_batch_points = torch.stack(cleaned_sampled_points_list, dim=0)
        sampled_batch_weights = torch.stack(cleaned_sampled_weights_list, dim=0)
        # todo for polyline and mesh, edges sampling are not supported
        new_shape = Shape()
        new_shape.set_data_with_refer_to(sampled_batch_points, input_shape)
        new_shape.set_weights(sampled_batch_weights)
        new_shape.set_scale(scale)
        if input_shape.pointfea is not None:
            cleaned_sampled_pointfea_list = []
            for sampled_pointfea in sampled_pointfea_list:
                nsample, fea_dim = sampled_pointfea.shape[0], sampled_pointfea.shape[-1]
                if nsample < max_len:
                    zeros_cat_pointfea = torch.zeros(
                        max_len - nsample, fea_dim, device=device
                    )
                    cleaned_sampled_pointfea_list.append(
                        torch.cat([sampled_pointfea, zeros_cat_pointfea], 0)
                    )
            sampled_batch_pointfea = torch.stack(cleaned_sampled_pointfea_list, dim=0)
            new_shape.set_pointfea(sampled_batch_pointfea)
        return new_shape

    return sampling


def point_uniform_sampler(num_sample, fixed_random_seed=True, sampled_by_weight=True):
    """
    :param num_sample: float
    :param rand_generator:
    :return:
    """

    uniform_point_sampler = uniform_sampler(
        num_sample, fixed_random_seed, sampled_by_weight
    )

    def sampling(input_shape):
        from robot.global_variable import Shape

        nbatch = input_shape.nbatch
        sampled_points_list = []
        sampled_weights_list = []
        sampled_pointfea_list = []
        for i in range(nbatch):
            points = input_shape.points[i]
            weights = input_shape.weights[i]
            sampled_points, sampled_weights, index = uniform_point_sampler(
                points, weights
            )
            sampled_weights = sampled_weights
            if input_shape.pointfea is not None:
                sampled_pointfea = input_shape.pointfea[i][index]
                sampled_pointfea_list.append(sampled_pointfea)
            sampled_points_list.append(sampled_points)
            sampled_weights_list.append(sampled_weights)
        sampled_batch_points = torch.stack(sampled_points_list, dim=0)
        sampled_batch_weights = torch.stack(sampled_weights_list, dim=0)
        # todo for polyline and mesh, edges sampling are not supported
        new_shape = Shape()
        new_shape.set_data_with_refer_to(sampled_batch_points, input_shape)
        new_shape.set_weights(sampled_batch_weights)
        new_shape.set_scale(num_sample)
        if input_shape.pointfea is not None:
            sampled_batch_pointfea = torch.stack(sampled_pointfea_list, dim=0)
            new_shape.set_pointfea(sampled_batch_pointfea)
        return new_shape

    return sampling


def point_fps_sampler(num_sample):
    fps_sampler = furthest_point_sample

    def sampling(input_shape):
        from robot.global_variable import Shape

        point_idx = fps_sampler(input_shape.points, num_sample)
        sampled_batch_points = index_points_gather(input_shape.points, point_idx)
        sampled_batch_weights = index_points_gather(input_shape.weights, point_idx)
        new_shape = Shape()
        new_shape.set_data_with_refer_to(sampled_batch_points, input_shape)
        new_shape.set_weights(sampled_batch_weights)
        new_shape.set_scale(num_sample)
        if input_shape.pointfea is not None:
            sampled_batch_pointfea = index_points_gather(
                input_shape.pointfea, point_idx
            )
            new_shape.set_pointfea(sampled_batch_pointfea)
        return new_shape

    return sampling


def batch_grid_sampler(scale):
    """
    :param scale: voxelgrid gather the point info inside grids of "scale" size
    :return:
    """
    sampler = grid_sampler(scale)

    def sampling(points, weights=None):
        points_list = []
        weights_list = []
        ind_list = []
        if weights is not None:
            for _points, _weights in zip(points, weights):
                spoints, sweights, sind = sampler(points, weights)
                points_list.append(spoints)
                weights_list.append(sweights)
                ind_list.append(sind)
            return (
                torch.stack(points_list, 0),
                torch.stack(weights_list, 0),
                torch.stack(ind_list, 0),
            )
        else:
            for _points in zip(points):
                spoints, sweights, sind = sampler(points)
                points_list.append(spoints)
                weights_list.append(sweights)
                ind_list.append(sind)
            return (
                torch.stack(points_list, 0),
                torch.stack(weights_list, 0),
                torch.stack(ind_list, 0),
            )

    return sampling


def batch_uniform_sampler(num_sample, fixed_random_seed=True, sampled_by_weight=True):
    """
    :param num_sample: float
    :param rand_generator:
    :return:
    """
    sampler = uniform_sampler(num_sample, fixed_random_seed, sampled_by_weight)

    def sampling(points, weights=None):
        points_list = []
        weights_list = []
        ind_list = []
        if weights is not None:
            for _points, _weights in zip(points, weights):
                spoints, sweights, sind = sampler(_points, _weights)
                points_list.append(spoints)
                weights_list.append(sweights)
                ind_list.append(sind)
            return (
                torch.stack(points_list, 0),
                torch.stack(weights_list, 0),
                torch.stack(ind_list, 0),
            )
        else:
            for _points in zip(points):
                spoints, sweights, sind = sampler(_points)
                points_list.append(spoints)
                weights_list.append(sweights)
                ind_list.append(sind)
            return (
                torch.stack(points_list, 0),
                torch.stack(weights_list, 0),
                torch.stack(ind_list, 0),
            )

    return sampling
