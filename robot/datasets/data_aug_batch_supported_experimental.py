import torch
import numpy as np
import random
from robot.utils.obj_factory import obj_factory
from robot.global_variable import Shape
from robot.utils.utils import get_grid_wrap_points
from robot.shape.point_sampler import (
    point_uniform_sampler,
    point_grid_sampler,
    uniform_sampler,
    grid_sampler,
)


def visualize(points, deformed_points, point_weights=None, deformed_point_weights=None):
    from robot.utils.visualizer import visualize_point_pair_overlap
    from robot.experiments.datasets.lung.lung_data_analysis import (
        flowed_weight_transform,
        target_weight_transform,
    )

    # visualize_point_pair_overlap(points, deformed_points,
    #                              flowed_weight_transform(point_weights, True),
    #                              target_weight_transform(deformed_point_weights, True),
    #                              title1="original", title2="deformed", rgb_on=False)
    visualize_point_pair_overlap(
        points,
        deformed_points,
        point_weights,
        deformed_point_weights,
        title1="original",
        title2="deformed",
        rgb_on=False,
    )


class PointAug(object):
    def __init__(self, aug_settings):
        self.remove_random_points = aug_settings["remove_random_points"]
        self.add_random_point_noise = aug_settings["add_random_point_noise"]
        self.add_random_weight_noise = aug_settings["add_random_weight_noise"]
        self.remove_random_points_by_ratio = aug_settings[
            "remove_random_points_by_ratio"
        ]
        self.add_random_point_noise_by_ratio = aug_settings[
            "add_random_point_noise_by_ratio"
        ]
        self.random_noise_raidus = aug_settings["random_noise_raidus"]
        self.normalize_weights = aug_settings["normalize_weights"]
        self.plot = aug_settings["plot"]

    def remove_random_points(self, points, point_weights, index):
        new_points_list, new_point_weights_list, new_index_list = [], [], []
        for _points, _point_weights, _index in zip(points, point_weights, index):
            npoints = _points.shape[0]
            nsample = int((1 - self.remove_random_points_by_ratio) * npoints)
            sampler = uniform_sampler(
                nsample, fixed_random_seed=False, sampled_by_weight=True
            )
            sampling_points, sampling_weights, sampling_index = sampler(
                _points, _point_weights
            )
            new_points_list.append(sampling_points)
            new_point_weights_list.append(sampling_weights)
            new_index_list.append(_index[sampling_index])
        return (
            torch.stack(new_points_list, 0),
            torch.stack(new_point_weights_list),
            torch.stack(new_index_list),
        )

    def add_noises_around_points(self, points, point_weights, index=None):
        new_points_list, new_point_weights_list, added_index_list = [], [], []
        for _points, _point_weights, _index in zip(points, point_weights, index):
            npoints, D = _points.shape[0], _points.shape[-1]
            nnoise = int(self.add_random_point_noise_by_ratio * npoints)
            noise_index = np.random.choice(list(range(npoints)), nnoise, replace=False)
            noise_disp = (
                torch.ones(nnoise, 3).to(_points.device).uniform_(-1, 1)
                * self.random_noise_raidus
            )
            noise = _points[noise_index] + noise_disp
            _points = torch.cat([_points, noise], 0)
            weights = torch.cat([_point_weights, _point_weights[noise_index]], 0)
            added_index = torch.tensor(list(range(npoints, npoints + nnoise))).to(
                _points.device
            )
            new_points_list.append(_points)
            new_point_weights_list.append(weights)
            added_index_list.append(added_index)
        return (
            torch.stack(new_points_list, 0),
            torch.stack(new_point_weights_list, 0),
            torch.cat([index, torch.stack(added_index_list, 0)], 1),
        )

    def add_random_noise_to_weights(self, points, point_weights, index=None):
        noise_std = (torch.min(point_weights) / 5).item()
        weights_noise = torch.ones_like(point_weights).normal_(0, noise_std)
        point_weights = point_weights + weights_noise
        return points, point_weights, index

    def __call__(self, points, point_weights):
        B, N, D = points.shape[0], points.shape[1], points.shape[2]
        device = points.device
        new_points, new_weights, new_index = (
            points,
            point_weights,
            torch.tensor(list(range(N))).repeat(B, 1).to(device),
        )
        if self.remove_random_points and self.remove_random_points_by_ratio != 0:
            new_points, new_weights, new_index = self.remove_random_points(
                new_points, new_weights, new_index
            )
        if self.add_random_point_noise and self.add_random_point_noise_by_ratio != 0:
            new_points, new_weights, new_index = self.add_noises_around_points(
                new_points, new_weights, new_index
            )
        if self.add_random_weight_noise:
            new_points, new_weights, new_index = self.add_random_noise_to_weights(
                new_points, new_weights, new_index
            )
        if self.normalize_weights:
            new_weights = new_weights * (point_weights.sum() / (new_weights.sum()))
        if self.plot:
            visualize(points[0], new_points[0], point_weights[0], new_weights[0])

        return new_points, new_weights, new_index


class SplineAug(object):
    """
    deform the point cloud via spline deform
    for the grid deformation the isotropic deformation should be used
    for the sampling deformation, either isotropic or anistropic deformation can be used
    for both deformation the nadwat interpolation is used
    :param deform_settings:
    :return:
    """

    def __init__(self, aug_settings):
        super(SplineAug, self).__init__()
        self.aug_settings = aug_settings
        self.do_grid_aug = aug_settings["do_grid_aug"]
        self.do_local_deform_aug = aug_settings["do_local_deform_aug"]
        grid_aug_settings = self.aug_settings["grid_spline_aug"]
        local_deform_aug_settings = self.aug_settings["local_deform_aug"]
        grid_spline_kernel_obj = grid_aug_settings[
            ("grid_spline_kernel_obj", "", "grid spline kernel object")
        ]
        local_deform_spline_kernel_obj = local_deform_aug_settings[
            ("local_deform_spline_kernel_obj", "", "local deform spline kernel object")
        ]
        knn_interp_kernel_obj = local_deform_aug_settings[
            (
                "knn_interp_kernel_obj",
                "",
                "local KNN interpolation kernel kernel object",
            )
        ]
        self.grid_spline_kernel = (
            obj_factory(grid_spline_kernel_obj) if grid_spline_kernel_obj else None
        )
        self.local_deform_spline_kernel = (
            obj_factory(local_deform_spline_kernel_obj)
            if local_deform_spline_kernel_obj
            else None
        )
        self.knn_interp_kernel = (
            obj_factory(knn_interp_kernel_obj) if knn_interp_kernel_obj else None
        )
        self.plot = aug_settings["plot"]

    def grid_spline_deform(self, points, point_weights, coupled_points=None):
        """

        :param points: BxNxD
        :param point_weights: BxNx1
        :return:
        """
        grid_aug_settings = self.aug_settings["grid_spline_aug"]
        grid_spacing = grid_aug_settings["grid_spacing"]
        scale = grid_aug_settings["disp_scale"]
        scale = scale * random.random()
        B = points.shape[0]
        deformed_points_list = []
        deformed_coupled_points_list = []
        coupled_points = [None] * B if coupled_points is None else coupled_points

        for _points, _weights, _coupled_points in zip(
            points, point_weights, coupled_points
        ):

            # grid_control_points, _ = get_grid_wrap_points(_weights, np.array([grid_spacing]*3).astype(np.float32))
            # grid_control_disp = torch.ones_like(grid_control_points).uniform_(-1,1)*scale
            # ngrids = grid_control_points.shape[0]
            # grid_control_points = grid_control_points[None]
            # grid_control_weights = torch.ones(1,ngrids, 1).to(points.device) / ngrids
            _points = _points[None]
            _weights = _weights[None]
            _coupled_points = _coupled_points[None]
            sampler = point_grid_sampler(grid_spacing)
            tosample_shape = Shape().set_data(points=_points, weights=_weights)
            sampled_shape = sampler(tosample_shape)
            grid_control_points, grid_control_weights = (
                sampled_shape.points,
                sampled_shape.weights,
            )
            grid_control_disp = (
                torch.ones_like(grid_control_points).uniform_(-1, 1) * scale
            )
            _points_disp = self.grid_spline_kernel(
                _points, grid_control_points, grid_control_disp, grid_control_weights
            )
            _deformed_points = _points + _points_disp
            deformed_points_list.append(_deformed_points)
            if coupled_points is None:
                _deformed_coupled_points = None
            else:
                _coupled_points_disp = self.grid_spline_kernel(
                    _coupled_points,
                    grid_control_points,
                    grid_control_disp,
                    grid_control_weights,
                )
                _deformed_coupled_points = _coupled_points + _coupled_points_disp
            deformed_coupled_points_list.append(_deformed_coupled_points)
        deformed_points = torch.cat(deformed_points_list, 0)
        deformed_coupled_points = (
            torch.cat(deformed_coupled_points_list, 0)
            if deformed_coupled_points_list[0] is not None
            else None
        )
        return deformed_points, point_weights, deformed_coupled_points

    def local_deform_spline_deform(self, points, point_weights, coupled_points=None):
        """

        :param points: BxNxD
        :param point_weights: BxNx1
        :return:
        """
        local_deform_aug_settings = self.aug_settings["local_deform_aug"]
        num_sample = local_deform_aug_settings["num_sample"]
        scale = local_deform_aug_settings["disp_scale"]
        sampler = point_uniform_sampler(num_sample, sampled_by_weight=False)
        tosample_shape = Shape().set_data(points=points, weights=point_weights)
        sampled_shape = sampler(tosample_shape)
        sampling_control_points, sampling_control_weights = (
            sampled_shape.points,
            sampled_shape.weights,
        )
        sampling_control_disp = (
            torch.ones_like(sampling_control_points).uniform_(-1, 1) * scale
        )
        points_disp = self.local_deform_spline_kernel(
            points,
            sampling_control_points,
            sampling_control_disp,
            sampling_control_weights,
        )
        deformed_points = points + points_disp
        if coupled_points is None:
            deformed_coupled_points = None
        else:
            coupled_points_disp = self.knn_interp_kernel(
                coupled_points, points, points_disp
            )
            deformed_coupled_points = coupled_points + coupled_points_disp
        return deformed_points, point_weights, deformed_coupled_points

    def __call__(self, points, point_weights, coupled_points=None):
        """

        :param points: torch.tensor BxNxD
        :param point_weights: torch.tensor BxNx1
        :return:
        """
        deformed_points = points
        deformed_weights = point_weights
        deformed_coupled_points = coupled_points
        if self.do_local_deform_aug:
            (
                deformed_points,
                deformed_weights,
                deformed_coupled_points,
            ) = self.local_deform_spline_deform(
                deformed_points, deformed_weights, deformed_coupled_points
            )
        if self.plot:
            visualize(
                points[0], deformed_points[0], point_weights[0], deformed_weights[0]
            )
        if self.do_grid_aug:
            (
                deformed_points,
                deformed_weights,
                deformed_coupled_points,
            ) = self.grid_spline_deform(
                deformed_points, deformed_weights, deformed_coupled_points
            )
        if self.plot:
            visualize(
                points[0], deformed_points[0], point_weights[0], deformed_weights[0]
            )
        return deformed_points, deformed_weights
