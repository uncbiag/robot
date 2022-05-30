import torch
import numpy as np
import random

from robot.shape.point_interpolator import KNNInterpolater
from robot.utils.obj_factory import obj_factory
from robot.global_variable import Shape
from robot.utils.utils import get_grid_wrap_points
from robot.experiments.datasets.lung.visualizer import lung_plot, camera_pos
from robot.utils.visualizer import visualize_point_pair_overlap, default_plot
from robot.shape.point_sampler import (
    point_uniform_sampler,
    point_grid_sampler,
    uniform_sampler,
    grid_sampler,
)


def visualize(
    points,
    deformed_points,
    point_weights=None,
    deformed_point_weights=None,
    deformed_coupled_points=None,
    title1 = "original",
    title2 = "deformed"
):
    ### visualization for general task
    # visualize_point_pair_overlap(points, deformed_points,
    #                              point_weights,
    #                              deformed_point_weights,
    #                              "original", "deformed",
    #                              rgb_on=False)
    #


    ### visualization for lung task

    visualize_point_pair_overlap(points, deformed_points,
                                 point_weights,
                                 deformed_point_weights,
                                 title1, title2,
                                 pc1_plot_func=lung_plot(color="source"),
                                 pc2_plot_func=lung_plot(color="target"),
                                 opacity=(1,1),
                                 light_mode="none",
                                 camera_pos =camera_pos)
    if deformed_coupled_points:
        visualize_point_pair_overlap(
            deformed_points,
            deformed_coupled_points,
            deformed_points,
            deformed_coupled_points,
            title2,
            title2 +" with coupled",
            rgb_on=False,
            show=True,
        )


class PointAug(object):
    def __init__(self, aug_settings):
        self.remove_random_points = aug_settings[
            (
                "remove_random_points",
                False,
                "randomly remove points from the uniform distribution",
            )
        ]
        self.add_random_point_noise = aug_settings[
            (
                "add_random_point_noise",
                False,
                "randomly add points from the uniform distribution",
            )
        ]
        self.add_random_weight_noise = aug_settings[
            (
                "add_random_weight_noise",
                False,
                "randomly add weight noise from normal distribution",
            )
        ]
        # self.remove_random_points_by_ratio = aug_settings[("remove_random_points_by_ratio", 0.01,"")]
        self.add_random_point_noise_by_ratio = aug_settings[
            ("add_random_point_noise_by_ratio", 0.01, "")
        ]
        self.random_noise_raidus = aug_settings[("random_noise_raidus", 0.1, "")]
        self.random_weight_noise_scale = aug_settings[
            (
                "random_weight_noise_scale",
                0.05,
                "scale factor on the average weight value",
            )
        ]
        self.normalize_weights = aug_settings[
            ("normalize_weights", False, "normalized the weight make it sum to 1")
        ]
        self.plot = aug_settings[("plot", False, "plot the shape")]

    # def remove_random_points(self, points, point_weights, index):
    #     npoints = points.shape[0]
    #     nsample = int((1 - self.remove_random_points_by_ratio) * npoints)
    #     sampler = uniform_sampler(nsample,fixed_random_seed=False,sampled_by_weight=True)
    #     sampling_points, sampling_weights, sampling_index = sampler(points, point_weights)
    #     return sampling_points, sampling_weights, index[sampling_index]

    def add_noises_around_points(self, points, point_weights, index=None):
        npoints, D = points.shape[0], points.shape[-1]
        nnoise = int(self.add_random_point_noise_by_ratio * npoints)
        rand_index = np.random.choice(list(range(npoints)), nnoise, replace=False)
        noise_disp = (
            torch.ones(nnoise, 3, device=points.device).uniform_(-1, 1)
            * self.random_noise_raidus
        )
        points[rand_index] = points[rand_index] + noise_disp
        return points, point_weights, index

    def add_random_noise_to_weights(self, points, point_weights, index=None):
        noise_std = (torch.min(point_weights) / 5).item()
        weights_noise = torch.ones_like(point_weights).normal_(0, noise_std)
        point_weights = torch.clamp(point_weights + weights_noise, min=1e-8)
        return points, point_weights, index

    def __call__(self, batch_points, batch_point_weights):
        N, D = batch_points.shape[1], batch_points.shape[2]
        device = batch_points.device
        new_points_list, new_weights_list, new_index_list = [], [], []
        for points, point_weights in zip(batch_points, batch_point_weights):
            new_points, new_weights, new_index = (
                points,
                point_weights,
                torch.tensor(list(range(N))).to(device),
            )
            # if self.remove_random_points and self.remove_random_points_by_ratio!=0:
            #     new_points, new_weights, new_index = self.remove_random_points(new_points, new_weights, new_index)
            if (
                self.add_random_point_noise
                and self.add_random_point_noise_by_ratio != 0
            ):
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
                visualize(points, new_points, point_weights, new_weights,title1="before point aug", title2="after point aug")
            new_points_list.append(new_points)
            new_weights_list.append(new_weights)
            new_index_list.append(new_index)

        return (
            torch.stack(new_points_list),
            torch.stack(new_weights_list),
            torch.stack(new_index_list),
        )


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
        self.do_rigid_aug = aug_settings["do_rigid_aug"]
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
        grid_aug_settings = self.aug_settings["grid_spline_aug"]
        grid_spacing = grid_aug_settings["grid_spacing"]
        scale = grid_aug_settings["disp_scale"]
        scale = scale * random.random()

        # grid_control_points, _ = get_grid_wrap_points(points, np.array([grid_spacing]*3).astype(np.float32))
        # grid_control_disp = torch.ones_like(grid_control_points).uniform_(-1,1)*scale
        # ngrids = grid_control_points.shape[0]
        # grid_control_weights = torch.ones(ngrids, 1).to(points.device) / ngrids

        sampler = grid_sampler(grid_spacing)
        grid_control_points, grid_control_weights, _ = sampler(points, point_weights)
        grid_control_disp = torch.ones_like(grid_control_points).uniform_(-1, 1) * scale
        points_disp = self.grid_spline_kernel(
            points[None],
            grid_control_points[None],
            grid_control_disp[None],
            grid_control_weights[None],
        )
        # visualize(points, grid_control_points, point_weights, grid_control_weights)

        points_disp = points_disp.squeeze()
        deformed_points = points + points_disp
        if coupled_points is None:
            deformed_coupled_points = None
        else:
            coupled_points_disp = self.grid_spline_kernel(
                coupled_points[None],
                grid_control_points[None],
                grid_control_disp[None],
                grid_control_weights[None],
            )
            deformed_coupled_points = coupled_points + coupled_points_disp
            deformed_coupled_points = deformed_coupled_points.squeeze()
        return deformed_points, point_weights, deformed_coupled_points

    def local_deform_spline_deform(self, points, point_weights, coupled_points=None):
        local_deform_aug_settings = self.aug_settings["local_deform_aug"]
        num_sample = local_deform_aug_settings["num_sample"]
        scale = local_deform_aug_settings["disp_scale"]
        sampler = uniform_sampler(
            num_sample, fixed_random_seed=False, sampled_by_weight=False
        )
        sampling_control_points, sampling_control_weights, _ = sampler(
            points, point_weights
        )
        sampling_control_disp = (
            torch.ones_like(sampling_control_points).uniform_(-1, 1) * scale
        )
        # visualize(points, sampling_control_points, point_weights, sampling_control_weights)
        points_disp = self.local_deform_spline_kernel(
            points[None],
            sampling_control_points[None],
            sampling_control_disp[None],
            sampling_control_weights[None],
        )
        points_disp = points_disp.squeeze()
        deformed_points = points + points_disp
        if coupled_points is None:
            deformed_coupled_points = None
        else:
            coupled_points_disp = self.knn_interp_kernel(
                coupled_points[None], points[None], points_disp[None]
            )
            deformed_coupled_points = coupled_points + coupled_points_disp
            deformed_coupled_points = deformed_coupled_points.squeeze()
        return deformed_points, point_weights, deformed_coupled_points

    def rigid_deform(self, points, point_weights=None, coupled_points=None):
        from scipy.spatial.transform import Rotation as R

        rigid_aug_settings = self.aug_settings["rigid_aug"]
        rotation_range = rigid_aug_settings["rotation_range"]
        scale_range = rigid_aug_settings["scale_range"]
        translation_range = rigid_aug_settings["translation_range"]
        scale = random.random() * (scale_range[1] - scale_range[0]) + scale_range[0]
        translation = (
            random.random() * (translation_range[1] - translation_range[0])
            + translation_range[0]
        )
        r = R.from_euler(
            "zyx",
            [
                random.random() * (rotation_range[1] - rotation_range[0])
                + rotation_range[0],
                random.random() * (rotation_range[1] - rotation_range[0])
                + rotation_range[0],
                random.random() * (rotation_range[1] - rotation_range[0])
                + rotation_range[0],
            ],
            degrees=True,
        )
        r_matrix = r.as_matrix() * scale
        r_matrix = torch.tensor(r_matrix, dtype=torch.float, device=points.device)
        deformed_points = points @ r_matrix + translation
        if coupled_points is None:
            deformed_coupled_points = None
        else:
            deformed_coupled_points = coupled_points @ r_matrix + translation
        return deformed_points, point_weights, deformed_coupled_points

    def __call__(self, batch_points, batch_point_weights, batch_coupled_points=None):
        """
        :param batch_points: torch.tensor  BxNxD
        :param batch_point_weights: torch.tensor BxNx1
        :param batch_coupled_points: torch.tensor BxNx1
        :return:
        """
        deformed_points_list, deformed_weights_list, deformed_coupled_points_list = (
            [],
            [],
            [],
        )
        B = batch_points.shape[0]
        batch_coupled_points = (
            [None] * B if batch_coupled_points is None else batch_coupled_points
        )
        for points, point_weights, coupled_points in zip(
            batch_points, batch_point_weights, batch_coupled_points
        ):
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
                        points,
                        deformed_points,
                        point_weights,
                        deformed_weights,
                        deformed_coupled_points,
                        title1="before local deform",
                        title2="after local deform"
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
                        points,
                        deformed_points,
                        point_weights,
                        deformed_weights,
                        deformed_coupled_points,
                        title1="before global deform",
                        title2="after global deform"
                    )
            if self.do_rigid_aug:
                (
                    deformed_points,
                    deformed_weights,
                    deformed_coupled_points,
                ) = self.rigid_deform(
                    deformed_points, deformed_weights, deformed_coupled_points
                )
                if self.plot:
                    visualize(
                        points,
                        deformed_points,
                        point_weights,
                        deformed_weights,
                        deformed_coupled_points,
                        title1="before rigid deform",
                        title2="after rigid deform"
                    )
            deformed_points_list.append(deformed_points)
            deformed_weights_list.append(deformed_weights)
            deformed_coupled_points_list.append(deformed_coupled_points)
        if deformed_coupled_points_list[0] is None:
            return torch.stack(deformed_points_list), torch.stack(deformed_weights_list)
        else:
            return (
                torch.stack(deformed_points_list),
                torch.stack(deformed_weights_list),
                torch.stack(deformed_coupled_points_list),
            )
