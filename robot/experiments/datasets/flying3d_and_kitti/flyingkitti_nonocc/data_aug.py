"""keops failed on multi-processing data loader,  so the synthsis process are put before the network"""
import os, sys

sys.path.insert(0, os.path.abspath("../../../.."))
import random
import time
from robot.experiments.datasets.lung.lung_data_analysis import *
from robot.global_variable import *
from robot.datasets.data_aug import SplineAug, PointAug
from robot.utils.module_parameters import ParameterDict
from robot.utils.utils import enlarge_by_factor
from functools import partial


def flyingkitti_nonocc_synth_data(**kwargs):
    aug_settings = ParameterDict()
    aug_settings["do_local_deform_aug"] = (
        kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    )
    aug_settings["do_grid_aug"] = (
        kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    )
    aug_settings["do_point_aug"] = (
        kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    )
    aug_settings["do_rigid_aug"] = (
        kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else False
    )
    aug_settings["plot"] = False
    local_deform_aug = aug_settings[
        (
            "local_deform_aug",
            {},
            "settings for uniform sampling based spline augmentation",
        )
    ]
    local_deform_aug["num_sample"] = 1000
    local_deform_aug["disp_scale"] = 2
    kernel_scale = 1.5
    spline_param = "cov_sigma_scale=0.02,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True".format(
        kernel_scale
    )
    local_deform_aug[
        "local_deform_spline_kernel_obj"
    ] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)
    grid_spline_aug = aug_settings[
        ("grid_spline_aug", {}, "settings for grid sampling based spline augmentation")
    ]
    grid_spline_aug["grid_spacing"] = 4
    grid_spline_aug["disp_scale"] = 5
    kernel_scale = 3
    grid_spline_aug[
        "grid_spline_kernel_obj"
    ] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(
        kernel_scale
    )
    rigid_aug_settings = aug_settings[
        ("rigid_aug", {}, "settings for rigid augmentation")
    ]
    rigid_aug_settings["rotation_range"] = [-50, 50]
    rigid_aug_settings["scale_range"] = [0.8, 1.2]
    rigid_aug_settings["translation_range"] = [-10, 10]

    spline_aug = SplineAug(aug_settings)

    points_aug = aug_settings[
        ("points_aug", {}, "settings for remove or add noise points")
    ]
    points_aug["remove_random_points"] = False
    points_aug["add_random_point_noise"] = False
    points_aug["add_random_weight_noise"] = False
    points_aug["remove_random_points_by_ratio"] = 0.05
    points_aug["add_random_point_noise_by_ratio"] = 0.01
    # points_aug["random_weight_noise_scale"] =0.01
    points_aug["random_noise_raidus"] = 1
    points_aug["normalize_weights"] = False
    points_aug["plot"] = False
    point_aug = PointAug(points_aug)

    def _synth(data_dict):
        synth_info = {"aug_settings": aug_settings}
        points, weights = data_dict["points"], data_dict["weights"]
        if aug_settings["do_point_aug"]:
            points, weights, corr_index = point_aug(points, weights)
            synth_info["corr_index"] = corr_index

        if aug_settings["do_local_deform_aug"] or aug_settings["do_spline_aug"]:
            points, weights = spline_aug(points, weights)
        data_dict["points"], data_dict["weights"] = points, weights
        return data_dict, synth_info

    return _synth


def flyingkitti_nonocc_aug_data(**kwargs):
    aug_settings = ParameterDict()
    aug_settings["do_local_deform_aug"] = (
        kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    )
    aug_settings["do_grid_aug"] = (
        kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    )
    aug_settings["do_point_aug"] = (
        kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else True
    )
    aug_settings["do_rigid_aug"] = (
        kwargs["do_rigid_aug"] if "do_rigid_aug" in kwargs else False
    )
    aug_settings["plot"] = False
    local_deform_aug = aug_settings[
        (
            "local_deform_aug",
            {},
            "settings for uniform sampling based spline augmentation",
        )
    ]
    local_deform_aug["num_sample"] = 1000
    local_deform_aug["disp_scale"] = 0.5
    kernel_scale = 1
    spline_param = "cov_sigma_scale=1,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True".format(
        kernel_scale
    )
    local_deform_aug[
        "local_deform_spline_kernel_obj"
    ] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)
    grid_spline_aug = aug_settings[
        ("grid_spline_aug", {}, "settings for grid sampling based spline augmentation")
    ]
    grid_spline_aug["grid_spacing"] = 4
    grid_spline_aug["disp_scale"] = 1
    kernel_scale = 3
    grid_spline_aug[
        "grid_spline_kernel_obj"
    ] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(
        kernel_scale
    )

    rigid_aug_settings = aug_settings[
        ("rigid_aug", {}, "settings for rigid augmentation")
    ]
    rigid_aug_settings["rotation_range"] = [-5, 5]
    rigid_aug_settings["scale_range"] = [0.95, 1.05]
    rigid_aug_settings["translation_range"] = [-0.5, 0.5]

    spline_aug = SplineAug(aug_settings)

    points_aug = aug_settings[
        ("points_aug", {}, "settings for remove or add noise points")
    ]
    points_aug["remove_random_points"] = False
    points_aug["add_random_point_noise"] = False
    points_aug["add_random_weight_noise"] = False
    points_aug["remove_random_points_by_ratio"] = 0.01
    points_aug["add_random_point_noise_by_ratio"] = 0.1
    points_aug["random_weight_noise_scale"] = 0.1
    points_aug["random_noise_raidus"] = 0.01
    points_aug["normalize_weights"] = False
    points_aug["plot"] = False
    point_aug = PointAug(points_aug)

    def _synth(data_dict):
        synth_info = {"aug_settings": aug_settings}
        points, weights = data_dict["points"], data_dict["weights"]

        if aug_settings["do_local_deform_aug"] or aug_settings["do_spline_aug"]:
            points, weights = spline_aug(points, weights)

        if aug_settings["do_point_aug"]:
            points, weights, corr_index = point_aug(points, weights)
            synth_info["corr_index"] = corr_index
        data_dict["points"], data_dict["weights"] = points, weights
        return data_dict, synth_info

    return _synth


if __name__ == "__main__":
    assert (
        shape_type == "pointcloud"
    ), "set shape_type = 'pointcloud'  in global_variable.py"
    device = torch.device("cpu")  # cuda:0  cpu
    reader_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_reader()"
    scale = (
        -1
    )  # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
    normalizer_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_normalizer()"
    sampler_obj = (
        "flyingkitti_nonocc_utils.flyingkitti_nonocc_sampler(num_sample=20000)"
    )
    use_local_mount = True
    remote_mount_transfer = lambda x: x.replace(
        "/playpen-raid1", "/home/zyshen/remote/llr11_mount"
    )
    path_transfer = (
        (lambda x: remote_mount_transfer(x)) if use_local_mount else (lambda x: x)
    )

    dataset_json_path = "/playpen-raid1/zyshen/data/flyingkitti_nonocc/train/pair_data.json"  # home/zyshen/remote/llr11_mount
    dataset_json_path = path_transfer(dataset_json_path)
    pair_name_list, pair_info_list = read_json_into_list(dataset_json_path)
    pair_path_list = [
        [pair_info["source"]["data_path"], pair_info["target"]["data_path"]]
        for pair_info in pair_info_list
    ]
    pair_id = 1002
    pair_path = pair_path_list[pair_id]
    pair_path = [path_transfer(path) for path in pair_path]
    get_obj_func = get_obj(
        reader_obj, normalizer_obj, sampler_obj, device, expand_bch_dim=True
    )
    source, source_interval = get_obj_func(pair_path[0])
    source_points, source_weights = source["points"], source["weights"]

    # set deformation
    aug_settings = ParameterDict()
    aug_settings["do_local_deform_aug"] = True
    aug_settings["do_grid_aug"] = True
    aug_settings["do_rigid_aug"] = True
    aug_settings["plot"] = True

    local_deform_aug = aug_settings[
        (
            "local_deform_aug",
            {},
            "settings for uniform sampling based spline augmentation",
        )
    ]
    local_deform_aug["num_sample"] = 1000
    local_deform_aug["disp_scale"] = 0.5  # 2
    kernel_scale = 1  # 1.5
    spline_param = "cov_sigma_scale=1,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True".format(
        kernel_scale
    )
    local_deform_aug[
        "local_deform_spline_kernel_obj"
    ] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)

    grid_spline_aug = aug_settings[
        ("grid_spline_aug", {}, "settings for grid sampling based spline augmentation")
    ]
    grid_spline_aug["grid_spacing"] = 4
    grid_spline_aug["disp_scale"] = 1  # 5
    kernel_scale = 3
    grid_spline_aug[
        "grid_spline_kernel_obj"
    ] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(
        kernel_scale
    )

    rigid_aug_settings = aug_settings[
        ("rigid_aug", {}, "settings for rigid augmentation")
    ]
    rigid_aug_settings["rotation_range"] = [-15, 15]
    rigid_aug_settings["scale_range"] = [0.9, 1.1]  # [0.8, 1.2]
    rigid_aug_settings["translation_range"] = [-1, 1]  # [-5, 5]

    st = time.time()
    spline_aug = SplineAug(aug_settings)
    print("it takes {} s".format(time.time() - st))
    spline_aug(source_points, source_weights)

    points_aug = aug_settings[
        ("points_aug", {}, "settings for remove or add noise points")
    ]
    points_aug["remove_random_points"] = False
    points_aug["add_random_point_noise"] = True
    points_aug["add_random_weight_noise"] = True
    points_aug["remove_random_points_by_ratio"] = 0.05
    points_aug["add_random_point_noise_by_ratio"] = 0.01
    points_aug["random_noise_raidus"] = 1
    points_aug["normalize_weights"] = False
    points_aug["plot"] = True
    point_aug = PointAug(points_aug)
    point_aug(source_points, source_weights)
