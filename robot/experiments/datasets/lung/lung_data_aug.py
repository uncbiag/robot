"""keops failed on multi-processing data loader,  so the synthsis process are put before the network"""
import os, sys
import pykeops
import subprocess

from robot.experiments.datasets.lung.global_variable import lung_expri_path
from robot.utils.visualizer import visualize_point_pair_overlap

sys.path.insert(0, os.path.abspath("../../../.."))
try:
    cache_path = "/playpen/zyshen/keops_cachev2"
    os.makedirs(cache_path, exist_ok=True)
    pykeops.set_bin_folder(cache_path)  # change the build folder
except:
    pass
# os.environ['DISPLAY'] = ':99.0'
# os.environ['PYVISTA_OFF_SCREEN'] = 'true'
# os.environ['PYVISTA_USE_IPYVTK'] = 'true'
# bashCommand ="Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & sleep 3"
# process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
# process.wait()
import random
import time
from robot.experiments.datasets.lung.lung_data_analysis import *
from robot.global_variable import *
from robot.experiments.datasets.lung.visualizer import camera_pos, lung_plot
from robot.datasets.data_aug import SplineAug, PointAug
from robot.utils.module_parameters import ParameterDict
from robot.utils.utils import enlarge_by_factor
from functools import partial


def lung_synth_data(**kwargs):
    aug_settings = ParameterDict()
    aug_settings["do_local_deform_aug"] = (
        kwargs["do_local_deform_aug"] if "do_local_deform_aug" in kwargs else True
    )
    aug_settings["do_grid_aug"] = (
        kwargs["do_grid_aug"] if "do_grid_aug" in kwargs else True
    )
    aug_settings["do_point_aug"] = (
        kwargs["do_point_aug"] if "do_point_aug" in kwargs else True
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
    local_deform_aug["disp_scale"] = 0.03
    kernel_scale = 0.04
    spline_param = "cov_sigma_scale=0.03,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True".format(
        kernel_scale
    )
    local_deform_aug[
        "local_deform_spline_kernel_obj"
    ] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)
    grid_spline_aug = aug_settings[
        ("grid_spline_aug", {}, "settings for grid sampling based spline augmentation")
    ]
    grid_spline_aug["grid_spacing"] = 0.9
    grid_spline_aug["disp_scale"] = 0.25
    kernel_scale = 0.25
    grid_spline_aug[
        "grid_spline_kernel_obj"
    ] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(
        kernel_scale
    )
    rigid_aug_settings = aug_settings[
        ("rigid_aug", {}, "settings for rigid augmentation")
    ]
    rigid_aug_settings["rotation_range"] = [-30, 30]
    rigid_aug_settings["scale_range"] = [0.8, 1.2]
    rigid_aug_settings["translation_range"] = [-0.1, 0.1]

    spline_aug = SplineAug(aug_settings)

    points_aug = aug_settings[
        ("points_aug", {}, "settings for remove or add noise points")
    ]
    points_aug["remove_random_points"] = False
    points_aug["add_random_point_noise"] = False
    points_aug["add_random_weight_noise"] = True
    points_aug["remove_random_points_by_ratio"] = 0.01
    points_aug["add_random_point_noise_by_ratio"] = 0.01
    points_aug["random_weight_noise_scale"] = 0.1
    points_aug["random_noise_raidus"] = 0.1
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


def lung_aug_data(**kwargs):
    aug_settings = ParameterDict()
    aug_settings["do_local_deform_aug"] = (
        kwargs["do_local_deform_aug"] if "do_local_deform_aug" in kwargs else True
    )
    aug_settings["do_grid_aug"] = (
        kwargs["do_grid_aug"] if "do_grid_aug" in kwargs else True
    )
    aug_settings["do_point_aug"] = (
        kwargs["do_point_aug"] if "do_point_aug" in kwargs else True
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
    local_deform_aug["disp_scale"] = 0.03
    kernel_scale = 0.04
    spline_param = "cov_sigma_scale=0.03,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True, self_center=True".format(
        kernel_scale
    )
    local_deform_aug[
        "local_deform_spline_kernel_obj"
    ] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)
    grid_spline_aug = aug_settings[
        ("grid_spline_aug", {}, "settings for grid sampling based spline augmentation")
    ]
    grid_spline_aug["grid_spacing"] = 0.9
    grid_spline_aug["disp_scale"] = 0.08
    kernel_scale = 0.15
    grid_spline_aug[
        "grid_spline_kernel_obj"
    ] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(
        kernel_scale
    )

    rigid_aug_settings = aug_settings[
        ("rigid_aug", {}, "settings for rigid augmentation")
    ]
    rigid_aug_settings["rotation_range"] = [-15, 15]
    rigid_aug_settings["scale_range"] = [0.8, 1.2]
    rigid_aug_settings["translation_range"] = [-0.1, 0.1]

    spline_aug = SplineAug(aug_settings)

    points_aug = aug_settings[
        ("points_aug", {}, "settings for remove or add noise points")
    ]
    points_aug["remove_random_points"] = False
    points_aug["add_random_point_noise"] = False
    points_aug["add_random_weight_noise"] = True
    points_aug["remove_random_points_by_ratio"] = 0.01
    points_aug["add_random_point_noise_by_ratio"] = 0.01
    points_aug["random_weight_noise_scale"] = 0.1
    points_aug["random_noise_raidus"] = 0.1
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


if __name__ == "__main__":
    assert (
        shape_type == "pointcloud"
    ), "set shape_type = 'pointcloud'  in global_variable.py"
    device = torch.device("cpu")  # cuda:0  cpu
    reader_obj = "lung_dataloader_utils.lung_reader()"
    scale = (
        -1
    )  # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
    normalizer_obj = (
        "lung_dataloader_utils.lung_normalizer(weight_scale=60000,scale=[100,100,100])"
    )
    sampler_obj = "lung_dataloader_utils.lung_sampler( method='combined',scale=0.0003,num_sample=60000,sampled_by_weight=True)"
    use_local_mount = True
    dataset_json_path = os.path.join(ROBOT_PATH, "demos/data/lung_data/lung_dataset_splits/train/pair_data.json")
    saving_output_path = os.path.join(lung_expri_path, "output/data_aug_visual")
    path_transfer = lambda x: x.replace('./', ROBOT_PATH + "/")
    dataset_json_path = path_transfer(dataset_json_path)
    saving_output_path = path_transfer(saving_output_path)
    os.makedirs(saving_output_path, exist_ok=True)
    pair_name_list, pair_info_list = read_json_into_list(dataset_json_path)
    pair_path_list = [
        [pair_info["source"]["data_path"], pair_info["target"]["data_path"]]
        for pair_info in pair_info_list
    ]
    # pair_id = 0

    pair_index_list = list(range(len(pair_name_list)))
    for pair_id in pair_index_list:
        pair_path = pair_path_list[pair_id]
        pair_path = [path_transfer(path) for path in pair_path]
        get_obj_func = get_obj(
            reader_obj, normalizer_obj, sampler_obj, device, expand_bch_dim=True
        )
        source, source_interval = get_obj_func(pair_path[0])
        target, target_interval = get_obj_func(pair_path[1])
        source_points, source_weights = source["points"], source["weights"]

        input_data = {"source": source, "target": target}
        create_shape_pair_from_data_dict = obj_factory(
            "shape_pair_utils.create_source_and_target_shape()"
        )
        source, target = create_shape_pair_from_data_dict(input_data)

        # set deformation
        aug_settings = ParameterDict()
        aug_settings["do_local_deform_aug"] = True
        aug_settings["do_grid_aug"] = True
        aug_settings["do_point_aug"] = True
        aug_settings["do_rigid_aug"] = False
        aug_settings["plot"] = True
        local_deform_aug = aug_settings[
            (
                "local_deform_aug",
                {},
                "settings for uniform sampling based spline augmentation",
            )
        ]
        local_deform_aug["num_sample"] = 1000
        local_deform_aug["disp_scale"] = 0.03
        kernel_scale = 0.05
        spline_param = "cov_sigma_scale=0.02,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True".format(
            kernel_scale
        )
        local_deform_aug[
            "local_deform_spline_kernel_obj"
        ] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)
        grid_spline_aug = aug_settings[
            (
                "grid_spline_aug",
                {},
                "settings for grid sampling based spline augmentation",
            )
        ]
        grid_spline_aug["grid_spacing"] = 0.9
        grid_spline_aug["disp_scale"] = 0.25
        kernel_scale = 0.25
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
        rigid_aug_settings["translation_range"] = [-0.3, 0.3]

        st = time.time()
        spline_aug = SplineAug(aug_settings)
        print("it takes {} s".format(time.time() - st))
        aug_points, aug_points_weights = spline_aug(source_points, source_weights)

        points_aug = aug_settings[
            ("points_aug", {}, "settings for remove or add noise points")
        ]
        points_aug["remove_random_points"] = False
        points_aug["add_random_point_noise"] = False
        points_aug["add_random_weight_noise"] = True
        points_aug["remove_random_points_by_ratio"] = 0.01
        points_aug["add_random_point_noise_by_ratio"] = 0.01
        points_aug["random_weight_noise_scale"] = 0.1
        points_aug["random_noise_raidus"] = 0.1
        points_aug["normalize_weights"] = False
        points_aug["plot"] = True
        point_aug = PointAug(points_aug)
        aug_points, aug_points_weights, _ = point_aug(aug_points, aug_points_weights)
        aug_shape = Shape().set_data(points=aug_points, weights=aug_points_weights)
        shape_pair = create_shape_pair(source, target)
        shape_pair.flowed = aug_shape
        shape_name = pair_info_list[pair_id]["source"]["name"]
        saving_capture_path = os.path.join(saving_output_path, shape_name)
        os.makedirs(saving_capture_path, exist_ok=True)
        saving_capture_path = os.path.join(
            saving_capture_path, "{}_synth.png".format(shape_name)
        )
        # visualize_point_pair_overlap(source_points, aug_points, source_weights, aug_points_weights, "source", "synth", rgb_on=False,  saving_capture_path=saving_capture_path, camera_pos=camera_pos,show=True)

        visualize_point_pair_overlap(
            source_points,
            aug_points,
            source_weights,
            aug_points_weights,
            "source",
            "synth",
            lung_plot(color="source"),
            lung_plot(color="target"),
            saving_capture_path=saving_capture_path,
            light_mode = "none",
            camera_pos = camera_pos,
            show=True
        )
        # visualize_point_overlap(source_points, aug_points, source_weights, aug_points_weights, "aug_overlap_target", point_size=[15,15],rgb_on=False,  saving_capture_path=saving_capture_path, camera_pos=camera_pos,show=True)
