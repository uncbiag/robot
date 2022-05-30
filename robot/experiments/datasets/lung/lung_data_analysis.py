import os, sys
import subprocess

# os.environ['DISPLAY'] = ':99.0'
# os.environ['PYVISTA_OFF_SCREEN'] = 'true'
# os.environ['PYVISTA_USE_IPYVTK'] = 'true'
# bashCommand ="Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & sleep 3"
# process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
# process.wait()
sys.path.insert(0, os.path.abspath("../../../.."))
from copy import deepcopy
import numpy as np
import torch
import pyvista as pv
import matplotlib.pyplot as plt
from robot.global_variable import Shape, shape_type
from robot.datasets.data_utils import read_json_into_list, get_obj, get_file_name
from robot.shape.shape_pair_utils import create_shape_pair
from robot.utils.obj_factory import obj_factory
from robot.utils.visualizer import (
    visualize_point_fea,
    visualize_point_pair,
    visualize_multi_point,
)
from robot.utils.local_feature_extractor import *


def get_pair(source_path, target_path, expand_bch_dim=True, return_tensor=True):

    get_obj_func = get_obj(
        reader_obj,
        normalizer_obj,
        sampler_obj,
        device,
        expand_bch_dim=expand_bch_dim,
        return_tensor=return_tensor,
    )
    source_obj, source_interval = get_obj_func(source_path)
    target_obj, target_interval = get_obj_func(target_path)
    return source_obj, target_obj


def plot_pair_weight_distribution(
    source_weight, target_weight, use_log=False, title="", show=True, save_path=None
):
    plt.style.use("bmh")
    fig, ax = plt.subplots()
    source_weight = np.log(source_weight) if use_log else source_weight
    target_weight = np.log(target_weight) if use_log else target_weight
    ax.hist(source_weight, bins=1000, density=0, histtype="stepfilled", alpha=0.7)
    ax.hist(target_weight, bins=1000, density=0, histtype="stepfilled", alpha=0.5)
    title += "weight" if not use_log else "log_weight"
    ax.set_title(title)
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.clf()


def plot_pair_weight_distribution_before_and_after_radius_matching(
    source_weight1,
    target_weight1,
    source_weight2,
    target_weight2,
    use_log=False,
    title="",
    show=True,
    save_path=None,
):
    plt.style.use("bmh")

    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flatten()
    source_weight_matched1 = matching_np_radius(source_weight1, target_weight1)
    smw_sum1, sw_sum1, tp_sum1 = (
        source_weight_matched1.sum(),
        source_weight1.sum(),
        target_weight1.sum(),
    )

    source_weight1 = np.log(source_weight1) if use_log else source_weight1
    target_weight1 = np.log(target_weight1) if use_log else target_weight1
    ax0.hist(source_weight1, bins=1000, density=0, histtype="stepfilled", alpha=0.7)
    ax0.hist(target_weight1, bins=1000, density=0, histtype="stepfilled", alpha=0.5)
    ax0.set_title("sw_sum: {:.3f}, tp_sum:{:.3f}".format(sw_sum1, tp_sum1), fontsize=10)
    source_weight_matched1_norm = (
        np.log(source_weight_matched1) if use_log else source_weight_matched1
    )
    ax1.hist(
        source_weight_matched1_norm,
        bins=1000,
        density=0,
        histtype="stepfilled",
        alpha=0.7,
    )
    ax1.hist(target_weight1, bins=1000, density=0, histtype="stepfilled", alpha=0.5)
    ax1.set_title(
        "smw_sum: {:.3f}, tp_sum:{:.3f}".format(smw_sum1, tp_sum1), fontsize=10
    )

    source_weight_matched2 = matching_np_radius(source_weight2, target_weight2)
    smw_sum2, sw_sum2, tp_sum2 = (
        source_weight_matched2.sum(),
        source_weight2.sum(),
        target_weight2.sum(),
    )
    source_weight2 = np.log(source_weight2) if use_log else source_weight2
    target_weight2 = np.log(target_weight2) if use_log else target_weight2
    ax2.hist(source_weight2, bins=1000, density=0, histtype="stepfilled", alpha=0.7)
    ax2.hist(target_weight2, bins=1000, density=0, histtype="stepfilled", alpha=0.5)
    ax2.set_title("sw_sum: {:.3f}, tp_sum:{:.3f}".format(sw_sum2, tp_sum2), fontsize=10)
    source_weight_matched2_norm = (
        np.log(source_weight_matched2) if use_log else source_weight_matched2
    )
    ax3.hist(
        source_weight_matched2_norm,
        bins=1000,
        density=0,
        histtype="stepfilled",
        alpha=0.7,
    )
    ax3.hist(target_weight2, bins=1000, density=0, histtype="stepfilled", alpha=0.5)
    ax3.set_title(
        "smw_sum: {:.3f}, tp_sum:{:.3f}".format(smw_sum2, tp_sum2), fontsize=10
    )

    fig.subplots_adjust(hspace=0.3)
    fig.suptitle(title)
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.clf()
    return source_weight_matched1, source_weight_matched2


def get_half_lung(lung, normalize_weight=False):
    weights = lung.weights.detach()
    points = lung.points.detach()
    pos_filter = points[..., 0] < 0
    points = points[pos_filter][None]
    weights = weights[pos_filter][None]
    weights = weights
    weights = weights / weights.sum() if normalize_weight else weights
    half_lung = Shape()
    half_lung.set_data(points=points, weights=weights)
    return half_lung


def get_key_vessel(lung, thre=2e-05):
    weights = lung.weights.detach()
    points = lung.points.detach()
    mask = (lung.weights > thre)[..., 0]
    weights = weights[mask][None]
    points = points[mask][None]
    key_lung = Shape()
    key_lung.set_data(points=points, weights=weights)
    return key_lung


def sampled_via_radius(source, target):
    min_npoints = min(source.npoints, target.npoints)
    tw = target.weights.squeeze()
    sw = source.weights.squeeze()
    t_sorted, t_indices = torch.sort(tw, descending=True)
    s_sorted, s_indices = torch.sort(sw, descending=True)
    t_sampled_indices = t_indices[:min_npoints]
    s_sampled_indices = s_indices[:min_npoints]
    tp_sampled = target.points[:, t_sampled_indices]
    sp_sampled = source.points[:, s_sampled_indices]
    tw_sampled = target.weights[:, t_sampled_indices]
    sw_sampled = source.weights[:, s_sampled_indices]
    target_sampled, source_sampled = Shape(), Shape()
    target_sampled.set_data(points=tp_sampled, weights=tw_sampled)
    source_sampled.set_data(points=sp_sampled, weights=sw_sampled)
    return source_sampled, target_sampled


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.
    Code adapted from
    http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def matching_np_radius(source_weights, target_weights):
    """

    :param source_weights: Nx1
    :param target_weights: Mx1
    :param matched_weights: Nx1
    :return:
    """

    ns = source_weights.shape[0]
    sw = source_weights.squeeze()
    tw = target_weights.squeeze()
    range = [min(sw.min(), tw.min()), max(sw.max(), tw.max())]
    resol = 10000
    interp = (range[1] - range[0]) / resol
    bins = np.linspace(range[0] - 2 * interp, range[1] + 2 * interp, resol)
    sw_indice = np.digitize(sw, bins, right=False)
    tw_indice = np.digitize(tw, bins, right=False)
    sw_digitize = bins[sw_indice]
    tw_digitize = bins[tw_indice]
    sw_transformed = hist_match(sw_digitize, tw_digitize)
    return sw_transformed.reshape(ns, 1).astype(np.float32)


def matching_shape_radius(source, target, sampled_by_radius=False, show=True):
    if sampled_by_radius:
        source, target = sampled_via_radius(source, target)
    device = source.points.device
    sn = source.npoints
    tn = target.npoints
    sw = source.weights.squeeze().cpu().numpy()
    tw = target.weights.squeeze().cpu().numpy()
    range = [min(sw.min(), tw.min()), max(sw.max(), tw.max())]
    resol = 10000
    interp = (range[1] - range[0]) / resol
    bins = np.linspace(range[0] - 2 * interp, range[1] + 2 * interp, resol)
    sw_indice = np.digitize(sw, bins, right=False)
    tw_indice = np.digitize(tw, bins, right=False)
    sw_digitize = bins[sw_indice]
    tw_digitize = bins[tw_indice]
    sw_transformed = hist_match(sw_digitize, tw_digitize)
    if show:
        plot_pair_weight_distribution(sw_digitize, tw_digitize, use_log=True)
        plot_pair_weight_distribution(sw_transformed, tw_digitize, use_log=True)
        visualize_point_pair(
            source.points,
            target.points,
            source.weights,
            target.weights,
            title1="source(before)",
            title2="target(before)",
        )

        visualize_point_pair(
            source.points,
            target.points,
            sw_transformed,
            tw_digitize,
            title1="source(after)",
            title2="target(after)",
        )
    source.weights = (
        torch.tensor(sw_transformed.astype(np.float32)).to(device).view(1, sn, 1)
    )
    target.weights = (
        torch.tensor(tw_digitize.astype(np.float32)).to(device).view(1, tn, 1)
    )
    return source, target


def source_weight_transform(weights, compute_on_half_lung=False):
    weights = weights * 1
    weights_cp = deepcopy(weights)
    thre = 1.9e-05
    thre = thre  # if not compute_on_half_lung else thre*2
    weights[weights_cp < thre] = 1e-7
    return weights


def flowed_weight_transform(weights, compute_on_half_lung=False):
    weights = weights * 1
    weights_cp = deepcopy(weights)
    thre = 1.9e-05
    thre = thre  # if not compute_on_half_lung else thre * 2
    weights[weights_cp < thre] = 1e-7
    return weights


def target_weight_transform(weights, compute_on_half_lung=False):
    weights = weights * 1
    weights_cp = deepcopy(weights)
    thre = 1.9e-05
    thre = thre  # if not compute_on_half_lung else thre * 2
    weights[weights_cp < thre] = 1e-7
    # weights[weights_cp > 1.1e-05] = 1e-7
    return weights


def pair_shape_transformer(init_thres=2.9e-5, nstep=5):
    # todo the next step of the transformer is to return a smoothed mask to constrain the movement of the lung
    def transform(source, target, cur_step):
        min_weights = min(torch.min(source.weights), torch.min(target.weights))
        max_weights = min(torch.max(source.weights), torch.max(target.weights))
        max_weights = max_weights.item()
        cur_step = cur_step.item()
        assert init_thres > min_weights
        thres = init_thres - (init_thres - min_weights) / nstep * cur_step
        s_weights = source.weights.clone()
        t_weights = target.weights.clone()
        s_weights[source.weights < thres] = 1e-7
        t_weights[target.weights < thres] = 1e-7
        s_transformed, t_transformed = Shape(), Shape()
        s_transformed.set_data(
            points=source.points, weights=s_weights, pointfea=source.pointfea
        )
        t_transformed.set_data(
            points=target.points, weights=t_weights, pointfea=target.pointfea
        )
        print("the weight of the lung pair are updated")
        return s_transformed, t_transformed

    return transform


def capture_plotter(save_source=False):
    from robot.utils.visualizer import visualize_point_pair_overlap

    inner_count = 0

    def save(record_path, name_suffix, shape_pair):
        nonlocal inner_count
        source, flowed, target = shape_pair.source, shape_pair.flowed, shape_pair.target
        for sp, fp, tp, sw, fw, tw, pair_name in zip(
            source.points,
            flowed.points,
            target.points,
            source.weights,
            flowed.weights,
            target.weights,
            pair_name_list,
        ):
            if inner_count == 0 or save_source:
                path = os.path.join(
                    record_path, "source_target" + "_" + name_suffix + ".png"
                )
                visualize_point_pair_overlap(
                    sp,
                    tp,
                    flowed_weight_transform(fw, True),
                    target_weight_transform(tw, True),
                    title1="source",
                    title2="target",
                    rgb_on=False,
                    saving_capture_path=path,
                    show=False,
                )
            path_1 = os.path.join(
                record_path,
                pair_name + "_flowed_target" + "_main_" + name_suffix + ".png",
            )
            path_2 = os.path.join(
                record_path,
                pair_name + "_flowed_target" + "_whole_" + name_suffix + ".png",
            )
            visualize_point_pair_overlap(
                fp,
                tp,
                flowed_weight_transform(fw, True),
                target_weight_transform(tw, True),
                title1="flowed",
                title2="target",
                rgb_on=False,
                saving_capture_path=path_1,
                show=False,
            )
            visualize_point_pair_overlap(
                fp,
                tp,
                fw,
                tw,
                title1="flowed",
                title2="target",
                rgb_on=False,
                saving_capture_path=path_2,
                show=False,
            )
        inner_count += 1

    return save


def lung_isolated_leaf_clean_up(
    lung, radius=0.032, principle_weight=None, normalize_weights=True
):

    points = lung.points.detach()
    weights = lung.weights.detach()
    mass, dev, cov = compute_local_moments(points, radius=radius)
    eigenvector_main = compute_local_fea_from_moments(
        "eigenvector_main", weights, mass, dev, cov
    )
    filter = mass[..., 0].squeeze() > 2
    to_remove = ~filter
    print(
        "In the first step, num of points are removed {}, {}".format(
            torch.sum(to_remove), torch.sum(to_remove) / len(filter)
        )
    )
    points_toremove = points[:, to_remove]
    mass_toremove = mass[:, to_remove]
    mass = mass[:, filter]
    points = points[:, filter]
    weights = weights[:, filter]
    eigenvector_main = eigenvector_main[:, filter]
    visualize_point_fea_with_arrow(points, mass, eigenvector_main * 0.01, rgb_on=False)
    visualize_point_overlap(
        points,
        points_toremove,
        mass,
        mass_toremove,
        title="cleaned points",
        point_size=(10, 20),
        rgb_on=False,
        opacity=("linear", 1.0),
    )

    Gamma = compute_anisotropic_gamma_from_points(
        points,
        cov_sigma_scale=radius,
        aniso_kernel_scale=radius,
        principle_weight=principle_weight,
    )
    mass, dev, cov = compute_aniso_local_moments(points, Gamma)
    eigenvector_main = compute_local_fea_from_moments(
        "eigenvector_main", weights, mass, dev, cov
    )
    filter = mass[..., 0].squeeze() > 2.5
    to_remove = ~filter
    print(
        "In the second step, num of points are removed {}, {}".format(
            torch.sum(to_remove), torch.sum(to_remove) / len(filter)
        )
    )
    points_toremove = points[:, to_remove]
    mass_toremove = mass[:, to_remove]
    mass = mass[:, filter]
    points = points[:, filter]
    weights = weights[:, filter]
    eigenvector_main = eigenvector_main[:, filter]
    visualize_point_fea_with_arrow(points, mass, eigenvector_main * 0.01, rgb_on=False)
    visualize_point_overlap(
        points,
        points_toremove,
        mass,
        mass_toremove,
        title="cleaned points",
        point_size=(10, 20),
        rgb_on=False,
        opacity=("linear", 1.0),
    )

    Gamma = compute_anisotropic_gamma_from_points(
        points,
        cov_sigma_scale=radius,
        aniso_kernel_scale=radius,
        principle_weight=principle_weight,
    )
    mass, dev, cov = compute_aniso_local_moments(points, Gamma)
    eigenvector_main = compute_local_fea_from_moments(
        "eigenvector_main", weights, mass, dev, cov
    )
    filter = mass[..., 0].squeeze() > 3
    to_remove = ~filter
    print(
        "In the third step, num of points are removed {}, {}".format(
            torch.sum(to_remove), torch.sum(to_remove) / len(filter)
        )
    )
    points_toremove = points[:, to_remove]
    mass_toremove = mass[:, to_remove]
    mass = mass[:, filter]
    points = points[:, filter]
    weights = weights[:, filter]
    eigenvector_main = eigenvector_main[:, filter]
    visualize_point_fea_with_arrow(points, mass, eigenvector_main * 0.01, rgb_on=False)
    visualize_point_overlap(
        points,
        points_toremove,
        mass,
        mass_toremove,
        title="cleaned points",
        point_size=(10, 20),
        rgb_on=False,
        opacity=("linear", 1.0),
    )

    cleaned_lung = Shape()
    cleaned_lung.points, cleaned_lung.weights = (
        points,
        weights / torch.sum(weights) if normalize_weights else weights,
    )
    return cleaned_lung


def analysis_large_vessel(
    source,
    target,
    source_weight_transform=source_weight_transform,
    target_weight_transform=target_weight_transform,
    title1="source",
    title2="target",
):
    source_points, source_weights, = (
        source.points.detach().cpu(),
        source.weights.detach().cpu(),
    )
    target_points, target_weights, = (
        target.points.detach().cpu(),
        target.weights.detach().cpu(),
    )
    plot_pair_weight_distribution(
        source_weight_transform(source_weights).squeeze().numpy(),
        target_weight_transform(target_weights).squeeze().numpy(),
        use_log=True,
    )
    visualize_point_pair(
        source_points,
        target_points,
        source_weight_transform(source_weights),
        target_weight_transform(target_weights),
        title1=title1,
        title2=title2,
    )


def compute_atlas(weight_list):
    atlas_weight = np.concatenate(weight_list)
    return atlas_weight


def transfer_radius_and_save_sample(
    cur_obj, atlas_distri, radius_transfered_saing_path
):
    cur_obj["weights"] = matching_np_radius(cur_obj["weights"], atlas_distri)
    data = pv.PolyData(cur_obj["points"])
    for key, item in cur_obj.items():
        if key not in ["points"]:
            data.point_arrays[key] = item
    data.save(radius_transfered_saing_path)
    return cur_obj


if __name__ == "__main__":
    assert (
        shape_type == "pointcloud"
    ), "set shape_type = 'pointcloud'  in global_variable.py"
    device = torch.device("cpu")  # cuda:0  cpu
    reader_obj = "lung_dataloader_utils.lung_reader()"
    normalizer_obj = (
        "lung_dataloader_utils.lung_normalizer(weight_scale=60000,scale=[100,100,100])"
    )
    phase = "train"

    use_local_mount = False
    remote_mount_transfer = lambda x: x.replace(
        "/playpen-raid1", "/home/zyshen/remote/llr11_mount"
    )
    path_transfer = (
        (lambda x: remote_mount_transfer(x)) if use_local_mount else (lambda x: x)
    )

    dataset_json_path = (
        "/playpen-raid1/zyshen/data/lung_expri/{}/pair_data.json".format(phase)
    )
    dataset_json_path = path_transfer(dataset_json_path)

    sampler_obj = "lung_dataloader_utils.lung_sampler( method='voxelgrid',scale=0.0003)"
    get_obj_func = get_obj(
        reader_obj,
        normalizer_obj,
        sampler_obj,
        device,
        expand_bch_dim=False,
        return_tensor=False,
    )
    altas_path = "/playpen-raid1/Data/UNC_vesselParticles/10067M_INSP_STD_MSM_COPD_wholeLungVesselParticles.vtk"
    altas_path = path_transfer(altas_path)
    atlas, _ = get_obj_func(altas_path)
    sampler_obj = "lung_dataloader_utils.lung_sampler( method='combined',scale=0.0003,num_sample=30000,sampled_by_weight=True)"
    get_obj_func = get_obj(
        reader_obj,
        normalizer_obj,
        sampler_obj,
        device,
        expand_bch_dim=False,
        return_tensor=False,
    )
    sampled_atlas, _ = get_obj_func(altas_path)

    radius_transfered_saing_path = "/playpen-raid1/zyshen/data/lung_atlas/{}".format(
        phase
    )
    radius_transfered_saing_path = path_transfer(radius_transfered_saing_path)
    os.makedirs(radius_transfered_saing_path, exist_ok=True)

    pair_name_list, pair_info_list = read_json_into_list(dataset_json_path)
    pair_path_list = [
        [pair_info["source"]["data_path"], pair_info["target"]["data_path"]]
        for pair_info in pair_info_list
    ]
    pair_id = 3
    output_path = "/playpen-raid1/zyshen/data/lung_data_analysis/val"
    for pair_id in range(len(pair_name_list)):
        pair_path = pair_path_list[pair_id]
        pair_path = [path_transfer(path) for path in pair_path]
        sampler_obj = (
            "lung_dataloader_utils.lung_sampler( method='voxelgrid',scale=0.0003)"
        )

        ########################
        plot_saving_path = os.path.join(radius_transfered_saing_path, "origin_plots")
        os.makedirs(plot_saving_path, exist_ok=True)

        source_path, target_path = pair_path_list[pair_id]
        source, target = get_pair(
            source_path, target_path, expand_bch_dim=False, return_tensor=False
        )

        saving_path = os.path.join(plot_saving_path, pair_name_list[pair_id] + ".png")
        camera_pos = [
            (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
            (0.0, 0.0, 0.0),
            (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
        ]
        visualize_point_pair(
            source["points"],
            target["points"],
            source["weights"],
            target["weights"],
            title1="source",
            title2="target",
            saving_capture_path=saving_path,
            camera_pos=camera_pos,
            show=False,
        )
        plot_saving_path = os.path.join(radius_transfered_saing_path, "plots")
        os.makedirs(plot_saving_path, exist_ok=True)

        # vtk_saving_path = os.path.join(radius_transfered_saing_path,"data")
        # os.makedirs(vtk_saving_path,exist_ok=True)
        # saving_path = os.path.join(vtk_saving_path,get_file_name(source_path)+".vtk")
        # mapped_source = transfer_radius_and_save_sample(source, atlas["weights"], saving_path)
        # saving_path = os.path.join(vtk_saving_path,get_file_name(target_path)+".vtk")
        # mapped_target = transfer_radius_and_save_sample(target, atlas["weights"], saving_path)
        # plot_saving_path = os.path.join(radius_transfered_saing_path, "plots")
        # source_vg_weight, target_vg_weight = source["weights"], target["weights"]
        # sampler_obj ="lung_dataloader_utils.lung_sampler( method='combined',scale=0.0003,num_sample=30000,sampled_by_weight=True)"
        # source, target = get_pair(source_path, target_path, expand_bch_dim=False, return_tensor=False)
        # source_combined_weight, target_combined_weight = source["weights"], target["weights"]
        # os.makedirs(plot_saving_path,exist_ok=True)
        # saving_file_path = os.path.join(plot_saving_path,pair_info_list[pair_id]["source"]["name"]+"_weights_distribution.png")
        # title = pair_info_list[pair_id]["source"]["name"] + "_" +"n_sp:{} ".format(len(source_vg_weight))+"n_tp:{}".format(len(atlas["weights"]))
        # _,source_combined_mapped_weight =plot_pair_weight_distribution_before_and_after_radius_matching(source_vg_weight, atlas["weights"],source_combined_weight,sampled_atlas["weights"], use_log=True,title=title,show=False,save_path=saving_file_path)
        # saving_file_path = os.path.join(plot_saving_path,    pair_info_list[pair_id]["target"]["name"] + "_weights_distribution.png")
        # title = pair_info_list[pair_id]["target"]["name"] + "_" + "n_sp:{} ".format(len(target_vg_weight)) + "n_tp:{}".format(len(atlas["weights"]))
        # _,target_combined_mapped_weight =plot_pair_weight_distribution_before_and_after_radius_matching(target_vg_weight, atlas["weights"], target_combined_weight, sampled_atlas["weights"],use_log=True, title=title, show=False,save_path=saving_file_path)

        # saving_path = os.path.join(plot_saving_path, pair_name_list[pair_id]+"_mapped.png")
        # camera_pos = [(-4.924379645467042, 2.17374925796456, 1.5003730890759344), (0.0, 0.0, 0.0),
        #               (0.40133888001174545, 0.31574165540339943, 0.8597873634998591)]
        # visualize_point_pair(source["points"], target["points"],
        #                      source_combined_mapped_weight,
        #                      target_combined_mapped_weight,
        #                      title1="source", title2="target", rgb_on=False,saving_capture_path=saving_path,camera_pos=camera_pos,show=False )

    # source, target = get_pair(*pair_path)
    # source_vg_weight, target_vg_weight = source["weights"], target["weights"]
    # title = pair_name_list[pair_id] + "_" +"n_sp:{} ".format(len(source_vg_weight))+"n_tp:{}".format(len(target_vg_weight))
    # sampler_obj ="lung_dataloader_utils.lung_sampler( method='combined',scale=0.0003,num_sample=30000,sampled_by_weight=True)"
    # source, target = get_pair(source_path, target_path, expand_bch_dim=False, return_tensor=False)
    # source_combined_weight, target_combined_weight = source["weights"], target["weights"]
    # plot_saving_path = os.path.join(radius_transfered_saing_path,"plots")
    # saving_folder_path = os.path.join(output_path,pair_name_list[pair_id])
    # os.makedirs(saving_folder_path,exist_ok=True)
    # saving_file_path = os.path.join(saving_folder_path,pair_name_list[pair_id]+"_weights_distribution.png")
    # plot_pair_weight_distribution_before_and_after_radius_matching(source_vg_weight, target_vg_weight,source_combined_weight,target_combined_weight, use_log=True,title=title,show=False,save_path=saving_file_path)
    #
    # visualize_point_pair(source["points"], target["points"],
    #                 source["weights"],
    #                  target["weights"],
    #                  title1="source", title2="target", rgb_on=False)

    #
    #
    # shape_pair = create_shape_pair(source, target)
    # source_half = get_half_lung(source)
    # target_half = get_half_lung(target)
    # cleaned_source_half = lung_isolated_leaf_clean_up(source_half,radius=0.02, principle_weight=[2,1,1], normalize_weights=False)
    # # visualize_point_pair(source_half.points, cleaned_source_half.points,
    # #                      source_weight_transform(source_half.weights),
    # #                      source_weight_transform(cleaned_source_half.weights),
    # #                      title1="source", title2="cleaned_source", rgb_on=False)
    # #
    # # plot_pair_weight_distribution(source_weight_transform(source_half.weights).cpu().squeeze().numpy(),
    # #                               target_weight_transform(target_half.weights).cpu().squeeze().numpy(),
    # #                               use_log=True)
    #
    # visualize_point_pair(source_half.points, target_half.points,
    #                      source_weight_transform(source_half.weights),
    #                      target_weight_transform(target_half.weights),
    #                      title1="source", title2="target", rgb_on=False)
