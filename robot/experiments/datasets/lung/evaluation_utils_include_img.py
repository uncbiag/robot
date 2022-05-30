import os
import numpy as np
import torch
import pyvista as pv
import SimpleITK as sitk
from robot.global_variable import Shape
from robot.experiments.datasets.lung.global_variable import NORMALIZE_SCALE, lung_expri_path
from robot.datasets.vtk_utils import read_vtk
from robot.shape.point_interpolator import NadWatIsoSpline
from robot.utils.shape_visual_utils import save_shape_into_files
from robot.utils.utils import identity_map
from robot.utils.visualizer import (
    capture_plotter,
    visualize_source_flowed_target_overlap,
)



dirlab_landmarks_folder_path = os.path.join(lung_expri_path,"dirlab_landmarks")
dirlab_image_folder_path = os.path.join(lung_expri_path,"dirlab_images")
#dirlab_landmarks_folder_path = "/playpen-raid1/Data/copd/processed_nobias/landmark_processed_colored"


def get_flowed(shape_pair, interp_kernel):
    flowed_points = interp_kernel(
        shape_pair.toflow.points,
        shape_pair.source.points,
        shape_pair.flowed.points,
        shape_pair.source.weights,
    )
    flowed = Shape()
    flowed.set_data_with_refer_to(flowed_points, shape_pair.toflow)
    shape_pair.set_flowed(flowed)
    return shape_pair

def get_flowed_disp_version(shape_pair, interp_kernel):
    flow_points = interp_kernel(
        shape_pair.toflow.points,
        shape_pair.source.points,
        shape_pair.flowed.points -shape_pair.source.points ,
        shape_pair.source.weights,
    )
    flowed = Shape()
    flowed.set_data_with_refer_to(shape_pair.toflow.points + flow_points, shape_pair.toflow)
    shape_pair.set_flowed(flowed)
    return shape_pair


def get_landmarks(source_landmarks_path, target_landmarks_path):
    source_landmarks = read_vtk(source_landmarks_path)["points"]
    target_landmarks = read_vtk(target_landmarks_path)["points"]
    return source_landmarks, target_landmarks

def get_landmarks_range(landmarks_range_path):
    landmarks_range = np.load(landmarks_range_path)
    return landmarks_range

def get_landmarks_dirlab_mapping_info(landmarks_range_path):
    import json
    with open(landmarks_range_path) as f:
        data_dict = json.load(f)
    return data_dict

def map_to_dirlab_coord(landmark_pos, dirlab_mapping_info):
    raw_origin, raw_spacing, dirlab_spacing, raw_shape = \
        dirlab_mapping_info["raw_origin"],dirlab_mapping_info["raw_spacing"],\
        dirlab_mapping_info["dirlab_spacing"],dirlab_mapping_info["raw_shape"]
    z_bias = dirlab_mapping_info["z_bias"]
    raw_origin, raw_spacing, dirlab_spacing, raw_shape = \
        np.array(raw_origin),  np.array(raw_spacing),  np.array(dirlab_spacing), np.array(raw_shape)
    dirlab_coord = (landmark_pos - raw_origin)/raw_spacing
    dirlab_coord[:,2] = (raw_shape[2] - (dirlab_coord[:,2] -z_bias))/4
    dirlab_center_coord = np.floor(dirlab_coord+0.5)+0.5
    dirlab_center_coord = dirlab_center_coord*dirlab_spacing
    return dirlab_center_coord, dirlab_coord.astype(np.float32)


def eval_landmark(model, shape_pair, batch_info, alias, eval_ot_map=False):
    record_path = os.path.join(
        batch_info["record_path"],
        "3d",
        "{}_epoch_{}".format(batch_info["phase"], batch_info["epoch"]),
    )
    s_name_list = batch_info["source_info"]["name"]
    t_name_list = batch_info["target_info"]["name"]
    landmarks_toflow_list, target_landmarks_list = [], []
    target_landmarks_range_list = []
    target_landmark_dirlab_coord_list = []
    target_landmarks_dirlab_mapping_info_list = []
    for i, (s_name, t_name) in enumerate(zip(s_name_list, t_name_list)):
        source_landmarks_path = os.path.join(
            dirlab_landmarks_folder_path, s_name + ".vtk"
        )
        target_landmarks_path = os.path.join(
            dirlab_landmarks_folder_path, t_name + ".vtk"
        )
        target_landmarks_range_path = target_landmarks_path.replace(".vtk","_range.npy")
        target_landmarks_info_path = target_landmarks_path.replace(".vtk","_info.json")
        landmarks_toflow, target_landmarks = get_landmarks(
            source_landmarks_path, target_landmarks_path
        )
        target_landmarks_range = get_landmarks_range(target_landmarks_range_path)
        target_landmarks_dirlab_mapping_info = get_landmarks_dirlab_mapping_info(target_landmarks_info_path)
        to_np = lambda x: x.detach().cpu().numpy()
        s_shift = shape_pair.source.extra_info["transform"]["shift"][i]
        s_scale = shape_pair.source.extra_info["transform"]["scale"][i]
        t_shift = shape_pair.target.extra_info["transform"]["shift"][i]
        t_scale = shape_pair.target.extra_info["transform"]["scale"][i]
        target_landmark_dirlab_coord, _ = map_to_dirlab_coord(target_landmarks, target_landmarks_dirlab_mapping_info)

        landmarks_toflow = (landmarks_toflow - to_np(s_shift)) / to_np(s_scale)
        target_landmarks = (target_landmarks - to_np(t_shift)) / to_np(t_scale)
        target_landmarks_range = (target_landmarks_range - to_np(t_shift)[...,None])/(to_np(t_scale)[...,None])
        landmarks_toflow_list.append(landmarks_toflow)
        target_landmarks_list.append(target_landmarks)
        target_landmarks_range_list.append(target_landmarks_range)
        target_landmark_dirlab_coord_list.append(target_landmark_dirlab_coord)
        target_landmarks_dirlab_mapping_info_list.append(target_landmarks_dirlab_mapping_info)
    device = shape_pair.source.points.device
    flowed_cp = shape_pair.flowed
    landmarks_toflow = torch.Tensor(np.stack(landmarks_toflow_list, 0)).to(device)
    target_landmarks_points = torch.Tensor(np.stack(target_landmarks_list, 0)).to(device)
    target_landmarks_range = torch.Tensor(np.stack(target_landmarks_range_list,0)).to(device)
    target_landmark_dirlab_coord = torch.Tensor(np.stack(target_landmark_dirlab_coord_list,0)).to(device)
    toflow = Shape().set_data(
        points=landmarks_toflow, weights=torch.ones_like(landmarks_toflow)
    )
    shape_pair.toflow = toflow
    gt_landmark = Shape().set_data(
        points=target_landmarks_points, weights=torch.ones_like(target_landmarks_points)
    )
    if not eval_ot_map:
        shape_pair = model.flow(shape_pair)
    else:
        interp_kernel = NadWatIsoSpline(exp_order=2, kernel_scale=0.05)
        shape_pair = get_flowed_disp_version(shape_pair, interp_kernel)
    flowed_landmarks_points = shape_pair.flowed.points
    source_scale, source_shift = shape_pair.source.extra_info["transform"]["scale"], shape_pair.source.extra_info["transform"]["shift"]
    target_scale, target_shift = shape_pair.target.extra_info["transform"]["scale"], shape_pair.target.extra_info["transform"]["shift"]
    source_landmarks_points_raw_coord = landmarks_toflow*source_scale + source_shift
    flowed_landmarks_points_raw_coord = flowed_landmarks_points*target_scale + target_shift
    target_landmarks_points_raw_coord = target_landmarks_points*target_scale + target_shift
    flowed_points_raw_coord = flowed_cp.points*target_scale + target_shift
    flowed_raw = Shape().set_data(
        points=flowed_points_raw_coord, weights=shape_pair.source.weights
    )

    save_shape_into_files(
        record_path, alias+"_flowed_raw", batch_info["pair_name"], flowed_raw
    )
    flowed_landmarks_points_raw_coord_np = to_np(flowed_landmarks_points_raw_coord)
    flowed_landmarks_dirlab_coord_list = [map_to_dirlab_coord(flowed_landmarks_points_raw_coord_np[i],target_landmarks_dirlab_mapping_info_list[i])[0]
                                     for i in range(len(target_landmarks_list))]
    flowed_landmarks_dirlab_coord = torch.Tensor(np.stack(flowed_landmarks_dirlab_coord_list,0)).to(device)
    diff = (target_landmarks_points - flowed_landmarks_points) * NORMALIZE_SCALE
    diff_range = (target_landmarks_range-flowed_landmarks_points[...,None])*NORMALIZE_SCALE
    diff_tovoxel = torch.where(diff_range[...,0].abs()<diff_range[...,1].abs(), diff_range[...,0],diff_range[...,1])
    in_voxel = torch.logical_and(flowed_landmarks_points>target_landmarks_range[...,0], flowed_landmarks_points<target_landmarks_range[...,1])
    in_voxel = torch.all(in_voxel,2)
    acc_voxel = in_voxel.sum(1) / in_voxel.shape[1]
    diff_tovoxel = diff_tovoxel* ((1-in_voxel.float())[...,None])
    diff_dirlab_coord = flowed_landmarks_dirlab_coord - target_landmark_dirlab_coord
    shape_pair.flowed.weights, gt_landmark.weights = diff, diff

    shape_pair.flowed.pointfea = None
    toflow.weights = diff_tovoxel
    save_shape_into_files(
        record_path, "landmark" + alias + "_toflow_point2voxel_error", batch_info["pair_name"], toflow
    )
    toflow.weights = diff_dirlab_coord
    save_shape_into_files(
        record_path, "landmark" + alias + "_toflow_mapped2dirlab_error", batch_info["pair_name"], toflow
    )
    toflow.weights = diff
    save_shape_into_files(
        record_path, "landmark" + alias + "_toflow_point2point_error", batch_info["pair_name"], toflow
    )

    source_landmarks_raw_coord = Shape().set_data(
        points=source_landmarks_points_raw_coord, weights=diff
    )
    save_shape_into_files(
        record_path, "landmark" + alias + "_source_raw_point2point_error", batch_info["pair_name"],
        source_landmarks_raw_coord
    )
    target_landmarks_raw_coord = Shape().set_data(
        points=target_landmarks_points_raw_coord, weights=diff
    )
    save_shape_into_files(
        record_path, "landmark" + alias + "_target_raw_point2point_error", batch_info["pair_name"],
        target_landmarks_raw_coord
    )


    flowed_landmarks_raw_coord = Shape().set_data(
        points=flowed_landmarks_points_raw_coord, weights = diff
    )
    save_shape_into_files(
        record_path, "landmark" + alias + "_flowed_raw_point2point_error", batch_info["pair_name"],
        flowed_landmarks_raw_coord
    )
    # save_shape_into_files(
    #     record_path,
    #     "landmark" + alias + "_flowed",
    #     batch_info["pair_name"],
    #     shape_pair.flowed,
    # )
    save_shape_into_files(
        record_path,
        "landmark" + alias + "_target",
        batch_info["pair_name"],
        gt_landmark,
    )

    shape_pair.flowed = flowed_cp  # compatible to save function
    shape_pair.toflow = None  # compatible to save function

    return diff, diff_tovoxel, acc_voxel, diff_dirlab_coord




#########
CENTER_BIAS=2

def transfer_landmarks_from_dirlab_to_high(dirlab_index, high_shape):
    new_index= dirlab_index.copy()
    new_index[:,-1] =(high_shape[-1]- dirlab_index[:,-1]*4) + CENTER_BIAS
    return new_index

def process_grids(img_info, is_insp=False):
    grid_size = img_info["dirlab_shape"]
    id_map = identity_map(grid_size, [1,1,1])+1 # +1 is due to dirlab convention
    index = id_map.transpose(1,2,3,0).reshape(-1,3)
    # unordered_index = index
    # index = np.zeros_like(unordered_index)
    # index[...,0] = unordered_index[...,2]
    # index[...,1] = unordered_index[...,1]
    # index[...,2] = unordered_index[...,0]
    img_shape_itk, spacing_itk, origin_itk = np.array(img_info["raw_shape"]), np.array(img_info["raw_spacing"]), np.array(
        img_info["raw_origin"])
    transfered_index = transfer_landmarks_from_dirlab_to_high(index, img_shape_itk)
    physical_points = transfered_index * spacing_itk + origin_itk
    # physical_points = physical_points.reshape(grid_size+[3])
    # physical_points = physical_points.transpose(2,1,0,3).reshape(-1,3)
    return physical_points.astype(np.float32)


def save_deformed_image(flowed,source_info, target_info, saving_path=""):
    grid_size = source_info["dirlab_shape"]
    target_img_path = os.path.join(dirlab_image_folder_path,target_info["name"]+".nii.gz")
    target_img = sitk.GetArrayFromImage(sitk.ReadImage(target_img_path)).astype(np.float32)
    target_img = torch.Tensor(target_img[None][None])
    inv_grid = torch.Tensor(flowed.reshape(grid_size+[3])[None])
    spacing = 1/(np.array(grid_size) -1) .astype(np.float32)
    inv_grid = inv_grid*(spacing[None])
    inv_grid = inv_grid*2-1
    inv_warped = torch.nn.functional.grid_sample(target_img, inv_grid, mode="bilinear", padding_mode="border",align_corners=True)
    inv_warped = inv_warped.cpu().numpy().squeeze()
    inv_warped= inv_warped.transpose([2,1,0])
    inv_warped = sitk.GetImageFromArray(inv_warped)
    inv_warped.SetSpacing(source_info["dirlab_spacing"])
    sitk.WriteImage(inv_warped,saving_path)





def eval_img(model, shape_pair, batch_info, alias, eval_ot_map=False):
    if not eval_ot_map:
        return
    record_path = os.path.join(
        batch_info["record_path"],
        "3d",
        "{}_epoch_{}".format(batch_info["phase"], batch_info["epoch"]),
    )
    s_name_list = batch_info["source_info"]["name"]
    t_name_list = batch_info["target_info"]["name"]
    landmarks_toflow_list = []
    source_info_list, target_info_list = [], []
    to_np = lambda x: x.detach().cpu().numpy()

    for i, (s_name, t_name) in enumerate(zip(s_name_list, t_name_list)):
        source_landmarks_path = os.path.join(
            dirlab_landmarks_folder_path, s_name + ".vtk"
        )
        target_landmarks_path = os.path.join(dirlab_landmarks_folder_path, t_name + ".vtk")
        source_landmarks_info_path = source_landmarks_path.replace(".vtk","_info.json")
        target_landmarks_info_path = target_landmarks_path.replace(".vtk","_info.json")
        source_landmarks_dirlab_mapping_info = get_landmarks_dirlab_mapping_info(source_landmarks_info_path)
        target_landmarks_dirlab_mapping_info = get_landmarks_dirlab_mapping_info(target_landmarks_info_path)
        landmarks_toflow = process_grids(source_landmarks_dirlab_mapping_info)
        s_shift = shape_pair.source.extra_info["transform"]["shift"][i]
        s_scale = shape_pair.source.extra_info["transform"]["scale"][i]
        landmarks_toflow = (landmarks_toflow - to_np(s_shift)) / to_np(s_scale)
        landmarks_toflow_list.append(landmarks_toflow)
        source_info_list.append(source_landmarks_dirlab_mapping_info)
        target_info_list.append(target_landmarks_dirlab_mapping_info)
    device = shape_pair.source.points.device
    flowed_cp = shape_pair.flowed
    landmarks_toflow = torch.Tensor(np.stack(landmarks_toflow_list, 0)).to(device)
    toflow = Shape().set_data(
        points=landmarks_toflow, weights=torch.ones_like(landmarks_toflow)
    )
    shape_pair.toflow = toflow
    if not eval_ot_map:
        shape_pair = model.flow(shape_pair)
    else:
        interp_kernel = NadWatIsoSpline(exp_order=2, kernel_scale=0.1)
        shape_pair = get_flowed_disp_version(shape_pair, interp_kernel)
    flowed_landmarks_points = shape_pair.flowed.points
    target_scale, target_shift = shape_pair.target.extra_info["transform"]["scale"], shape_pair.target.extra_info["transform"]["shift"]
    flowed_landmarks_points_raw_coord = flowed_landmarks_points*target_scale + target_shift

    flowed_landmarks_points_raw_coord_np = to_np(flowed_landmarks_points_raw_coord)
    flowed_landmarks_dirlab_img_coord_list = [map_to_dirlab_coord(flowed_landmarks_points_raw_coord_np[i],target_info_list[i])[1]
                                     for i in range(len(flowed_landmarks_points_raw_coord_np))]
    for (flowed_img_coord, source_info, target_info, t_name) in zip(flowed_landmarks_dirlab_img_coord_list, source_info_list,target_info_list,t_name_list):
        saving_path = os.path.join(record_path,t_name+'_img_warped.nii.gz')
        save_deformed_image(flowed_img_coord, source_info, target_info,saving_path)
    shape_pair.flowed = flowed_cp  # compatible to save function
    shape_pair.toflow = None  # compatible to save function


#########


def visualize_feature(shape_pair, batch_info):
    from sklearn.manifold import TSNE

    flowed_fea = shape_pair.flowed.pointfea.detach().cpu().numpy()
    target_fea = shape_pair.target.pointfea.detach().cpu().numpy()
    nbatch, npoints = flowed_fea.shape[0], flowed_fea.shape[1]
    camera_pos = [
        (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
        (0.0, 0.0, 0.0),
        (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
    ]
    record_path = os.path.join(batch_info["record_path"], "fea_visual")
    os.makedirs(record_path, exist_ok=True)
    for b in range(nbatch):
        fea_high = np.concatenate([flowed_fea[b, :, 3:], target_fea[b, :, 3:]], 0)
        fea_embedded = TSNE(n_components=3, perplexity=30, n_jobs=5).fit_transform(
            fea_high
        )
        fea_normalized = (
            (fea_embedded - fea_embedded.min())
            / (fea_embedded.max() - fea_embedded.min() + 1e-7)
            * 3
        )
        flowed_embedding, target_embedded = (
            fea_normalized[:npoints],
            fea_normalized[npoints:],
        )
        pair_name = batch_info["pair_name"][b]
        saving_path = os.path.join(record_path, pair_name + ".png")
        visualize_source_flowed_target_overlap(
            shape_pair.source.points,
            shape_pair.flowed.points,
            shape_pair.target.points,
            flowed_embedding,
            flowed_embedding,
            target_embedded,
            title1="source",
            title2="flowed",
            title3="target",
            rgb_on=True,
            saving_capture_path=saving_path,
            camera_pos=camera_pos,
            add_bg_contrast=False,
            show=False,
        )
        data = pv.PolyData(shape_pair.source.points[b].detach().cpu().numpy())
        data.point_arrays["pointfea"] = flowed_embedding
        dcn = lambda x: x.detach().cpu().numpy()
        data.point_arrays["weights"] = dcn(shape_pair.source.weights[b])
        saving_path = os.path.join(record_path, pair_name + "_source.vtk")
        data.save(saving_path)
        data = pv.PolyData(dcn(shape_pair.flowed.points[b]))
        data.point_arrays["pointfea"] = flowed_embedding
        data.point_arrays["weights"] = dcn(shape_pair.source.weights[b])
        saving_path = os.path.join(record_path, pair_name + "_flowed.vtk")
        data.save(saving_path)
        data = pv.PolyData(dcn(shape_pair.target.points[b]))
        data.point_arrays["pointfea"] = target_embedded
        data.point_arrays["weights"] = dcn(shape_pair.target.weights[b])
        saving_path = os.path.join(record_path, pair_name + "_target.vtk")
        data.save(saving_path)


def evaluate_res(visualize_fea=False):
    def eval(metrics, shape_pair, batch_info, additional_param=None, alias=""):
        phase = batch_info["phase"]
        if phase == "val" or phase == "test":
            if visualize_fea and "mapped_position" not in additional_param:
                visualize_feature(shape_pair, batch_info)
            model = additional_param["model"]
            flowed_points_cp = shape_pair.flowed.points
            shape_pair.control_points = additional_param["initial_nonp_control_points"]
            eval_ot_map = "mapped_position" in additional_param
            has_prealign = (
                "prealign_param" in additional_param
                and additional_param["prealign_param"] is not None
            )
            record_path = os.path.join(
                batch_info["record_path"],
                "3d",
                "{}_epoch_{}".format(batch_info["phase"], batch_info["epoch"]),
            )
            os.makedirs(record_path, exist_ok=True)
            if additional_param is not None and not eval_ot_map:
                source_scale, source_shift = shape_pair.source.extra_info["transform"]["scale"], \
                                             shape_pair.source.extra_info["transform"]["shift"]
                target_scale, target_shift = shape_pair.target.extra_info["transform"]["scale"], \
                                             shape_pair.target.extra_info["transform"]["shift"]
                source_points_raw_coord = shape_pair.source.points * source_scale + source_shift
                target_points_raw_coord = shape_pair.target.points * target_scale + target_shift
                source_raw = Shape().set_data(
                    points=source_points_raw_coord, weights=shape_pair.source.weights
                )
                target_raw = Shape().set_data(
                    points=target_points_raw_coord, weights=shape_pair.target.weights
                )
                save_shape_into_files(
                    record_path, "source_raw", batch_info["pair_name"], source_raw
                )
                save_shape_into_files(
                    record_path, "target_raw", batch_info["pair_name"], target_raw
                )


            if additional_param is not None and has_prealign and not eval_ot_map:
                save_shape_into_files(
                    record_path,
                    alias + "_prealigned",
                    batch_info["pair_name"],
                    additional_param["prealigned"],
                )
                reg_param = additional_param["prealign_param"].detach().cpu().numpy()
                for pid, pair_name in enumerate(batch_info["pair_name"]):
                    np.save(
                        os.path.join(
                            record_path, pair_name + alias + "_prealigned_reg_param.npy"
                        ),
                        reg_param[pid],
                    )
            if additional_param is not None and eval_ot_map:
                shape_pair.flowed.points = additional_param["mapped_position"]
                save_shape_into_files(
                    record_path,
                    alias + "_flowed",
                    batch_info["pair_name"],
                    shape_pair.flowed,
                )
            diff, diff_tovoxel, acc_voxel, diff_dirlab_coord = eval_landmark(
                model, shape_pair, batch_info, alias, eval_ot_map=eval_ot_map
            )
            eval_img(model, shape_pair, batch_info, alias, eval_ot_map=eval_ot_map
            )
            diff_var = (diff - diff.mean(1, keepdim=True)) ** 2
            diff_var = diff_var.sum(2).mean(1)
            diff_norm_mean = diff.norm(p=2, dim=2).mean(1)
            diff_tovoxel_var = (diff_tovoxel - diff_tovoxel.mean(1, keepdim=True)) ** 2
            diff_tovoxel_var = diff_tovoxel_var.sum(2).mean(1)
            diff_tovoxel_norm_mean = diff_tovoxel.norm(p=2, dim=2).mean(1)
            diff_dirlab_coord_var = (diff_dirlab_coord - diff_dirlab_coord.mean(1, keepdim=True)) ** 2
            diff_dirlab_coord_var = diff_dirlab_coord_var.sum(2).mean(1)
            diff_dirlab_coord_norm_mean = diff_dirlab_coord.norm(p=2, dim=2).mean(1)
            metrics.update(
                {
                    "lmk_diff_mean"
                    + alias: [
                        _diff_norm_mean.item() for _diff_norm_mean in diff_norm_mean
                    ],
                    "lmk_diff_var"
                    + alias: [_diff_var.item() for _diff_var in diff_var],
                    "lmk_tovoxel_diff_mean"
                    + alias: [
                        _diff_norm_mean.item() for _diff_norm_mean in diff_tovoxel_norm_mean
                    ],
                    "lmk_tovoxel_diff_var"
                    + alias: [_diff_var.item() for _diff_var in diff_tovoxel_var],
                    "lmk_acc_voxel"
                    + alias: [_acc.item() for _acc in acc_voxel],
                    "lmk_dirlab_coord_diff_mean"
                    + alias: [_diff_norm_mean.item() for _diff_norm_mean in diff_dirlab_coord_norm_mean],
                    "lmk_dirlab_coord_diff_var"
                    + alias: [_diff_var.item() for _diff_var in diff_dirlab_coord_var]

                }
            )
            shape_pair.flowed.points = flowed_points_cp
        return metrics

    return eval
