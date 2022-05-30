import torch

from robot.shape.shape_pair import ShapePair
from robot.global_variable import Shape, shape_type
from robot.shape.point_interpolator import (
    nadwat_kernel_interpolator,
    ridge_kernel_intepolator,
)

use_fast_fps = False
try:
    from pointnet2.lib.pointnet2_utils import FurthestPointSampling

    use_fast_fps = True
except:
    Warning("failed to import Furthest point sampling from pointnet2 ")


def reg_param_initializer():
    def init(input_data):
        reg_param = torch.zeros_like(input_data["source"]["points"])
        return reg_param

    return init


def create_source_and_target_shape():
    def create(input_dict):
        source_dict, target_dict = input_dict["source"], input_dict["target"]
        source_shape = Shape()
        source_shape.set_data(**source_dict)
        target_shape = Shape()
        target_shape.set_data(**target_dict)
        return source_shape, target_shape

    return create


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = torch.zeros(B, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def create_shape_pair(
    source, target, toflow=None, pair_name=None, n_control_points=-1, extra_info={}
):
    shape_pair = ShapePair()
    shape_pair.set_source_and_target(source, target)
    shape_pair.extra_info = extra_info
    if n_control_points > 0:
        if not use_fast_fps:
            control_idx = farthest_point_sample(
                source.points, n_control_points
            )  # non-gpu accerlatation
        else:
            control_idx = FurthestPointSampling.apply(source.points, n_control_points)
        assert control_idx.shape[0] == 1
        control_idx = control_idx.squeeze().long()
        control_points = source.points[:, control_idx]
        shape_pair.control_points = control_points
        device = source.points.device
        shape_pair.control_weights = (
            torch.ones(shape_pair.nbatch, n_control_points, 1, device=device)
            / n_control_points
        )
        shape_pair.dense_mode = False
    if toflow is not None:
        shape_pair.set_toflow(toflow)
    if pair_name is not None:
        shape_pair.set_pair_name(pair_name)
    return shape_pair


def prepare_shape_pair(n_control_points=-1):
    def prepare(source, target, toflow=None, pair_name=None, extra_info={}):
        return create_shape_pair(
            source,
            target,
            toflow=toflow,
            pair_name=pair_name,
            n_control_points=n_control_points,
            extra_info=extra_info,
        )

    return prepare


def create_shape_from_data_dict(attr_list=None):
    def create(data_dict):
        shape = Shape()
        shape.set_data(**data_dict)
        if attr_list is not None:
            # todo test
            for attr in attr_list:
                if attr in data_dict:
                    setattr(shape, attr, data_dict[attr])
        return shape

    return create


def create_shape_pair_from_data_dict(attr_list=None):
    def create(data_dict):
        shape_pair = ShapePair()
        source_dict, target_dict = data_dict["source"], data_dict["target"]
        source = Shape()
        source.set_data(**source_dict)
        target = Shape()
        target.set_data(**target_dict)
        shape_pair.set_source_and_target(source, target)
        if "toflow" in data_dict:
            toflow = Shape()
            toflow.set_data(**data_dict["toflow"])
            shape_pair.toflow = toflow
        if "flowed" in data_dict:
            flowed = Shape()
            flowed.set_data(**data_dict["flowed"])
            shape_pair.flowed = flowed
        if "reg_param" in data_dict:
            shape_pair.reg_param = data_dict["reg_param"]
        if "control_points" in data_dict:
            shape_pair.control_points = data_dict["control_points"]
        if "control_weights" in data_dict:
            shape_pair.control_weights = data_dict["control_weights"]
        if "flowed_control_points" in data_dict:
            shape_pair.flowed_control_points = data_dict["flowed_control_points"]
        if "extra_info" in data_dict:
            shape_pair.extra_info = data_dict["extra_info"]
        if attr_list is not None:
            # todo test
            for attr in attr_list:
                setattr(shape_pair, attr, data_dict["attr"])
        return shape_pair

    return create


def decompose_shape_into_dict():
    def decompose(shape):
        data_dict = {
            attr: getattr(shape, attr)
            for attr in shape.attr_list
            if getattr(shape, attr) is not None
        }
        if shape.extra_info is not None:
            data_dict["extra_info"] = shape.extra_info
        return data_dict

    return decompose


def decompose_shape_pair_into_dict():
    def decompose(shape_pair):
        data_dict = {}
        data_dict["source"] = decompose_shape_into_dict()(shape_pair.source)
        data_dict["target"] = decompose_shape_into_dict()(shape_pair.target)
        if shape_pair.toflow is not None:
            data_dict["toflow"] = decompose_shape_into_dict()(shape_pair.toflow)
        if shape_pair.flowed is not None:
            data_dict["flowed"] = decompose_shape_into_dict()(shape_pair.flowed)
        if shape_pair.reg_param is not None:
            data_dict["reg_param"] = shape_pair.reg_param
        if shape_pair.control_points is not None:
            data_dict["control_points"] = shape_pair.control_points
        if shape_pair.control_weights is not None:
            data_dict["control_weights"] = shape_pair.control_weights
        if shape_pair.flowed_control_points is not None:
            data_dict["flowed_control_points"] = shape_pair.flowed_control_points
        if shape_pair.extra_info is not None:
            data_dict["extra_info"] = shape_pair.extra_info
        return data_dict

    return decompose
