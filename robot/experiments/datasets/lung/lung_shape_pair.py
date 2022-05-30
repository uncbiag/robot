import torch
from robot.shape.shape_pair import ShapePair
import pointnet2.lib.pointnet2_utils as pointutils


def reg_param_initializer():
    def init(input_data):
        reg_param = torch.zeros_like(input_data["source"]["points"])
        return reg_param

    return init


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # import ipdb; ipdb.set_trace()
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


def create_shape_pair(source, target, toflow=None, pair_name=None, n_control_points=-1):
    shape_pair = ShapePair()
    shape_pair.set_source_and_target(source, target)
    if n_control_points > 0:
        control_idx = farthest_point_sample(
            source.points, n_control_points
        )  # non-gpu accerlatation
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
