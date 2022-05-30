"""
PointConv util functions
Author: Wenxuan Wu
Date: May 2020
"""

import torch
import torch.nn as nn
from pykeops.torch import LazyTensor
import torch.nn.functional as F
import pointnet2.lib.pointnet2_utils as pointnet2_utils
from robot.shape.point_interpolator import nadwat_kernel_interpolator
from robot.utils.knn_utils import KNN, AnisoKNN

LEAKY_RATE = 0.1
use_bn = False


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        use_leaky=True,
        bn=use_bn,
    ):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = (
            nn.ReLU(inplace=True)
            if not use_leaky
            else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        )

        self.composed_module = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu,
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """

    # sqrdists = square_distance(new_xyz, xyz)
    # _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    # return group_idx
    new_xyz = LazyTensor(new_xyz[:, :, None].contiguous())  # BxSx1xC
    xyz = LazyTensor(xyz[:, None].contiguous())  # Bx1xNxC
    dist2 = new_xyz.sqdist(xyz)
    group_idx = dist2.argKmin(nsample, dim=2)
    return group_idx


def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(
        points_flipped, knn_idx.int()
    ).permute(0, 2, 3, 1)

    return new_points


def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, S, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


def aniso_group_query(cov_sigma_scale, aniso_kernel_scale):
    # compatible to code in pointnet2
    aniso_knn = AnisoKNN(
        cov_sigma_scale=cov_sigma_scale,
        aniso_kernel_scale=aniso_kernel_scale,
        return_value=False,
    )

    def group_query(nsample, s_xyz, xyz, s_points):
        """
        Input:
            nsample: scalar
            s_xyz: input points position data, [B, N, C]
            s_points: input points data, [B, N, D]
            xyz: input points position data, [B, S, C]
        Return:
            new_xyz: sampled points position data, [B, 1, C]
            new_points: sampled points data, [B, S, N, C+D]
        """
        B, N, C = s_xyz.shape
        S = xyz.shape[1]
        new_xyz = xyz
        idx = aniso_knn(new_xyz, s_xyz, nsample)
        grouped_xyz = index_points_group(s_xyz, idx)  # [B, npoint, nsample, C]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
        if s_points is not None:
            grouped_points = index_points_group(s_points, idx)
            new_points = torch.cat(
                [grouped_xyz_norm, grouped_points], dim=-1
            )  # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm
        return new_points, grouped_xyz_norm

    return group_query


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], bn=use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights = F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights


class PointConv(nn.Module):
    def __init__(
        self, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True
    ):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = (
            nn.ReLU(inplace=True)
            if not use_leaky
            else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        )

    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(
            input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)
        ).view(B, N, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points


class PointConvD(nn.Module):
    def __init__(
        self,
        npoint,
        nsample,
        in_channel,
        out_channel,
        weightnet=16,
        bn=use_bn,
        use_leaky=True,
        group_all=False,
        use_aniso_kernel=False,
        cov_sigma_scale=0.02,
        aniso_kernel_scale=0.08,
    ):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        self.group_all = group_all
        self.use_aniso_kernel = use_aniso_kernel
        self.group_query = (
            group_query
            if not self.use_aniso_kernel
            else aniso_group_query(
                cov_sigma_scale=abs(cov_sigma_scale),
                aniso_kernel_scale=abs(aniso_kernel_scale),
            )
        )
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = (
            nn.ReLU(inplace=True)
            if not use_leaky
            else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        )

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # import ipdb; ipdb.set_trace()
        npoint = self.npoint if self.npoint > 0 else xyz.shape[1]
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1).contiguous()
        points = points.permute(0, 2, 1).contiguous()
        if not self.group_all:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint)
            new_xyz = index_points_gather(xyz, fps_idx)
        else:
            fps_idx = torch.arange(N, device=xyz.device).repeat(B, 1)
            new_xyz = xyz
        new_points, grouped_xyz_norm = self.group_query(
            self.nsample, xyz, new_xyz, points
        )

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(
            input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)
        ).view(B, npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx.type(torch.LongTensor)


class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn=use_bn, use_leaky=True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = (
            nn.ReLU(inplace=True)
            if not use_leaky
            else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        )

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1)  # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat(
            [grouped_points1, grouped_points2, direction_xyz], dim=-1
        )  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1

        point_to_patch_cost = torch.sum(weights * new_points, dim=2)  # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1)  # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1
        grouped_point_to_patch_cost = index_points_group(
            point_to_patch_cost.permute(0, 2, 1), knn_idx
        )  # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(
            weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim=2
        )  # B C N

        return patch_to_patch_cost


class ContiguousBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()


class PointWarping(nn.Module):
    def forward(self, xyz1, xyz2, flow1=None, resol_factor=None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1)  # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1)  # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        knn_idx = knn_point(3, xyz1_to_2, xyz2)
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(
            B, N2, 1, C
        )  # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim=2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1)  # B 3 N2

        return warped_xyz2


class PointWarping2(nn.Module):
    def __init__(self, initial_radius):
        super(PointWarping2, self).__init__()
        self.initial_radius = initial_radius

    def forward(self, xyz1, xyz2, flow1=None, resol_factor=1):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1 = xyz1.permute(0, 2, 1).contiguous()  # B N1 3
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1).contiguous()  # B N1 3
        xyz2 = xyz2.permute(0, 2, 1).contiguous()  # B N2  3
        flow1 = flow1.permute(0, 2, 1).contiguous()
        weight = torch.ones(B, N2, 1, device=xyz1.device)
        interpolator = nadwat_kernel_interpolator(
            scale=self.initial_radius * resol_factor
        )
        flow2 = ContiguousBackward.apply(interpolator(xyz2, xyz1_to_2, flow1, weight))
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1)
        return warped_xyz2


class PointWarping3(nn.Module):
    def __init__(self, initial_radius):
        super(PointWarping3, self).__init__()
        self.initial_radius = initial_radius
        self.knn = KNN(return_value=False)

    def forward(self, xyz1, xyz2, flow1=None, resol_factor=1, K=5):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1 = xyz1.permute(0, 2, 1).contiguous()  # B N1 3
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1).contiguous()  # B N1 3
        xyz2 = xyz2.permute(0, 2, 1).contiguous()  # B N2  3
        flow1 = flow1.permute(0, 2, 1).contiguous()
        weight = torch.ones(B, N2, 1, device=xyz1.device)

        if self.initial_radius > 0:
            interpolator = nadwat_kernel_interpolator(
                scale=self.initial_radius * resol_factor
            )
            flow2 = ContiguousBackward.apply(
                interpolator(xyz2, xyz1_to_2, flow1, weight)
            )
            warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1)
        else:
            index = self.knn(xyz2, xyz1_to_2, K)
            grouped_flow2 = index_points_group(flow1, index)
            flow2 = torch.mean(grouped_flow2, dim=2)
            warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1)
        return warped_xyz2


class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow, resol_factor=None):
        # import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1)  # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1)  # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1)  # B S 3
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(
            B, N, 1, C
        )
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim=2).permute(
            0, 2, 1
        )
        return dense_flow


class UpsampleFlow2(nn.Module):
    def __init__(self, initial_radius):
        super(UpsampleFlow2, self).__init__()
        self.initial_radius = initial_radius

    def forward(self, xyz, sparse_xyz, sparse_flow, resol_factor=1):
        radius = self.initial_radius * resol_factor
        B, S = sparse_xyz.shape[0], sparse_xyz.shape[2]
        xyz = xyz.permute(0, 2, 1).contiguous()  # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1).contiguous()  # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1).contiguous()  # B S D
        sparse_weight = torch.ones(B, S, 1, device=xyz.device)

        interpolator = nadwat_kernel_interpolator(scale=radius)
        dense_flow = ContiguousBackward.apply(
            interpolator(xyz, sparse_xyz, sparse_flow, sparse_weight)
        )
        return dense_flow.permute(0, 2, 1)


class UpsampleFlow3(nn.Module):
    def __init__(
        self,
        initial_radius,
    ):
        super(UpsampleFlow3, self).__init__()
        self.initial_radius = initial_radius
        self.knn = KNN(return_value=False if self.initial_radius < 0 else True)

    def forward(self, xyz, sparse_xyz, sparse_flow, resol_factor=1, K=5):
        xyz = xyz.permute(0, 2, 1).contiguous()  # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1).contiguous()  # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1).contiguous()
        if self.initial_radius > 0:
            sigma = self.initial_radius * resol_factor
            K_dist, index = self.knn(xyz / sigma, sparse_xyz / sigma, K)
            grouped_flow = index_points_group(sparse_flow, index)
            K_w = torch.nn.functional.softmax(-K_dist, dim=2)
            dense_flow = torch.sum(K_w[..., None] * grouped_flow, dim=2)
        else:
            index = self.knn(xyz, sparse_xyz, K)
            grouped_flow = index_points_group(sparse_flow, index)
            dense_flow = torch.mean(grouped_flow, dim=2)
        return dense_flow.permute(0, 2, 1)


class SceneFlowEstimatorPointConv(nn.Module):
    def __init__(
        self,
        feat_ch,
        cost_ch,
        flow_ch=3,
        channels=[128, 128],
        mlp=[128, 64],
        neighbors=9,
        weightnet=16,
        clamp=[-200, 200],
        use_leaky=True,
    ):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(
                neighbors,
                last_channel + 3,
                ch_out,
                weightnet=weightnet,
                bn=True,
                use_leaky=True,
            )
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        """
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        """
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])


class SceneFlowEstimatorPointConv2(nn.Module):
    def __init__(
        self,
        feat_ch,
        cost_ch,
        flow_ch=3,
        channels=[128, 128],
        mlp=[128, 64],
        neighbors=9,
        weightnet=16,
        clamp=[-200, 200],
        use_leaky=True,
    ):
        super(SceneFlowEstimatorPointConv2, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(
                neighbors,
                last_channel + 3,
                ch_out,
                weightnet=weightnet,
                bn=True,
                use_leaky=True,
            )
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)
        self.fea_fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        """
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        """
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        fea_flow = self.fea_fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1]), fea_flow


class SceneFlowEstimatorPointConv3(nn.Module):
    def __init__(
        self,
        feat_ch,
        cost_ch,
        flow_ch=3,
        channels=[128, 128],
        mlp=[128, 64],
        neighbors=9,
        weightnet=16,
        clamp=[-200, 200],
        use_leaky=True,
    ):
        super(SceneFlowEstimatorPointConv3, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(
                neighbors,
                last_channel + 3,
                ch_out,
                weightnet=weightnet,
                bn=True,
                use_leaky=True,
            )
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)
        self.fea_fc = nn.Conv1d(last_channel, 3, 1)
        self.shift_fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        """
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        """
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        fea_flow = self.fea_fc(new_points)
        shift_flow = self.shift_fc(new_points)
        return (
            new_points,
            flow.clamp(self.clamp[0], self.clamp[1]),
            fea_flow,
            shift_flow,
        )
