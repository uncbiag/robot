import torch
import torch.nn as nn
from robot.kernels.keops_kernels import LazyKeopsKernel
from pointnet2.lib import pointnet2_utils as pointutils
from robot.shape.point_interpolator import nadwat_kernel_interpolator

LEAKY_RATE = 0.1


class ChannelConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        use_leaky=True,
        bn=False,
    ):
        super(ChannelConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = (
            nn.ReLU(inplace=True)
            if not use_leaky
            else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        )

        self.composed_module = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels, bias=True),
                nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
                relu,
            ]
        )

    def forward(self, x):
        x = self.composed_module[0](x)
        x = self.composed_module[1](x.permute(0, 2, 1).contiguous())
        x = self.composed_module[2](x.permute(0, 2, 1).contiguous())
        return x


class GaussSpatialConv(nn.Module):
    def __init__(self, sigma):
        super(GaussSpatialConv, self).__init__()
        self.kernel = LazyKeopsKernel(kernel_type="normalized_gauss", sigma=sigma)

    def forward(self, x, y, y_fea):
        x, y, fea = x.contiguous(), y.contiguous(), y_fea.contiguous()
        return self.kernel(x, y, y_fea)


class MultiGaussSpatialConv(nn.Module):
    def __init__(self, sigma_list, weight_list):
        super(MultiGaussSpatialConv, self).__init__()
        self.kernel = LazyKeopsKernel(
            kernel_type="normalized_multi_gauss",
            sigma_list=sigma_list,
            weight_list=weight_list,
        )

    def forward(self, x, y, y_fea):
        x, y, fea = x.contiguous(), y.contiguous(), y_fea.contiguous()
        return self.kernel(x, y, y_fea)


class AnisoMultiGaussSpatialConv(nn.Module):
    def __init__(self, sigma_list, weight_list):
        super(AnisoMultiGaussSpatialConv, self).__init__()
        self.kernel = LazyKeopsKernel(
            kernel_type="aniso_multi_gauss",
            sigma_list=sigma_list,
            weight_list=weight_list,
        )

    def forward(self, x, y, y_fea, gamma):
        x, y, fea = x.contiguous(), y.contiguous(), y_fea.contiguous()
        return self.kernel(x, y, y_fea, gamma)


class MultiNadwatSpatialConv(nn.Module):
    def __init__(self, sigma_list, weight_list, iso=True, self_center=False):
        super(MultiNadwatSpatialConv, self).__init__()
        self.iso = iso
        self.kernel = nadwat_kernel_interpolator(
            scale=sigma_list, weight=weight_list, iso=iso, self_center=self_center
        )

    def forward(self, x, y, y_fea, y_weight=None, gamma=None):
        x, y, y_fea = x.contiguous(), y.contiguous(), y_fea.contiguous()
        if y_weight is None:
            B, M, device = y.shape[0], y.shape[1], y.device
            y_weight = torch.ones(B, M, 1, device=device) / M
        if self.iso:
            return self.kernel(x, y, y_fea, y_weight)
        else:
            return self.kernel(x, y, y_fea, y_weight, gamma)


def furthest_sampling(nsamples):
    """
    :param xyz: input points position data, [B, C, N]
    :param nsamples:
    :return: B,C, M
    """

    def sampling(xyz):
        xyz = xyz.permute(0, 2, 1).contiguous()
        fps_idx = pointutils.furthest_point_sample(xyz, nsamples)  # [B, N]
        new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, N, D]
        return new_xyz.permute(0, 2, 1).contiguous()

    return sampling
