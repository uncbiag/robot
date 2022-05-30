import torch.nn as nn
import torch
import numpy as np
from functools import partial
import torch.nn.functional as F
from robot.utils.utils import shrink_by_factor
from pointnet2.util import (
    PointNetSetAbstraction,
    PointNetFeaturePropogation,
    FlowEmbedding,
    PointNetSetUpConv,
)


class FlowNet3D(nn.Module):
    def __init__(
        self,
        input_channel=3,
        initial_radius=0.001,
        initial_npoints=4096,
        param_shrink_factor=1.0,
        init_neigh_num=16,
        predict_at_low_resl=False,
    ):
        super(FlowNet3D, self).__init__()
        sbf = partial(shrink_by_factor, factor=param_shrink_factor)
        self.sa1 = PointNetSetAbstraction(
            npoint=initial_npoints,
            radius=20 * initial_radius,
            nsample=init_neigh_num,
            in_channel=input_channel,
            mlp=sbf([32, 32, 64]),
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=shrink_by_factor(initial_npoints, 4),
            radius=40 * initial_radius,
            nsample=16,
            in_channel=sbf(64),
            mlp=sbf([64, 64, 128]),
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=shrink_by_factor(initial_npoints, 16),
            radius=80 * initial_radius,
            nsample=8,
            in_channel=sbf(128),
            mlp=sbf([128, 128, 256]),
            group_all=False,
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=shrink_by_factor(initial_npoints, 64),
            radius=160 * initial_radius,
            nsample=8,
            in_channel=sbf(256),
            mlp=sbf([256, 256, 512]),
            group_all=False,
        )

        self.fe_layer = FlowEmbedding(
            radius=500 * initial_radius,
            nsample=64,
            in_channel=sbf(128),
            mlp=sbf([128, 128, 128]),
            pooling="max",
            corr_func="concat",
        )

        self.su1 = PointNetSetUpConv(
            nsample=8,
            radius=96 * initial_radius,
            f1_channel=sbf(256),
            f2_channel=sbf(512),
            mlp=[],
            mlp2=sbf([256, 256]),
        )
        self.su2 = PointNetSetUpConv(
            nsample=8,
            radius=48 * initial_radius,
            f1_channel=sbf(128 + 128),
            f2_channel=sbf(256),
            mlp=sbf([128, 128, 256]),
            mlp2=sbf([256]),
        )
        self.su3 = PointNetSetUpConv(
            nsample=8,
            radius=24 * initial_radius,
            f1_channel=sbf(64),
            f2_channel=sbf(256),
            mlp=sbf([128, 128, 256]),
            mlp2=sbf([256]),
        )
        self.fp = PointNetFeaturePropogation(
            in_channel=sbf(256) + input_channel, mlp=sbf([256, 256])
        )

        self.conv1 = nn.Conv1d(sbf(256), sbf(128), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(sbf(128))
        self.conv2 = nn.Conv1d(sbf(128), 3, kernel_size=1, bias=True)

    def forward(self, pc1, pc2, feature1, feature2):
        l1_pc1, l1_feature1, _ = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1, _ = self.sa2(l1_pc1, l1_feature1)

        l1_pc2, l1_feature2, _ = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2, _ = self.sa2(l1_pc2, l1_feature2)

        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        l3_pc1, l3_feature1, _ = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1, _ = self.sa4(l3_pc1, l3_feature1)

        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(
            l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1
        )
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)

        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        nonp_param = self.conv2(x)
        nonp_param = nonp_param.transpose(2, 1).contiguous()
        return nonp_param, {"predict_at_low_resl": False}


class FlowNet3DIMP(nn.Module):
    def __init__(
        self,
        input_channel=3,
        initial_radius=0.001,
        initial_npoints=4096,
        param_shrink_factor=1.0,
        predict_at_low_resl=False,
        init_neigh_num=16,
        use_aniso_kernel=True,
    ):
        super(FlowNet3DIMP, self).__init__()
        sbf = partial(shrink_by_factor, factor=param_shrink_factor)
        self.predict_at_low_resl = predict_at_low_resl
        self.sa0 = PointNetSetAbstraction(
            npoint=initial_npoints,
            radius=20 * initial_radius,
            nsample=16,
            in_channel=input_channel,
            mlp=sbf([16, 16, 24]),
            group_all=True,
            use_aniso_kernel=use_aniso_kernel,
            cov_sigma_scale=initial_radius * 20,
            aniso_kernel_scale=initial_radius * 80,
        )
        self.sa1 = PointNetSetAbstraction(
            npoint=initial_npoints,
            radius=20 * initial_radius,
            nsample=init_neigh_num,
            in_channel=sbf(24),
            mlp=sbf([64, 64, 64]),
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=shrink_by_factor(initial_npoints, 4),
            radius=40 * initial_radius,
            nsample=16,
            in_channel=sbf(64),
            mlp=sbf([64, 64, 128]),
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=shrink_by_factor(initial_npoints, 16),
            radius=80 * initial_radius,
            nsample=8,
            in_channel=sbf(128),
            mlp=sbf([128, 128, 256]),
            group_all=False,
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=shrink_by_factor(initial_npoints, 64),
            radius=160 * initial_radius,
            nsample=8,
            in_channel=sbf(256),
            mlp=sbf([256, 256, 512]),
            group_all=False,
        )

        self.fe_layer = FlowEmbedding(
            radius=500 * initial_radius,
            nsample=64,
            in_channel=sbf(128),
            mlp=sbf([128, 128, 128]),
            pooling="max",
            corr_func="concat",
        )

        self.su1 = PointNetSetUpConv(
            nsample=8,
            radius=96 * initial_radius,
            f1_channel=sbf(256),
            f2_channel=sbf(512),
            mlp=[],
            mlp2=sbf([256, 256]),
        )
        self.su2 = PointNetSetUpConv(
            nsample=8,
            radius=48 * initial_radius,
            f1_channel=sbf(128 + 128),
            f2_channel=sbf(256),
            mlp=sbf([128, 128, 256]),
            mlp2=sbf([256]),
        )
        self.su3 = PointNetSetUpConv(
            nsample=8,
            radius=24 * initial_radius,
            f1_channel=sbf(64),
            f2_channel=sbf(256),
            mlp=sbf([128, 128, 256]),
            mlp2=sbf([256]),
        )
        self.fp = PointNetFeaturePropogation(
            in_channel=sbf(256) + input_channel, mlp=sbf([256, 256])
        )

        self.conv1 = nn.Conv1d(sbf(256), sbf(128), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(sbf(128))
        self.conv2 = nn.Conv1d(sbf(128), 3, kernel_size=1, bias=True)

    def forward(self, pc1, pc2, feature1, feature2):
        l0_pc1, l0_feature1, _ = self.sa0(pc1, feature1)
        l1_pc1, l1_feature1, low_idx = self.sa1(l0_pc1, l0_feature1)
        l2_pc1, l2_feature1, _ = self.sa2(l1_pc1, l1_feature1)

        l0_pc2, l0_feature2, _ = self.sa0(pc2, feature2)
        l1_pc2, l1_feature2, _ = self.sa1(l0_pc2, l0_feature2)
        l2_pc2, l2_feature2, _ = self.sa2(l1_pc2, l1_feature2)

        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        l3_pc1, l3_feature1, _ = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1, _ = self.sa4(l3_pc1, l3_feature1)

        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(
            l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1
        )
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        if not self.predict_at_low_resl:
            l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        else:
            l0_fnew1 = l1_fnew1
        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        nonp_param = self.conv2(x)
        nonp_param = nonp_param.transpose(2, 1).contiguous()
        return (
            nonp_param,
            {
                "control_points": l1_pc1.transpose(2, 1),
                "control_points_idx": low_idx,
                "predict_at_low_resl": True,
            }
            if self.predict_at_low_resl
            else {"predict_at_low_resl": False},
        )


if __name__ == "__main__":
    import os
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input = torch.randn((8, 3, 2048)).cuda()
    label = torch.randn(8, 16).cuda()
    model = FlowNet3D().cuda()
    output = model(input, input, input, input)
    # print(output.size())
