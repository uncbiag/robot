import torch
import torch.nn as nn
from robot.modules_reg.networks.geo_net_utils import *

SpatialConv = GaussSpatialConv


class GeoFlowNet(nn.Module):
    def __init__(self, input_channel=1, initial_radius=0.001, initial_npoints=8192):
        super(GeoFlowNet, self).__init__()

        # level 0 down
        self.channel_conv0_0 = ChannelConv(input_channel, 16)
        self.channel_conv0_1 = ChannelConv(16, 16)
        radius = initial_radius
        self.spatial_conv0 = SpatialConv(radius * 5)  # conv level0 from level0
        self.channel_conv0_2 = ChannelConv(16, 16)
        self.channel_conv0_3 = ChannelConv(16, 16)

        # level 1 down
        self.sampler1 = furthest_sampling(int(initial_npoints))
        radius = initial_radius * 4
        self.spatial_conv1_0 = SpatialConv(radius * 5)  # conv level1 from level0
        self.channel_conv1_0 = ChannelConv(16, 32)
        self.channel_conv1_1 = ChannelConv(32, 32)
        self.spatial_conv1_1 = SpatialConv(radius * 5)  # conv level1 from level1
        self.channel_conv1_2 = ChannelConv(
            32, 32
        )  # cat point 1 - point 2  and point2-point1
        self.channel_conv1_3 = ChannelConv(32, 32)

        # level 2 down
        radius = initial_radius * 8
        self.sampler2 = furthest_sampling(int(initial_npoints / 4))
        self.spatial_conv2_0 = SpatialConv(radius * 5)  # conv level2 from level1
        self.channel_conv2_0 = ChannelConv(32, 64)
        self.channel_conv2_1 = ChannelConv(64, 64)
        self.spatial_conv2_1 = SpatialConv(radius * 5)  # conv level2 from level2
        self.channel_conv2_pc2_1 = ChannelConv(64, 64)
        self.channel_conv2_2 = ChannelConv(
            128, 64
        )  # cat point 1 - point 2  and point2-point1
        self.channel_conv2_pc2_2 = ChannelConv(64, 64)

        # level 3 down from now on , only point 1 is concerned,  to take care of the keops efficiency, we limit channel less than 64
        radius = initial_radius * 16
        self.sampler3 = furthest_sampling(int(initial_npoints / 16))
        self.spatial_conv3_0 = SpatialConv(radius * 5)  # conv level3 from level2
        self.channel_conv3_0 = ChannelConv(64, 64)
        self.channel_conv3_1 = ChannelConv(64, 64)
        self.spatial_conv3_1 = SpatialConv(
            radius * 5
        )  # conv level3 from level3, do twice
        self.channel_conv3_2_list = nn.ModuleList([ChannelConv(128, 64)] * 2)

        # level 2 up
        radius = initial_radius * 8
        self.spatial_conv2_2 = SpatialConv(
            radius * 5
        )  # conv level2 from level3, do twice
        self.channel_conv2_3 = ChannelConv(64 + 128, 96)
        self.channel_conv2_4 = ChannelConv(96, 96)
        self.spatial_conv2_3 = SpatialConv(
            radius * 5
        )  # conv level2 from level2, do twice
        self.channel_conv2_5 = ChannelConv(96, 96)

        # level 1 up
        radius = initial_radius * 4
        self.spatial_conv1_2 = SpatialConv(radius * 5)  # conv level1 from level2
        self.channel_conv1_4 = ChannelConv(96 + 32, 64)
        self.channel_conv1_5 = ChannelConv(64, 64)
        self.spatial_conv1_3 = SpatialConv(radius * 5)  # conv level2 from level2,
        self.channel_conv1_6 = ChannelConv(64, 64)

        # level 1 up
        radius = initial_radius
        self.spatial_conv0_2 = SpatialConv(radius * 5)  # conv level0 from level1
        self.channel_conv0_4 = ChannelConv(64 + 16, 32)
        self.channel_conv0_5 = ChannelConv(32, 32)
        self.spatial_conv0_3 = SpatialConv(radius * 5)  # conv level0 from level0
        self.channel_conv0_6 = ChannelConv(32, 32)
        self.refine = nn.Linear(32, 3, bias=True)

    def forward(self, pc1, pc2, feature1, feature2):
        # level0 down
        l0_pc1, l0_pc2 = pc1, pc2
        l0_fea1 = self.channel_conv0_1(self.channel_conv0_0(feature1))
        l0_fea1 = (
            self.channel_conv0_2(self.spatial_conv0(l0_pc1, l0_pc1, l0_fea1)) + l0_fea1
        )
        l0_fea1 = self.channel_conv0_3(l0_fea1)
        l0_fea2 = self.channel_conv0_1(self.channel_conv0_0(feature2))
        l0_fea2 = (
            self.channel_conv0_2(self.spatial_conv0(l0_pc2, l0_pc2, l0_fea2)) + l0_fea2
        )
        l0_fea2 = self.channel_conv0_3(l0_fea2)

        # level1 down
        l1_pc1, l1_pc2 = self.sampler1(l0_pc1), self.sampler1(l0_pc2)
        l1_fea1 = torch.cat([self.spatial_conv1_0(l1_pc1, l0_pc1, l0_fea1), l1_pc1], 2)
        l1_fea1 = self.channel_conv1_1(self.channel_conv1_0(l1_fea1))
        l1_fea1 = (
            self.channel_conv1_2(self.spatial_conv1_1(l1_pc1, l1_pc1, l1_fea1))
            + l1_fea1
        )
        l1_fea1 = self.channel_conv1_3(l1_fea1)
        l1_fea2 = torch.cat([self.spatial_conv1_0(l1_pc2, l0_pc2, l0_fea2), l1_pc2], 2)
        l1_fea2 = self.channel_conv1_1(self.channel_conv1_0(l1_fea2))
        l1_fea2 = (
            self.channel_conv1_2(self.spatial_conv1_1(l1_pc2, l1_pc2, l1_fea2))
            + l1_fea2
        )
        l1_fea2 = self.channel_conv1_3(l1_fea2)

        # level2 down
        l2_pc1, l2_pc2 = self.sampler2(l1_pc1), self.sampler2(l1_pc2)
        l2_fea1 = torch.cat([self.spatial_conv2_0(l2_pc1, l1_pc1, l1_fea1), l2_pc1], 2)
        l2_fea1_ = self.channel_conv2_1(self.channel_conv2_0(l2_fea1))
        l2_fea2 = torch.cat([self.spatial_conv2_0(l2_pc2, l1_pc2, l1_fea2), l2_pc2], 2)
        l2_fea2_ = self.channel_conv2_1(self.channel_conv2_0(l2_fea2))

        l2_fea1 = self.channel_conv2_pc2_1(
            self.spatial_conv2_1(l2_pc1, l2_pc2, l2_fea2_)
        )
        l2_fea1 = self.channel_conv2_2(torch.cat([l2_fea1, l2_fea1_], 2))
        l2_fea2 = self.channel_conv2_pc2_2(l2_fea2_)

        # level3 down
        l3_pc1, l3_pc2 = self.sampler3(l2_pc1), self.sampler3(l2_pc2)
        l3_fea1 = torch.cat([self.spatial_conv3_0(l3_pc1, l2_pc1, l2_fea1), l3_pc1], 2)
        l3_fea1_ = self.channel_conv3_1(self.channel_conv3_0(l3_fea1))
        l3_fea2 = torch.cat([self.spatial_conv3_0(l3_pc2, l2_pc2, l2_fea2), l3_pc2], 2)
        l3_fea2_ = self.channel_conv3_1(self.channel_conv3_0(l3_fea2))

        l3_fea1 = self.spatial_conv3_1(l3_pc1, l3_pc2, l3_fea2_)
        l3_fea1 = torch.cat([l3_fea1, l3_fea1_], 2)
        l3_fea1 = [conv(l3_fea1) for conv in self.channel_conv3_2_list]

        # level2 up
        l2_fea1_up = [self.spatial_conv2_2(l2_pc1, l3_pc1, fea) for fea in l3_fea1]
        l2_fea1 = self.channel_conv2_4(
            self.channel_conv2_3(torch.cat(l2_fea1_up + [l2_fea1], 2))
        )
        l2_fea1 = self.channel_conv2_5(l2_fea1)

        # level1 up
        l1_fea1_up = self.spatial_conv1_2(l1_pc1, l2_pc1, l2_fea1)
        l1_fea1 = self.channel_conv1_5(
            self.channel_conv1_4(torch.cat([l1_fea1_up, l1_fea1], 2))
        )
        l1_fea1 = self.channel_conv1_6(l1_fea1)

        # level0 up
        l0_fea1_up = self.spatial_conv0_2(l0_pc1, l1_pc1, l1_fea1)
        l0_fea1 = self.channel_conv0_5(
            self.channel_conv0_4(torch.cat([l0_fea1_up, l0_fea1], 2))
        )
        l0_fea1 = self.channel_conv0_6(l0_fea1)
        flow = self.refine(l0_fea1)
        return flow, None


if __name__ == "__main__":
    import os
    import pykeops

    print(pykeops.config.bin_folder)  # display default build_folder
    cache_path = "/playpen/zyshen/keops_cachev2"
    os.makedirs(cache_path, exist_ok=True)
    pykeops.set_bin_folder(cache_path)
    from robot.utils.net_utils import print_model

    model = GeoFlowNet(input_channel=3, initial_npoints=4096).cuda()
    print_model(model)
    input1 = torch.rand(2, 20000, 3).cuda()
    input2 = torch.rand(2, 20000, 3).cuda()
    fea1 = torch.rand(2, 20000, 3).cuda()
    fea2 = torch.rand(2, 20000, 3).cuda()
    flow, _ = model(input1, input2, fea1, fea2)
    flow.mean(2).mean().backward()
