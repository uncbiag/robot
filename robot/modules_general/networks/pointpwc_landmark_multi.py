"""
PointPWC-Net model and losses
Author: Wenxuan Wu
Date: May 2020
"""
from functools import partial
import torch.nn as nn
import torch
from robot.modules_reg.networks.pointconv_util import (
    PointConv,
    PointConvD,
    UpsampleFlow3,
)
from robot.modules_reg.networks.pointconv_util import Conv1d
from robot.utils.utils import shrink_by_factor

scale = 1.0


class PointPWC_Landmark_Multi(nn.Module):
    def __init__(
        self,
        input_channel=3,
        initial_input_radius=1.0,
        first_sampling_npoints=2048,
        predict_at_low_resl=False,
        param_shrink_factor=1,
        output_channels=8,
        use_aniso_kernel=False,
    ):
        super(PointPWC_Landmark_Multi, self).__init__()

        feat_nei = 16
        sbf = partial(shrink_by_factor, factor=param_shrink_factor)
        self.predict_at_low_resl = predict_at_low_resl
        self.scale = 1
        self.res_scale = 0.1
        self.heatmap_threshold = 0.9
        # l0: 8192
        self.level0 = Conv1d(input_channel, sbf(32))
        self.level0_1 = Conv1d(sbf(32), sbf(32))
        self.level0_2 = Conv1d(sbf(32), sbf(64))

        # l1: 2048
        self.level1 = PointConvD(
            first_sampling_npoints,
            feat_nei,
            sbf(64) + 3,
            sbf(64),
            use_aniso_kernel=use_aniso_kernel,
            cov_sigma_scale=initial_input_radius * 20,
            aniso_kernel_scale=initial_input_radius * 80,
        )
        self.level1_0 = Conv1d(sbf(64), sbf(64))
        self.level1_1 = Conv1d(sbf(64), sbf(128))
        self.level1_2 = PointConvD(
            int(first_sampling_npoints),
            feat_nei,
            sbf(64 + 32) + 3,
            sbf(64),
            group_all=True,
        )
        self.heatmap1_re = PointConv(9, sbf(64) + 3, sbf(64), bn=True, use_leaky=True)
        self.heatmap1_conv = Conv1d(sbf(64), output_channels)

        # l2: 512
        self.level2 = PointConvD(
            int(first_sampling_npoints / 4), feat_nei, sbf(128) + 3, sbf(128)
        )
        self.level2_0 = Conv1d(sbf(128), sbf(128))
        self.level2_1 = Conv1d(sbf(128), sbf(256))
        self.level2_2 = PointConvD(
            int(first_sampling_npoints / 4),
            feat_nei,
            sbf(128 + 64) + 3,
            sbf(128),
            group_all=True,
        )
        self.heatmap2_re = PointConv(9, sbf(128) + 3, sbf(128), bn=True, use_leaky=True)
        self.heatmap2_conv = Conv1d(sbf(128), output_channels)

        # l3: 256
        self.level3 = PointConvD(
            int(first_sampling_npoints / 8), feat_nei, sbf(256) + 3, sbf(256)
        )
        self.level3_0 = Conv1d(sbf(256), sbf(256))
        self.level3_1 = Conv1d(sbf(256), sbf(512))
        self.level3_2 = PointConvD(
            int(first_sampling_npoints / 8),
            feat_nei,
            sbf(256 + 64) + 3,
            sbf(256),
            group_all=True,
        )
        self.heatmap3 = PointConv(9, sbf(256) + 3, sbf(256), bn=True, use_leaky=True)
        self.heatmap3_conv = Conv1d(sbf(256), output_channels)

        # l4: 64
        self.level4 = PointConvD(
            int(first_sampling_npoints / 32), feat_nei, sbf(512) + 3, sbf(256)
        )
        self.level4_0 = PointConvD(
            int(first_sampling_npoints / 32), feat_nei, sbf(256) + 3, sbf(256)
        )

        # deconv
        self.deconv4_3 = Conv1d(sbf(256), sbf(64))
        self.deconv3_2 = Conv1d(sbf(256), sbf(64))
        self.deconv2_1 = Conv1d(sbf(128), sbf(32))

        self.heatmap0_re = PointConv(9, sbf(64) + 3, sbf(64), bn=True, use_leaky=True)
        self.heatmap0_conv = Conv1d(sbf(64), output_channels)

        # upsample
        self.upsample = UpsampleFlow3(-1)

    def normalize(self, heatmap, points):
        B, L = heatmap.shape[:2]
        hm_min = heatmap.min(2)[0].view(B, L, 1)
        hm_max = heatmap.max(2)[0].view(B, L, 1)
        heatmap = (heatmap - hm_min) / (hm_max - hm_min)
        heatmap_cp = heatmap.clone()
        heatmap_cp[heatmap < self.heatmap_threshold] = 0
        heatmap = heatmap_cp / (heatmap_cp.sum(-1, keepdim=True) + 1e-9)
        landmarks = torch.einsum("bln,bdn -> bdl", heatmap, points)
        return landmarks

    def forward(self, xyz1, color1):
        # xyz1, xyz2: B, N, 3
        # color1, color2: B, N, 3

        # l0
        _scale = 2
        scaler = lambda x: self.scale * (_scale ** x)
        pc1_l0 = xyz1.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1)  # B 3 N
        feat1_l0 = self.level0(color1)
        feat1_l0 = self.level0_1(feat1_l0)
        feat1_l0_1 = self.level0_2(feat1_l0)

        # l1
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)
        feat1_l1_2 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1_2)

        # l2
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
        feat1_l2_3 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2_3)

        # l3
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
        feat1_l3_4 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3_4)

        # l4
        pc1_l4, feat1_l4, _ = self.level4(pc1_l3, feat1_l3_4)
        _, feat1_l4_3, _ = self.level4_0(pc1_l4, feat1_l4)

        # l3
        feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4, resol_factor=scaler(7))
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)
        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim=1)
        _, c_feat1_l3, _ = self.level3_2(pc1_l3, c_feat1_l3)
        heatmap3 = self.heatmap3_conv(self.heatmap3(pc1_l3, c_feat1_l3))

        # l2
        heatmap3_up = self.upsample(pc1_l2, pc1_l3, heatmap3, resol_factor=scaler(5))
        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, c_feat1_l3, resol_factor=scaler(5))
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)
        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim=1)
        _, c_feat1_l2, _ = self.level2_2(pc1_l2, c_feat1_l2)
        heatmap_re2 = (
            self.heatmap2_conv(self.heatmap2_re(pc1_l2, c_feat1_l2)) * self.res_scale
        )
        heatmap2 = heatmap3_up + heatmap_re2

        # l1
        heatmap2_up = self.upsample(pc1_l1, pc1_l2, heatmap2, resol_factor=scaler(5))
        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, c_feat1_l2, resol_factor=scaler(5))
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)
        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim=1)
        _, c_feat1_l1, _ = self.level1_2(pc1_l1, c_feat1_l1)
        heatmap_re1 = (
            self.heatmap1_conv(self.heatmap1_re(pc1_l1, c_feat1_l1)) * self.res_scale
        )
        heatmap1 = heatmap2_up + heatmap_re1

        if not self.predict_at_low_resl:
            heatmap1_up = self.upsample(
                pc1_l0, pc1_l1, heatmap1, resol_factor=scaler(5)
            )
            feat1_l1_0 = self.upsample(
                pc1_l0, pc1_l1, c_feat1_l1, resol_factor=scaler(5)
            )
            heatmap_re0 = (
                self.heatmap0_conv(self.heatmap0_re(pc1_l0, feat1_l1_0))
                * self.res_scale
            )
            heatmap0 = heatmap1_up + heatmap_re0
            return heatmap0, pc1_l0.transpose(2, 1).contiguous()  # BxCxN,  BxNxD

        else:
            return heatmap1, pc1_l1.transpose(2, 1).contiguous()  # BxCxN,  BxNxD
