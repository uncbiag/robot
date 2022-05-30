
from functools import partial
import torch.nn as nn
import torch
from pykeops.torch import LazyTensor
import numpy as np
import torch.nn.functional as F
from robot.modules_reg.networks.pointconv_util import (
    PointConv,
    PointConvD,
    PointWarping3,
    UpsampleFlow3,
    PointWarping,
    UpsampleFlow,
    PointConvFlow,
    SceneFlowEstimatorPointConv2,
)
from robot.modules_reg.networks.pointconv_util import (
    index_points_gather as index_points,
    index_points_group,
    Conv1d,
    square_distance,
)
import time
from robot.utils.utils import shrink_by_factor
from robot.shape.point_interpolator import nadwat_kernel_interpolator

scale = 1.0


class PointConvSceneFlowPWC2_5(nn.Module):
    def __init__(
        self,
        input_channel=3,
        initial_input_radius=1.0,
        initial_input_npoints=40000,
        first_sampling_npoints=2048,
        predict_at_low_resl=False,
        param_shrink_factor=1,
    ):
        super(PointConvSceneFlowPWC2_5, self).__init__()

        flow_nei = 32
        feat_nei = 16
        sbf = partial(shrink_by_factor, factor=param_shrink_factor)
        self.predict_at_low_resl = predict_at_low_resl
        self.scale = 1
        # l0: 8192
        self.level0 = Conv1d(input_channel, sbf(32))
        self.level0_1 = Conv1d(sbf(32), sbf(32))
        self.cost0 = PointConvFlow(flow_nei, sbf(32 + 32 + 32 + 32) + 3, sbf([32, 32]))
        self.flow0 = SceneFlowEstimatorPointConv2(
            sbf(32 + 64), sbf(32), channels=sbf([128, 128]), mlp=sbf([128, 64])
        )
        self.level0_2 = Conv1d(sbf(32), sbf(64))

        # l1: 2048
        self.level1 = PointConvD(first_sampling_npoints, feat_nei, sbf(64) + 3, sbf(64))
        self.cost1 = PointConvFlow(flow_nei, sbf(64 + 32 + 64 + 32) + 3, sbf([64, 64]))
        self.flow1 = SceneFlowEstimatorPointConv2(
            sbf(64 + 64), sbf(64), channels=sbf([128, 128]), mlp=sbf([128, 64])
        )
        self.level1_0 = Conv1d(sbf(64), sbf(64))
        self.level1_1 = Conv1d(sbf(64), sbf(128))

        # l2: 512
        self.level2 = PointConvD(
            int(first_sampling_npoints / 4), feat_nei, sbf(128) + 3, sbf(128)
        )
        self.cost2 = PointConvFlow(
            flow_nei, sbf(128 + 64 + 128 + 64) + 3, sbf([128, 128])
        )
        self.flow2 = SceneFlowEstimatorPointConv2(
            sbf(128 + 64), sbf(128), channels=sbf([128, 128]), mlp=sbf([128, 64])
        )
        self.level2_0 = Conv1d(sbf(128), sbf(128))
        self.level2_1 = Conv1d(sbf(128), sbf(256))

        # l3: 256
        self.level3 = PointConvD(
            int(first_sampling_npoints / 8), feat_nei, sbf(256) + 3, sbf(256)
        )
        self.cost3 = PointConvFlow(
            flow_nei, sbf(256 + 64 + 256 + 64) + 3, sbf([256, 256])
        )
        self.flow3 = SceneFlowEstimatorPointConv2(
            sbf(256), sbf(256), flow_ch=0, channels=sbf([128, 128]), mlp=sbf([128, 64])
        )
        self.level3_0 = Conv1d(sbf(256), sbf(256))
        self.level3_1 = Conv1d(sbf(256), sbf(512))

        # l4: 64
        self.level4 = PointConvD(
            int(first_sampling_npoints / 32), feat_nei, sbf(512) + 3, sbf(256)
        )

        # deconv
        self.deconv4_3 = Conv1d(sbf(256), sbf(64))
        self.deconv3_2 = Conv1d(sbf(256), sbf(64))
        self.deconv2_1 = Conv1d(sbf(128), sbf(32))
        self.deconv1_0 = Conv1d(sbf(64), sbf(32))

        # warping
        self.warping = PointWarping3(initial_input_radius)

        # upsample
        self.upsample = UpsampleFlow3(initial_input_radius)

    def forward(self, xyz1, xyz2, color1, color2):
        # xyz1, xyz2: B, N, 3
        # color1, color2: B, N, 3

        # l0
        _scale = 2
        scaler = lambda x: self.scale * (_scale ** x)
        pc1_l0 = xyz1.permute(0, 2, 1)
        pc2_l0 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1)  # B 3 N
        color2 = color2.permute(0, 2, 1)  # B 3 N
        feat1_l0 = self.level0(color1)
        feat1_l0 = self.level0_1(feat1_l0)
        feat1_l0_1 = self.level0_2(feat1_l0)

        feat2_l0 = self.level0(color2)
        feat2_l0 = self.level0_1(feat2_l0)
        feat2_l0_1 = self.level0_2(feat2_l0)

        # l1
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)
        feat1_l1_2 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1_2)

        pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2_l0, feat2_l0_1)
        feat2_l1_2 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1_2)

        # l2
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
        feat1_l2_3 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2_3)

        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l1_2)
        feat2_l2_3 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2_3)

        # l3
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
        feat1_l3_4 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3_4)

        pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
        feat2_l3_4 = self.level3_0(feat2_l3)
        feat2_l3_4 = self.level3_1(feat2_l3_4)

        # l4
        pc1_l4, feat1_l4, _ = self.level4(pc1_l3, feat1_l3_4)
        feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4, resol_factor=scaler(7))
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)

        pc2_l4, feat2_l4, _ = self.level4(pc2_l3, feat2_l3_4)
        feat2_l4_3 = self.upsample(pc2_l3, pc2_l4, feat2_l4, resol_factor=scaler(7))
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)

        # l3
        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim=1)
        c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim=1)
        cost3 = self.cost3(pc1_l3, pc2_l3, c_feat1_l3, c_feat2_l3)
        feat3, flow3, fea_flow3 = self.flow3(pc1_l3, feat1_l3, cost3)

        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_l3, resol_factor=scaler(5))
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_l3, resol_factor=scaler(5))
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim=1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim=1)

        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_l2, resol_factor=scaler(4))
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_l2, resol_factor=scaler(4))
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim=1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim=1)

        feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_l1, resol_factor=scaler(3))
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)

        feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_l1, resol_factor=scaler(3))
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim=1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim=1)

        # l2
        up_flow2 = self.upsample(pc1_l2, pc1_l3, flow3, resol_factor=scaler(5))
        up_feaflow2 = self.upsample(pc1_l2, pc1_l3, fea_flow3, resol_factor=scaler(5))
        pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2, resol_factor=scaler(4))
        cost2 = self.cost2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2)

        feat3_up = self.upsample(pc1_l2, pc1_l3, feat3, resol_factor=scaler(5))
        new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim=1)
        feat2, resid_flow2, resid_fea_flow2 = self.flow2(
            pc1_l2, new_feat1_l2, cost2, up_flow2
        )
        flow2 = up_flow2 + resid_flow2
        fea_flow2 = up_feaflow2 + resid_fea_flow2

        # l1
        up_flow1 = self.upsample(pc1_l1, pc1_l2, flow2, resol_factor=scaler(4))
        up_feaflow1 = self.upsample(pc1_l1, pc1_l2, fea_flow2, resol_factor=scaler(4))
        pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1, resol_factor=scaler(3))
        cost1 = self.cost1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1)

        feat2_up = self.upsample(pc1_l1, pc1_l2, feat2, resol_factor=scaler(4))
        new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim=1)
        feat1, resid_flow1, resid_fea_flow1 = self.flow1(
            pc1_l1, new_feat1_l1, cost1, up_flow1
        )
        flow1 = up_flow1 + resid_flow1
        fea_flow1 = up_feaflow1 + resid_fea_flow1
        # l0
        up_flow0 = self.upsample(pc1_l0, pc1_l1, flow1, resol_factor=scaler(3))
        up_fea_flow0 = self.upsample(pc1_l0, pc1_l1, fea_flow1, resol_factor=scaler(3))
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0, resol_factor=scaler(1))
        cost0 = self.cost0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0)

        feat1_up = self.upsample(pc1_l0, pc1_l1, feat1, resol_factor=scaler(3))
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim=1)
        _, resid_flow0, resid_fea_flow0 = self.flow0(
            pc1_l0, new_feat1_l0, cost0, up_flow0
        )
        flow0 = up_flow0 + resid_flow0
        fea_flow0 = up_fea_flow0 + resid_fea_flow0
        floweds = [
            flow0 + pc1_l0.detach(),
            flow1 + pc1_l1.detach(),
            flow2 + pc1_l2.detach(),
            flow3 + pc1_l3.detach(),
        ]
        floweds = [flow.transpose(2, 1).contiguous() for flow in floweds]
        fps_pc1_idxs = [fps_pc1_l1, fps_pc1_l2, fps_pc1_l3]
        if not self.predict_at_low_resl:

            additional_param = {
                "control_points": pc1_l1.transpose(2, 1).contiguous(),
                "control_points_idx": fps_pc1_l1,
                "predict_at_low_resl": self.predict_at_low_resl,
            }
            additional_param.update({"floweds": floweds, "fps_pc1_idxs": fps_pc1_idxs})
            return fea_flow0.transpose(2, 1).contiguous(), additional_param
        else:
            additional_param = {
                "control_points": pc1_l1.transpose(2, 1).contiguous(),
                "control_points_idx": fps_pc1_l1,
                "predict_at_low_resl": self.predict_at_low_resl,
            }
            additional_param.update({"floweds": floweds, "fps_pc1_idxs": fps_pc1_idxs})
            return fea_flow1.transpose(2, 1).contiguous(), additional_param
