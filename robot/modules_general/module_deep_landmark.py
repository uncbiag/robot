import math
import torch
import torch.nn as nn
from robot.global_variable import Shape
from robot.utils.obj_factory import obj_factory

from robot.modules_general.networks.pointpwc_landmark import PointPWC_Landmark
from robot.modules_general.networks.pointpwc_landmark_multi import (
    PointPWC_Landmark_Multi,
)
from robot.modules_general.networks.pointpwc_landmark_multiflow import (
    PointPWC_Landmark_MultiFlow,
)
from robot.utils.utils import sigmoid_decay


class PointConvLandmarkPredictor(nn.Module):
    def __init__(self, opt):
        super(PointConvLandmarkPredictor, self).__init__()
        self.opt = opt
        local_feature_extractor_obj = self.opt[
            (
                "local_feature_extractor_obj",
                "",
                "function object for local_feature_extractor",
            )
        ]
        self.local_feature_extractor = (
            obj_factory(local_feature_extractor_obj)
            if len(local_feature_extractor_obj)
            else self.default_local_feature_extractor
        )
        self.input_channel = self.opt[("input_channel", 1, "input channel")]
        self.num_landmarks = self.opt[("num_landmarks", 6, "num of landmarks")]
        self.predict_at_low_resl = self.opt[
            ("predict_at_low_resl", False, "predict at low resoltuion")
        ]
        self.initial_npoints = self.opt[
            ("initial_npoints", 8192, "initial sampled points")
        ]
        self.initial_radius = self.opt[
            ("initial_radius", 0.001, "initial radius, only set when use aniso kernel")
        ]
        self.param_shrink_factor = self.opt[
            ("param_shrink_factor", 2, "network parameter shrink factor")
        ]
        self.use_aniso_kernel = self.opt[
            ("use_aniso_kernel", False, "use the aniso kernel in first sampling layer")
        ]
        self.network_name = self.opt[
            (
                "network_name",
                "pointconv",
                "pointconv, pointconv_multi,pointconv_multiflow",
            )
        ]
        self.prediction_strategy = (
            "heatmap" if self.network_name not in ["pointconv_multiflow"] else "points"
        )
        self.normalized_strategy = self.opt[
            "normalized_strategy", "sigmoid", "sigmoid or lineaer"
        ]
        self.heatmap_threshold = self.opt[
            "heatmap_threshold", 0.9, "set heat map value to 0 if < threshold"
        ]
        self.init_deep_landmark_predictor()
        self.buffer = {}
        self.iter = 0

    def default_local_feature_extractor(self, input_shape):
        input_shape.pointfea = (
            input_shape.points.clone()
        )  # torch.cat([cur_source.points, cur_source.weights], 2)
        return input_shape

    def init_deep_landmark_predictor(self):
        network = {
            "pointconv": PointPWC_Landmark,
            "pointconv_multi": PointPWC_Landmark_Multi,
            "pointconv_multiflow": PointPWC_Landmark_MultiFlow,
        }
        self.predictor = network[self.network_name](
            input_channel=self.input_channel,
            first_sampling_npoints=self.initial_npoints,
            output_channels=self.num_landmarks,
            predict_at_low_resl=self.predict_at_low_resl,
            param_shrink_factor=self.param_shrink_factor,
            use_aniso_kernel=self.use_aniso_kernel,
        )

    def compute_landmark(self, heatmap, points):
        """
        normalized the heat map  BxLxN  and compute landdmark
        :param heatmap: BxLxN
        :param points: BxNxD
        :return:  BxLxD
        """
        landmarks = None

        if self.normalized_strategy == "sigmoid":
            heatmap = torch.sigmoid(heatmap)
            heatmap = heatmap
            heatmap_cp = heatmap.clone()
            heatmap_cp[heatmap < self.heatmap_threshold] = 0
            heatmap = heatmap_cp / (heatmap_cp.sum(-1, keepdim=True) + 1e-9)
            landmarks = (heatmap[..., None] * points[:, None]).sum(2)
        elif self.normalized_strategy == "linear":
            B, L = heatmap.shape[:2]
            hm_min = heatmap.min(2)[0].view(B, L, 1)
            hm_max = heatmap.max(2)[0].view(B, L, 1)
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)
            heatmap_cp = heatmap.clone()
            heatmap_cp[heatmap < self.heatmap_threshold] = 0
            heatmap = heatmap_cp / (heatmap_cp.sum(-1, keepdim=True) + 1e-9)
            landmarks = (heatmap[..., None] * points[:, None]).sum(2)

        elif self.normalized_strategy == "topK":
            B, L = heatmap.shape[:2]
            D = points.shape[2]
            hm_min = heatmap.min(2)[0].view(B, L, 1)
            hm_max = heatmap.max(2)[0].view(B, L, 1)
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)
            topK_val, topk_ind = torch.topk(
                heatmap, k=3, dim=2, largest=True
            )  # BxLxK, BxLxK
            # topK_heat = torch.gather(heatmap,dim=2, index=topk_ind)
            # topK_heat = topK_heat / (topK_heat.sum(-1, keepdim=True) + 1e-9)  # BxLxK
            # points_t = points.transpose(2,1) # BxDxN
            # points_t_rep = points_t[:None].repeat(1,L,1,1) # BxLxDxN
            # index_rep = topk_ind[:,:,None].repeat(1,1,D,1)  # BxLxDxK
            # topK_pos_t = torch.gather(points_t_rep,dim=3, index=index_rep) # BxLxDxK
            # topK_pos = topK_pos_t.transpose(3,2).contiguous() # BxLxKxD
            # landmarks = (topK_heat[..., None] * topK_pos).sum(2) # BxLxKx1 * BxLxKxD
            heatmap_zero = torch.zeros_like(heatmap)
            heatmap_zero.scatter_(dim=2, index=topk_ind, src=topK_val)
            heatmap_zero = heatmap_zero / (heatmap_zero.sum(-1, keepdim=True) + 1e-9)
            landmarks = (heatmap_zero[..., None] * points[:, None]).sum(2)

        elif self.normalized_strategy == "combined":
            B, L = heatmap.shape[:2]
            hm_min = heatmap.min(2)[0].view(B, L, 1)
            hm_max = heatmap.max(2)[0].view(B, L, 1)
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)
            heatmap_cp = heatmap.clone()
            heatmap_cp[heatmap < self.heatmap_threshold] = 0
            heatmap = heatmap_cp / (heatmap_cp.sum(-1, keepdim=True) + 1e-9)
            landmarks_global = (heatmap[..., None] * points[:, None]).sum(2)
            topK_val, topk_ind = torch.topk(
                heatmap, k=5, dim=2, largest=True
            )  # BxLxK, BxLxK
            heatmap_zero = torch.zeros_like(heatmap)
            heatmap_zero.scatter_(dim=2, index=topk_ind, src=topK_val)
            heatmap_zero = heatmap_zero / (heatmap_zero.sum(-1, keepdim=True) + 1e-9)
            landmarks_local = (heatmap_zero[..., None] * points[:, None]).sum(2)
            landmarks = self.w * landmarks_global + (1 - self.w) * landmarks_local
        else:
            raise NotImplemented

        return landmarks

    def heatmap_landmark_predictor(self, input_shape):
        heatmap, sampled_points = self.predictor(
            input_shape.points, input_shape.pointfea
        )  # BxLxN,  BxNxD
        landmarks = self.compute_landmark(heatmap, sampled_points)  # BxCxN
        additional_param = {"heatmaps": heatmap, "control_points": sampled_points}
        return landmarks, additional_param

    def points_landmark_predictor(self, input_shape):
        landmarks, additional_param = self.predictor(
            input_shape.points, input_shape.pointfea
        )  # BxLxN,  BxNxD
        return landmarks, additional_param

    def deep_landmark_predictor(self, input_shape):
        if self.prediction_strategy == "heatmap":
            return self.heatmap_landmark_predictor(input_shape)
        elif self.prediction_strategy == "points":
            return self.points_landmark_predictor(input_shape)
        else:
            raise NotImplemented

    def setup(self, batch_info):
        epoch = batch_info["epoch"]
        factor = float(max(sigmoid_decay(epoch, static=40, k=6) * 1, 0.01))
        self.w = factor

    def __call__(self, input_shape, batch_info=None):
        self.setup((batch_info))
        input_shape = self.local_feature_extractor(input_shape)
        pred_landmarks, additional_param = self.deep_landmark_predictor(input_shape)
        output_shape = Shape().set_data_with_refer_to(input_shape.points, input_shape)
        output_shape.landmarks = pred_landmarks
        return output_shape, additional_param


class DeepLandmarkPredictorLoss(nn.Module):
    def __init__(self, opt):
        super(DeepLandmarkPredictorLoss, self).__init__()
        self.opt = opt
        self.loss_type = self.opt[("loss_type", "mse")]

    def mse(self, pred, gt, mask):
        """

        :param pred: BxLxD
        :param gt: BxLxD
        :return:
        """
        return (((gt - pred) ** 2) * mask).sum(2).mean(1)

    def ce(self, pred_heatmap, gt, points, mask, sigma=0.01):
        sigma = 1.41421356237 * sigma
        pred_heatmap = (pred_heatmap).log_softmax(dim=2)  # BxLxN
        gt_points_dist = ((gt[:, :, None] / sigma - points[:, None] / sigma) ** 2).sum(
            -1
        )  # BxLx1xD- Bx1xNxD,  BxLxN
        soften_gtmap = (-gt_points_dist).softmax(dim=2)
        sim_loss = (-pred_heatmap * soften_gtmap).sum(2, keepdim=True) * mask
        return sim_loss.sum(2).mean(1)

    def multi_mse(self, pred_list, gt, weight_list, mask):
        sim_loss = [
            w * self.mse(pred, gt, mask) for w, pred in zip(weight_list, pred_list)
        ]
        return sum(sim_loss)

    def multi_ce(self, pred_heatmap_list, gt, points_list, weight_list, mask):
        sim_loss = [
            w * self.ce(pred_heatmap, gt, points, mask)
            for w, pred_heatmap, points in zip(
                weight_list, pred_heatmap_list, points_list
            )
        ]
        return sum(sim_loss)

    def __call__(self, refer_shape, output_shape, additional_param, batch_info):
        pred = output_shape.landmarks
        gt = refer_shape.landmarks
        mask = output_shape.extra_info["landmark_masks"]
        if additional_param.get("multi_scale", False):
            pred = additional_param["landmarks"]
            weights = additional_param["weights"]
            self.loss_type = "multi_mse" if self.loss_type == "mse" else self.loss_type
            self.loss_type = "multi_ce" if self.loss_type == "ce" else self.loss_type
        if self.loss_type == "mse":
            sim_loss = self.mse(pred, gt, mask)
            return sim_loss, sim_loss * 0
        elif self.loss_type == "multi_mse":
            sim_loss = self.multi_mse(pred, gt, weights, mask)
            return sim_loss, sim_loss * 0
        elif self.loss_type == "ce":
            heatmap = additional_param["heatmaps"]
            points = additional_param["control_points"]
            sim_loss = self.ce(heatmap, gt, points, mask)
            return sim_loss, sim_loss * 0
        elif self.loss_type == "multi_ce":
            heatmap = additional_param["heatmaps"]
            points = additional_param["control_points"]
            sim_loss = self.multi_ce(heatmap, gt, points, weights, mask)
            return sim_loss, sim_loss * 0

        else:
            raise NotImplemented


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def grad_hook(grad):
    # import pydevd
    # pydevd.settrace(suspend=False, trace_only_current_thread=True)
    print("debugging info, the grad_norm is {} ".format(grad.norm()))
    return grad
