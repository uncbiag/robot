from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from pykeops.torch import LazyTensor
import torch.nn.functional as F
from robot.global_variable import Shape
from robot.utils.obj_factory import obj_factory
from pointnet2.util import (
    PointNetSetAbstraction,
    PointNetSetUpConv,
    PointNetFeaturePropogation,
)
from robot.metrics.reg_losses import GeomDistance
from robot.modules_reg.module_gradient_flow import wasserstein_barycenter_mapping
from robot.modules_reg.networks.dgcnn import DGCNN
from robot.modules_reg.networks.pointpwc_feature_net import PointConvFeature
from robot.utils.utils import shrink_by_factor


class PointNet2FeaExtractor(nn.Module):
    def __init__(self, opt):
        super(PointNet2FeaExtractor, self).__init__()
        self.opt = opt
        self.local_pair_feature_extractor = obj_factory(
            self.opt[
                (
                    "local_pair_feature_extractor_obj",
                    "local_feature_extractor.default_local_pair_feature_extractor()",
                    "function object for local_feature_extractor",
                )
            ]
        )
        self.input_channel = self.opt[("input_channel", 10, "input channel")]
        self.output_channel = self.opt[("output_channel", 12, "output channel")]
        self.initial_radius = self.opt[("initial_radius", 0.001, "initial radius")]
        self.initial_npoints = self.opt[("initial_npoints",4096, "initial_npoints")]
        self.group_via_knn = self.opt[("group_via_knn",False, "group_via_knn")]
        self.include_pos_in_final_feature = self.opt[
            (
                "include_pos_in_final_feature",
                True,
                "include_pos, here the pos refers to the relative postion not xyz",
            )
        ]
        self.include_pos_info_network = self.opt[
            ("include_pos_info_network", True, "include_pos, xyz")
        ]
        self.use_complex_network = self.opt[
            ("use_complex_network", False, "use a complex pointnet to extract feature ")
        ]
        self.param_shrink_factor = self.opt[
            (
                "param_shrink_factor",
                2,
                "shrink factor the model parameter, #params/param_shrink_factor",
            )
        ]
        self.pretrained_model_path = self.opt[
            ("pretrained_model_path", "", "the path of the pretrained model")
        ]
        if not self.use_complex_network:
            self.init_simple_deep_feature_extractor()
            self.deep_pair_feature_extractor = self.simple_deep_pair_feature_extractor
        else:
            self.init_complex_deep_feature_extractor()
            self.deep_pair_feature_extractor = self.complex_deep_pair_feature_extractor
        self.buffer = {}
        self.iter = 0
        self.load_from_pretrained_model()

    def load_from_pretrained_model(self):
        if self.pretrained_model_path:
            checkpoint = torch.load(self.pretrained_model_path, map_location="cpu")
            cur_state = self.state_dict()
            for key in list(checkpoint["state_dict"].keys()):
                if "module.pair_feature_extractor." in key:
                    replaced_key = key.replace("module.pair_feature_extractor.", "")
                    if replaced_key in cur_state:
                        cur_state[replaced_key] = checkpoint["state_dict"].pop(key)
                    else:
                        print("")
            self.load_state_dict(cur_state)
            print("load pretrained model from {}".format(self.pretrained_model_path))

    def init_simple_deep_feature_extractor(self):
        self.sa1 = PointNetSetAbstraction(
            npoint=int(self.initial_npoints),
            radius=self.initial_radius * 20,
            nsample=32,
            in_channel=self.input_channel,
            mlp=[16, 32],
            group_all=False,
            use_knn = self.group_via_knn,
            use_aniso_kernel = False,
            include_xyz=self.include_pos_info_network,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=int(self.initial_npoints/4),
            radius=self.initial_radius * 40,
            nsample=32,
            in_channel=32,
            mlp=[32, 64],
            group_all=False,
            use_knn=self.group_via_knn,
            use_aniso_kernel=False,
            include_xyz=self.include_pos_info_network,
        )
        self.su1 = PointNetSetUpConv(
            nsample=8,
            radius=self.initial_radius * 120,
            f1_channel=32,
            f2_channel=64,
            mlp=[32],
            mlp2=[32],
        )
        self.fp = PointNetFeaturePropogation(
            in_channel=32 + self.input_channel, mlp=[32]
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, self.output_channel, kernel_size=1, bias=True)

    def simple_deep_pair_feature_extractor(self, cur_source, target):
        pc1, pc2, feature1, feature2 = (
            cur_source.points,
            target.points,
            cur_source.pointfea,
            target.pointfea,
        )
        pc1, pc2, feature1, feature2 = (
            pc1.transpose(2, 1).contiguous(),
            pc2.transpose(2, 1).contiguous(),
            feature1.transpose(2, 1).contiguous(),
            feature2.transpose(2, 1).contiguous(),
        )

        l1_pc1, l1_feature1, _ = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1, _ = self.sa2(l1_pc1, l1_feature1)
        l1_fnew1 = self.su1(l1_pc1, l2_pc1, l1_feature1, l2_feature1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        x = F.relu(self.bn1(l0_fnew1))
        sf = self.conv2(x)

        l1_pc2, l1_feature2, _ = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2, _ = self.sa2(l1_pc2, l1_feature2)
        l1_fnew2 = self.su1(l1_pc2, l2_pc2, l1_feature2, l2_feature2)
        l0_fnew2 = self.fp(pc2, l1_pc2, feature2, l1_fnew2)
        x = F.relu(self.bn1(l0_fnew2))
        tf = self.conv2(x)

        # sf = (sf ) / (sf.std(2, keepdim=True) + 1e-7)
        # tf = (tf) / (tf.std(2, keepdim=True) + 1e-7)
        # sf = (sf-sf.mean(2,keepdim=True))/(sf.std(2,keepdim=True)+1e-7)
        # tf = (tf-tf.mean(2,keepdim=True))/(tf.std(2,keepdim=True)+1e-7)

        sf, tf = sf.transpose(2, 1).contiguous(), tf.transpose(2, 1).contiguous()

        if self.include_pos_in_final_feature:
            sf = torch.cat([cur_source.points, sf], 2)
            tf = torch.cat([target.points, tf], 2)

        cur_source.pointfea = sf
        target.pointfea = tf
        return cur_source, target


    def init_complex_deep_feature_extractor(self):
        sbf = partial(shrink_by_factor, factor=self.param_shrink_factor)

        self.sa0 = PointNetSetAbstraction(
            npoint=int(self.initial_npoints),
            radius=20 * self.initial_radius,
            nsample=16,
            in_channel=self.input_channel,
            mlp=(sbf([16, 16, 16])),
            group_all=True,
            use_aniso_kernel=True,
            cov_sigma_scale=self.initial_radius * 20,
            aniso_kernel_scale=self.initial_radius * 80,
        )
        self.sa1 = PointNetSetAbstraction(
            npoint=int(self.initial_npoints),
            radius=20 * self.initial_radius,
            nsample=16,
            in_channel=sbf(16),
            mlp=sbf([16, 16, 32]),
            group_all=False,
            include_xyz=self.include_pos_info_network,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=int(self.initial_npoints/4),
            radius=40 * self.initial_radius,
            nsample=16,
            in_channel=sbf(32),
            mlp=sbf([32, 32, 64]),
            group_all=False,
            include_xyz=self.include_pos_info_network,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=int(self.initial_npoints/16),
            radius=80 * self.initial_radius,
            nsample=8,
            in_channel=sbf(64),
            mlp=sbf([64, 64, 128]),
            group_all=False,
            include_xyz=self.include_pos_info_network,
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=int(self.initial_npoints/64),
            radius=160 * self.initial_radius,
            nsample=8,
            in_channel=sbf(128),
            mlp=sbf([128, 128, 256]),
            group_all=False,
            include_xyz=self.include_pos_info_network,
        )

        self.su1 = PointNetSetUpConv(
            nsample=8,
            radius=96 * self.initial_radius,
            f1_channel=sbf(128),
            f2_channel=sbf(256),
            mlp=[],
            mlp2=sbf([128, 128]),
        )
        self.su2 = PointNetSetUpConv(
            nsample=8,
            radius=48 * self.initial_radius,
            f1_channel=sbf(64),
            f2_channel=sbf(128),
            mlp=sbf([64, 64, 128]),
            mlp2=sbf([128]),
        )
        self.su3 = PointNetSetUpConv(
            nsample=8,
            radius=24 * self.initial_radius,
            f1_channel=sbf(32),
            f2_channel=sbf(128),
            mlp=sbf([64, 64, 128]),
            mlp2=sbf([128]),
        )
        self.fp = PointNetFeaturePropogation(
            in_channel=sbf(128) + self.input_channel, mlp=sbf([128, 128])
        )

        self.conv1 = nn.Conv1d(sbf(128), sbf(64), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(sbf(64))
        self.conv2 = nn.Conv1d(sbf(64), self.output_channel, kernel_size=1, bias=True)

    def complex_deep_pair_feature_extractor(self, cur_source, target):
        pc1, pc2, feature1, feature2 = (
            cur_source.points,
            target.points,
            cur_source.pointfea,
            target.pointfea,
        )
        pc1, pc2, feature1, feature2 = (
            pc1.transpose(2, 1).contiguous(),
            pc2.transpose(2, 1).contiguous(),
            feature1.transpose(2, 1).contiguous(),
            feature2.transpose(2, 1).contiguous(),
        )

        l0_pc1, l0_feature1, _ = self.sa0(pc1, feature1)
        l1_pc1, l1_feature1, _ = self.sa1(l0_pc1, l0_feature1)
        l2_pc1, l2_feature1, _ = self.sa2(l1_pc1, l1_feature1)
        l3_pc1, l3_feature1, _ = self.sa3(l2_pc1, l2_feature1)
        l4_pc1, l4_feature1, _ = self.sa4(l3_pc1, l3_feature1)
        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, l2_feature1, l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)
        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)

        l0_pc2, l0_feature2, _ = self.sa0(pc2, feature2)
        l1_pc2, l1_feature2, _ = self.sa1(l0_pc2, l0_feature2)
        l2_pc2, l2_feature2, _ = self.sa2(l1_pc2, l1_feature2)
        l3_pc2, l3_feature2, _ = self.sa3(l2_pc2, l2_feature2)
        l4_pc2, l4_feature2, _ = self.sa4(l3_pc2, l3_feature2)
        l3_fnew2 = self.su1(l3_pc2, l4_pc2, l3_feature2, l4_feature2)
        l2_fnew2 = self.su2(l2_pc2, l3_pc2, l2_feature2, l3_fnew2)
        l1_fnew2 = self.su3(l1_pc2, l2_pc2, l1_feature2, l2_fnew2)
        l0_fnew2 = self.fp(pc2, l1_pc2, feature2, l1_fnew2)
        x = F.relu(self.bn1(self.conv1(l0_fnew2)))
        tf = self.conv2(x)
        sf = (sf) / (sf.std(2, keepdim=True) + 1e-7)
        tf = (tf) / (tf.std(2, keepdim=True) + 1e-7)
        # sf = (sf-sf.mean(2,keepdim=True))/(sf.std(2,keepdim=True)+1e-7)
        # tf = (tf-tf.mean(2,keepdim=True))/(tf.std(2,keepdim=True)+1e-7)
        sf = sf.transpose(2, 1).contiguous()
        tf = tf.transpose(2, 1).contiguous()

        if self.include_pos_in_final_feature:
            sf = torch.cat([cur_source.points, sf], 2)
            tf = torch.cat([target.points, tf], 2)

        cur_source.pointfea = sf
        target.pointfea = tf
        return cur_source, target

    def normalize_fea(self):
        pass

    def __call__(self, cur_source, target, iter=-1):
        cur_source, target = self.local_pair_feature_extractor(cur_source, target)
        cur_source, target = self.deep_pair_feature_extractor(cur_source, target)

        return cur_source, target


class DGCNNFeaExtractor(nn.Module):
    def __init__(self, opt):
        super(DGCNNFeaExtractor, self).__init__()
        self.opt = opt
        self.local_pair_feature_extractor = obj_factory(
            self.opt[
                (
                    "local_pair_feature_extractor_obj",
                    "local_feature_extractor.default_local_pair_feature_extractor()",
                    "function object for local_feature_extractor",
                )
            ]
        )
        self.input_channel = self.opt[("input_channel", 4, "input channel")]
        self.output_channel = self.opt[("output_channel", 16, "output channel")]
        self.initial_radius = self.opt[("initial_radius", 0.001, "initial radius")]
        self.K_neigh = self.opt[("K_neigh", 20, "K_neigh")]
        self.param_shrink_factor = self.opt[
            ("param_shrink_factor", 2, "network parameter shrink factor")
        ]
        self.include_pos_in_final_feature = self.opt[
            ("include_pos_in_final_feature", True, "include_pos")
        ]
        self.init_deep_feature_extractor()
        self.buffer = {}
        self.iter = 0

    def init_deep_feature_extractor(self):
        self.extractor = DGCNN(
            input_channel=self.input_channel,
            output_channels=self.output_channel,
            k=self.K_neigh,
            param_shrink_factor=self.param_shrink_factor,
        )

    def deep_pair_feature_extractor(self, cur_source, target):
        sf = self.extractor(cur_source.points, cur_source.pointfea)
        tf = self.extractor(target.points, target.pointfea)
        sf = (sf - sf.mean(2, keepdim=True)) / (sf.std(2, keepdim=True) + 1e-7)
        tf = (tf - tf.mean(2, keepdim=True)) / (tf.std(2, keepdim=True) + 1e-7)

        if self.include_pos_in_final_feature:
            sf = torch.cat([cur_source.points, sf], 2)
            tf = torch.cat([target.points, tf], 2)
        new_cur_source = Shape().set_data(
            points=cur_source.points, weights=cur_source.weights, pointfea=sf
        )
        new_target = Shape().set_data(
            points=target.points, weights=target.weights, pointfea=tf
        )
        return new_cur_source, new_target

    def __call__(self, cur_source, target, iter=-1):
        cur_source, target = self.local_pair_feature_extractor(cur_source, target)
        cur_source, target = self.deep_pair_feature_extractor(cur_source, target)
        return cur_source, target


class PointConvFeaExtractor(nn.Module):
    def __init__(self, opt):
        super(PointConvFeaExtractor, self).__init__()
        self.opt = opt
        self.local_pair_feature_extractor = obj_factory(
            self.opt[
                (
                    "local_pair_feature_extractor_obj",
                    "local_feature_extractor.default_local_pair_feature_extractor()",
                    "function object for local_feature_extractor"
                )
            ]
        )
        self.input_channel = self.opt[("input_channel", 1, "input channel")]
        self.output_channel = self.opt[("output_channel", 6, "output channel")]
        self.initial_radius = self.opt[("initial_radius", 0.001, "initial radius")]
        self.initial_npoints = self.opt[("initial_npoints",2048, "inital_n")]
        self.param_shrink_factor = self.opt[
            ("param_shrink_factor", 2, "network parameter shrink factor")
        ]
        self.include_pos_in_final_feature = self.opt[
            ("include_pos_in_final_feature", True, "include_pos")
        ]
        self.pretrained_model_path = self.opt[
            ("pretrained_model_path", "", "the path of the pretrained model")
        ]
        self.use_aniso_kernel = self.opt[
            ("use_aniso_kernel", False, "use the aniso kernel in first sampling layer")
        ]
        self.init_deep_feature_extractor()
        self.buffer = {}
        self.iter = 0

    def init_deep_feature_extractor(self):
        self.extractor = PointConvFeature(
            input_channel=self.input_channel,
            output_channels=self.output_channel,
            param_shrink_factor=self.param_shrink_factor,
            use_aniso_kernel=self.use_aniso_kernel,
            first_sampling_npoints = self.initial_npoints
        )
        if self.pretrained_model_path:
            checkpoint = torch.load(self.pretrained_model_path, map_location="cpu")
            cur_state = self.state_dict()
            for key in list(checkpoint["state_dict"].keys()):
                if "module.pair_feature_extractor." in key:
                    replaced_key = key.replace("module.pair_feature_extractor.", "")
                    if replaced_key in cur_state:
                        cur_state[replaced_key] = checkpoint["state_dict"].pop(key)
                    else:
                        print("")
            self.load_state_dict(cur_state)
            print("load pretrained model from {}".format(self.pretrained_model_path))

    def deep_pair_feature_extractor(self, cur_source, target):
        sf = self.extractor(cur_source.points, cur_source.pointfea)
        tf = self.extractor(target.points, target.pointfea)

        sf = (sf - sf.mean(2, keepdim=True)) / (sf.std(2, keepdim=True) + 1e-7)
        tf = (tf - tf.mean(2, keepdim=True)) / (tf.std(2, keepdim=True) + 1e-7)

        # sf = (sf ) / (sf.norm(2,2, keepdim=True) + 1e-7)
        # tf = (tf ) / (tf.norm(2,2, keepdim=True) + 1e-7)
        if self.include_pos_in_final_feature:
            sf = torch.cat([cur_source.points, sf], 2)
            tf = torch.cat([target.points, tf], 2)

        # new_cur_source = Shape().set_data(
        #     points=cur_source.points, weights=cur_source.weights, pointfea=sf
        # )
        # new_target = Shape().set_data(
        #     points=target.points, weights=target.weights, pointfea=tf
        # )
        # return new_cur_source, new_target
        cur_source.pointfea= sf
        target.pointfea = tf
        return cur_source, target

    def __call__(self, cur_source, target, iter=-1):
        cur_source, target = self.local_pair_feature_extractor(cur_source, target)
        cur_source, target = self.deep_pair_feature_extractor(cur_source, target)
        return cur_source, target


class DeepFeatureLoss(nn.Module):
    def __init__(self, opt):
        super(DeepFeatureLoss, self).__init__()
        self.opt = opt
        self.loss_type = self.opt[
            ("loss_type", "ot_map", "naive_corres/ot_ce/ot_soften_ce/ot_map")
        ]
        self.geomloss_setting = deepcopy(self.opt["geomloss"])
        self.geomloss_setting.print_settings_off()
        self.pos_is_in_fea = self.opt[
            "pos_is_in_fea",
            False,
            "if the position info is in the feature this should be consistent with 'include_pos_in_final_feature' in the feature extractor model",
        ]
        self.soften_gt_sigma = self.opt[
            ("soften_gt_sigma", 0.005, "sigma to soften the one-hot gt")
        ]
        self.buffer = {"gt_one_hot": None, "gt_plan": None, "soften_gt_one_hot": None}
        kernel_obj = self.opt[("kernel_obj", {}, "settings for interpolation mode")]
        self.kernel = obj_factory(kernel_obj)

    def naive_corres(self, cur_source, target):
        """
        cur_source and target should have one-to-one ordered correspondence
        :param cur_source:
        :param target:
        :return:
        """
        points = cur_source.points
        pointfea1 = cur_source.pointfea
        pointfea2 = target.pointfea
        weights = cur_source.weights
        sigma = self.soften_gt_sigma
        x_i = LazyTensor(points[:, :, None] / sigma)  # BxNx1xD
        x_j = LazyTensor(points[:, None] / sigma)  # Bx1xNxD
        point_dist2 = -x_i.sqdist(x_j)  # BxNxNx1
        point_loggauss = point_dist2 - LazyTensor(
            point_dist2.logsumexp(dim=2)[..., None]
        )  # BxNxN

        fea_i = LazyTensor(pointfea1[:, :, None])  # BxNx1xd
        fea_j = LazyTensor(pointfea2[:, None])  # Bx1xNxd
        fea_dist2 = -fea_i.sqdist(fea_j)  # BxNxNx1
        fea_logsoftmax = fea_dist2 - LazyTensor(
            fea_dist2.logsumexp(dim=2)[..., None]
        )  # BxNxNx1
        ce_loss = -(point_loggauss.exp() * fea_logsoftmax).sum(2) * weights
        return ce_loss.sum(1)[..., 0]  # B

    def compute_gt_one_hot(self, refer_points):
        if (
            self.buffer["gt_one_hot"] is not None
            and self.buffer["gt_one_hot"].shape[0] == refer_points.shape[0]
        ):
            return self.buffer["gt_one_hot"]
        else:
            self.buffer["gt_one_hot"] = None
        if self.buffer["gt_one_hot"] is None:
            B, N = refer_points.shape[0], refer_points.shape[1]
            device = refer_points.device
            i = torch.arange(N, dtype=torch.float32, device=device)
            i = i.repeat(B, 1).contiguous()
            j = i
            i = LazyTensor(i[:, :, None, None])
            j = LazyTensor(j[:, None, :, None])
            id_matrix = (0.5 - (i - j) ** 2).step()  # BxNxN
            self.buffer["gt_one_hot"] = id_matrix
        return self.buffer["gt_one_hot"]

    def compute_soften_gt_one_hot(self, refer_points):
        sigma = 0.005 * 1.4142135623730951
        x_i = LazyTensor(refer_points[:, :, None] / sigma)  # BxNx1xD
        x_j = LazyTensor(refer_points[:, None] / sigma)  # Bx1xNxD
        point_dist2 = -x_i.sqdist(x_j)  # BxNxNx1
        point_loggauss = point_dist2 - LazyTensor(
            point_dist2.logsumexp(dim=2)[..., None]
        )  # BxNxN
        return point_loggauss.exp()

    def compute_gt_plan(self, refer_points, refer_weights):
        if (
            self.buffer["gt_plan"] is not None
            and self.buffer["gt_plan"].shape[0] == refer_points.shape[0]
        ):
            return self.buffer["gt_plan"]
        else:
            self.buffer["gt_plan"] = None

        if self.buffer["gt_plan"] is None:
            gt_one_hot = self.compute_gt_one_hot(refer_points)  # BxNxN
            refer_weights = LazyTensor(refer_weights[:, None])  # Bx1xN
            gt_plan = gt_one_hot * refer_weights
            self.buffer["gt_plan"] = gt_plan
        return self.buffer["gt_plan"]

    def ot_ce(self, cur_source, target):
        """
                cur_source and target should have one-to-one ordered correspondence

        :param cur_source:
        :param target:
        :return:
        """
        points = cur_source.points
        weights = cur_source.weights  # BxNx1
        lazy_gt_one_hot = (
            self.compute_gt_one_hot(points)
            if self.loss_type == "ot_ce"
            else self.compute_soften_gt_one_hot(points)
        )
        self.geomloss_setting["mode"] = "prob"
        _, lazy_log_prob = wasserstein_barycenter_mapping(
            cur_source, target, self.geomloss_setting
        )  # BxNxM
        ce_loss = -(lazy_gt_one_hot * lazy_log_prob).sum(2) * weights  # BxN
        ce_loss = torch.dot(ce_loss.view(-1), torch.ones_like(ce_loss).view(-1))
        B = points.shape[0]
        factor = 0.001
        return torch.stack([ce_loss * factor / B] * B, 0)

    def mapped_bary_center(self, cur_source, target, gt_flowed):

        self.geomloss_setting["mode"] = "soft"
        # if torch.is_grad_enabled():
        #     cur_source.pointfea.register_hook(grad_hook)
        #     #target.pointfea.register_hook(grad_hook)
        flowed, _ = wasserstein_barycenter_mapping(
            cur_source, target, self.geomloss_setting
        )  # BxNxM
        return (
            ((gt_flowed.points - flowed.points) ** 2).sum(2) * flowed.weights[..., 0]
        ).sum(1)

    def unsupervised_mapped_bary_center(self, cur_source, target, gt_flowed):
        self.geomloss_setting["mode"] = "soft"
        flowed, _ = wasserstein_barycenter_mapping(
            cur_source, target, self.geomloss_setting
        )  # BxNxM
        disp = flowed.points - cur_source.points
        smoothed_disp = self.kernel(
            cur_source.points, cur_source.points, disp, cur_source.weights
        )
        flowed_points = flowed.points + smoothed_disp
        flowed = Shape().set_data_with_refer_to(flowed_points, flowed)
        geomloss_setting = deepcopy(self.opt["geomloss"])
        geomloss_setting.print_settings_off()
        geomloss_setting["attr"] = "points"
        geom_loss = GeomDistance(geomloss_setting)
        wasserstein_dist = geom_loss(flowed, gt_flowed)
        return wasserstein_dist

    def reg_loss(self, cur_source, target):
        if self.pos_is_in_fea:
            reg_s = (cur_source.pointfea[:, :, 3:]) ** 2
            reg_t = (target.pointfea[:, :, 3:]) ** 2
        else:
            reg_s = (cur_source.pointfea) ** 2
            reg_t = (target.pointfea) ** 2
        return (reg_s + reg_t).mean(2).mean(1)

    def __call__(self, cur_source, target, gt_flowed, has_gt=True):

        reg_loss = self.reg_loss(cur_source, target)
        if has_gt:
            if self.loss_type == "ot_map":
                return self.mapped_bary_center(cur_source, target, gt_flowed), reg_loss
            elif self.loss_type == "ot_ce" or self.loss_type == "ot_soften_ce":
                return self.ot_ce(cur_source, target), reg_loss
            elif self.loss_type == "naive_corres":
                return self.naive_corres(cur_source, target), reg_loss
            else:
                raise NotImplemented
        else:
            return (
                self.unsupervised_mapped_bary_center(cur_source, target, gt_flowed),
                reg_loss,
            )


def grad_hook(grad):
    # import pydevd
    # pydevd.settrace(suspend=False, trace_only_current_thread=True)
    print("debugging info, the grad_norm is {} ".format(grad.norm()))
    return grad
