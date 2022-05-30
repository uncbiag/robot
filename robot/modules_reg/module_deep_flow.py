import torch
from copy import deepcopy
import torch.nn as nn
from pykeops.torch import LazyTensor
import torch.nn.functional as F
from robot.global_variable import Shape
from robot.utils.obj_factory import obj_factory
from pointnet2.util import (
    PointNetSetAbstraction,
    PointNetSetUpConv,
    PointNetFeaturePropogation,
    FlowEmbedding,
    index_points,
)
from robot.modules_reg.networks.pointpwcnet_original import (
    multiScaleChamferSmoothCurvature,
    PointConvSceneFlowPWC8192selfglobalPointConv,
)
from robot.modules_reg.networks.scene_flow import FLOT
from robot.metrics.reg_losses import GeomDistance, GMMLoss
from robot.modules_reg.module_gradient_flow import (
    wasserstein_barycenter_mapping,
    point_based_gradient_flow_guide,
)
from robot.modules_reg.networks.flownet3d import FlowNet3D, FlowNet3DIMP
from robot.modules_reg.networks.pointpwcnet2_2 import PointConvSceneFlowPWC2_2
from robot.modules_reg.networks.pointpwcnet2_4 import PointConvSceneFlowPWC2_4
from robot.modules_reg.networks.pointpwcnet2_6 import PointConvSceneFlowPWC2_6
from robot.metrics.reg_losses import CurvatureReg

PWC_POOL = {
    "disp": PointConvSceneFlowPWC2_2,
    "reg_param": PointConvSceneFlowPWC2_4,
    "adaptive": PointConvSceneFlowPWC2_6,
}


class DeepFlowNetRegParam(nn.Module):
    def __init__(self, opt):
        super(DeepFlowNetRegParam, self).__init__()
        self.opt = opt
        local_pair_feature_extractor_obj = self.opt[
            (
                "local_pair_feature_extractor_obj",
                "",
                "function object for local_feature_extractor",
            )
        ]
        self.local_pair_feature_extractor = (
            obj_factory(local_pair_feature_extractor_obj)
            if len(local_pair_feature_extractor_obj)
            else self.default_local_pair_feature_extractor
        )
        # todo test the case that the pointfea is initialized by the dataloader
        self.initial_npoints = self.opt[
            ("initial_npoints", 4096, "num of initial sampling points")
        ]
        self.input_channel = self.opt[("input_channel", 1, "input channel")]
        self.initial_radius = self.opt[("initial_radius", 0.001, "initial radius")]
        self.init_neigh_num = self.opt[("init_neigh_num", 16, "init_neigh_num")]
        self.param_shrink_factor = self.opt[
            (
                "param_shrink_factor",
                2,
                "shrink factor the model parameter, #params/param_shrink_factor",
            )
        ]
        self.predict_at_low_resl = self.opt[
            (
                "predict_at_low_resl",
                False,
                "the reg_param would be predicted for 'initi_npoints'",
            )
        ]
        self.use_aniso_kernel = self.opt[
            (
                "use_aniso_kernel",
                True,
                "use the aniso kernel in first layer feature extraction",
            )
        ]
        self.init_deep_feature_extractor()
        self.buffer = {}
        self.iter = 0

    def default_local_pair_feature_extractor(self, cur_source, target):
        cur_source.pointfea = cur_source.weights * 10000  # todo  lung fix in future
        target.pointfea = target.weights * 10000
        return cur_source, target

    def init_deep_feature_extractor(self):
        self.flow_predictor = FlowNet3DIMP(
            input_channel=self.input_channel,
            initial_radius=self.initial_radius,
            initial_npoints=self.initial_npoints,
            init_neigh_num=self.init_neigh_num,
            param_shrink_factor=self.param_shrink_factor,
            predict_at_low_resl=self.predict_at_low_resl,
            use_aniso_kernel=self.use_aniso_kernel,
        )

    def deep_flow(self, cur_source, target):
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
        nonp_param, additional_param = self.flow_predictor(pc1, pc2, feature1, feature2)
        return nonp_param, additional_param

    def normalize_fea(self):
        pass

    def forward(self, cur_source, shape_pair, iter=-1):
        cur_source, shape_pair.target = self.local_pair_feature_extractor(
            cur_source, shape_pair.target
        )
        shape_pair.reg_param, additional_param = self.deep_flow(
            cur_source, shape_pair.target
        )
        if self.predict_at_low_resl:
            shape_pair.dense_mode = False
            shape_pair.control_points = additional_param["control_points"].contiguous()
            shape_pair.control_weights = index_points(
                cur_source.weights, additional_param["control_points_idx"].long()
            ).contiguous()
            shape_pair.toflow = cur_source
        else:
            shape_pair.dense_mode = True
            shape_pair.control_points = cur_source.points
            shape_pair.control_weights = cur_source.weights
            shape_pair.toflow = cur_source
        return shape_pair, additional_param


class PointConvSceneFlowPWCRegParam(nn.Module):
    def __init__(self, opt):
        super(PointConvSceneFlowPWCRegParam, self).__init__()
        self.opt = opt
        local_pair_feature_extractor_obj = self.opt[
            (
                "local_pair_feature_extractor_obj",
                "",
                "function object for local_feature_extractor",
            )
        ]
        self.local_pair_feature_extractor = (
            obj_factory(local_pair_feature_extractor_obj)
            if len(local_pair_feature_extractor_obj)
            else self.default_local_pair_feature_extractor
        )
        # todo test the case that the pointfea is initialized by the dataloader
        self.input_channel = self.opt[("input_channel", 1, "input channel")]
        self.initial_npoints = self.opt[
            ("initial_npoints", 2048, "num of initial sampling points")
        ]
        self.initial_radius = self.opt[
            (
                "initial_radius",
                0.001,
                "initial radius or the resolution of the point cloud",
            )
        ]
        self.neigh_num = self.opt[
            ("neigh_num", 9, "num of neighbor considered by pointconv")
        ]
        self.weight_net_neigh_num = self.opt[
            ("weight_net_neigh_num", 16, "num of weightnet_neighbor considered by pointconv")
        ]
        self.pwc_mode = self.opt[
            (
                "pwc_mode",
                "reg_param",
                "network output mode, 'disp','reg_mode','adaptive' ",
            )
        ]
        self.param_shrink_factor = self.opt[
            (
                "param_shrink_factor",
                2,
                "control the number of factor, #params/param_shrink_factor",
            )
        ]
        self.delploy_original_model = self.opt[
            (
                "delploy_original_model",
                False,
                "delploy the original model in pointpwcnet paper",
            )
        ]
        self.predict_at_low_resl = self.opt[
            (
                "predict_at_low_resl",
                False,
                "the reg_param would be predicted for 'initi_npoints'",
            )
        ]
        self.use_aniso_kernel = self.opt[
            ("use_aniso_kernel", False, "use the aniso kernel in first sampling layer")
        ]
        self.control_shift_factor = self.opt[
            (
                "control_shift_factor",
                0.1,
                "factor scaling the predicted shift of the control points",
            )
        ]
        self.init_deep_feature_extractor()
        self.buffer = {}
        self.iter = 0

    def default_local_pair_feature_extractor(self, cur_source, target):
        cur_source.pointfea = (
            cur_source.points.clone()
        )  # torch.cat([cur_source.points, cur_source.weights], 2)
        target.pointfea = (
            target.points.clone()
        )  # torch.cat([target.points, target.weights], 2)
        return cur_source, target

    def init_deep_feature_extractor(self):
        self.load_pretrained_model = self.opt[
            ("load_pretrained_model", False, "load_pretrained_model")
        ]
        self.pretrained_model_path = self.opt[
            ("pretrained_model_path", "", "path of pretrained model")
        ]
        if not self.delploy_original_model:
            self.flow_predictor = PWC_POOL[self.pwc_mode](
                input_channel=self.input_channel,
                initial_input_radius=self.initial_radius,
                first_sampling_npoints=self.initial_npoints,
                neigh_num=self.neigh_num,
                weight_neigh_num=self.weight_net_neigh_num,
                predict_at_low_resl=self.predict_at_low_resl,
                param_shrink_factor=self.param_shrink_factor,
                use_aniso_kernel=self.use_aniso_kernel,
            )
        else:
            self.flow_predictor = PointConvSceneFlowPWC8192selfglobalPointConv(
                input_channel=self.input_channel,
                initial_npoints=self.initial_npoints,
                predict_at_low_resl=self.predict_at_low_resl,
            )
        if self.load_pretrained_model:
            checkpoint = torch.load(self.pretrained_model_path, map_location="cpu")
            self.flow_predictor.load_state_dict(checkpoint, strict=False)

    def deep_flow(self, cur_source, target):
        pc1, pc2, feature1, feature2 = (
            cur_source.points,
            target.points,
            cur_source.pointfea,
            target.pointfea,
        )
        nonp_param, additional_param = self.flow_predictor(pc1, pc2, feature1, feature2)
        return nonp_param, additional_param

    def forward(self, cur_source, shape_pair, iter=-1):
        cur_source, shape_pair.target = self.local_pair_feature_extractor(
            cur_source, shape_pair.target
        )
        shape_pair.reg_param, additional_param = self.deep_flow(
            cur_source, shape_pair.target
        )
        if self.predict_at_low_resl:
            shape_pair.dense_mode = False
            shape_pair.control_points = additional_param["control_points"].contiguous()
            shape_pair.control_weights = index_points(
                cur_source.weights, additional_param["control_points_idx"].long()
            ).contiguous()
            shape_pair.toflow = cur_source
        else:
            shape_pair.dense_mode = True
            shape_pair.control_points = cur_source.points
            shape_pair.control_weights = cur_source.weights
            shape_pair.toflow = cur_source
        return shape_pair, additional_param


class FLOTRegParam(nn.Module):
    def __init__(self, opt):
        super(FLOTRegParam, self).__init__()
        self.opt = opt
        self.input_channel = self.opt[("input_channel", 1, "input channel")]
        self.nb_iter = self.opt[("nb_iter", 1, "# iter for solving the ot problem")]
        self.predict_at_low_resl = False
        local_pair_feature_extractor_obj = self.opt[
            (
                "local_pair_feature_extractor_obj",
                "",
                "function object for local_feature_extractor",
            )
        ]
        self.local_pair_feature_extractor = (
            obj_factory(local_pair_feature_extractor_obj)
            if len(local_pair_feature_extractor_obj)
            else self.default_local_pair_feature_extractor
        )
        # todo test the case that the pointfea is initialized by the dataloader
        self.init_deep_feature_extractor()
        self.buffer = {}
        self.iter = 0

    def default_local_pair_feature_extractor(self, cur_source, target):
        cur_source.pointfea = (
            cur_source.points.clone()
        )  # torch.cat([cur_source.points, cur_source.weights], 2)
        target.pointfea = (
            target.points.clone()
        )  # torch.cat([target.points, target.weights], 2)
        return cur_source, target

    def init_deep_feature_extractor(self):
        self.load_pretrained_model = self.opt[
            ("load_pretrained_model", False, "load_pretrained_model")
        ]
        self.pretrained_model_path = self.opt[
            ("pretrained_model_path", "", "path of pretrained model")
        ]
        self.flow_predictor = FLOT(
            nb_iter=self.nb_iter, initial_channel=self.input_channel
        )
        if self.load_pretrained_model:
            checkpoint = torch.load(self.pretrained_model_path, map_location="cpu")
            self.flow_predictor.load_state_dict(checkpoint["model"], strict=False)
            print(
                "succeed loading the external model {}".format(
                    self.pretrained_model_path
                )
            )

    def deep_flow(self, cur_source, target):
        pc1, pc2 = cur_source.points, target.points
        nonp_param, additional_param = self.flow_predictor(pc1, pc2)
        return nonp_param, additional_param

    def forward(self, cur_source, shape_pair, iter=-1):
        cur_source, shape_pair.target = self.local_pair_feature_extractor(
            cur_source, shape_pair.target
        )
        shape_pair.reg_param, additional_param = self.deep_flow(
            cur_source, shape_pair.target
        )
        if self.predict_at_low_resl:
            shape_pair.dense_mode = False
            shape_pair.control_points = additional_param["control_points"].contiguous()
            shape_pair.control_weights = index_points(
                cur_source.weights, additional_param["control_points_idx"].long()
            ).contiguous()
            shape_pair.toflow = cur_source
        else:
            shape_pair.dense_mode = True
            shape_pair.control_points = cur_source.points
            shape_pair.control_weights = cur_source.weights
            shape_pair.toflow = cur_source
        return shape_pair, additional_param


class FlowModel(nn.Module):
    def __init__(self, opt):
        super(FlowModel, self).__init__()
        self.opt = opt
        self.model_type = opt[("model_type", "spline", "model type 'spline'/'lddmm'")]
        self.use_aniso_postgradientflow = opt[
            ("use_aniso_postgradientflow", False, "use_aniso_postgradientflow'")
        ]
        if self.model_type == "spline":
            self.flow_model = self.spline_forward
            self.regularization = self.spline_reg
        elif self.model_type == "lddmm":
            self.init_lddmm(opt["lddmm"])
            self.flow_model = self.lddmm_forward
            self.regularization = self.lddmm_reg
        elif self.model_type == "disp":
            self.flow_model = self.disp_forward
            self.regularization = self.disp_reg

        self.init_spline(opt["spline", {}, "settings for spline"])
        self.init_interp(opt[("interp", {}, "settings for interpolation mode")])
        self.init_aniso_postgradientflow(
            opt[("aniso_postgradientflow", {}, "settings for interpolation mode")]
        )

    def init_interp(self, opt_interp):
        interp_kernel_obj = opt_interp[
            (
                "interp_kernel_obj",
                "point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.005)",
                "shape interpolator",
            )
        ]
        self.interp_kernel = obj_factory(interp_kernel_obj)

    def init_aniso_postgradientflow(self, opt_aniso_postgradientflow):
        aniso_kernel_obj = opt_aniso_postgradientflow[
            (
                "aniso_kernel_obj",
                "point_interpolator.NadWatAnisoSpline(exp_order=2, cov_sigma_scale=0.02,aniso_kernel_scale={},eigenvalue_min=0.2,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True, requires_grad=False)",
                "shape interpolator",
            )
        ]
        self.aniso_post_kernel = obj_factory(aniso_kernel_obj)
        self.geomloss_for_aniso_postgradientflow = opt_aniso_postgradientflow[
            "geomloss", {}, "settings for geomloss"
        ]

    def flow(self, shape_pair):
        toflow_points = shape_pair.toflow.points
        if self.model_type == "disp" or self.model_type == "spline":
            control_points = shape_pair.control_points
            flowed_control_points = shape_pair.flowed_control_points
            control_weights = shape_pair.control_weights
            # interp kernel size should be different for dense_mode and non_dense_mode
            kernel = self.interp_kernel if shape_pair.dense_mode else self.spline_kernel
            return kernel(
                toflow_points, control_points, flowed_control_points, control_weights
            )
        elif self.model_type == "lddmm":
            control_points = shape_pair.control_points
            self.lddmm_module.set_mode("flow")
            momentum = shape_pair.reg_param
            _, flowed_control_points, flowed_points = self.integrator.solve(
                (momentum, control_points, toflow_points)
            )
            return flowed_control_points, flowed_points
        else:
            raise NotImplemented

    def disp_forward(self, toflow, shape_pair, additional_param):
        shape_pair.flowed_control_points = (
            shape_pair.control_points + shape_pair.reg_param
        )
        if shape_pair.dense_mode:
            flowed_points = shape_pair.flowed_control_points
        else:
            # similar to the spline
            flow_points = self.spline_kernel(
                toflow.points,
                shape_pair.control_points,
                shape_pair.reg_param,
                shape_pair.control_weights,
            )
            flowed_points = toflow.points + flow_points
        return flowed_points, None

    def disp_reg(self, reg_param, reg_additional_input=None):
        dist = reg_param ** 2
        dist = dist.sum(2).mean(1)
        return dist

    def init_spline(self, spline_opt):
        spline_kernel_obj = spline_opt[
            (
                "spline_kernel_obj",
                "point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)",
                "shape interpolator",
            )
        ]
        self.spline_kernel = obj_factory(spline_kernel_obj)

    def spline_forward(self, toflow, shape_pair, additional_param):
        """
        input the reg_param and output the displacement
        :param reg_param:
        :return:
        """
        control_disp = self.spline_kernel(
            shape_pair.control_points,
            shape_pair.control_points,
            shape_pair.reg_param,
            shape_pair.control_weights,
        )
        shape_pair.flowed_control_points = shape_pair.control_points + control_disp
        if shape_pair.dense_mode:
            sm_disp = control_disp
            flowed_points = shape_pair.flowed_control_points
        else:
            sm_disp = self.spline_kernel(
                toflow.points,
                shape_pair.control_points,
                shape_pair.reg_param,
                shape_pair.control_weights,
            )  # todo check if detach the points
            flowed_points = toflow.points + sm_disp
        return flowed_points, (sm_disp, toflow)

    def spline_reg(self, disp, reg_additional_input):
        # sm_disp, toflow = reg_additional_input
        # toflow_points, to_flow_weights = toflow.points, toflow.weights
        # sm_disp = self.spline_kernel(toflow_points,toflow_points,disp,to_flow_weights)
        # dist = disp * sm_disp
        dist = disp ** 2
        dist = dist.sum(2).mean(1)
        return dist

    def init_lddmm(self, lddmm_opt):
        from robot.modules_reg.ode_int import ODEBlock
        from robot.modules_reg.module_lddmm import LDDMMHamilton, LDDMMVariational

        self.module_type = lddmm_opt[
            ("module", "hamiltonian", "lddmm module type: hamiltonian or variational")
        ]
        assert self.module_type in ["hamiltonian", "variational"]
        self.lddmm_module = (
            LDDMMHamilton(lddmm_opt[("hamiltonian", {}, "settings for hamiltonian")])
            if self.module_type == "hamiltonian"
            else LDDMMVariational(
                lddmm_opt[("variational", {}, "settings for variational")]
            )
        )
        self.lddmm_kernel = self.lddmm_module.kernel
        self.integrator_opt = lddmm_opt[("integrator", {}, "settings for integrator")]
        self.integrator = ODEBlock(self.integrator_opt)
        self.integrator.set_func(self.lddmm_module)

    def lddmm_forward(self, toflow, shape_pair, additional_param):
        toflow_points = toflow.points
        momentum = shape_pair.reg_param
        momentum = momentum.clamp(
            -0.5, 0.5
        )  # todo  this is a temporal setting for data normalized into [0,1] should be set in class attribute
        if shape_pair.dense_mode:
            self.lddmm_module.set_mode("shooting")
            _, flowed_control_points = self.integrator.solve(
                (momentum, shape_pair.control_points)
            )
            flowed_points = flowed_control_points
            shape_pair.flowed_control_points = flowed_control_points
        else:
            self.lddmm_module.set_mode("flow")
            _, flowed_control_points, flowed_points = self.integrator.solve(
                (momentum, shape_pair.control_points, toflow_points)
            )
            shape_pair.flowed_control_points = flowed_control_points
        return flowed_points, shape_pair.control_points

    def lddmm_reg(self, momentum, toflow_points):
        momentum = momentum.clamp(-0.5, 0.5)
        dist = momentum * self.lddmm_kernel(
            toflow_points, toflow_points, momentum
        )  # todo check if detach the points
        dist = dist.sum(2).mean(1)
        return dist

    def aniso_postgradientflow(self, flowed, target):
        geomloss_setting = deepcopy(self.geomloss_for_aniso_postgradientflow)
        geomloss_setting.print_settings_off()
        geomloss_setting["mode"] = "analysis"
        geomloss_setting["attr"] = "points"
        use_bary_map = geomloss_setting[
            (
                "use_bary_map",
                False,
                "take wasserstein baryceneter if there is little noise or outlier,  otherwise use gradient flow",
            )
        ]

        if use_bary_map:
            (
                mapped_target_index,
                mapped_topK_target_index,
                mapped_position,
            ) = wasserstein_barycenter_mapping(
                flowed, target, geomloss_setting
            )  # BxN
        else:
            mapped_position, wasserstein_dist = point_based_gradient_flow_guide(
                flowed, target, geomloss_setting
            )
        disp = mapped_position - flowed.points
        smoothed_disp = self.aniso_post_kernel(
            flowed.points, flowed.points, disp, flowed.weights
        )
        flowed_points = flowed.points + smoothed_disp
        new_flowed = Shape().set_data_with_refer_to(flowed_points, flowed)
        return new_flowed

    def forward(self, cur_source, shape_pair, additional_param):
        flowed_points, reg_additional_input = self.flow_model(
            cur_source, shape_pair, additional_param
        )

        reg_loss = self.regularization(shape_pair.reg_param, reg_additional_input)
        flowed = Shape().set_data_with_refer_to(flowed_points, cur_source)
        if self.use_aniso_postgradientflow:
            flowed = self.aniso_postgradientflow(flowed, shape_pair.target)
        return flowed, reg_loss


class DeepFlowLoss(nn.Module):
    def __init__(self, opt):
        super(DeepFlowLoss, self).__init__()
        self.opt = opt
        self.loss_type = self.opt[("loss_type", "disp_l2", "disp_l2 / ot_dist")]
        self.include_curv_constrain = self.opt[
            ("include_curv_constrain", False, "add constrain on the curvature")
        ]
        self.curv_factor = self.opt[("curv_factor", 1.0, "curv_factor")]
        self.use_gmm_as_unsupervised_metric = self.opt[
            ("use_gmm_as_unsupervised_metric", False, "use_gmm_as_unsupervised_metric")
        ]

    def disp_l2(self, flowed, target):
        """
        the gt correspondence should be known and the number of points should be the same
        :param flowed:
        :param target:
        :return:
        """
        fp = flowed.points
        tp = target.points
        fw = flowed.weights
        l2_loss = (((fp - tp) ** 2).sum(2, keepdim=True) * fw).sum(1)  # todo test
        return l2_loss[..., 0]  # remove the last 1 dim

    def ot_distance(self, flowed, target):
        """
        the gt correspondence can be unknown, the point number can be inconsistent
        :param flowed: shape,
        :param target: shape
        :return:
        """
        geomloss_setting = deepcopy(self.opt["geomloss"])
        geomloss_setting.print_settings_off()
        geomloss_setting["attr"] = "points"
        self.geom_loss = GeomDistance(geomloss_setting)
        return self.geom_loss(flowed, target)

    def geo_distance(self, flowed, target):
        """
        compute the local feature of  the flowed and the target, compare there l2 difference
        :param flowed:
        :param target:
        :return:
        """
        local_pair_feature_extractor_obj = self.opt[
            (
                "local_pair_feature_extractor_obj",
                "",
                "function object for local_feature_extractor",
            )
        ]
        local_pair_feature_extractor = (
            obj_factory(local_pair_feature_extractor_obj)
            if len(local_pair_feature_extractor_obj)
            else None
        )
        flowed, target = local_pair_feature_extractor(flowed, target)
        fp = flowed.pointfea
        tp = target.pointfea
        fw = flowed.weights
        l2_loss = (((fp - tp) ** 2).sum(2, keepdim=True) * fw).sum(1)  # todo test
        return l2_loss[..., 0]

    def gmm(self, flowed, target):
        gmm_setting = deepcopy(self.opt["gmm"])
        gmm_setting.print_settings_off()
        gmm_setting["attr"] = "points"
        gmm_loss = GMMLoss(gmm_setting)
        return gmm_loss(flowed, target)[..., 0] * 100  # todo fix this factor

    def curvature_diff(self, source, flowed, target):
        curloss_setting = deepcopy(self.opt["curloss"])
        curloss = CurvatureReg(curloss_setting)
        curvature_loss = curloss(source, flowed, target)
        return curvature_loss

    def control_shift_reg(self, additional_param):
        if "control_shift" in additional_param:
            return (additional_param["control_shift"] ** 2).sum(2).mean(1)  # 100
        else:
            return 0

    def forward(self, flowed, target, gt_flowed, additional_param=None, has_gt=True):
        curv_reg = (
            self.curv_factor
            * self.curvature_diff(additional_param["source"], flowed, target)
            if self.include_curv_constrain
            else 0
        )

        curv_reg += self.control_shift_reg(additional_param)
        if not has_gt:
            if not self.use_gmm_as_unsupervised_metric:
                return self.ot_distance(flowed, target) + curv_reg
            else:
                return self.gmm(flowed, target) + curv_reg
        else:
            if self.loss_type == "disp_l2":
                return self.disp_l2(flowed, gt_flowed) + curv_reg
            elif self.loss_type == "ot_dist":
                return self.ot_distance(flowed, target) + curv_reg


class PWCLoss(nn.Module):
    def __init__(self, opt):
        super(PWCLoss, self).__init__()
        self.opt = opt
        self.multi_scale_weight = opt[
            ("multi_scale_weight", [0.02, 0.04, 0.08, 0.16], "weight for each scale")
        ]
        self.loss_type = opt[("loss_type", "disp_l2", "disp_l2 / chamfer_self")]
        self.use_chamfer_as_self_supervised_loss = opt[
            ("use_chamfer_as_self_supervised_loss", False, "use_self_supervised_loss")
        ]
        self.loss_order = opt[
            ("loss_order", 2, " order of distance if set 2, l2 distance")
        ]

    def multiScaleLoss(self, flowed, gt_flowed, additional_param):
        weights = flowed.weights
        alpha = self.multi_scale_weight
        floweds, fps_idxs = (
            additional_param["floweds"],
            additional_param["fps_pc1_idxs"],
        )  # todo to be fixed
        num_scale = len(floweds)
        offset = len(fps_idxs) - num_scale + 1
        num_scale = len(floweds)
        # generate GT list and mask1s
        gt = [gt_flowed.points]
        w = [weights]
        for i in range(len(fps_idxs)):
            fps_idx = fps_idxs[i]
            sub_gt_flow = index_points(gt[-1], fps_idx)
            sub_w = index_points(w[-1], fps_idx)
            gt.append(sub_gt_flow)
            w.append(sub_w / sub_w.sum(1, keepdim=True))
        total_loss = 0
        dist_fn = (
            lambda x: x ** self.loss_order
            if self.loss_order % 2 == 0
            else torch.abs(x) ** self.loss_order
        )
        for i in range(num_scale):
            diff_flow = floweds[i] - gt[i + offset]
            total_loss += alpha[i] * (
                dist_fn(diff_flow).sum(dim=2, keepdim=True) * w[i + offset]
            ).sum(1)
        w = flowed.weights / flowed.weights.sum(1, keepdim=True)
        additional_loss = (
            (dist_fn(flowed.points - gt_flowed.points)).sum(2, keepdim=True) * w).sum(1)  # todo test
        reg = 0
        if "control_shift" in additional_param:
            reg = dist_fn(additional_param["control_shift"]).sum(2).mean(1) * 10

        return total_loss[..., 0] + additional_loss[..., 0] + reg

    def multiScaleLoss_with_OT(self, flowed, target, gt_flowed, additional_param):
        weights = flowed.weights
        alpha = self.multi_scale_weight
        floweds, fps_idxs = (
            additional_param["floweds"],
            additional_param["fps_pc1_idxs"],
        )  # todo to be fixed
        num_scale = len(floweds)
        offset = len(fps_idxs) - num_scale + 1
        num_scale = len(floweds)
        # generate GT list and mask1s
        gt = [gt_flowed.points]
        w = [weights]

        for i in range(len(fps_idxs)):
            fps_idx = fps_idxs[i]
            sub_gt_flow = index_points(gt[-1], fps_idx)
            sub_w = index_points(w[-1], fps_idx)
            gt.append(sub_gt_flow)
            w.append(sub_w / sub_w.sum(1, keepdim=True))
        total_loss = 0
        dist_fn = (
            lambda x: x ** self.loss_order
            if self.loss_order % 2 == 0
            else torch.abs(x) ** self.loss_order
        )

        for i in range(num_scale):
            diff_flow = floweds[i] - gt[i + offset]
            total_loss += alpha[i] * (
                dist_fn(diff_flow).sum(dim=2, keepdim=True) * w[i + offset]
            ).sum(1)
        additional_loss = self.ot_distance(flowed, target) * alpha[-1]
        reg = 0
        if "control_shift" in additional_param:
            reg = dist_fn(additional_param["control_shift"]).sum(2).mean(1) * 10
        return total_loss[..., 0] + additional_loss + reg

    def chamfer_self_loss(self, flowed, target, additional_param):
        pred_flows, pc1, pc2 = (
            additional_param["flows"],
            additional_param["pc1"],
            additional_param["pc2"],
        )
        (
            total_loss,
            chamfer_loss,
            curvature_loss,
            smoothness_loss,
        ) = multiScaleChamferSmoothCurvature(pc1, pc2, pred_flows)
        return total_loss

    def ot_distance(self, flowed, target):
        """
        the gt correspondence can be unknown, the point number can be inconsistent
        :param flowed: shape,
        :param target: shape
        :return:
        """
        geomloss_setting = deepcopy(self.opt["geomloss"])
        geomloss_setting.print_settings_off()
        geomloss_setting["attr"] = "points"
        self.geom_loss = GeomDistance(geomloss_setting)
        return self.geom_loss(flowed, target)

    def forward(self, flowed, target, gt_flowed, additional_param=None, has_gt=True):
        if not has_gt:
            if self.use_chamfer_as_self_supervised_loss:
                return self.chamfer_self_loss(flowed, target, additional_param)
            else:
                return self.ot_distance(flowed, target)
        elif has_gt:
            if self.loss_type == "disp_l2":
                return self.multiScaleLoss(flowed, gt_flowed, additional_param)
            elif self.loss_type == "chamfer_dist":
                return self.chamfer_self_loss(flowed, target, additional_param)
            elif self.loss_type == "ot_dist":
                return self.multiScaleLoss_with_OT(
                    flowed, target, gt_flowed, additional_param
                )

        else:
            return self.ot_distance(flowed, target)


if __name__ == "__main__":
    import torch
    from pykeops.torch import LazyTensor

    def deep_corres(points, weights, pointfea):
        x_i = LazyTensor(points[:, :, None])  # BxNx1xD
        x_j = LazyTensor(points[:, None])  # Bx1xNxD
        fea_i = LazyTensor(pointfea[:, :, None])  # BxNx1xd
        fea_j = LazyTensor(pointfea[:, None])  # Bx1xNxd
        weight = LazyTensor(weights[:, None])  # Bx1xNx1
        point_dist2 = x_i.sqdist(x_j)  # BxNxNx1
        fea_dist2 = fea_i.sqdist(fea_j)  # BxNxNx1
        point_dist2 = point_dist2 - point_dist2.max()  # BxNxN
        fea_dist2 = fea_dist2 - fea_dist2.max()  # BxNxN
        point_logsoftmax = point_dist2 - LazyTensor(
            point_dist2.logsumexp(dim=2)[..., None]
        )  # BxNxN
        fea_logsoftmax = fea_dist2 - LazyTensor(
            fea_dist2.logsumexp(dim=2)[..., None]
        )  # BxNxN
        loss = ((point_logsoftmax.exp() - fea_logsoftmax.exp()) * weight) ** 2
        return loss.sum(2).sum(1)

    points = torch.rand(2, 1000, 3)
    fea = torch.rand(2, 1000, 5)
    weights = torch.rand(2, 1000, 1)
    deep_corres(points, weights, fea)
