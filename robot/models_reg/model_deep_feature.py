from copy import deepcopy
from robot.modules_reg.module_deep_feature import *
from robot.modules_reg.module_gradient_flow import point_based_gradient_flow_guide
from robot.utils.utils import sigmoid_decay
from robot.metrics.reg_losses import Loss

DEEP_EXTRACTOR = {
    "pointnet2_extractor": PointNet2FeaExtractor,
    "dgcnn_extractor": DGCNNFeaExtractor,
    "pointconv_extractor": PointConvFeaExtractor,
}


class DeepFeature(nn.Module):
    """
    In this class, a deep feature extractor is trained,
    the synth data is used for training
    additionally, a spline model is included for evaluation
    """

    def __init__(self, opt):
        super(DeepFeature, self).__init__()
        self.opt = opt
        create_shape_pair_from_data_dict = opt[
            (
                "create_shape_pair_from_data_dict",
                "shape_pair_utils.create_shape_pair_from_data_dict()",
                "generator func",
            )
        ]
        self.create_shape_pair_from_data_dict = obj_factory(
            create_shape_pair_from_data_dict
        )
        decompose_shape_pair_into_dict = opt[
            (
                "decompose_shape_pair_into_dict",
                "shape_pair_utils.decompose_shape_pair_into_dict()",
                "decompose shape pair into dict",
            )
        ]
        self.decompose_shape_pair_into_dict = obj_factory(
            decompose_shape_pair_into_dict
        )
        spline_kernel_obj = self.opt[
            ("spline_kernel_obj", "", "shape interpolator in multi-scale solver")
        ]
        self.spline_kernel = (
            obj_factory(spline_kernel_obj) if spline_kernel_obj else None
        )
        interp_kernel_obj = self.opt[
            (
                "interp_kernel_obj",
                "point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)",
                "kernel for multi-scale interpolation",
            )
        ]
        self.interp_kernel = obj_factory(interp_kernel_obj)
        deep_extractor = self.opt[
            ("deep_extractor", "pointnet2_extractor", "name of deep feature extractor")
        ]
        self.pair_feature_extractor = DEEP_EXTRACTOR[deep_extractor](
            self.opt[deep_extractor, {}, "settings for the deep extractor"]
        )
        self.loss = DeepFeatureLoss(
            self.opt[("deepfea_loss", {}, "settings for deep feature loss")]
        )
        self.geom_loss_opt_for_eval = opt[
            (
                "geom_loss_opt_for_eval",
                {},
                "settings for sim_loss_opt, the sim_loss here is not used for training but for evaluation",
            )
        ]
        # self.reg_loss_fn = self.regularization

        external_evaluate_metric_obj = self.opt[
            ("external_evaluate_metric_obj", "", "external evaluate metric")
        ]
        self.external_evaluate_metric = (
            obj_factory(external_evaluate_metric_obj)
            if external_evaluate_metric_obj
            else None
        )
        self.register_buffer("local_iter", torch.Tensor([0]))
        self.print_step = self.opt[("print_step", 5, "print every n iteration")]
        self.buffer = {}

    def check_if_update_lr(self):
        return None, None

    def set_cur_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch

    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.local_iter = self.local_iter * 0

    def flow(self, shape_pair):
        """
        :param shape_pair:
        :return:
        """

        toflow_points = shape_pair.toflow.points
        control_points = shape_pair.control_points
        flowed_control_points = shape_pair.flowed_control_points
        control_weights = shape_pair.control_weights
        flowed_points = self.interp_kernel(
            toflow_points, control_points, flowed_control_points, control_weights
        )
        flowed = Shape()
        flowed.set_data_with_refer_to(flowed_points, shape_pair.toflow)
        shape_pair.set_flowed(flowed)
        return shape_pair

    def model_eval(self, input_data, batch_info=None):
        """
        for  deep approach, we assume the source points = control points
        :param shape_pair:
        :param batch_info:
        :return:
        """
        if batch_info["corr_source_target"]:
            loss, shape_data_dict = self.forward(input_data, batch_info)
            shape_pair = self.create_shape_pair_from_data_dict(shape_data_dict)

        else:
            shape_pair = self.create_shape_pair_from_data_dict(input_data)
            shape_pair.source, shape_pair.target = self.pair_feature_extractor(
                shape_pair.source, shape_pair.target
            )
            loss = torch.tensor([-1] * shape_pair.source.nbatch)

        geomloss_setting = deepcopy(self.opt["deepfea_loss"]["geomloss"])
        geomloss_setting.print_settings_off()
        geomloss_setting["mode"] = "analysis"
        geomloss_setting["attr"] = "pointfea"
        if self.cur_epoch == 0:
            print(
                "In the first epoch, the validation/debugging output is the baseline "
            )
            geomloss_setting["attr"] = "points"
        (
            mapped_target_index,
            mapped_topK_target_index,
            bary_mapped_position,
        ) = wasserstein_barycenter_mapping(
            shape_pair.source, shape_pair.target, geomloss_setting
        )  # BxN
        mapped_position = bary_mapped_position
        source_points = shape_pair.source.points
        source_weights = shape_pair.source.weights
        disp = mapped_position - source_points
        # self.spline_kernel = None
        smoothed_disp = (
            self.spline_kernel(source_points, source_points, disp, source_weights)
            if self.spline_kernel is not None
            else disp
        )
        flowed_points = source_points + smoothed_disp
        shape_pair.flowed = Shape().set_data_with_refer_to(
            flowed_points, shape_pair.source
        )
        geomloss_setting = deepcopy(self.geom_loss_opt_for_eval)
        geomloss_setting["attr"] = "points"
        geomloss = GeomDistance(geomloss_setting)
        wasserstein_dist = geomloss(shape_pair.flowed, shape_pair.target)
        source_points = shape_pair.source.points
        B, N = source_points.shape[0], source_points.shape[1]
        device = source_points.device
        print(
            "the current data is {}".format(
                "synth" if batch_info["is_synth"] else "real"
            )
        )

        if batch_info["corr_source_target"]:
            # compute mapped acc
            gt_index = torch.arange(N, device=device).repeat(B, 1)  # B,N
            acc = (mapped_target_index == gt_index).sum(1) / N
            topk_acc = (
                (mapped_topK_target_index == (gt_index[..., None])).sum(2) > 0
            ).sum(1) / N
            metrics = {
                "score": [_topk_acc.item() for _topk_acc in topk_acc],
                "loss": [_loss.item() for _loss in loss],
                "_acc": [_acc.item() for _acc in acc],
                "topk_acc": [_topk_acc.item() for _topk_acc in topk_acc],
                "ot_dist": [_ot_dist.item() for _ot_dist in wasserstein_dist],
            }
        else:
            metrics = {
                "score": [_ot_dist.item() for _ot_dist in wasserstein_dist],
                "loss": [_loss.item() for _loss in loss],
                "ot_dist": [_ot_dist.item() for _ot_dist in wasserstein_dist],
            }
        if self.external_evaluate_metric is not None:
            shape_pair.control_points = shape_pair.source.points
            shape_pair.control_weights = shape_pair.source.weights
            shape_pair.flowed_control_points = shape_pair.flowed.points
            additional_param = {
                "model": self,
                "initial_nonp_control_points": shape_pair.source.points,
            }
            self.external_evaluate_metric(
                metrics,
                shape_pair,
                batch_info,
                additional_param=additional_param,
                alias="",
            )
            additional_param.update({"mapped_position": mapped_position})
            self.external_evaluate_metric(
                metrics, shape_pair, batch_info, additional_param, "_and_gf"
            )
        return metrics, self.decompose_shape_pair_into_dict(shape_pair)

    def get_factor(self):
        """
        get the regularizer factor according to training strategy

        :return:
        """
        sim_factor = self.opt[("sim_factor", 1, "similarity factor")]
        reg_factor_init = self.opt[
            ("reg_factor_init", 100, "initial regularization factor")
        ]
        reg_factor_decay = self.opt[
            ("reg_factor_decay", 6, "regularization decay factor")
        ]
        static_epoch = self.opt[
            ("static_epoch", 5, "first # epoch the factor doesn't change")
        ]
        min_threshold = 0
        reg_factor = float(
            max(
                sigmoid_decay(self.cur_epoch, static=static_epoch, k=reg_factor_decay)
                * reg_factor_init,
                min_threshold,
            )
        )
        return sim_factor, reg_factor

    def forward(self, input_data, batch_info=None):
        shape_pair = self.create_shape_pair_from_data_dict(input_data)
        has_gt = batch_info["has_gt"]
        gt_flowed_points = (
            shape_pair.target.points.clone()
            if not has_gt
            else shape_pair.extra_info.get("gt_flowed",None)
        )
        gt_flowed = Shape().set_data_with_refer_to(gt_flowed_points, shape_pair.source)
        cur_source, shape_pair.target = self.pair_feature_extractor(
            shape_pair.source, shape_pair.target
        )
        sim_loss, reg_loss = self.loss(
            cur_source, shape_pair.target, gt_flowed, has_gt=has_gt
        )
        self.buffer["sim_loss"] = sim_loss.detach()
        self.buffer["reg_loss"] = reg_loss.detach()
        sim_factor, reg_factor = self.get_factor()
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        loss = sim_loss + reg_loss
        shape_pair.source = cur_source

        if self.local_iter % self.print_step == 0:
            # print(
            #     "debugging: feature norm {}".format(
            #         shape_pair.source.pointfea.norm(2, 2).mean(1)
            #     )
            # )
            print(
                "{} th step, {} sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}".format(
                    self.local_iter.item(),
                    "synth_data" if batch_info["is_synth"] else "real_data",
                    sim_loss.mean().item(),
                    reg_loss.mean().item(),
                    sim_factor,
                    reg_factor,
                )
            )
            # self.debug_mode(shape_pair)
        self.local_iter += 1

        return loss, self.decompose_shape_pair_into_dict(shape_pair)

    def debug_mode(self, shape_pair):
        """remove in the furture """
        import os
        from robot.utils.visualizer import (
            visualize_point_pair_overlap,
            visualize_source_flowed_target_overlap,
        )

        source, flowed, target = shape_pair.source, shape_pair.flowed, shape_pair.target
        if self.local_iter % 50 != 0:
            return
        # saving_capture_path = None if not self.saving_running_result_visualize else os.path.join(self.record_path,"debugging")
        saving_capture_path = (
            "/playpen-raid1/zyshen/debug/debug_onecase_deepfeature_nonaug"
        )

        geomloss_setting = deepcopy(self.opt["deepfea_loss"]["geomloss"])
        geomloss_setting.print_settings_off()
        geomloss_setting["mode"] = "analysis"
        geomloss_setting["attr"] = "pointfea"
        if self.cur_epoch == 0:
            print(
                "In the first epoch, the validation/debugging output is the baseline "
            )
            geomloss_setting["attr"] = "points"
        (
            mapped_target_index,
            mapped_topK_target_index,
            bary_mapped_position,
        ) = wasserstein_barycenter_mapping(
            shape_pair.source, shape_pair.target, geomloss_setting
        )  # BxN
        mapped_position = bary_mapped_position
        source_points = shape_pair.source.points
        source_weights = shape_pair.source.weights
        disp = mapped_position - source_points
        flowed_points = source_points + disp
        flowed = Shape().set_data_with_refer_to(flowed_points, shape_pair.source)

        if saving_capture_path:
            os.makedirs(saving_capture_path, exist_ok=True)
            saving_capture_path = os.path.join(
                saving_capture_path,
                "training_iter_{}".format(self.local_iter.item()) + ".png",
            )
        visualize_source_flowed_target_overlap(
            source.points[0],
            flowed.points[0],
            target.points[0],
            source.weights[0],
            flowed.weights[0],
            target.weights[0],
            "cur_source",
            "flowed",
            "target",
            camera_pos=[
                (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
                (0.0, 0.0, 0.0),
                (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
            ],
            rgb_on=[False, False, False],
            show=False,
            saving_capture_path=saving_capture_path,
        )
