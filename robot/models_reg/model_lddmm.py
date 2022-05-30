from copy import deepcopy
import torch
import torch.nn as nn
from robot.modules_reg.module_lddmm import LDDMMHamilton, LDDMMVariational
from robot.modules_reg.module_gradient_flow import gradient_flow_guide
from robot.global_variable import Shape
from robot.metrics.reg_losses import Loss
from robot.modules_reg.ode_int import ODEBlock
from robot.modules_reg.opt_flowed_eval import opt_flow_model_eval
from robot.utils.utils import sigmoid_decay
from robot.utils.obj_factory import obj_factory


class LDDMMOPT(nn.Module):
    """
    the class implements the LDDMM approach
    (compared with discrete flow, the source shape is fixed here, shooting path starts from the source)
    1. standard LDDMM

    2. gradient flow guided LDDMM
        M(0) = Source,
        for t = 0...T:
            I.  compute gradient flow between M(t) and target T,  get the gradflow result G(t)
            II.  for n iteration, use the G(t) with MSE loss to guide lddmm, updating its initial momentum,
            III . The shooting result is set as M(t+1)


    """

    def __init__(self, opt):
        super(LDDMMOPT, self).__init__()
        self.opt = opt
        self.module_type = self.opt[
            ("module", "hamiltonian", "lddmm module type: hamiltonian or variational")
        ]
        assert self.module_type in ["hamiltonian", "variational"]
        self.lddmm_module = (
            LDDMMHamilton(self.opt[("hamiltonian", {}, "settings for hamiltonian")])
            if self.module_type == "hamiltonian"
            else LDDMMVariational(
                self.opt[("variational", {}, "settings for variational")]
            )
        )
        self.lddmm_kernel = self.lddmm_module.kernel
        self.interp_kernel = self.lddmm_kernel
        pair_feature_extractor_obj = opt[
            ("pair_feature_extractor_obj", "", "feature extraction function")
        ]
        self.pair_feature_extractor = (
            obj_factory(pair_feature_extractor_obj)
            if pair_feature_extractor_obj
            else None
        )
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.sim_loss_fn = Loss(sim_loss_opt)
        self.reg_loss_fn = self.geodesic_distance
        self.integrator_opt = self.opt[("integrator", {}, "settings for integrator")]
        self.integrator = ODEBlock(self.integrator_opt)
        self.integrator.set_func(self.lddmm_module)
        self.call_thirdparty_package = False
        self.register_buffer(
            "local_iter", torch.Tensor([0])
        )  # iteration record in single scale
        self.register_buffer(
            "global_iter", torch.Tensor([0])
        )  # iteration record in multi-scale
        self.print_step = self.opt[("print_step", 10, "print every n iteration")]
        self.use_gradflow_guided = self.opt[
            (
                "use_gradflow_guided",
                False,
                "optimization guided by gradient flow, to accererate the convergence",
            )
        ]
        self.gradflow_guided_buffer = {}
        self.geom_loss_opt_for_eval = opt[
            (
                "geom_loss_opt_for_eval",
                {},
                "settings for sim_loss_opt, the sim_loss here is not used for optimization but for evaluation",
            )
        ]
        external_evaluate_metric_obj = self.opt[
            ("external_evaluate_metric_obj", "", "external evaluate metric")
        ]
        self.external_evaluate_metric = (
            obj_factory(external_evaluate_metric_obj)
            if external_evaluate_metric_obj
            else None
        )

    def set_record_path(self, record_path):
        self.record_path = record_path

    def init_reg_param(self, shape_pair, force=False):
        if shape_pair.reg_param is None or force:
            reg_param = torch.zeros_like(shape_pair.get_control_points()).normal_(
                0, 1e-7
            )
            reg_param.requires_grad_()
            shape_pair.set_reg_param(reg_param)
        return shape_pair

    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.local_iter = self.local_iter * 0
        self.gradflow_guided_buffer = {}

    def clean(self):
        self.local_iter = self.local_iter * 0
        self.global_iter = self.global_iter * 0
        self.gradflow_guided_buffer = {}

    def shooting(self, shape_pair):
        momentum = shape_pair.reg_param
        momentum = momentum.clamp(-1, 1)
        control_points = shape_pair.get_control_points()
        self.lddmm_module.set_mode("shooting")
        _, flowed_control_points = self.integrator.solve((momentum, control_points))
        shape_pair.set_flowed_control_points(flowed_control_points)
        return shape_pair

    def flow(self, shape_pair):
        momentum = shape_pair.reg_param
        control_points = shape_pair.control_points
        toflow_points = shape_pair.get_toflow_points()
        self.lddmm_module.set_mode("flow")
        _, flowed_control_points, flowed_points = self.integrator.solve(
            (momentum, control_points, toflow_points)
        )
        shape_pair.flowed_control_points = flowed_control_points
        flowed = Shape()
        flowed.set_data_with_refer_to(flowed_points, shape_pair.source)
        shape_pair.set_flowed(flowed)
        return shape_pair

    def geodesic_distance(self, momentum, control_points):
        momentum = momentum.clamp(-1, 1)
        dist = momentum * self.lddmm_kernel(control_points, control_points, momentum)
        dist = dist.mean(2).mean(1)
        return dist

    def get_factor(self):
        """
        get the regularizer factor according to training strategy

        :return:
        """
        sim_factor = 100
        reg_factor_init = 0.1  # self.initial_reg_factor
        static_epoch = 100
        min_threshold = reg_factor_init / 10
        decay_factor = 8
        reg_factor = float(
            max(
                sigmoid_decay(
                    self.local_iter.item(), static=static_epoch, k=decay_factor
                )
                * reg_factor_init,
                min_threshold,
            )
        )
        return sim_factor, reg_factor

    def update_reg_param_from_low_scale_to_high_scale(
        self, shape_pair_low, shape_pair_high
    ):
        def estimate_factor_ratio(
            control_points_low, control_points_high, point_scale=False
        ):
            """
            an estimation of the scaling factor when upsampling the momentum
            :param control_points_low:
            :param control_points_high:
            :return:
            """
            device = control_points_low.device
            if not point_scale:
                ones_low = torch.ones(
                    control_points_low.shape[0], control_points_low.shape[1], 1
                ).to(device)
                weight_low = self.interp_kernel(
                    control_points_low, control_points_low, ones_low
                )
                weight_high_low = self.interp_kernel(
                    control_points_high, control_points_low, ones_low
                )
                weight_low_new = self.interp_kernel(
                    control_points_low, control_points_high, weight_high_low
                )
                weight_high_ratio = self.interp_kernel(
                    control_points_high, control_points_low, weight_low / weight_low_new
                )
                weight_high_ratio.clamp_(0, 1)
                # inf_mask = torch.isinf(weight_high_ratio)
                # if torch.sum(inf_mask)>0:
                #     print(" {} inf detected".format(torch.sum(inf_mask).item()))
                #     weight_high_ratio[inf_mask] = 0
            else:
                weight_high_ratio = torch.ones(
                    control_points_high.shape[0], control_points_high.shape[1], 1
                ).to(device)
                weight_high_ratio = (
                    weight_high_ratio
                    * control_points_low.shape[1]
                    / control_points_high.shape[1]
                )
            return weight_high_ratio

        control_points_high = shape_pair_high.get_control_points()
        control_points_low = shape_pair_low.get_control_points()
        reg_param_low = shape_pair_low.reg_param
        reg_param_high = self.interp_kernel(
            control_points_high, control_points_low, reg_param_low
        )
        reg_param_high = (
            estimate_factor_ratio(control_points_low, control_points_high)
            * reg_param_high
        )
        reg_param_high.detach_()
        reg_param_high.requires_grad_()
        shape_pair_high.set_reg_param(reg_param_high)
        return shape_pair_high

    def wasserstein_gradient_flow_guidence(self, flowed, target):
        """
        wassersten gradient flow has a reasonable behavior only when set self.pair_feature_extractor = None
        """
        gradflow_guided_opt = self.opt[
            ("gradflow_guided", {}, "settings for gradflow guidance")
        ]
        self.update_gradflow_every_n_step = gradflow_guided_opt[
            ("update_gradflow_every_n_step", 10, "update the gradflow every # step")
        ]
        gradflow_blur_init = gradflow_guided_opt[
            (
                "gradflow_blur_init",
                0.05,
                "the inital 'blur' parameter in geomloss setting",
            )
        ]
        update_gradflow_blur_by_raito = gradflow_guided_opt[
            (
                "update_gradflow_blur_by_raito",
                0.05,
                "the raito that updates the 'blur' parameter in geomloss setting",
            )
        ]
        gradflow_blur_min = gradflow_guided_opt[
            (
                "gradflow_blur_min",
                0.05,
                "the minium value of the 'blur' parameter in geomloss setting",
            )
        ]
        gradflow_reach_init = gradflow_guided_opt[
            (
                "gradflow_reach_init",
                1.0,
                "the inital 'reach' parameter in geomloss setting",
            )
        ]
        update_gradflow_reach_by_raito = gradflow_guided_opt[
            (
                "update_gradflow_reach_by_raito",
                0.8,
                "the raito that updates the 'reach' parameter in geomloss setting",
            )
        ]
        gradflow_reach_min = gradflow_guided_opt[
            (
                "gradflow_reach_min",
                1.0,
                "the minium value of the 'reach' parameter in geomloss setting",
            )
        ]

        pair_shape_transformer_obj = gradflow_guided_opt[
            (
                "pair_shape_transformer_obj",
                "",
                "shape pair transformer before put into gradient guidance",
            )
        ]
        gradflow_mode = gradflow_guided_opt[
            ("mode", "grad_forward", "grad_forward or ot_mapping")
        ]

        post_kernel_obj = gradflow_guided_opt[
            ("post_kernel_obj", "", "shape interpolator")
        ]
        post_kernel_obj = obj_factory(post_kernel_obj) if post_kernel_obj else None

        if (
            self.global_iter % self.update_gradflow_every_n_step == 0
            or len(self.gradflow_guided_buffer) == 0
        ):
            cur_iter = self.global_iter / self.update_gradflow_every_n_step
            if pair_shape_transformer_obj:
                pair_shape_transformer = obj_factory(pair_shape_transformer_obj)
                flowed, target = pair_shape_transformer(flowed, target, cur_iter)
            n_update = self.global_iter.item() / self.update_gradflow_every_n_step
            cur_blur = max(
                gradflow_blur_init * (update_gradflow_blur_by_raito ** n_update),
                gradflow_blur_min,
            )
            if gradflow_reach_init > 0:
                cur_reach = max(
                    gradflow_reach_init * (update_gradflow_reach_by_raito ** n_update),
                    gradflow_reach_min,
                )
            else:
                cur_reach = None
            geomloss_setting = deepcopy(self.opt["gradflow_guided"]["geomloss"])
            geomloss_setting["geom_obj"] = geomloss_setting["geom_obj"].replace(
                "blurplaceholder", str(cur_blur)
            )
            geomloss_setting["geom_obj"] = geomloss_setting["geom_obj"].replace(
                "reachplaceholder", str(cur_reach) if cur_reach else "None"
            )
            geomloss_setting["mode"] = "soft"
            geomloss_setting["attr"] = "pointfea"
            guide_fn = gradient_flow_guide(gradflow_mode)
            gradflowed, _ = guide_fn(flowed, target, geomloss_setting, self.local_iter)
            if post_kernel_obj is not None:
                disp = gradflowed.points - flowed.points
                flowed_points = flowed.points
                smoothed_disp = post_kernel_obj(
                    flowed_points, flowed_points, disp, flowed.weights
                )
                gradflowed.points = flowed_points + smoothed_disp
            gradflowed.points = gradflowed.points.detach()
            self.gradflow_guided_buffer["gradflowed"] = gradflowed
            # self.debug(flowed, gradflowed, target)
        return flowed, self.gradflow_guided_buffer["gradflowed"]

    def extract_point_fea(self, flowed, target, iter=-1):
        flowed.pointfea = flowed.points
        target.pointfea = target.points
        return flowed, target

    def extract_fea(self, flowed, target):
        """LDDMMM support feature extraction"""
        if not self.pair_feature_extractor:
            return self.extract_point_fea(flowed, target, self.global_iter)
        else:
            return self.pair_feature_extractor(flowed, target, self.global_iter)

    def forward(self, shape_pair):
        if shape_pair.dense_mode:
            shape_pair = self.shooting(shape_pair)
            shape_pair.infer_flowed()
        else:
            shape_pair = self.flow(shape_pair)
        flowed, target = self.extract_fea(shape_pair.flowed, shape_pair.target)
        if self.use_gradflow_guided:
            flowed, target = self.wasserstein_gradient_flow_guidence(flowed, target)
        sim_loss = self.sim_loss_fn(flowed, target)
        reg_loss = self.reg_loss_fn(
            shape_pair.reg_param, shape_pair.get_control_points()
        )
        sim_factor, reg_factor = self.get_factor()
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        if self.local_iter % self.print_step == 0:
            print(
                "{} th step, sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}".format(
                    self.local_iter.item(),
                    sim_loss.item(),
                    reg_loss.item(),
                    sim_factor,
                    reg_factor,
                )
            )
            # if self.local_iter%10==0:
            #     self.debug(shape_pair.source, shape_pair.flowed, shape_pair.target, alias="{}_{}".format(shape_pair.pair_name[0], int(self.local_iter.item())))
        loss = sim_loss + reg_loss
        self.local_iter += 1
        self.global_iter += 1
        return loss

    def model_eval(self, shape_pair, batch_info=None):
        """
        for  deep approach, we assume the source points = control points
        :param shape_pair:
        :param batch_info:
        :return:
        """
        return opt_flow_model_eval(
            shape_pair,
            model=self,
            batch_info=batch_info,
            geom_loss_opt_for_eval=self.geom_loss_opt_for_eval,
            external_evaluate_metric=self.external_evaluate_metric,
        )

    def debug(self, source, flowed, target, alias="debug"):
        from robot.utils.visualizer import (
            visualize_multi_point,
            visualize_point_pair_overlap,
        )
        import os

        camera_pos = [
            (-4.924379645467042, 2.17374925796456, 1.5003730890759344),
            (0.0, 0.0, 0.0),
            (0.40133888001174545, 0.31574165540339943, 0.8597873634998591),
        ]
        #
        # visualize_point_pair_overlap(flowed.points, target.points,
        #                          flowed.weights,target.weights,
        #                              camera_pos=camera_pos,
        #                          title1="flowed",title2="target", rgb_on=False)
        camera_toy_pos = [
            (-5.5034147913360005, 5.520778675107747, 10.458100554989956),
            (0.0, 0.0, 0.0),
            (0.28747814320872545, 0.901163583812316, -0.32443876523591686),
        ]
        visualize_multi_point(
            [source.points, flowed.points, target.points],
            [source.points, source.points, target.points],
            ["source", "flowed", "target"],
            # gradflowed_disp,
            camera_pos=camera_toy_pos,
            show=True,
            saving_capture_path=os.path.join(self.record_path, alias),
        )
        # visualize_point_pair_overlap(flowed.points, target.points,
        #                              source.points.squeeze(), target.points.squeeze(),
        #                              title1="flowed", title2="target", rgb_on=True)
