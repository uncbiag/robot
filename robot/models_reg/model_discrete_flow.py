import os
from copy import deepcopy
import torch
import torch.nn as nn
from robot.global_variable import Shape
from robot.metrics.reg_losses import Loss, GeomDistance
from robot.modules_reg.opt_flowed_eval import opt_flow_model_eval
from robot.utils.obj_factory import obj_factory, partial_obj_factory
from robot.utils.utils import sigmoid_decay
from robot.modules_reg.module_gradient_flow import (
    gradient_flow_guide,
    wasserstein_barycenter_mapping,
)


class DiscreteFlowOPT(nn.Module):
    """
    flow the source via n step, in each step with the #current# source X get updated, the target Y is fixed
    In this class, we provide two approaches to solve this task:

    1. standard spline registration
    for each step a spline registration problem can be defined as
    p^* = argmin_p <p,Kp> + Sim(X+Kp,Y)

    where p is the momentum or registration parameter,
    K is the kernel metric
    an #user-defined# the optimizer is used to solve this problem
    with new  X = X+Kp


    2. gradient flow guided spline registration
    for each step, a rigid kernel regression problem is to be solved:
     p^* = argmin_p <p,Kp> + \lambda \|Kp-v\|^2
    where p is the momentum or registration parameter,  v is the velocity guidence solved from gradient flow defined by wasserstein distance
    this problem can be either solved via standard conjunction gradient solver
    or a fast approximate \tilde{v}=Kv, where K should be Nadaraya-Watson kernel



    """

    def __init__(self, opt):
        super(DiscreteFlowOPT, self).__init__()
        self.opt = opt
        self.record_path = None
        self.gradient_flow_mode = self.opt[
            (
                "gradient_flow_mode",
                False,
                "if true, work on gradient flow guided spline registration, otherwise standard spline registration",
            )
        ]
        self.drift_every_n_iter = self.opt[
            (
                "drift_every_n_iter",
                10,
                "if n is set bigger than -1, then after every n iteration, the current flowed shape is set as the source shape",
            )
        ]
        spline_kernel_obj = self.opt[
            (
                "spline_kernel_obj",
                "point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)",
                "shape interpolator in multi-scale solver",
            )
        ]
        self.spline_kernel = obj_factory(spline_kernel_obj)
        interp_kernel_obj = self.opt[
            (
                "interp_kernel_obj",
                "point_interpolator.NadWatIsoSpline(exp_order=2,kernel_scale=0.05)",
                "kernel for multi-scale interpolation",
            )
        ]
        self.interp_kernel = obj_factory(interp_kernel_obj)
        pair_feature_extractor_obj = self.opt[
            ("pair_feature_extractor_obj", "", "feature extraction function")
        ]
        self.pair_feature_extractor = (
            obj_factory(pair_feature_extractor_obj)
            if pair_feature_extractor_obj
            else None
        )
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.sim_loss_fn = Loss(sim_loss_opt)
        self.reg_loss_fn = self.regularization
        self.geom_loss_opt_for_eval = opt[
            (
                "geom_loss_opt_for_eval",
                {},
                "settings for sim_loss_opt, the sim_loss here is not used for optimization but for evaluation",
            )
        ]
        self.call_thirdparty_package = False
        self.register_buffer("local_iter", torch.Tensor([0]))
        self.register_buffer("global_iter", torch.Tensor([0]))
        self.print_step = self.opt[("print_step", 1, "print every n iteration")]
        self.running_result_visualize = self.opt[
            ("running_result_visualize", False, "visualize the intermid results")
        ]
        self.saving_running_result_visualize = self.opt[
            ("saving_running_result_visualize", False, "save the visualize results")
        ]
        external_evaluate_metric_obj = self.opt[
            ("external_evaluate_metric_obj", "", "external evaluate metric")
        ]
        self.external_evaluate_metric = (
            obj_factory(external_evaluate_metric_obj)
            if external_evaluate_metric_obj
            else None
        )
        self.drift_buffer = {}
        if self.gradient_flow_mode:
            print("in gradient flow mode, points drift every iteration")
            self.drift_every_n_iter = 1

    def set_record_path(self, record_path):
        self.record_path = record_path

    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.local_iter = self.local_iter * 0

    def clean(self):
        self.local_iter = self.local_iter * 0
        self.global_iter = self.global_iter * 0
        self.drift_buffer = {}

    def flow(self, shape_pair):
        """
        todo  for the rigid kernel regression, anistropic kernel interpolation is not supported yet
        the flow approach is designed for two proposes: interpolate the flowed points from control points / ambient space interpolation
        Attention:
        the flow function only get reasonable result if the source and target has similar topology
        If the topology difference between the source to target is large, this flow function is not recommended,
        to workaround, 1. use dense_mode in shape_pair  2. if use multi-scale solver, set the last scale to be -1 with enough iterations
        :param shape_pair:
        :return:
        """

        toflow_points = shape_pair.toflow.points
        control_points = shape_pair.control_points
        control_weights = shape_pair.control_weights

        moving_control_points = self.drift_buffer["moving_control_points"]
        interped_control_points_high = self.interp_kernel(
            toflow_points, control_points, moving_control_points, control_weights
        )
        self.spline_kernel.set_flow(True)
        interped_control_points_disp = self.spline_kernel(
            toflow_points, control_points, shape_pair.reg_param, control_weights
        )
        self.spline_kernel.set_flow(False)
        flowed_points = interped_control_points_high + interped_control_points_disp
        flowed = Shape()
        flowed.set_data_with_refer_to(flowed_points, shape_pair.source)
        shape_pair.set_flowed(flowed)
        return shape_pair
    #
    # def flow(self, shape_pair):
    #     """
    #
    #     :param shape_pair:
    #     :return:
    #     """
    #
    #     toflow_points = shape_pair.toflow.points
    #     control_points = shape_pair.control_points
    #     control_weights = shape_pair.control_weights
    #     flowed_control_points = shape_pair.flowed_control_points
    #
    #     flowed_points = self.interp_kernel(
    #         toflow_points, control_points, flowed_control_points, control_weights
    #     )
    #     flowed = Shape()
    #     flowed.set_data_with_refer_to(flowed_points, shape_pair.source)
    #     shape_pair.set_flowed(flowed)
    #     return shape_pair

    def regularization(self, sm_flow, flow):
        dist = sm_flow * flow
        dist = dist.mean(2).mean(1)

        return dist

    def get_factor(self):
        """
        get the regularizer factor according to training strategy

        :return:
        """
        sim_factor = self.opt[("sim_factor", 100, "similarity factor")]
        init_reg_factor = self.opt[("init_reg_factor", 100, "regularization factor")]
        min_reg_factor = self.opt[
            ("min_reg_factor", init_reg_factor / 10, "min_reg_factor")
        ]
        decay_factor = self.opt[("decay_factor", 8, "decay_factor")]
        sim_factor = sim_factor
        reg_factor_init = init_reg_factor  # self.initial_reg_factor
        static_epoch = 10
        min_threshold = min_reg_factor
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
        """todo   do interpolation on disp"""
        # assert self.drift_every_n_iter==-1, "the drift mode is not fully tested in multi-scale mode, disabled for now"
        control_points_high = shape_pair_high.get_control_points()
        control_points_low = shape_pair_low.get_control_points()
        control_weights_low = shape_pair_low.control_weights
        moving_control_points_low = (
            control_points_low
            if len(self.drift_buffer) == 0
            else self.drift_buffer["moving_control_points"]
        )
        interped_control_points_high = self.interp_kernel(
            control_points_high,
            control_points_low,
            moving_control_points_low,
            control_weights_low,
        )
        self.spline_kernel.set_interp(
            True
        )  # the spline kernel interpolates the displacement
        interped_control_points_disp_high = self.spline_kernel(
            control_points_high,
            control_points_low,
            shape_pair_low.reg_param,
            control_weights_low,
        )
        self.spline_kernel.set_interp(False)
        flowed_control_points_high = (
            interped_control_points_high + interped_control_points_disp_high
        )
        shape_pair_high.set_control_points(control_points_high)
        self.init_reg_param(shape_pair_high)
        self.drift_buffer[
            "moving_control_points"
        ] = flowed_control_points_high.detach().clone()
        return shape_pair_high

    # def update_reg_param_from_low_scale_to_high_scale(
    #     self, shape_pair_low, shape_pair_high
    # ):
    #     """todo   do interpolation on disp"""
    #     # assert self.drift_every_n_iter==-1, "the drift mode is not fully tested in multi-scale mode, disabled for now"
    #     control_points_high = shape_pair_high.get_control_points()
    #     control_points_low = shape_pair_low.get_control_points()
    #     control_weights_low = shape_pair_low.control_weights
    #     flowed_control_points_low = shape_pair_low.flowed_control_points
    #     flowed_control_points_high = self.interp_kernel(
    #         control_points_high,
    #         control_points_low,
    #         flowed_control_points_low,
    #         control_weights_low,
    #     )
    #     shape_pair_high.set_control_points(control_points_high)
    #     self.init_reg_param(shape_pair_high)
    #     self.drift_buffer[
    #         "moving_control_points"
    #     ] = flowed_control_points_high.detach().clone()
    #     return shape_pair_high

    def drift(self, shape_pair):
        """
        Under drift strategy, the source updates during the optimization,  we introduce a "moving" to record the #current# source
        """
        flowed = shape_pair.flowed if self.local_iter > 0 else shape_pair.source
        # for multi scale solver, we need to load the moving_control_points from the last scale
        flowed_control_points = (
            self.drift_buffer["moving_control_points"]
            if self.drift_buffer
            else shape_pair.control_points
        )
        flowed_control_points = (
            shape_pair.flowed_control_points
            if self.local_iter > 0
            else flowed_control_points
        )
        if self.local_iter % self.drift_every_n_iter == 0:
            print(
                "drift at the {} th step, the moving (current source) is updated".format(
                    self.local_iter.item()
                )
            )
            moving = Shape()
            moving.set_data(
                points=flowed.points.detach().clone(),
                weights=flowed.weights.detach().clone(),
                pointfea=flowed.pointfea,
            )
            self.drift_buffer["moving"] = moving
            self.drift_buffer[
                "moving_control_points"
            ] = flowed_control_points.detach().clone()
            self.reset_reg_param(shape_pair)

    def init_reg_param(self, shape_pair):
        reg_param = torch.zeros_like(shape_pair.get_control_points()).normal_(0, 1e-7)
        reg_param.requires_grad_()
        shape_pair.set_reg_param(reg_param)
        return shape_pair

    def reset_reg_param(self, shape_pair):
        shape_pair.reg_param.data[:] = torch.zeros_like(shape_pair.reg_param).normal_(
            0, 1e-7
        )

    def extract_point_fea(self, flowed, target, iter=-1):
        flowed.pointfea = flowed.points
        target.pointfea = target.points
        return flowed, target

    def extract_fea(self, flowed, target):
        """DiscreteFlowOPT supports feature extraction"""
        if not self.pair_feature_extractor:
            return self.extract_point_fea(flowed, target, self.global_iter)
        else:
            return self.pair_feature_extractor(flowed, target, self.global_iter)

    def standard_spline_forward(self, shape_pair):
        """
        In this forward, the solver should be built either via build_multi_scale_solver / build_single_scale_general_solver
        reg_param here is the momentum p before kernel operation K,  \tilde{v} = Kp
        :param shape_pair:
        :return:
        """
        if self.drift_every_n_iter == -1:
            control_points = shape_pair.get_control_points()
        else:
            self.drift(shape_pair)
            control_points = self.drift_buffer["moving_control_points"]
        # todo check the behavior of spline kernel with shape_pair.control points as input
        smoothed_reg_param = self.spline_kernel(
            control_points,
            control_points,
            shape_pair.reg_param,
            shape_pair.control_weights,
        )
        flowed_control_points = control_points + smoothed_reg_param
        shape_pair.set_flowed_control_points(flowed_control_points)
        flowed_has_inferred = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        shape_pair.flowed, shape_pair.target = self.extract_fea(
            shape_pair.flowed, shape_pair.target
        )
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        reg_loss = self.reg_loss_fn(smoothed_reg_param, shape_pair.reg_param)
        sim_factor, reg_factor = self.get_factor()
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        if self.local_iter % 30 == 0:
            print(
                "{} th step, sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}".format(
                    self.local_iter.item(),
                    sim_loss.item(),
                    reg_loss.item(),
                    sim_factor,
                    reg_factor,
                )
            )
            if self.running_result_visualize or self.saving_running_result_visualize:
                self.visualize_discreteflow(shape_pair)
        loss = sim_loss + reg_loss
        self.local_iter += 1
        self.global_iter += 1
        return loss


    def NN_guidence(self,flowed, target):
        from robot.utils.knn_utils import NN
        nn_guided_opt = deepcopy(
            self.opt[("nn_guided", {}, "settings for nearest point guidance")]
        )
        post_kernel_obj = nn_guided_opt[
            ("post_kernel_obj", "", "shape interpolator")
        ]
        post_kernel = obj_factory(post_kernel_obj) if post_kernel_obj else None
        nn_points,_ = NN(return_value=False, return_pos=True)(flowed.points, target.points)
        nnflowed = Shape().set_data_with_refer_to(nn_points, flowed)
        if post_kernel is not None:
            disp = nnflowed.points - flowed.points
            flowed_points = flowed.points
            smoothed_disp = post_kernel(
                flowed_points, flowed_points, disp, flowed.weights
            )
            nnflowed.points = flowed_points + smoothed_disp
        nnflowed_disp = (nnflowed.points - flowed.points).detach()
        if self.running_result_visualize or self.saving_running_result_visualize:
            self.visualize_gradflow(flowed, nnflowed_disp, target, None)
        return nnflowed, nnflowed_disp

    # mapped_position = NNInterpolater()(shape_pair.flowed.points, shape_pair.target.points,shape_pair.target.points)
    # wasserstein_dist =  torch.Tensor([-1]*shape_pair.nbatch)

    def wasserstein_gradient_flow_guidence(self, flowed, target):
        """
        wassersten gradient flow has a reasonable behavior only when set self.pair_feature_extractor = None
        """
        gradflow_guided_opt = deepcopy(
            self.opt[("gradflow_guided", {}, "settings for gradflow guidance")]
        )
        gradflow_guided_opt.print_settings_off()
        post_kernel_obj = gradflow_guided_opt[
            ("post_kernel_obj", "", "shape interpolator")
        ]
        post_kernel = obj_factory(post_kernel_obj) if post_kernel_obj else None
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
                0.3,
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
        n_update = self.global_iter.item()
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
        geomloss_setting.print_settings_off()

        geomloss_setting["geom_obj"] = geomloss_setting["geom_obj"].replace(
            "blurplaceholder", str(cur_blur)
        )
        geomloss_setting["geom_obj"] = geomloss_setting["geom_obj"].replace(
            "reachplaceholder", str(cur_reach) if cur_reach else "None"
        )
        print(geomloss_setting["geom_obj"])
        guide_fn = gradient_flow_guide(gradflow_mode)
        flowed_weights_cp = flowed.weights
        if pair_shape_transformer_obj:
            pair_shape_transformer = obj_factory(pair_shape_transformer_obj)
            flowed, target = pair_shape_transformer(flowed, target, self.local_iter)
        gradflowed, weight_map_ratio = guide_fn(
            flowed, target, geomloss_setting, self.local_iter
        )
        gradflowed.points = gradflowed.points.detach()
        if post_kernel is not None:
            disp = gradflowed.points - flowed.points
            flowed_points = flowed.points
            smoothed_disp = post_kernel(
                flowed_points, flowed_points, disp, flowed.weights
            )
            gradflowed.points = flowed_points + smoothed_disp
        gradflowed_disp = (gradflowed.points - flowed.points).detach()
        if self.running_result_visualize or self.saving_running_result_visualize:
            self.visualize_gradflow(flowed, gradflowed_disp, target, None)
        gradflowed.weights = flowed_weights_cp
        return gradflowed, gradflowed_disp

    def gradflow_spline_foward(self, shape_pair):
        """
        In this forward, the solver should be built via build_single_scale_model_embedded_solver
         gradient flow has a reasonable behavior only when set self.pair_feature_extractor = None
         reg_param here is the momentum p before kernel operation K,  \tilde{v} = Kp
        :param shape_pair:
        :return:
        """
        self.drift(shape_pair)
        moving, control_points = (
            self.drift_buffer["moving"],
            self.drift_buffer["moving_control_points"],
        )
        if self.local_iter == 0:
            moving, shape_pair.target = self.extract_fea(moving, shape_pair.target)
        grad_flowed, gradflowed_disp = self.wasserstein_gradient_flow_guidence(  #self.NN_guidence( #self.wasserstein_gradient_flow_guidence(
            moving, shape_pair.target
        )
        gradflowed_disp = gradflowed_disp  # if self.local_iter<3 else gradflowed_disp/(self.local_iter.item())# todo control the step size, should be set as a controlable parameter
        if not shape_pair.dense_mode:
            smoothed_reg_param = self.spline_kernel(
                control_points, moving.points, gradflowed_disp, moving.weights
            )
        else:
            smoothed_reg_param = gradflowed_disp
        flowed_control_points = control_points + smoothed_reg_param
        shape_pair.set_flowed_control_points(flowed_control_points)
        # flowed_has_inferred = shape_pair.infer_flowed()
        # shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        shape_pair.flowed = grad_flowed

        # the following loss are not used for param update, only used for result analysis
        # shape_pair.flowed, shape_pair.target = self.extract_fea(shape_pair.flowed, shape_pair.target)
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        reg_loss = self.reg_loss_fn(smoothed_reg_param, shape_pair.reg_param)
        sim_factor, reg_factor = 1, 1
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        if self.local_iter % 1 == 0:
            print(
                "{} th step, sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}".format(
                    self.local_iter.item(),
                    sim_loss.mean().item(),
                    reg_loss.mean().item(),
                    sim_factor,
                    reg_factor,
                )
            )
            if self.running_result_visualize or self.saving_running_result_visualize:
                self.visualize_discreteflow(shape_pair)
        loss = sim_loss + reg_loss
        self.local_iter += 1
        self.global_iter += 1
        return loss.sum()

    def forward(self, shape_pair):
        if self.gradient_flow_mode:
            return self.gradflow_spline_foward(shape_pair)
        else:
            return self.standard_spline_forward(shape_pair)

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

    def visualize_discreteflow(self, shape_pair):
        from robot.utils.visualizer import (
            visualize_point_pair_overlap,
            visualize_source_flowed_target_overlap,
            default_plot
        )
        from robot.experiments.datasets.lung.lung_data_analysis import (
            flowed_weight_transform,
            target_weight_transform,
        )

        saving_capture_path = (
            None
            if not self.saving_running_result_visualize
            else os.path.join(self.record_path, "debugging")
        )

        source, flowed, target = shape_pair.source, shape_pair.flowed, shape_pair.target
        visualize_source_flowed_target_overlap(
            source.points,
            flowed.points,
            target.points,
            source.points,
            source.points,
            target.points,
            "source",
            "flowed",
            "target",
            source_plot_func=default_plot(cmap="magma", rgb=False),
            flowed_plot_func=default_plot(cmap="magma", rgb=False),
            target_plot_func=default_plot(cmap="viridis", rgb=False),
            flow=flowed.points-source.points,
            opacity=(1.0, 1.0, 1.0),
            add_bg_contrast=True,
            show=self.running_result_visualize,
            saving_capture_path=saving_capture_path,
        )
        camera_pos = None
        # camera_pos =[(-4.924379645467042, 2.17374925796456, 1.5003730890759344),(0.0, 0.0, 0.0),(0.40133888001174545, 0.31574165540339943, 0.8597873634998591)]
        # visualize_source_flowed_target_overlap(source.points, flowed.points, target.points,
        #                                        source.weights, flowed.weights, target.weights,
        #                                        "cur_source", "gradflowed", "target",
        #                                        # gradflowed_disp,
        #                                        rgb_on=[False, False, False],
        #                                        add_bg_contrast=False,
        #                                        camera_pos=camera_pos,
        #                                        show=self.running_result_visualize,
        #                                        saving_capture_path=saving_capture_path)
        if saving_capture_path:
            os.makedirs(saving_capture_path, exist_ok=True)
            saving_capture_path = os.path.join(
                saving_capture_path,
                "discreteflow_iter_{}".format(self.local_iter.item()) + ".png",
            )

    def visualize_gradflow(self, flowed, gradflowed_disp, target, mapped_mass_ratio):
        from robot.utils.visualizer import visualize_source_flowed_target_overlap, default_plot

        saving_capture_path = (
            None
            if not self.saving_running_result_visualize
            else os.path.join(self.record_path)
        )
        if saving_capture_path:
            os.makedirs(saving_capture_path, exist_ok=True)
            saving_capture_path = os.path.join(
                saving_capture_path,
                "gradflow_iter_{}".format(self.local_iter.item()) + ".png",
            )
        camera_pos = None

        # camera_pos =[(-4.924379645467042, 2.17374925796456, 1.5003730890759344),(0.0, 0.0, 0.0),(0.40133888001174545, 0.31574165540339943, 0.8597873634998591)]

        visualize_source_flowed_target_overlap(
            flowed.points,
            flowed.points + gradflowed_disp,
            target.points,
            flowed.points,
            flowed.points,
            target.points,
            "cur_source",
            "gradflowed",
            "target",
            source_plot_func=default_plot(cmap="magma", rgb=False),
            flowed_plot_func=default_plot(cmap="magma", rgb=False),
            target_plot_func=default_plot(cmap="viridis", rgb=False),
            flow=gradflowed_disp,
            opacity=(1.0,1.0,1.0),
            add_bg_contrast=True,
            camera_pos=camera_pos,
            show=self.running_result_visualize,
            saving_capture_path=saving_capture_path,

        )
