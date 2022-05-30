import torch
import torch.nn as nn
from robot.global_variable import Shape

from robot.modules_reg.opt_flowed_eval import opt_flow_model_eval
from robot.modules_reg.module_probreg import ProbReg
from robot.utils.obj_factory import obj_factory


class ProRegOPT(nn.Module):
    """
    an interface for the third party package
    https://github.com/neka-nat/probreg
    """

    def __init__(self, opt):
        super(ProRegOPT, self).__init__()
        self.opt = opt
        self.probreg_module = ProbReg(self.opt)
        self.call_thirdparty_package = True
        self.sim_loss_fn = lambda x, y: torch.tensor(-1)
        self.reg_loss_fn = lambda x: torch.tensor(-1)
        interpolator_obj = self.opt[
            (
                "interpolator_obj",
                "point_interpolator.nadwat_kernel_interpolator(scale=0.1, exp_order=2)",
                "shape interpolator in multi-scale solver",
            )
        ]
        self.interp_kernel = obj_factory(interpolator_obj)
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
        self.register_buffer(
            "local_iter", torch.Tensor([0])
        )  # iteration record in single scale
        self.register_buffer(
            "global_iter", torch.Tensor([0])
        )  # iteration record in multi-scale
        self.print_step = self.opt[
            ("print_step", 10, "print every n iteration, disabled in teaser")
        ]
        self.drift_buffer = {}

    def clean(self):
        self.local_iter = self.local_iter * 0
        self.global_iter = self.global_iter * 0

    def set_record_path(self, record_path):
        self.record_path = record_path

    def init_reg_param(self, shape_pair):
        reg_param = shape_pair.source.points.clone()
        reg_param.requires_grad_()
        shape_pair.set_reg_param(reg_param)

    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.local_iter = self.local_iter * 0

    def flow(self, shape_pair):
        flowed_control_points = shape_pair.flowed_control_points
        toflow_points = shape_pair.toflow.points
        control_points = shape_pair.control_points
        control_weights = shape_pair.control_weights
        flowed_points = self.interp_kernel(
            toflow_points, control_points, flowed_control_points, control_weights
        )
        flowed = Shape()
        flowed.set_data_with_refer_to(flowed_points, shape_pair.source)
        shape_pair.set_flowed(flowed)
        return shape_pair

    def update_reg_param_from_low_scale_to_high_scale(
        self, shape_pair_low, shape_pair_high
    ):
        control_points_high = shape_pair_high.get_control_points()
        control_points_low = shape_pair_low.get_control_points()
        control_weights_low = shape_pair_low.control_weights
        reg_param_low = shape_pair_low.reg_param
        reg_param_high = self.interp_kernel(
            control_points_high, control_points_low, reg_param_low, control_weights_low
        )
        reg_param_high.detach_()
        reg_param_high.requires_grad_()
        shape_pair_high.set_reg_param(reg_param_high)
        return shape_pair_high

    def get_factor(self):
        return 1, 1

    def forward(self, shape_pair):
        """
        for Probreg tasks, during optimization, the registration only works for dense mode
        :param shape_pair:
        :return:
        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        assert shape_pair.dense_mode == True
        shape_pair.set_control_points(
            shape_pair.source.points.clone(), shape_pair.source.weights
        )
        flowed_points = self.probreg_module(
            shape_pair.source, shape_pair.target, return_tranform_param=False
        )
        end.record()
        torch.cuda.synchronize()
        print(
            "{}, it takes {} ms".format(shape_pair.pair_name, start.elapsed_time(end))
        )

        shape_pair.flowed_control_points = flowed_points.detach().clone()
        flowed = Shape().set_data_with_refer_to(flowed_points, shape_pair.source)
        shape_pair.flowed = flowed
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        reg_loss = self.reg_loss_fn(flowed_points)
        sim_factor, reg_factor = self.get_factor()
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        if self.local_iter % 10 == 0:
            print(
                "{} th step, sim_loss is {}, reg_loss is {}, sim_factor is {}, reg_factor is {}".format(
                    self.local_iter.item(),
                    sim_loss.item(),
                    reg_loss.item(),
                    sim_factor,
                    reg_factor,
                )
            )
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
