import torch
import torch.nn as nn

from robot.modules_reg.opt_flowed_eval import opt_flow_model_eval
from robot.modules_reg.module_teaser import Teaser

# from robot.modules_reg.probreg_module import ProbReg
from robot.modules_reg.module_gradflow_prealign import GradFlowPreAlign
from robot.global_variable import Shape
from robot.metrics.reg_losses import Loss
from robot.utils.utils import sigmoid_decay
from robot.utils.obj_factory import obj_factory
from robot.utils.utils import timming


class PrealignOPT(nn.Module):
    def __init__(self, opt):
        super(PrealignOPT, self).__init__()
        self.opt = opt
        self.module_type = self.opt[
            ("module_type", "gradflow_prealign", "gradflow_prealign")
        ]
        assert self.module_type in ["probreg", "teaser", "gradflow_prealign"]
        # here we treat gradflow_prealign as a self-completed module for affine optimization
        self.thirdparty_package = ["probreg", "teaser", "gradflow_prealign"]
        module_dict = {
            "teaser": Teaser,
            "gradflow_prealign": GradFlowPreAlign,
        }  # "probreg":ProbReg,
        self.prealign_module = module_dict[self.module_type](
            self.opt[(self.module_type, {}, "settings for prealign module")]
        )
        self.prealign_module.set_mode("prealign")
        sim_loss_opt = opt[("sim_loss", {}, "settings for sim_loss_opt")]
        self.call_thirdparty_package = self.module_type in self.thirdparty_package
        self.sim_loss_fn = (
            Loss(sim_loss_opt)
            if not self.call_thirdparty_package
            else lambda x, y: torch.tensor(-1)
        )
        self.reg_loss_fn = self.compute_regularization
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
        batch, dim, device = (
            shape_pair.source.nbatch,
            shape_pair.source.dimension,
            shape_pair.source.points.device,
        )
        reg_param = torch.zeros([batch, dim + 1, dim], device=device).normal_(0, 1e-7)
        for i in range(dim):
            reg_param[:, i, i] = 1.0
        reg_param.requires_grad_()
        self.identity_param = reg_param.clone().detach()
        shape_pair.set_reg_param(reg_param)

    def set_loss_fn(self, loss_fn):
        self.sim_loss_fn = loss_fn

    def reset(self):
        self.local_iter = self.local_iter * 0

    def apply_prealign_transform(self, prealign_param, points):
        """

        :param prealign_param: Bx(D+1)xD: BxDxD transfrom matrix and Bx1xD translation
        :param points: BxNxD
        :return:
        """
        dim = points.shape[-1]
        points = torch.bmm(points, prealign_param[:, :dim, :])
        points = prealign_param[:, dim:, :].contiguous() + points
        return points

    def prealign(self, shape_pair):
        source = shape_pair.source
        target = shape_pair.target
        control_points = shape_pair.get_control_points()
        prealign_param = self.prealign_module(source, target, shape_pair.reg_param)
        shape_pair.reg_param = prealign_param
        flowed_control_points = self.apply_prealign_transform(
            prealign_param, control_points
        )
        shape_pair.set_flowed_control_points(flowed_control_points)
        return shape_pair, prealign_param

    def flow(self, shape_pair):
        prealign_params = shape_pair.reg_param
        toflow_points = shape_pair.get_toflow_points()
        flowed_points = self.apply_prealign_transform(prealign_params, toflow_points)
        flowed = Shape()
        flowed.set_data_with_refer_to(flowed_points, shape_pair.toflow)
        shape_pair.set_flowed(flowed)
        return shape_pair

    def update_reg_param_from_low_scale_to_high_scale(
        self, shape_pair_low, shape_pair_high
    ):
        shape_pair_high.set_reg_param(shape_pair_low.reg_param.detach())
        return shape_pair_high

    def compute_regularization(self, prealign_params):
        if self.call_thirdparty_package:
            return torch.tensor(-1)
        else:
            return torch.norm(prealign_params - self.identity_param)

    def get_factor(self):
        """
        get the regularizer factor according to training strategy

        :return:
        """
        if self.call_thirdparty_package:
            return 1, 1
        sim_factor = 10
        reg_factor_init = 1  # self.initial_reg_factor
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

    def forward(self, shape_pair):
        """
        for affine tasks, during optimization, there is no difference between toflow points and control points
        the similarity is computed based on the control points
        :param shape_pair:
        :return:
        """
        shape_pair, prealign_param = timming(self.prealign)(shape_pair)
        flowed_has_inferred = shape_pair.infer_flowed()
        shape_pair = self.flow(shape_pair) if not flowed_has_inferred else shape_pair
        sim_loss = self.sim_loss_fn(shape_pair.flowed, shape_pair.target)
        reg_loss = self.reg_loss_fn(prealign_param)
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
