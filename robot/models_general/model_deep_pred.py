from robot.modules_general.module_deep_landmark import *
from robot.utils.utils import sigmoid_decay

DEEP_PREDICTOR = {"pointconv_landmark_predictor": PointConvLandmarkPredictor}
DEEP_LOSS = {"deep_landmark_loss": DeepLandmarkPredictorLoss}


class DeepPredictor(nn.Module):
    """
    In this class, a deep feature predictor is trained,
    the synth data is used for training
    additionally, a spline model is included for evaluation
    """

    def __init__(self, opt):
        super(DeepPredictor, self).__init__()
        self.opt = opt
        create_shape_from_data_dict = opt[
            (
                "create_shape_from_data_dict",
                "shape_pair_utils.create_shape_from_data_dict()",
                "generator func",
            )
        ]
        self.create_shape_from_data_dict = obj_factory(create_shape_from_data_dict)
        decompose_shape_into_dict = opt[
            (
                "decompose_shape_into_dict",
                "shape_pair_utils.decompose_shape_into_dict()",
                "decompose shape into dict",
            )
        ]
        self.decompose_shape_into_dict = obj_factory(decompose_shape_into_dict)

        predictor = self.opt[
            ("predictor", "pointnet2_predictor", "name of deep feature predictor")
        ]
        loss_name = self.opt[("deep_loss", "loss_name", "name of loss")]
        self.deep_predictor = DEEP_PREDICTOR[predictor](
            self.opt[predictor, {}, "settings for the deep predictor"]
        )
        self.loss = DEEP_LOSS[loss_name](
            self.opt[
                loss_name, {}, "settings for the deep registration parameter generator"
            ]
        )
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

    def model_eval(self, input_data, batch_info=None):
        """
        for  deep approach
        :param shape:
        :param batch_info:
        :return:
        """
        loss, shape_data_dict = self.forward(input_data, batch_info)
        shape = self.create_shape_from_data_dict(shape_data_dict)
        metrics = {"score": [_loss.item() for _loss in loss]}
        if self.external_evaluate_metric is not None:
            additional_param = {"model": self}
            self.external_evaluate_metric(
                metrics, shape, batch_info, additional_param=additional_param, alias=""
            )
        return metrics, self.decompose_shape_into_dict(shape)

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
        input_shape = self.create_shape_from_data_dict(input_data)
        output_shape, additional_param = self.deep_predictor(input_shape, batch_info)
        sim_loss, reg_loss = self.loss(
            input_shape, output_shape, additional_param, batch_info
        )
        self.buffer["sim_loss"] = sim_loss.detach()
        self.buffer["reg_loss"] = reg_loss.detach()
        sim_factor, reg_factor = self.get_factor()
        sim_loss = sim_loss * sim_factor
        reg_loss = reg_loss * reg_factor
        loss = sim_loss + reg_loss
        if self.local_iter % self.print_step == 0:
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
        return loss, self.decompose_shape_into_dict(output_shape)
