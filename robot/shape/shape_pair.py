"""
a ShapePair object records the registration parameters of the source shape to the target shape
"""
import torch
from robot.global_variable import Shape


class ShapePair:
    """
    1. During training, the control points are flowed to get flowed control points.

    if the dense mode is true, which refers the control points is the same as the source,
    the flowed then can be directly created from the flowed control points
    (toflow and flowed are implicitly set)

    Examples:
        >>> shape_pair = ShapePair(dense_mode=True)
        >>> shape_pair.set_source_and_target(source, target)
        >>> do_registration(shape_pair)


    if the dense mode is false, which refers the control points are different from the toflow points,
    , (the toflow by default is set as the source points), an additional forward on the toflow is needed to get the flowed points
    The similarity measure will be computed between the flowed and the target

    Examples:
        >>> shape_pair = ShapePair(dense_mode=False)
        >>> shape_pair.set_source_and_target(source, target)
        >>> shape_pair.set_control_points(contorl_points)
        >>> do_registration(shape_pair)
        >>> do_flow(shape_pair)


    2. During external inference, e.g. given ambient points, in this case, assume the reg_param
    and the control points have already known. toflow need to be externally initialized as the given ambient points .
    the dense mode is set to false,  The flowed ambient points can be return after the inference.

    Examples:
        >>> ....
        >>> shape_pair.set_toflow(toflow)
        >>> do_flow(shape_pair)
    """

    def __init__(self, dense_mode=True):
        self.source = None
        self.target = None
        self.toflow = None
        self.flowed = None
        self.reg_param = None
        self.control_points = None
        self.control_weights = None
        self.flowed_control_points = None
        self.dense_mode = dense_mode
        self.pair_name = None
        self.extra_info = {}
        self.shape_type = None
        self.dimension = None
        self.nbatch = -1

    def set_source_and_target(self, source, target):
        self.source = source
        self.target = target
        self.toflow = source
        self.shape_type = self.source.type
        self.nbatch = source.nbatch
        self.dimension = source.dimension

    def set_pair_name(self, pair_name):
        self.pair_name = pair_name

    def get_pair_name(self):
        if self.pair_name is not None:
            return self.pair_name
        if len(self.source.name_list) and len(self.target.name_list):
            self.pair_name = [
                s_name + "_" + t_name
                for s_name, t_name in zip(self.source.name_list, self.target.name_list)
            ]
            return self.pair_name
        return "not_given"

    def set_toflow(self, toflow):
        self.toflow = toflow
        self.dense_mode = False

    def set_flowed(self, flowed):
        self.flowed = flowed

    def set_reg_param(self, reg_param):
        self.reg_param = reg_param

    def set_extra_info(self, value, name):
        self.extra_info.update({name: value})

    def set_flowed_control_points(self, flowed_control_points):
        self.flowed_control_points = flowed_control_points

    def infer_flowed(self):
        if self.dense_mode:
            self.flowed = Shape()
            self.flowed.set_data_with_refer_to(self.flowed_control_points, self.toflow)
            return True
        else:
            return False

    def set_control_points(self, control_points, control_weights=None):
        self.control_points = control_points
        if control_weights is None and self.control_weights is None:
            control_weights = torch.ones(
                control_points.shape[0], control_points.shape[1], 1
            )
            control_weights = control_weights / control_points.shape[1]
            control_weights = control_weights.to(control_points.device)
        if control_weights is not None:
            self.control_weights = control_weights
        # self.control_points.requires_grad_()

    def get_control_points(self, detach=False):
        if self.control_points is None:
            self.control_points = self.source.points.clone()
            self.control_weights = self.source.weights
        return self.control_points if not detach else self.control_points.detach()

    def get_toflow_points(self, detach=False):
        return self.toflow.points if not detach else self.toflow.points.detach()

    def get_flowed_points(self, detach=False):
        return self.flowed.points if not detach else self.flowed.points.detach()
