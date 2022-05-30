from robot.utils.local_feature_extractor import *
from robot.utils.obj_factory import *
from robot.utils.module_parameters import ParameterDict
from robot.global_variable import ROBOT_PATH
import robot.modules_reg.module_deep_feature as deep_feature_module
import torch.nn as nn

# lung_pair_feature_extractor = pair_feature_extractor


def update_weight(weight_list, iter):
    max_weight = 1
    if iter > 2:
        weight_list = [
            min(max(0.1 * iter + weight, max_weight), 0.5) for weight in weight_list
        ]
    return weight_list


def lung_pair_feature_extractor(
    fea_type_list, weight_list=None, radius=0.01, std_normalize=True, include_pos=False
):
    fea_extractor = feature_extractor(fea_type_list, radius, std_normalize, include_pos)
    aniso_kernel_scale = 0.03
    get_anistropic_gamma_obj = "local_feature_extractor.compute_anisotropic_gamma_from_points(cov_sigma_scale=0.02,aniso_kernel_scale={},principle_weight=(2.,1.,1.),eigenvalue_min=0.1,iter_twice=True)".format(
        aniso_kernel_scale
    )
    get_anistropic_gamma = partial_obj_factory(get_anistropic_gamma_obj)

    def extract(flowed, target, iter=-1, flowed_gamma=None, target_gamma=None):
        if flowed_gamma is None:
            flowed_gamma = get_anistropic_gamma(flowed.points)
        if target_gamma is None:
            target_gamma = get_anistropic_gamma(target.points)
        cur_weight_list = (
            update_weight(weight_list, iter) if weight_list is not None else None
        )
        flowed.pointfea, mean, std, _ = fea_extractor(
            flowed.points,
            flowed.weights,
            weight_list=cur_weight_list,
            gamma=flowed_gamma,
            return_stats=True,
        )
        target.pointfea, _ = fea_extractor(
            target.points,
            target.weights,
            weight_list=weight_list,
            gamma=target_gamma,
            mean=mean,
            std=std,
        )
        return flowed, target

    return extract


class LungFeatureExtractor(object):
    def __init__(
        self,
        fea_type_list,
        weight_list=None,
        radius=0.01,
        get_anistropic_gamma_obj=None,
        std_normalize=True,
        include_pos=False,
        fixed=False,
    ):
        """

        :param fea_type_list:
        :param weight_list:
        :param radius:  used only for isotropic kernel
        :param get_anistropic_gamma_obj: a function object to compute anisotropic gamma
        :param std_normalize:
        :param include_pos:
        :param fixed:
        """
        self.fea_type_list = fea_type_list
        self.feature_extractor = feature_extractor(
            fea_type_list, radius, std_normalize, include_pos=False
        )
        self.weight_list = weight_list
        self.radius = radius
        self.get_anistropic_gamma = (
            None
            if get_anistropic_gamma_obj is None
            else partial_obj_factory(get_anistropic_gamma_obj)
        )
        self.std_normalize = std_normalize
        self.include_pos = include_pos
        self.fixed = fixed
        self.buffer = {"flowed_gamma": None, "target_gamma": None}

    # def update_weight(self, weight_list, iter):
    #     max_weight = 1
    #     if iter > 2:
    #         weight_list = [min(max(0.1 * iter + weight, max_weight), 0.5) for weight in weight_list]
    #     return weight_list

    def update_weight(self, weight_list, iter):
        return weight_list

    def __call__(self, flowed, target, iter=0):
        flowed_gamma = None
        target_gamma = None
        if (
            self.buffer["flowed_gamma"] is None
            and self.get_anistropic_gamma is not None
        ):
            flowed_gamma = self.get_anistropic_gamma(flowed.points)
        if (
            self.buffer["target_gamma"] is None
            and self.get_anistropic_gamma is not None
        ):
            target_gamma = self.get_anistropic_gamma(target.points)
        cur_weight_list = (
            update_weight(self.weight_list, iter)
            if self.weight_list is not None
            else None
        )
        if not self.fixed or iter == 0:
            flowed_pointfea, mean, std, _ = self.feature_extractor(
                flowed.points,
                flowed.weights,
                weight_list=cur_weight_list,
                gamma=flowed_gamma,
                return_stats=True,
            )
            target_pointfea, _ = self.feature_extractor(
                target.points,
                target.weights,
                weight_list=cur_weight_list,
                gamma=target_gamma,
                mean=mean,
                std=std,
            )
            self.buffer["flowed_pointfea"] = flowed_pointfea.detach()
            self.buffer["target_pointfea"] = target_pointfea.detach()
        elif self.fixed and iter > 0:
            flowed_pointfea = self.buffer["flowed_pointfea"]
            target_pointfea = self.buffer["target_pointfea"]
        if self.include_pos:
            flowed_pointfea = (
                torch.cat([flowed.points, flowed_pointfea], -1)
                if flowed_pointfea is not None
                else flowed.points
            )
            target_pointfea = (
                torch.cat([target.points, target_pointfea], -1)
                if target_pointfea is not None
                else target.points
            )

        flowed.pointfea = flowed_pointfea
        target.pointfea = target_pointfea
        return flowed, target


class LungDeepFeatureExtractor(nn.Module):
    def __init__(self, fixed=False):
        """
        :param fixed:
        """
        super(LungDeepFeatureExtractor, self).__init__()

        self.fixed = fixed
        deep_opt = ParameterDict()
        deep_opt[
            "local_pair_feature_extractor_obj"
        ] = "lung_feature_extractor.get_naive_lung_feature(include_xyz=False, weight_factor=1000)"
        deep_opt["input_channel"] = 1
        deep_opt["output_channel"] = 15
        deep_opt["param_shrink_factor"] = 1
        deep_opt["initial_npoints"] = 4096
        deep_opt["initial_radius"] = 0.001
        deep_opt["include_pos_in_final_feature"] = False
        deep_opt["use_aniso_kernel"] = False
        deep_opt[
            "pretrained_model_path"
        ] = os.path.join(ROBOT_PATH,"demos/pretrained_models/lung_model/pretrained_deep_feature")
        self.feature_extractor = deep_feature_module.PointConvFeaExtractor(deep_opt)
        self.buffer = {}

    def forward(self, flowed, target, iter=0):
        with torch.no_grad():
            if not self.fixed or iter == 0:
                flowed, target = self.feature_extractor(flowed, target)
                flowed.pointfea = flowed.pointfea.detach()
                target.pointfea = target.pointfea.detach()
                self.buffer["flowed_pointfea"] = flowed.pointfea
                self.buffer["target_pointfea"] = target.pointfea
            elif self.fixed and iter > 0:
                flowed_pointfea = self.buffer["flowed_pointfea"]
                target_pointfea = self.buffer["target_pointfea"]
                flowed.pointfea = flowed_pointfea
                target.pointfea = target_pointfea
        return flowed, target


def get_naive_lung_feature(include_xyz=True, weight_factor=10000):
    def get_fea(flowed, target):
        if include_xyz:
            flowed.pointfea = torch.cat(
                [flowed.points, flowed.weights * weight_factor], -1
            )
            target.pointfea = torch.cat(
                [target.points, target.weights * weight_factor], -1
            )
        else:
            flowed.pointfea = flowed.weights * weight_factor
            target.pointfea = target.weights * weight_factor
        return flowed, target

    return get_fea
