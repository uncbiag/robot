from robot.utils.local_feature_extractor import *
from robot.utils.obj_factory import *
from robot.utils.module_parameters import ParameterDict
from robot.global_variable import ROBOT_PATH
import robot.modules_reg.module_deep_feature as deep_feature_module
import torch.nn as nn

class ModelNetDeepFeatureExtractor(nn.Module):
    def __init__(self, fixed=False,**kwargs):
        """
        :param fixed:
        """
        super(ModelNetDeepFeatureExtractor, self).__init__()

        self.fixed = fixed
        deep_opt = ParameterDict()
        deep_opt[
            "local_pair_feature_extractor_obj"
        ] = "local_feature_extractor.default_local_pair_feature_extractor()"
        deep_opt["input_channel"] = 3
        deep_opt["output_channel"] = 30
        deep_opt["initial_radius"] = 0.001
        # deep_opt["initial_npoints"] = 1024
        # deep_opt["group_via_knn"] = True
        deep_opt["include_pos_in_final_feature"] = False
        deep_opt["use_aniso_kernel"] = False
        deep_opt[
            "pretrained_model_path"
        ] = os.path.join(ROBOT_PATH,"experiments/datasets/modelnet40/model/simple")
        self.feature_extractor = deep_feature_module.PointNet2FeaExtractor(deep_opt)
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

