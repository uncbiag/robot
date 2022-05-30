import os
import numpy as np
from robot.models_reg.model_base import ModelBase
from robot.global_variable import *
from robot.utils.net_utils import print_model
from robot.models_reg.multiscale_optimization import build_multi_scale_solver
from robot.utils.shape_visual_utils import save_shape_pair_into_files
from robot.shape.shape_pair_utils import create_shape_pair
from robot.utils.utils import timming


class OptModel(ModelBase):
    """
    Optimization models_reg include two step optimization: affine and non-parametric optimization
    the affine optimization  (optional, gradient flow for default)
    the non-parametric optimization (optional, gradient flow for default) is implemented multi-scale optimization framework


    """

    def name(self):
        return "Optimization Model"

    def initialize(self, opt, device, gpus=None):
        """
        initialize variable settings of Optimization Approches
        multi-gpu is not supported for optimization tasks

        :param opt: ParameterDict, task settings
        :return:
        """
        ModelBase.initialize(self, opt, device, gpus)
        self.init_optimization_env(self.opt, device)
        self.step_count = 0
        """ count of the step"""
        self.cur_epoch = 0
        """visualize condition"""
        prepare_shape_pair_obj = opt[
            (
                "prepare_shape_pair",
                "shape_pair_utils.prepare_shape_pair()",
                "shape pair initialization func",
            )
        ]
        self.prepare_shape_pair = obj_factory(prepare_shape_pair_obj)
        create_shape_pair_from_data_dict_obj = opt[
            (
                "create_shape_pair_from_data_dict",
                "shape_pair_utils.create_shape_pair_from_data_dict()",
                "input data generator func",
            )
        ]
        self.create_shape_pair_from_data_dict = obj_factory(
            create_shape_pair_from_data_dict_obj
        )
        prepare_input_object = opt[
            ("prepare_input_object", "", "input processing function")
        ]
        self.prepare_input = (
            obj_factory(prepare_input_object)
            if prepare_input_object
            else self._set_input
        )
        analyzer_obj = opt[("analyzer_obj", "", "result analyzer")]
        self.external_analyzer = obj_factory(analyzer_obj) if analyzer_obj else None

    def init_optimization_env(self, opt, device):
        method_name = opt[("method_name", "lddmm_opt", "specific optimization method")]
        prealign_opt = opt[("prealign_opt", {}, "method settings")]
        if "_and_prealign_opt" in method_name:
            self.run_prealign = True
            self.run_nonparametric = True
            self._prealign_model = MODEL_POOL["prealign_opt"](prealign_opt).to(device)
            method_name = method_name.replace("_and_prealign_opt", "")
            method_opt = opt[(method_name, {}, "method settings")]
            self._model = MODEL_POOL[method_name](method_opt).to(device)

        elif method_name in [
            "lddmm_opt",
            "discrete_flow_opt",
            "gradient_flow_opt",
            "barycenter_opt",
            "probreg_opt",
        ]:
            self.run_prealign = False
            self.run_nonparametric = True
            self._prealign_model = None
            method_opt = opt[(method_name, {}, "method settings")]
            self._model = MODEL_POOL[method_name](method_opt).to(device)
        elif method_name == "prealign_opt":
            self.run_prealign = True
            self.run_nonparametric = False
            self._prealign_model = MODEL_POOL["prealign_opt"](prealign_opt).to(device)
            self._model = None
        else:
            raise NotImplemented
        # if gpus and len(gpus) >= 1:
        #     self._model = nn.DataParallel(self._model, gpus)
        if self._prealign_model is not None:
            print(
                "---------- A model instance for {} is initialized -------------".format(
                    "prealign_opt"
                )
            )
            print_model(self._prealign_model)
            print("-----------------------------------------------")
        if self._model is not None:
            print(
                "---------- A model instance for {} is initialized -------------".format(
                    method_name
                )
            )
            print_model(self._model)
            print("-----------------------------------------------")

    def _set_input(self, input_data, batch_info):
        input_data["extra_info"] = {
            key: item
            for key, item in input_data["target"].get("extra_info", {}).items()
        }

        if "gt_flow" in input_data["source"].get("extra_info", {}):
            input_data["extra_info"]["gt_flow"] = input_data["source"]["extra_info"][
                "gt_flow"
            ]
            input_data["extra_info"]["gt_flowed"] = (
                input_data["extra_info"]["gt_flow"] + input_data["source"]["points"]
            )
            batch_info["corr_source_target"] = True
            batch_info["has_gt"] = True
            return input_data, batch_info
        else:
            batch_info["corr_source_target"] = False
            batch_info["has_gt"] = False
            return input_data, batch_info

    def set_input(self, input_data, device, phase=None):
        """
        :param input_data:
        :param is_train:
        :return:
        """

        def to_device(item, device):
            if isinstance(item, dict):
                return {key: to_device(_item, device) for key, _item in item.items()}
            else:
                return item.to(device)

        batch_info = {
            "pair_name": input_data["pair_name"],
            "source_info": input_data["source_info"],
            "target_info": input_data["target_info"],
            "is_synth": False,
            "phase": phase,
            "epoch": self.cur_epoch,
        }
        input_data["source"] = to_device(input_data["source"], device)
        input_data["target"] = to_device(input_data["target"], device)
        batch_info["record_path"] = self.record_path
        input_data, self.batch_info = self.prepare_input(input_data, batch_info)

        return input_data

    def get_debug_info(self):
        """ get filename of the failed cases"""
        info = {"file_name": self.batch_info["fname_list"]}
        return info

    def optimize_parameters(self, data=None):
        """
        forward and backward the model, optimize parameters and manage the learning rate

        :param data: input_data(not used
        :return:
        """
        shape_pair = self.create_shape_pair_from_data_dict(data)
        source, target = shape_pair.source, shape_pair.target
        shape_pair = self.prepare_shape_pair(
            source, target, extra_info=shape_pair.extra_info
        )
        shape_pair.set_pair_name(self.batch_info["pair_name"])
        forward_t_prealign, forward_t_nonp = 0, 0
        if self.run_prealign:
            multi_scale_opt = self.opt[
                (
                    "multi_scale_optimization_prealign",
                    {},
                    "settings for multi_scale_optimization_prealign",
                )
            ]
            multi_scale_opt["record_path"] = self.record_path
            sovler = build_multi_scale_solver(multi_scale_opt, self._prealign_model)
            shape_pair, forward_t_prealign = timming(sovler, return_t=True)(shape_pair)
            save_shape_pair_into_files(
                self.record_path,
                "shape_prealigned",
                shape_pair.get_pair_name(),
                shape_pair,
            )
            if self.run_nonparametric:
                shape_pair.control_points = shape_pair.flowed_control_points.detach()
                shape_pair.source.points = shape_pair.flowed.points.detach()
        if self.run_nonparametric:
            multi_scale_opt = self.opt[
                (
                    "multi_scale_optimization",
                    {},
                    "settings for multi_scale_optimization",
                )
            ]
            multi_scale_opt["record_path"] = self.record_path
            sovler = build_multi_scale_solver(multi_scale_opt, self._model)
            shape_pair, forward_t_nonp = timming(sovler, return_t=True)(shape_pair)
            save_shape_pair_into_files(
                self.record_path,
                "shape_nonparametric",
                shape_pair.get_pair_name(),
                shape_pair,
            )
        forward_t = forward_t_prealign + forward_t_nonp
        return shape_pair, forward_t

    def get_evaluation(self, input_data):
        """
        get
        :param input_data:
        :return:
        """
        shape_pair, forward_t = self.optimize_parameters(input_data)
        scores = {}
        if self._model is not None:
            scores, shape_pair = self._model.model_eval(shape_pair, self.batch_info)
        elif self._prealign_model is not None:
            scores, shape_pair = self._prealign_model.model_eval(
                shape_pair, self.batch_info
            )
        scores.update(
            {"forward_t": [forward_t / shape_pair.nbatch] * shape_pair.nbatch}
        )

        return scores, shape_pair

    def save_visual_res(self, save_visual_results, input_data, eval_res, phase):
        """ handle in optimizer"""
        pass

    def analyze_res(self, eval_res, cache_res=True):
        import numpy as np

        eval_metrics, shape_pair = eval_res
        assert (
            "score" in eval_metrics
        ), "at least there should be one metric named 'score'"
        if self.external_analyzer:
            self.external_analyzer(
                shape_pair, self.batch_info["pair_name"], self._model, self.record_path
            )
        if cache_res:
            if len(self.caches) == 0:
                self.caches.update(eval_metrics)
                self.caches.update({"pair_name": self.batch_info["pair_name"]})
            else:
                for metric in eval_metrics:
                    self.caches[metric] += eval_metrics[metric]
                self.caches["pair_name"] += self.batch_info["pair_name"]

        if len(eval_metrics):
            return np.array(eval_metrics["score"]).mean(), eval_metrics
        else:
            return -1, np.array([-1])

    def get_extra_to_plot(self):
        """
        extra image to be visualized

        :return: image (BxCxXxYxZ), name
        """
        return self._model.get_extra_to_plot()

    def set_test(self):
        torch.set_grad_enabled(True)
