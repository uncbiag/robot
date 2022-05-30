import os
import torch.nn as nn
from robot.models_reg.model_base import ModelBase
from robot.global_variable import *
from robot.utils.net_utils import print_model
from robot.utils.shape_visual_utils import save_shape_pair_into_files
from robot.modules_reg.optimizer import optimizer_builder
from robot.modules_reg.scheduler import scheduler_builder


class DeepModel(ModelBase):
    """
    Deep models_reg
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

        self.opt_optim = opt[("optim", {}, "setting for the optimizer")]
        self.opt_scheduler = opt[("scheduler", {}, "setting for the scheduler")]
        self.criticUpdates = opt["tsk_set"][
            ("criticUpdates", 1, "update parameter every # iter")
        ]
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
        self.init_learning_env(self.opt, device, gpus)
        self.step_count = 0
        """ count of the step"""
        self.cur_epoch = 0
        prepare_input_object = opt[
            ("prepare_input_object", "", "input processing function")
        ]
        self.prepare_input = (
            obj_factory(prepare_input_object)
            if prepare_input_object
            else self._set_input
        )
        capture_plotter_obj = opt[
            (
                "capture_plot_obj",
                "visualizer.capture_plotter()",
                "factory object for 2d capture plot",
            )
        ]
        self.capture_plotter = obj_factory(capture_plotter_obj)
        analyzer_obj = opt[("analyzer_obj", "", "result analyzer")]
        self.external_analyzer = obj_factory(analyzer_obj) if analyzer_obj else None

    def init_learning_env(self, opt, device, gpus):
        method_name = opt[
            ("method_name", "discrete_flow_deep", "specific optimization method")
        ]
        if method_name in ["feature_deep", "discrete_flow_deep", "flow_deep"]:
            method_opt = opt[(method_name, {}, "method settings")]
            self._model = MODEL_POOL[method_name](method_opt)
        else:
            raise ValueError("method not supported")
        self._model = nn.DataParallel(self._model, gpus)
        self._model.to(device)
        self.optimizer = optimizer_builder(self.opt_optim)(self._model.parameters())
        self.lr_scheduler = scheduler_builder(self.opt_scheduler)(self.optimizer)
        self.shape_folder_3d = os.path.join(self.record_path, "3d")
        os.makedirs(self.shape_folder_3d, exist_ok=True)
        self.shape_folder_2d = os.path.join(self.record_path, "2d")
        os.makedirs(self.shape_folder_2d, exist_ok=True)

        if self._model is not None:
            print(
                "---------- A model instance for {} is initialized -------------".format(
                    method_name
                )
            )
            print_model(self._model)
            print("-----------------------------------------------")

    def rebuild_lr_scheduler(self, base_epoch=0.0):
        self.lr_scheduler_base_epoch = base_epoch
        self.lr_scheduler = scheduler_builder(self.opt_scheduler)(self.optimizer)

    def reset_lr_optimizer(self, new_lr=0.001):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
            param_group["initial_lr"] = new_lr
        print(" the learning rate now is set to {}".format(new_lr))

    def update_learning_rate(self, new_lr=-1):
        """
        set new learning rate

        :param new_lr: new learning rate
        :return:
        """
        if new_lr < 0:
            lr = self.opt_optim["lr"]
        else:
            lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        print(" the learning rate now is set to {}".format(lr))

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
            batch_info["corr_source_target"] = all(input_data["source"]["extra_info"].get("corr_source_target",False))
            batch_info["has_gt"] = all(input_data["source"]["extra_info"].get("has_gt",False))
            return input_data, batch_info

    def set_input(self, input_data, device, phase=None):
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
            "record_path": self.record_path,
        }
        input_data["source"] = to_device(input_data["source"], device)
        input_data["target"] = to_device(input_data["target"], device)
        input_data, self.batch_info = self.prepare_input(input_data, batch_info)

        ##################### unknown bug during single gpu training, when gpu_id!=0 and call pointnet2_utils.py an ugly workaround  todo debug the pointnet2_utils
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] != 0:
            from pykeops.torch import LazyTensor

            a = LazyTensor(input_data["source"]["points"][:, :, None])
            a.sum(1)

        ################################################################################################
        return input_data

    def do_some_clean(self):
        self.batch_info = {}

    def get_batch_names(self):
        return self.batch_info["pair_name"]

    def get_debug_info(self):
        """ get filename of the failed cases"""
        info = {"file_name": self.batch_info["fname_list"]}
        return info

    def forward(self, input_data=None):
        self._model.module.set_cur_epoch(self.cur_epoch)
        loss, shape_data_dict = self._model(input_data, self.batch_info)
        return loss, shape_data_dict

    def backward_net(self, loss):
        loss.backward()

    def optimize_parameters(self, input_data=None):
        """
        forward and backward the model, optimize parameters and manage the learning rate
        :return:
        """
        if self.is_train:
            self.iter_count += 1
        loss, shape_data_dict = self.forward(input_data)
        loss = loss.mean()
        self.backward_net(loss / self.criticUpdates)
        self.loss = loss.item()
        update_lr, lr = self._model.module.check_if_update_lr()
        if update_lr:
            self.update_learning_rate(lr)
        if self.iter_count % self.criticUpdates == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return shape_data_dict

    def get_current_errors(self):
        return self.loss

    def update_scheduler(self, epoch):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch - self.lr_scheduler_base_epoch)

        for param_group in self.optimizer.param_groups:
            print(
                "the current epoch is {} with learining rate set at {}".format(
                    epoch, param_group["lr"]
                )
            )

    def get_evaluation(self, input_data):
        """
        get
        :param input_data:
        :return:
        """
        self._model.module.set_cur_epoch(self.cur_epoch)
        scores, shape_data_dict = self._model.module.model_eval(
            input_data, self.batch_info
        )
        shape_pair = self.create_shape_pair_from_data_dict(shape_data_dict)
        return scores, shape_pair

    def save_visual_res(self, save_fig_on, input_data, eval_res, phase):
        scores, shape_pair = eval_res
        save_shape_pair_into_files(
            self.shape_folder_3d,
            "{}_epoch_{}".format(phase, self.cur_epoch),
            self.batch_info["pair_name"],
            shape_pair,
        )
        if save_fig_on:
            self.capture_plotter(
                self.shape_folder_2d,
                "{}_epoch_{}".format(phase, self.cur_epoch),
                self.batch_info["pair_name"],
                shape_pair,
            )

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

    def save_res(self, phase, saving=True):
        import pandas as pd

        if saving:
            saving_pred_path = os.path.join(
                self.record_path, "score_{}_{}.csv".format(phase, self.cur_epoch)
            )
            submission_df = pd.DataFrame(self.caches)
            submission_df.to_csv(saving_pred_path, index=False)
        self.caches = {}

    def get_extra_to_plot(self):
        """
        extra image to be visualized

        :return: image (BxCxXxYxZ), name
        """
        return self._model.module.get_extra_to_plot()

    def set_train(self):
        self._model.train(True)
        self.is_train = True
        torch.set_grad_enabled(True)

    def set_val(self):
        self._model.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)

    def set_debug(self):
        self._model.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)

    def set_test(self):
        self._model.train(False)
        self.is_train = False
        torch.set_grad_enabled(False)
