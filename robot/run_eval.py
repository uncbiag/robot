"""
run shape learning/optimization
"""

import os, sys
import subprocess
sys.path.insert(0, os.path.abspath(".."))

from robot.datasets.data_utils import get_file_name, cp_file

os.environ["DISPLAY"] = ":99.0"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_IPYVTK"] = "true"
bashCommand = "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & sleep 3"
process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
process.wait()
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import robot.utils.module_parameters as pars
from abc import ABCMeta, abstractmethod
from robot.pipeline.run_pipeline import run_one_task


class BaseTask:
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def save(self):
        pass


class ModelTask(BaseTask):
    """
    base module for task setting files (.json)
    """

    def __init__(self, name, path="../settings/base_task_settings.json"):
        super(ModelTask, self).__init__(name)
        self.task_par = pars.ParameterDict()
        self.task_par.load_JSON(path)

    def save(self, path="../settings/task_settings.json"):
        self.task_par.write_ext_JSON(path)


def init_eval_env(setting_path, output_root_path, data_json_path):
    """
    create task environment.

    :param setting_path: the path to load 'task_setting.json'
    :param output_root_path: the output path
    :return:
    """
    eval_data_folder = os.path.join(output_root_path, "test")
    os.makedirs(eval_data_folder, exist_ok=True)
    os.makedirs(os.path.join(output_root_path, "res"), exist_ok=True)
    if data_json_path:
        data_json_name = get_file_name(data_json_path)
        cp_file(data_json_path, os.path.join(eval_data_folder, data_json_name + ".json"))
    os.makedirs(output_root_path, exist_ok=True)
    tsm_json_path = os.path.join(setting_path, "task_setting.json")
    assert os.path.isfile(tsm_json_path), "task setting:{} not exists".format(
        tsm_json_path
    )
    tsm = ModelTask("task", tsm_json_path)
    tsm.task_par["tsk_set"]["task_name"] = "res"
    tsm.task_par["tsk_set"]["output_root_path"] = output_root_path
    return tsm


def do_evaluation(args):
    """
    set running env and run the task
    :param args: the parsed arguments
    """
    setting_folder_path = args.setting_folder_path
    output_root_path = args.output_root_path
    data_json_path = args.data_json
    task_name = "eval"
    task_output_path = os.path.join(output_root_path, task_name)
    print("task output path: {}".format(task_output_path))
    tsm = init_eval_env(setting_folder_path, task_output_path, data_json_path)
    tsm = addition_test_setting(args, tsm)
    tsm.task_par["tsk_set"]["gpu_ids"] = args.gpus
    tsm_json_path = os.path.join(task_output_path, "task_setting.json")
    tsm.save(tsm_json_path)
    pipeline = run_one_task(tsm_json_path, is_train=False)
    return pipeline


def addition_test_setting(args, tsm):
    model_path = args.model_path
    if model_path is not None:
        assert os.path.isfile(model_path), "the model {} not exist".format_map(
            model_path
        )
        tsm.task_par["tsk_set"]["model_path"] = model_path
    tsm.task_par["tsk_set"]["is_train"] = False
    tsm.task_par["tsk_set"]["continue_train"] = False
    return tsm


if __name__ == "__main__":
    """
    An evaluation interface on new data.
    make sure the file is saved in a compatible way to your task specific reader. e.g. 'lung_reader' in lung_dataloader_utils.py
    Arguments:
        --data_json/ -dj: path of data json
        --output_root_folder/ -o: the path of output root folder, we assume the tasks under this folder share the same dataset
        --setting_folder_path/ -ts: path of the folder where settings are saved,should include task_setting.json
        --model_path/ -m: for learning based approach, the model checkpoint should either provided here (first priority) or set in task_setting.json (second priority)
        --gpu_id/ -g: gpu_id to use
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="An easy interface for training classification models_reg"
    )
    parser.add_argument(
        "-dj",
        "--data_json",
        required=False,
        type=str,
        default=None,
        help="the path of data json file",
    )
    parser.add_argument(
        "-o",
        "--output_root_path",
        required=True,
        type=str,
        default=None,
        help="the path of output root folder",
    )
    parser.add_argument(
        "-ts",
        "--setting_folder_path",
        required=True,
        type=str,
        default=None,
        help="path of the folder where settings are saved,should include task_setting.json",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        required=False,
        default=None,
        help="the path of trained model",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        default=None,
        nargs="+",
        type=int,
        metavar="N",
        help="list of gpu ids to use",
    )
    args = parser.parse_args()
    print(args)
    do_evaluation(args)
