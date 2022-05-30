"""
run shape learning/optimization
"""

import os, sys
import subprocess
from distutils.dir_util import copy_tree

os.environ["DISPLAY"] = ":99.0"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_IPYVTK"] = "true"
bashCommand = "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & sleep 3"
process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
process.wait()

sys.path.insert(0, os.path.abspath(".."))
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


def init_task_env(setting_path, output_root_path, task_name):
    """
    create task environment.

    :param setting_path: the path to load 'task_setting.json'
    :param output_root_path: the output path
    :param task_name: task name i.e. run_unet, run_with_ncc_loss
    :return:
    """
    tsm_json_path = os.path.join(setting_path, "task_setting.json")
    assert os.path.isfile(tsm_json_path), "task setting:{} not exists".format(
        tsm_json_path
    )
    tsm = ModelTask("task_reg", tsm_json_path)
    tsm.task_par["tsk_set"]["task_name"] = task_name
    tsm.task_par["tsk_set"]["output_root_path"] = output_root_path
    return tsm


def _do_learning(args):
    """
    set running env and run the task

    :param args: the parsed arguments
    :param pipeline:a Pipeline object
    :return: a Pipeline object
    """
    output_root_path = args.output_root_path
    dataset_path = args.dataset_folder
    task_name = args.task_name
    setting_folder_path = args.setting_folder_path
    task_output_path = os.path.join(output_root_path, task_name)
    print("debugging {}".format(task_output_path))

    if os.path.isdir(output_root_path):
        print(
            "the output folder {} exists, skipping copying the dataset json files".format(
                output_root_path
            )
        )
    else:
        print("copy dataset json files from {} to {}".format(dataset_path, output_root_path))
        try:
            [
                copy_tree(
                    os.path.join(dataset_path, phase), os.path.join(output_root_path, phase)
                )
                for phase in ["train", "val", "test", "debug"]
            ]
        except:
            Warning("Failed to find train/val/test/debug splits, ignore this warnning if you use your customized dataloader")
    os.makedirs(task_output_path, exist_ok=True)
    tsm = init_task_env(setting_folder_path, output_root_path, task_name)
    if args.eval:
        tsm = addition_test_setting(args, tsm)
    tsm.task_par["tsk_set"]["gpu_ids"] = args.gpus
    tsm_json_path = os.path.join(task_output_path, "task_setting.json")
    tsm.save(tsm_json_path)
    pipeline = run_one_task(tsm_json_path, not args.eval)
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


def do_learning(args):
    """

    :param args: the parsed arguments
    :return: None
    """
    task_name = args.task_name
    args.task_name_record = task_name
    _do_learning(args)


if __name__ == "__main__":
    """
    An interface for shape learning approaches.
    Assume there is two level folder, output_root_folder/task_name
    The inference mode here is for learning pipeline, namely estimated on the test set. if you want to use model on wild data, please refer to run_eval.py
    Arguments:
        --eval: run in inference mode
        --dataset_folder/ -ds: the path including the dataset splits, which contains train/val/test/debug subfolders
        --output_root_folder/ -o: the path of output root folder, we assume the tasks under this folder share the same dataset
        --task_name / -tn: task name
        --setting_folder_path/ -ts: path of the folder where settings are saved,should include task_setting.json
        --gpu_id/ -g: gpu_id to use
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="An easy interface for training classification models_reg"
    )
    parser.add_argument("--eval", action="store_true", help="training the task")
    parser.add_argument(
        "-ds",
        "--dataset_folder",
        required=False,
        type=str,
        default=None,
        help="the path of dataset splits, must be provided unless using customized dataloader",
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
        "-tn",
        "--task_name",
        required=True,
        type=str,
        default=None,
        help="the name of the task",
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
    do_learning(args)
