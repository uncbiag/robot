import os, sys
from robot.utils.utils import set_device
from torch.utils.tensorboard import SummaryWriter
import robot.utils.module_parameters as pars
from robot.datasets.data_manager import DataManager


class Initializer:
    """
    The initializer for data manager,  log env and task settings
    """

    class Logger(object):
        """
        redirect the stdout into files
        """

        def __init__(self, task_path):
            self.terminal = sys.stdout
            self.log = open(os.path.join(task_path, "logfile.log"), "a", buffering=1)

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            os.fsync(self.log.fileno())

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            pass

    def init_task_option(self, setting_path="../settings/task_settings.json"):
        """
        load task settings from the task setting json
        :param setting_path: the path of task setting json
        :return: ParameterDict, task settings
        """
        opt = pars.ParameterDict()
        opt.load_JSON(setting_path)
        self.task_opt = opt["tsk_set"]
        self.task_name = self.task_opt["task_name"]
        self.task_root_path = self.task_opt["output_root_path"]
        self.task_path = os.path.join(self.task_root_path, self.task_name)
        dataset_opt = opt["dataset"]
        self.data_manager.set_data_path(self.task_root_path)
        self.data_manager.set_data_opt(dataset_opt)
        return self.task_opt

    def get_task_option(self):
        """
        get current task settings
        :return:
        """
        return self.task_opt

    def initialize_compute_env(self):
        import torch

        torch.backends.cudnn.benchmark = True
        gpu_id = self.task_opt["gpu_ids"]
        device, gpus = set_device(gpu_id)
        return device, gpus

    def initialize_data_manager(self):
        """
        if the data processing settings are given, then set the data manager according to the setting
        if not, then assume the settings are included in task settings, no further actions need to be set in data manager
        :param setting_path: the path of the data processing json
        :param task_path: the path of the task setting json (disabled)
        :return: None
        """
        self.data_manager = DataManager()

    def build_data_loader(self):
        """
        get task related setttings for data manager
        """
        batch_size = self.task_opt[
            (
                "batch_sz",
                1,
                "list of batch size, refers to train, val, test,debug respectively",
            )
        ]
        is_train = self.task_opt[("is_train", False, "train the model")]
        return self.data_manager.build_data_loaders(
            batch_size=batch_size, is_train=is_train
        )

    def setting_folder(self):
        for item in self.path:
            if not os.path.exists(self.path[item]):
                os.makedirs(self.path[item])

    def initialize_log_env(
        self,
    ):
        """
        initialize log environment for the task.
        including
        task_path/checkpoints:  saved checkpoints for learning methods every # epoch
        task_path/logdir: saved logs for tensorboard
        task_path/records: saved 2d and 3d images for analysis
        :return: tensorboard writer
        """
        logdir = os.path.join(self.task_path, "log")
        check_point_path = os.path.join(self.task_path, "checkpoints")
        record_path = os.path.join(self.task_path, "records")
        model_path = self.task_opt[
            ("model_path", "", "if continue_train, the model path should be given here")
        ]
        self.task_opt[("path", {}, "task path settings")]
        self.task_opt["path"]["expr_path"] = self.task_path
        self.task_opt["path"]["logdir"] = logdir
        self.task_opt["path"]["check_point_path"] = check_point_path
        self.task_opt["path"]["model_load_path"] = model_path
        self.task_opt["path"]["record_path"] = record_path
        self.path = {
            "logdir": logdir,
            "check_point_path": check_point_path,
            "record_path": record_path,
        }
        self.setting_folder()
        sys.stdout = self.Logger(self.task_path)
        print("start logging:")
        self.writer = SummaryWriter(logdir, self.task_name)
        return self.writer
