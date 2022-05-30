from robot.pipeline.build_model import build_model
from robot.pipeline.train_model import train_model
from robot.pipeline.test_model import eval_model
from robot.pipeline.initializer import Initializer


class Pipline:
    """
    Pipeline class,
    initialize env : data_manager, log settings and task settings
    run_task : run training based model or evaluation based model
    """

    def initialize(self, task_setting_pth="../settings/task_settings.json"):
        """
        initialize task environment
        :param task_setting_pth: the path of current task setting file
        :return: None
        """
        initializer = Initializer()
        initializer.initialize_data_manager()
        self.task_setting_pth = task_setting_pth
        self.tsk_opt = initializer.init_task_option(task_setting_pth)
        self.writer = initializer.initialize_log_env()
        self.tsk_opt = initializer.get_task_option()
        self.data_loaders = initializer.build_data_loader()
        self.device, self.gpus = initializer.initialize_compute_env()
        self.model = build_model(self.tsk_opt, self.device, self.gpus)

    def clean_up(self):
        """
        clean the environment settings, but keep the dataloader
        :return: None
        """
        self.tsk_opt = None
        self.writer = None
        self.model = None

    def run_task(self, is_train=True):
        """
        run training based model or evaluation based model
        :return: None
        """
        _run_model = train_model if is_train else eval_model
        _run_model(
            self.tsk_opt, self.model, self.data_loaders, self.writer, self.device
        )
        saving_comment_path = self.task_setting_pth.replace(".json", "_comment.json")
        self.tsk_opt.write_JSON_comments(saving_comment_path)


def run_one_task(task_setting_pth="../settings/task_settings.json", is_train=True):
    pipline = Pipline()
    pipline.initialize(task_setting_pth)
    pipline.run_task(is_train)
    return pipline


if __name__ == "__main__":
    pipline = Pipline()
    pipline.initialize()
    pipline.run_task()
