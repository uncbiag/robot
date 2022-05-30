from copy import deepcopy
import random
from robot.utils.obj_factory import obj_factory


class AugShape(object):
    def __init__(self, data_aug_obj=None, test_time_randomize=False, aug_ratio=0.9):
        super(AugShape, self).__init__()
        self.data_aug = obj_factory(data_aug_obj) if data_aug_obj else None
        self.aug_ratio = aug_ratio
        self.sampler = None
        self.test_time_randomize = test_time_randomize

    def prepare_synth_input(self, input_data, batch_info):
        aug_data, synth_info = self.data_aug(deepcopy(input_data["shape"]))
        input_data["shape"] = aug_data
        batch_info["is_synth"] = True
        return input_data, batch_info

    def do_nothing(self, input_data, batch_info):
        batch_info["is_synth"] = False
        return input_data, batch_info

    def planner(self, phase):
        if phase == "debug":
            return False
        elif phase == "val":
            return False
        elif phase == "test":
            return False
        elif phase == "train":
            use_synth = random.random() < self.aug_ratio
            return use_synth

    def __call__(self, input_data, batch_info):
        use_synth = self.planner(batch_info["phase"])
        if use_synth:
            input_data, batch_info = self.prepare_synth_input(input_data, batch_info)
        else:
            input_data, batch_info = self.do_nothing(input_data, batch_info)

        return input_data, batch_info
