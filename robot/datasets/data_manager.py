import torch
import numpy as np
from robot.global_variable import DATASET_POOL
from robot.utils.obj_factory import partial_obj_factory

# todo reformat the import style
class DataManager(object):
    def __init__(
        self,
    ):
        """
        the class for data management
        return a dict, each train/val/test/debug phase has its own dataloader
        """
        self.data_path = None
        self.data_opt = None
        self.phases = []

    def set_data_path(self, data_path):
        """
        set data path
        :param data_path:
        :return:
        """
        self.data_path = data_path

    def set_data_opt(self, data_opt):
        """
        set data opt
        :param data_opt:
        :return:
        """
        self.data_opt = data_opt

    def init_dataset_loader(self, transformed_dataset, batch_size):
        """
        initialize the data loaders: set work number, set work type( shuffle for trainning, order for others)
        :param transformed_dataset:
        :param batch_size: the batch size of each iteration
        :return: dict of dataloaders for train|val|test|debug
        """

        def _init_fn(worker_id):
            np.random.seed(12 + worker_id)

        num_workers_reg = {
            "train": 8,
            "val": 8,
            "test": 8,
            "debug": 4,
        }  # {'train':0,'val':0,'test':0,'debug':0}#{'train':8,'val':4,'test':4,'debug':4}
        shuffle_list = {"train": True, "val": False, "test": False, "debug": False}
        batch_size = (
            [batch_size] * 4 if not isinstance(batch_size, list) else batch_size
        )
        batch_size = {
            "train": batch_size[0],
            "val": batch_size[1],
            "test": batch_size[2],
            "debug": batch_size[3],
        }
        dataloaders = {
            x: torch.utils.data.DataLoader(
                transformed_dataset[x],
                batch_size=batch_size[x],
                shuffle=shuffle_list[x],
                num_workers=num_workers_reg[x],
                worker_init_fn=_init_fn,
                pin_memory=True,
            )
            for x in self.phases
        }
        return dataloaders

    def build_data_loaders(self, batch_size=20, is_train=True):
        """
        build the data_loaders for the train phase and the test phase
        :param batch_size: the batch size for each iteration
        :param is_train: in train mode or not
        :return: dict of dataloaders for train phase or the test phase
        """
        if is_train:
            self.phases = ["train", "val", "debug"]
        else:
            self.phases = ["test"]
        name = self.data_opt["name"]
        dataset_opt = self.data_opt[(name, {}, "settings for {} dataset".format(name))]
        assert name in DATASET_POOL, "{} not in dataset pool {}".format(
            name, DATASET_POOL
        )
        if name!="custom_dataset":
            transformed_dataset = {
                phase: DATASET_POOL[name](self.data_path, dataset_opt, phase=phase)
                for phase in self.phases
            }
        else:
            transformed_dataset = {
                phase: partial_obj_factory(dataset_opt["name"])(self.data_path, dataset_opt, phase=phase)
                for phase in self.phases
            }
        dataloaders = self.init_dataset_loader(transformed_dataset, batch_size)
        dataloaders["data_size"] = {
            phase: len(dataloaders[phase]) for phase in self.phases
        }
        print("dataloader is ready")

        return dataloaders
