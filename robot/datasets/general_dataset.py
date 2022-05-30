from __future__ import print_function, division
import os
import time
import blosc
import torch
import random
import numpy as np
from robot.datasets.data_utils import read_json_into_list, split_dict
from torch.utils.data import Dataset
from robot.utils.obj_factory import obj_factory
from multiprocessing import *

blosc.set_nthreads(1)
from tqdm import tqdm


class GeneralDataset(Dataset):
    """registration dataset."""

    def __init__(self, data_root_path, option=None, phase=None):
        """
        the dataloader for  task, to avoid frequent disk communication, all shapes can be optionally compressed into memory
        :param data_root_path:  string, path to the data
            the data should be preprocessed and saved into txt
        :param phase:  string, 'train'/'val'/ 'test'/ 'debug' , debug here means a subset of train data, to check if model is overfitting
        : option:  pars, settings for classification/segmentation task

        """
        self.phase = phase
        self.data_path = os.path.join(data_root_path, phase)
        self.transform = ToTensor()
        ind = ["train", "val", "test", "debug"].index(phase)
        max_num_for_loading = option[
            "max_num_for_loading",
            (-1, -1, -1, -1),
            "the max number of pairs to be loaded, set -1 if there is no constraint,[max_train, max_val, max_test, max_debug]",
        ]
        self.max_num_for_loading = max_num_for_loading[ind]
        """ the max number of pairs to be loaded into the memory,[max_train, max_val, max_test, max_debug]"""
        self.file_name_list = []
        self.file_info_list = []
        self.file_list = []
        self.get_file_list()
        self.reg_option = option
        self.reader = obj_factory(option[("reader", "", "a reader instance")])
        self.sampler = obj_factory(
            option[
                (
                    "sampler",
                    "",
                    "a sampler instance, the goal of sampling here is for batch consistency, where we pick the same index order from the source and the target",
                )
            ]
        )
        self.normalizer = obj_factory(
            option[("normalizer", "", "a normalizer instance")]
        )
        shape_postprocess_obj = option[
            ("shape_postprocess_obj", "", "a file_postprocess instance")
        ]
        self.shape_postprocess = (
            obj_factory(shape_postprocess_obj) if shape_postprocess_obj else None
        )
        load_training_data_into_memory = option[
            (
                "load_training_data_into_memory",
                True,
                "when train network, load all training sample into memory can relieve disk burden",
            )
        ]
        self.load_into_memory = (
            load_training_data_into_memory if phase == "train" else False
        )
        self.enlarge_dataset_size_by_factor = option[
            (
                "enlarge_dataset_size_by_factor",
                100.0,
                " during the training, increase the dataset size to  factor*len(dataset) ",
            )
        ]
        if self.load_into_memory:
            self._init_data_pool()

    def get_file_list(self):
        """"""
        if not os.path.exists(self.data_path):
            raise IOError("Non data detected")
        self.file_name_list, self.file_info_list = read_json_into_list(
            os.path.join(self.data_path, "shape_data.json")
        )
        read_num = min(self.max_num_for_loading, len(self.file_info_list))
        step = int(len(self.file_info_list) / read_num)
        if self.max_num_for_loading > 0:
            self.file_info_list = self.file_info_list[::step]
            self.file_name_list = self.file_name_list[::step]

    def _init_data_pool(self):
        """"""
        manager = Manager()
        data_dic = manager.dict()
        data_info_dic = {}
        _file_name_list = []
        for file_info in self.file_info_list:
            fname = file_info["name"]
            if fname not in data_info_dic:
                data_info_dic[fname] = file_info
            _file_name_list.append(fname)

        #  multi process
        num_of_workers = 12
        num_of_workers = num_of_workers if len(_file_name_list) > 12 else 2
        dict_splits = split_dict(data_info_dic, num_of_workers)
        procs = []
        for i in range(num_of_workers):
            p = Process(
                target=self._data_into_zipnp,
                args=(
                    dict_splits[i],
                    data_dic,
                ),
            )
            p.start()
            print("pid:{} start:".format(p.pid))
            procs.append(p)
        for p in procs:
            p.join()
        print(
            "the loading phase finished, total {} data have been loaded".format(
                len(data_dic)
            )
        )
        data_dic = dict(data_dic)

        # organize data into pair list
        for file_name in _file_name_list:
            self.file_list.append(data_dic[file_name])

    def _preprocess_data(self, file_info):
        """
        preprocess the data :
        1. read the data into dict
        2. normalize the data dict
        :param path: data_path
        :return: data_dict, shape_type
        """
        case_dict = self.reader(file_info)
        case_dict = self.normalizer(case_dict)
        return case_dict

    def _data_into_zipnp(self, data_path_dic, data_dict):
        """
        compress the data into zip to save memory
        :param data_path_dic:
        :param data_dict:
        :return:
        """

        def zip_fn(item):
            if isinstance(item, dict):
                return {key: zip_fn(_item) for key, _item in item.items()}
            else:
                return blosc.pack_array(item)

        for fn in tqdm(data_path_dic):
            case_dict = self._preprocess_data(data_path_dic[fn])
            data_dict[fn] = zip_fn(case_dict)

    def setup_random_seed(self):
        """due to the property of the dataloader, we manually set the random seed here"""
        if self.phase != "train":
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
        else:
            torch.manual_seed(int(time.time()))
            np.random.seed(int(time.time()))
            random.seed(int(time.time()))

    def __len__(self):
        # to make the epoch size always meet the setting, we scale the dataset when training dataset size is too small
        return (
            int(len(self.file_name_list) * self.enlarge_dataset_size_by_factor)
            if self.phase == "train"
            else len(self.file_name_list)
        )

    def __getitem__(self, idx):
        """
        get pair data_dict
        {"source": source_dict, "target":target_dict,
                           "shape_type":shape_type, "file_name":file_name,
                           "source_info":source_info, "target_info":target_info}

        :param idx: id of the items
        :return: file_data_dict, file_name

        """

        def unzip_shape_fn(item):
            if isinstance(item, dict):
                return {key: unzip_shape_fn(_item) for key, _item in item.items()}
            else:
                return blosc.unpack_array(item)

        # print(idx)
        self.setup_random_seed()
        idx = idx % len(self.file_name_list)
        file_info = self.file_info_list[idx]
        file_name = self.file_name_list[idx]
        if not self.load_into_memory:
            shape_dict = self._preprocess_data(file_info)
        else:
            zip_shape_dict = self.file_list[idx]
            shape_dict = unzip_shape_fn(zip_shape_dict)

        shape_dict = self.shape_postprocess(
            shape_dict, phase=self.phase, sampler=self.sampler
        )
        if self.transform:
            shape_dict = {key: self.transform(fea) for key, fea in shape_dict.items()}
        sample = {"shape": shape_dict, "file_name": file_name, "shape_info": file_info}
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if isinstance(sample, dict):
            return {item: ToTensor()(sample[item]) for item in sample}
        else:
            n_tensor = torch.from_numpy(sample)
            return n_tensor
