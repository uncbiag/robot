import os
import copy
from robot.datasets.data_utils import saving_shape_info, divide_sess_set

sesses = ["train", "val", "test", "debug"]
number_of_workers = 10
warning_once = True
import random


class BaseDataSet(object):
    def __init__(self):
        """
        :param name: name of data set
        :param data_path: path of the dataset
        """

        self.data_path = None
        """path of the dataset"""
        self.output_path = None
        """path of the output directory"""
        self.max_train_pairs = -1
        self.max_total_pairs = -1
        self.divided_ratio = (0.7, 0.1, 0.2)
        """divided the data into train, val, test set"""

    def set_data_path(self, path):
        self.data_path = path

    def set_output_path(self, path):
        self.output_path = path
        os.makedirs(path, exist_ok=True)

    def set_divided_ratio(self, ratio):
        """set dataset divide ratio, (train_ratio, val_ratio, test_ratio)"""
        self.divided_ratio = ratio

    def save_sess_to_txt(self, info=None):
        pass

    def gen_sess_dic(self):
        pass

    def check_settings(self):
        pass

    def prepare_data(self):
        """
        preprocessig  data for each dataset
        :return:
        """
        self.check_settings()
        print("start preparing data..........")
        print("the output file path is: {}".format(self.output_path))
        info_dict = self.gen_sess_dic()
        self.save_sess_to_txt(copy.deepcopy(info_dict))
        print("data preprocessing finished")


class GeneralDataSet(BaseDataSet):
    """"""

    def __init__(self):
        BaseDataSet.__init__(self)
        self.id_sess_dic = None
        self.file_list = None

    def set_file_list(self, file_list):
        self.file_list = file_list

    def set_id_sess_dic(self, id_sess_dic):
        """
        {"train": id_list, "val":id_list, "test":id_list, "debug": id_list}
        :return:
        """
        self.id_sess_dic = id_sess_dic

    def __gen_pair(self, pair_fn, pair_list, pair_num_limit=1000):
        obj_list_1, obj_list_2 = pair_list
        pair_list = pair_fn(obj_list_1, obj_list_2)

        if pair_num_limit >= 0:
            num_limit = min(len(pair_list), pair_num_limit)
            pair_list = random.sample(pair_list, num_limit)
            return pair_list
        else:
            return pair_list

    def gen_sess_dic(self):
        file_list = self.file_list
        num_file = len(file_list)
        if self.id_sess_dic is None:
            sub_folder_dic, id_sess_dic = divide_sess_set(
                self.output_path, num_file, self.divided_ratio
            )
        else:
            sub_folder_dic = {
                x: os.path.join(self.output_path, x)
                for x in ["train", "val", "test", "debug"]
            }
            id_sess_dic = self.id_sess_dic
        ind_filter = lambda x_list, ind_list: [x_list[ind] for ind in ind_list]
        shape_sess_dic = {
            sess: ind_filter(file_list, id_sess_dic[sess])
            for sess in ["train", "val", "test", "debug"]
        }
        if self.max_train_pairs > -1:
            shape_sess_dic["train"] = shape_sess_dic["train"][: self.max_train_pairs]
        return sub_folder_dic, shape_sess_dic

    def save_sess_to_txt(self, info_dict=None):
        """
        the output txt file only contains the source path and the target path
        1. if the source and the target obj do not have extra info,
        then each line refers to the source_obj_path and the  target_obj_path
        2. if the source and the target obj include extra info
        the each line refers to a source_txt and a target txt saved under output_path/info
         this additional txt file  records the path of obj_path, and paths of extra info

        :param info:
        :return:
        """
        sub_folder_dic, shape_sess_dic = info_dict
        saving_shape_info(sub_folder_dic, shape_sess_dic)
