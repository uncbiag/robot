import os
import copy
from robot.datasets.prepare_data import BaseDataSet
from robot.datasets.data_utils import saving_pair_info, divide_sess_set

sesses = ["train", "val", "test", "debug"]
number_of_workers = 10
warning_once = True
import random


class GeneralDataSet(BaseDataSet):
    """"""

    def __init__(self):
        BaseDataSet.__init__(self)
        self.reg_coupled_pair = False
        # set either coupled_pair_path_list or self_cross_path_list
        self.coupled_pair_list = None
        self.self_cross_list = None
        self.id_sess_dic = None

    def set_coupled_pair_list(self, coupled_pair_list):
        """
        [source_list, target_list], where source and the target has one to one correspondence
        :param coupled_pair_path_list:
        :return:
        """
        self.coupled_pair_list = coupled_pair_list
        self.reg_coupled_pair = True

    def set_self_cross_list(self, self_cross_list):
        """
        a list that pairs are randomly selected from all possible pairs
        :param self_cross_path_list:
        :return:
        """
        self.self_cross_list = self_cross_list
        self.reg_coupled_pair = False

    def set_id_sess_dic(self, id_sess_dic):
        """
        {"train": id_list, "val":id_list, "test":id_list, "debug": id_list}
        :return:
        """
        self.id_sess_dic = id_sess_dic

    def check_settings(self):
        if self.reg_coupled_pair:
            assert self.coupled_pair_list is not None and self.self_cross_list is None
        if not self.reg_coupled_pair:
            assert self.coupled_pair_list is None and self.self_cross_list is not None

    @staticmethod
    def __gen_pair_list_from_two_list(obj_list_1, obj_list_2):
        pair_list = []
        num_obj_1 = len(obj_list_1)
        num_obj_2 = len(obj_list_2)
        name_list_1 = [obj["name"] for obj in obj_list_1]
        name_list_2 = [obj["name"] for obj in obj_list_2]
        for i in range(num_obj_1):
            count_max = 100  # -1
            pair_list_tmp = []
            for j in range(num_obj_2):
                if name_list_1[i] == name_list_2[j]:
                    continue
                pair_list_tmp.append([obj_list_1[i], obj_list_2[j]])
            if len(pair_list_tmp) > count_max and count_max > 0:
                pair_list_tmp = random.sample(pair_list_tmp, count_max)
            pair_list += pair_list_tmp
        return pair_list

    @staticmethod
    def __gen_pair_list_with_coupled_list(obj_list_1, obj_list_2):
        pair_list = []
        num_obj_1 = len(obj_list_1)
        num_obj_2 = len(obj_list_2)
        assert num_obj_1 == num_obj_2
        name_list_1 = [obj["name"] for obj in obj_list_1]
        name_list_2 = [obj["name"] for obj in obj_list_2]
        for i in range(num_obj_1):
            if name_list_1[i] == name_list_2[i]:
                continue
            pair_list.append([obj_list_1[i], obj_list_2[i]])
        return pair_list

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
        if not self.reg_coupled_pair:
            obj_list = self.self_cross_list
            pair_list = [obj_list, obj_list]
            gen_pair_list_func = self.__gen_pair_list_from_two_list
        else:
            coupled_pair_list = self.coupled_pair_list
            pair_list = coupled_pair_list
            gen_pair_list_func = self.__gen_pair_list_with_coupled_list
        num_pair = len(pair_list[0])
        if self.id_sess_dic is None:
            sub_folder_dic, id_sess_dic = divide_sess_set(
                self.output_path, num_pair, self.divided_ratio
            )
        else:
            sub_folder_dic = {
                x: os.path.join(self.output_path, x)
                for x in ["train", "val", "test", "debug"]
            }
            id_sess_dic = self.id_sess_dic
        ind_filter = lambda x_list, ind_list: [x_list[ind] for ind in ind_list]
        sub_pair_sess_dic = {
            sess: [
                ind_filter(pair_list[0], id_sess_dic[sess]),
                ind_filter(pair_list[1], id_sess_dic[sess]),
            ]
            for sess in ["train", "val", "test", "debug"]
        }
        sess_ratio = {
            "train": self.divided_ratio[0],
            "val": self.divided_ratio[1],
            "test": self.divided_ratio[2],
            "debug": self.divided_ratio[1],
        }
        if self.max_train_pairs > -1:
            sub_pair_sess_dic["train"] = sub_pair_sess_dic["train"][
                : self.max_train_pairs
            ]
        pair_list_dic = {
            sess: self.__gen_pair(
                gen_pair_list_func,
                sub_pair_sess_dic[sess],
                int(self.max_total_pairs * sess_ratio[sess])
                if self.max_total_pairs > 0
                else -1,
            )
            for sess in sesses
        }
        return sub_folder_dic, pair_list_dic

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
        sub_folder_dic, pair_list_dic = info_dict
        saving_pair_info(sub_folder_dic, pair_list_dic)


if __name__ == "__main__":
    synth_data1_list = [
        {
            "name": "img_0_{}".format(i),
            "data_path": "path_for_img_0_{}".format(i),
            "extra_info": {"info": "info_0_{}".format(i)},
        }
        for i in range(120)
    ]
    synth_data2_list = [
        {
            "name": "img_1_{}".format(i),
            "data_path": "path_for_img_1_{}".format(i),
            "extra_info": {"info": "info_1_{}".format(i)},
        }
        for i in range(120)
    ]
    # for task where source and target has specific relation, e.g., longitudinal registration
    # the source list and the target list need to be given, two lists should have one-to-one correspondence
    # both list would be divided into train part, val part and test part, debug part(sub_train part)
    # these sessions are divided according to id_sess_dic
    # if the id_sess_dic is not set, these sessions would be automatically divided according to divided_ratio
    # the pair can be get by [source_list[ind], target_list[ind]]
    # then final num of pairs per session is determined by divided_ratio*max_total_pairs
    dataset = GeneralDataSet()
    dataset.set_output_path("./debug/datasets/prepare_data/func_debug")
    dataset.set_coupled_pair_list([synth_data1_list, synth_data2_list])
    dataset.set_divided_ratio((0.6, 0.3, 0.1))
    dataset.prepare_data()

    # for task where source and target are randomly picked, e.g., cross-object registration
    # a list need to be given
    # the list would be divided into train part, val part and test part, debug part(sub_train part)
    # these sessions are divided according to id_sess_dic
    # if the id_sess_dic is not set, these sessions would be automatically divided according to divided_ratio
    # the pair will be randomly collected inside each session
    # then final num of pairs per session is determined by divided_ratio*max_total_pairs
    dataset = GeneralDataSet()
    dataset.set_output_path("./debug/datasets/prepare_data/func_debug2")
    dataset.set_self_cross_list(synth_data1_list)
    dataset.set_id_sess_dic(
        {
            "train": list(range(80)),
            "val": list(range(80, 100)),
            "test": list(range(100, 120)),
            "debug": list(range(0, 30)),
        }
    )
    dataset.max_total_pairs = 400
    dataset.prepare_data()
