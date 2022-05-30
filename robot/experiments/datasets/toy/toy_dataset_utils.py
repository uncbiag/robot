"""
data reader for the toys
given a file path, the reader will return a dict
{"points":Nx3, "weights":Nx1, "faces":Nx3}
"""
import numpy as np
from robot.datasets.vtk_utils import read_vtk


def toy_reader():
    """
    :return:
    """
    reader = read_vtk

    def read(file_info):
        path = file_info["data_path"]
        raw_data_dict = reader(path)
        data_dict = {}
        data_dict["points"] = raw_data_dict["points"]
        data_dict["faces"] = raw_data_dict["faces"]
        return data_dict

    return read


def toy_sampler():
    """
    :param args:
    :return:
    """

    def do_nothing(data_dict):
        return data_dict, None

    return do_nothing


def toy_normalizer(scale=1, add_random_noise_on_weight=False):
    """
    :return:
    """

    def scale_data(data_dict):
        data_dict["points"] = data_dict["points"] *scale
        return data_dict
    def randomized_weight(data_dict):
        weights = data_dict["weights"]
        min_weight = np.min(weights)
        npoints = len(weights)
        rand_noise =np.random.rand(npoints) * abs(min_weight)/10
        weights = weights + rand_noise
        data_dict["weights"] = weights/np.sum(weights)
        return data_dict
    return scale_data if not add_random_noise_on_weight else randomized_weight


if __name__ == "__main__":
    from robot.utils.obj_factory import obj_factory

    reader_obj = "toy_dataset_utils.toy_reader()"
    sampler_obj = "toy_dataset_utils.toy_sampler()"
    normalizer_obj = "toy_dataset_utils.toy_normalizer()"
    reader = obj_factory(reader_obj)
    normalizer = obj_factory(normalizer_obj)
    sampler = obj_factory(sampler_obj)
    file_path = "/playpen-raid1/zyshen/proj/robot/settings/datasets/toy/toy_synth/divide_3d_sphere_level1.vtk"
    file_info = {"name": file_path, "data_path": file_path}
    raw_data_dict = reader(file_info)
    normalized_data_dict = normalizer(raw_data_dict)
    sampled_data_dict = sampler(normalized_data_dict)
