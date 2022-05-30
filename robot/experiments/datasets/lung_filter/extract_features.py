import argparse
import enum
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from robot.experiments.datasets.lung_filter.medical_image_utils import load_ITK, resample
from robot.experiments.datasets.lung_filter.key_point_extractor import foerstner_kpts
import torch

parser = argparse.ArgumentParser(description="Prepare data for training")
parser.add_argument('-o', '--output_path', required=True, type=str,
                    default=None, help='the path to the root of dataset folders')
parser.add_argument('-d', '--dataset_name', required=True, type=str,
                    default='', help='the name of the dataset')


class FILE_TYPE(enum.Enum):
    nii = 1
    copd = 2
    dct = 3
    copd_highres = 4


COPD_spacing = {"copd1": [0.625, 0.625, 2.5],
                "copd2": [0.645, 0.645, 2.5],
                "copd3": [0.652, 0.652, 2.5],
                "copd4": [0.590, 0.590, 2.5],
                "copd5": [0.647, 0.647, 2.5],
                "copd6": [0.633, 0.633, 2.5],
                "copd7": [0.625, 0.625, 2.5],
                "copd8": [0.586, 0.586, 2.5],
                "copd9": [0.664, 0.664, 2.5],
                "copd10": [0.742, 0.742, 2.5]}
COPD_shape = {"copd1": [121, 512, 512],
              "copd2": [102, 512, 512],
              "copd3": [126, 512, 512],
              "copd4": [126, 512, 512],
              "copd5": [131, 512, 512],
              "copd6": [119, 512, 512],
              "copd7": [112, 512, 512],
              "copd8": [115, 512, 512],
              "copd9": [116, 512, 512],
              "copd10": [135, 512, 512]}
FDCT_spacing = {"dct1": [0.97, 0.97, 2.5],
                "dct2": [1.16, 1.16, 2.5],
                "dct3": [1.15, 1.15, 2.5],
                "dct4": [1.13, 1.13, 2.5],
                "dct5": [1.10, 1.10, 2.5],
                "dct6": [0.97, 0.97, 2.5],
                "dct7": [0.97, 0.97, 2.5],
                "dct8": [0.97, 0.97, 2.5],
                "dct9": [0.97, 0.97, 2.5],
                "dct10": [0.97, 0.97, 2.5]}
FDCT_shape = {"dct1": [94, 256, 256],
              "dct2": [112, 256, 256],
              "dct3": [104, 256, 256],
              "dct4": [99, 256, 256],
              "dct5": [106, 256, 256],
              "dct6": [128, 512, 512],
              "dct7": [136, 512, 512],
              "dct8": [128, 512, 512],
              "dct9": [128, 512, 512],
              "dct10": [120, 512, 512]
              }

Data_Info = {
    "train":
        # {
        # "data_list_path": "/home/zyshen/data/raw_lung_data",
        # "file_type": FILE_TYPE.nii,
        # "idx_file_name": "data_id"},
        {
            "data_list_path":"/playpen-raid1/Data/Lung_Registration",
        "file_type":FILE_TYPE.nii,
        "idx_file_name":"data_id"},

    "val": [
        {
            "data_list_path": "/playpen-raid1/lin.tian/data/raw/copd",
            "file_type": FILE_TYPE.copd,
            "idx_file_name": "copd_data_id"
        },
        {
            "data_list_path": "/playpen-raid1/lin.tian/data/raw/4DCT",
            "file_type": FILE_TYPE.dct,
            "idx_file_name": "dct_data_id"
        },
        {
            "data_list_path": "/playpen-raid1/lin.tian/data/raw/DIRLABCasesHighRes",
            "file_type": FILE_TYPE.copd_highres,
            "idx_file_name": "copd_highres_data_id"
        }
    ]
}


def normalize_intensity(img, linear_clip=False):
    # TODO: Lin-this line is for CT. Modify it to make this method more general.
    img[img < -1024] = -1024
    return img
    # if linear_clip:
    #     img = img - img.min()
    #     normalized_img =img / np.percentile(img, 95) * 0.95
    # else:
    #     min_intensity = img.min()
    #     max_intensity = img.max()
    #     normalized_img = (img-img.min())/(max_intensity - min_intensity)
    # normalized_img = normalized_img*2 - 1
    # return normalized_img


def process_single_file(path_pair, spacing, device=torch.device("cpu")):
    ori_source, ori_spacing, ori_sz, source_origin = load_ITK(path_pair[0])
    ori_source = np.swapaxes(sitk.GetArrayFromImage(ori_source), 0, 2)
    source, _, _ = resample(ori_source, ori_spacing, spacing)

    ori_target, ori_spacing, ori_sz, target_origin = load_ITK(path_pair[1])
    ori_target = np.swapaxes(sitk.GetArrayFromImage(ori_target), 0, 2)
    target, _, _ = resample(ori_target, ori_spacing, spacing)

    ori_source_seg, ori_spacing, ori_sz, _ = load_ITK(path_pair[2])
    ori_source_seg_np = sitk.GetArrayFromImage(ori_source_seg)
    ori_source_seg_np[ori_source_seg_np > 0] = 1
    ori_source_seg = np.swapaxes(ori_source_seg_np, 0, 2)
    source_seg, _, _ = resample(ori_source_seg, ori_spacing, spacing, mode="nearest")

    ori_target_seg, ori_spacing, ori_sz, _ = load_ITK(path_pair[3])
    ori_target_seg_np = sitk.GetArrayFromImage(ori_target_seg)
    ori_target_seg_np[ori_target_seg_np > 0] = 1
    ori_target_seg = np.swapaxes(ori_target_seg_np, 0, 2)
    target_seg, _, _ = resample(ori_target_seg, ori_spacing, spacing, mode="nearest")
    source[source < -1000] = -1000
    target[target < -1000] = -1000

    pts_source = foerstner_kpts(
        (torch.from_numpy(source).to(device).unsqueeze(0).unsqueeze(0).float().clamp_(-1000, 1500) + 1000) / 2500,
        torch.from_numpy(source_seg).to(device).unsqueeze(0).unsqueeze(0), d=5, num_points=None).cpu()
    pts_target = foerstner_kpts(
        (torch.from_numpy(target).to(device).unsqueeze(0).unsqueeze(0).float().clamp_(-1000, 1500) + 1000) / 2500,
        torch.from_numpy(target_seg).to(device).unsqueeze(0).unsqueeze(0), d=5, num_points=None).cpu()

    return source, target, source_seg, target_seg, pts_source, pts_target, source_origin, target_origin


def read_data_list(data_folder_path):
    '''
        load data files and set the path into a list.
        Inspiration image and label are at pos 0 and pos 2.
        Expiration image and label are at pos 1 and pos 3.
    '''
    case_list = os.listdir(data_folder_path)
    return_list = []
    for case in case_list:
        case_dir = os.path.join(data_folder_path, case)
        case_data = ["", "", "", "", case]
        files = os.listdir(case_dir)
        for f in files:
            if "_EXP_" in f:
                if "_img." in f:
                    case_data[1] = os.path.join(case_dir, f)
                elif "_label." in f:
                    case_data[3] = os.path.join(case_dir, f)
            elif "_INSP_" in f:
                if "_img." in f:
                    case_data[0] = os.path.join(case_dir, f)
                elif "_label." in f:
                    case_data[2] = os.path.join(case_dir, f)
        return_list.append(case_data)
    return return_list


def read_copd_highres_data_list(data_folder_path):
    # This is the dirlab copd corresponding data in Raul's data
    case_list = os.listdir(data_folder_path)
    return_list = []
    for case in case_list:
        # if case not in ['copd3', 'copd4', 'copd5', 'copd9']:
        #     continue
        case_dir = os.path.join(data_folder_path, case)
        case_data = ["", "", "", "", f"{case}_highres"]
        case_data[0] = os.path.join(case_dir, case + '_INSP.nrrd')
        case_data[1] = os.path.join(case_dir, case + '_EXP.nrrd')
        return_list.append(case_data)
    return return_list


def read_copd_data_list(data_folder_path):
    case_list = os.listdir(data_folder_path)
    return_list = []
    for case in case_list:
        case_dir = os.path.join(data_folder_path, case + '/' + case)
        case_data = ["", "", "", "", case]
        case_data[0] = os.path.join(case_dir, case + '_iBHCT.img')
        case_data[1] = os.path.join(case_dir, case + '_eBHCT.img')
        return_list.append(case_data)
    return return_list


def read_dct_data_list(data_folder_path):
    case_list = os.listdir(data_folder_path)
    return_list = []
    for case in case_list:
        case_id = case.lower()[0:case.find('Pack')]
        case_dir = os.path.join(data_folder_path, case + '/Images')

        # rename the file
        # for f in os.listdir(case_dir):
        #     series_id = f[f.find('_T')+1: f.find('_T')+4]
        #     os.rename(os.path.join(case_dir, f), os.path.join(case_dir, case_id+'_'+series_id+'.img'))

        case_data = ["", "", "", "", "dct" + case_id[4:]]
        case_data[0] = os.path.join(case_dir, case_id + '_T00.img')
        case_data[1] = os.path.join(case_dir, case_id + '_T50.img')
        return_list.append(case_data)
    return return_list


def plot_preprocessed(source, target, save_path, source_seg=None, target_seg=None):
    if source_seg is not None and target_seg is not None:
        fig, axes = plt.subplots(4, 4)
    else:
        fig, axes = plt.subplots(2, 4)
    for i in range(4):
        axes[0, i].imshow(source[:, 30 * i, :])
        axes[1, i].imshow(target[:, 30 * i, :])
        if source_seg is not None and target_seg is not None:
            axes[2, i].imshow(source_seg[:, 30 * i, :])
            axes[3, i].imshow(target_seg[:, 30 * i, :])
    axes[0, 0].set_ylabel("Source")
    axes[1, 0].set_ylabel("Target")
    if source_seg is not None and target_seg is not None:
        axes[2, 0].set_ylabel("Source Seg")
        axes[3, 0].set_ylabel("Target Seg")
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def preprocess(data_folder_path, preprocessed_path, case_num=200):
    if not os.path.exists(data_folder_path):
        print("Did not find data list file at %s" % data_folder_path)
        return

    file_list = read_data_list(data_folder_path)

    if len(file_list) > case_num:
        file_list = file_list[:case_num]

    case_id_list = []
    data_count = len(file_list)
    for i in range(data_count):
        case_id = file_list[i][4]
        case_id_list.append(case_id)
        print("Preprocessing %i/%i %s" % (i, data_count, case_id))

        source, target, source_seg, target_seg, source_pts, target_pts, source_origin, target_origin = process_single_file(
            file_list[i],
            np.array((1, 1, 1)))
        np.save(os.path.join(preprocessed_path, "%s_target.npy" % case_id), target)
        np.save(os.path.join(preprocessed_path, "%s_source.npy" % case_id), source)
        np.save(os.path.join(preprocessed_path, "%s_source_seg.npy" % case_id), source_seg)
        np.save(os.path.join(preprocessed_path, "%s_target_seg.npy" % case_id), target_seg)
        np.save(os.path.join(preprocessed_path, "%s_source_pts.npy" % case_id), source_pts)
        np.save(os.path.join(preprocessed_path, "%s_target_pts.npy" % case_id), target_pts)
        np.save(os.path.join(preprocessed_path, "%s_source_origin.npy" % case_id), source_origin)
        np.save(os.path.join(preprocessed_path, "%s_target_origin.npy" % case_id), target_origin)

    return case_id_list


def save_id_list(task_root, file_name, case_id_list, mode="train"):
    if mode == "train":
        train_data_id_path = task_root + "/train"
        if not os.path.exists(train_data_id_path):
            os.makedirs(train_data_id_path)

        val_data_id_path = task_root + "/val"
        if not os.path.exists(val_data_id_path):
            os.makedirs(val_data_id_path)

        debug_data_id_path = task_root + "/debug"
        if not os.path.exists(debug_data_id_path):
            os.makedirs(debug_data_id_path)
        case_count = len(case_id_list)
        np.random.shuffle(case_id_list)
        train_list = case_id_list[:int(case_count * 4 / 5)]
        val_list = case_id_list[int(case_count * 4 / 5):case_count]

        np.save(os.path.join(train_data_id_path, file_name), train_list)
        np.save(os.path.join(debug_data_id_path, file_name), train_list)
        np.save(os.path.join(val_data_id_path, file_name), val_list)
    elif mode == "test":
        test_data_id_path = task_root + "/test"
        if not os.path.exists(test_data_id_path):
            os.makedirs(test_data_id_path)
        np.save(os.path.join(test_data_id_path, file_name), case_id_list)


if __name__ == "__main__":
    # After processing, the ct image should in SAR orientation.
    args = parser.parse_args()

    task_root = os.path.join(os.path.abspath(args.output_path), args.dataset_name)
    preprocessed_path = task_root + "/preprocessed/"
    if not os.path.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    data_list_path = Data_Info["train"]["data_list_path"]
    file_type = Data_Info["train"]["file_type"]
    id_file_name = Data_Info["train"]["idx_file_name"]
    case_id_list = preprocess(data_list_path, preprocessed_path, case_num=10)

