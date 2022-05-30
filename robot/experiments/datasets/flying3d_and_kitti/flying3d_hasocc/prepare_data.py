import os
import glob
import numpy as np
from robot.datasets.data_utils import get_file_name, save_json

# Get list of filenames / directories
root_dir = "/playpen-raid1/Data/data_processed_maxcut_35_20k_2k_8192"
output_dir = "/playpen-raid1/zyshen/data/flying3d_hasocc"
for mode in ["train", "val", "test"]:
    if mode == "train" or mode == "val":
        pattern = "TRAIN_*.npz"
    elif mode == "test":
        pattern = "TEST_*.npz"
    else:
        raise ValueError("Mode " + str(mode) + "unknown.")
    file_path_list = glob.glob(os.path.join(root_dir, pattern))

    # Remove one sample containing a nan value in train set
    scan_with_nan_value = os.path.join(root_dir, "TRAIN_C_0140_left_0006-0.npz")
    if scan_with_nan_value in file_path_list:
        file_path_list.remove(scan_with_nan_value)

    # Remove samples with all points occluded in train set
    scan_with_points_all_occluded = [
        "TRAIN_A_0364_left_0008-0.npz",
        "TRAIN_A_0364_left_0009-0.npz",
        "TRAIN_A_0658_left_0014-0.npz",
        "TRAIN_B_0053_left_0009-0.npz",
        "TRAIN_B_0053_left_0011-0.npz",
        "TRAIN_B_0424_left_0011-0.npz",
        "TRAIN_B_0609_right_0010-0.npz",
    ]
    for f in scan_with_points_all_occluded:
        if os.path.join(root_dir, f) in file_path_list:
            file_path_list.remove(os.path.join(root_dir, f))

    # Remove samples with all points occluded in test set
    scan_with_points_all_occluded = [
        "TEST_A_0149_right_0013-0.npz",
        "TEST_A_0149_right_0012-0.npz",
        "TEST_A_0123_right_0009-0.npz",
        "TEST_A_0123_right_0008-0.npz",
    ]
    for f in scan_with_points_all_occluded:
        if os.path.join(root_dir, f) in file_path_list:
            file_path_list.remove(os.path.join(root_dir, f))

    # Train / val / test split
    if mode == "train" or mode == "val":
        ind_val = set(np.linspace(0, len(file_path_list) - 1, 2000).astype("int"))
        ind_all = set(np.arange(len(file_path_list)).astype("int"))
        ind_train = ind_all - ind_val
        assert (
            len(ind_train.intersection(ind_val)) == 0
        ), "Train / Val not split properly"
        file_path_list = np.sort(file_path_list)
        if mode == "train":
            file_path_list = file_path_list[list(ind_train)]
        elif mode == "val":
            file_path_list = file_path_list[list(ind_val)]

    output_dict = {}
    for file_path in file_path_list:
        pair_name = get_file_name(file_path)
        output_dict[pair_name] = {}
        output_dict[pair_name]["source"] = {}
        output_dict[pair_name]["target"] = {}
        output_dict[pair_name]["source"]["name"] = pair_name + "_source"
        output_dict[pair_name]["source"]["type"] = "source"
        output_dict[pair_name]["target"]["name"] = pair_name + "_target"
        output_dict[pair_name]["target"]["type"] = "target"
        output_dict[pair_name]["source"]["data_path"] = file_path
        output_dict[pair_name]["target"]["data_path"] = file_path
    os.makedirs(os.path.join(output_dir, mode), exist_ok=True)
    save_json(os.path.join(output_dir, mode, "pair_data.json"), output_dict)
    if mode == "train":
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)
        save_json(os.path.join(output_dir, "debug", "pair_data.json"), output_dict)
