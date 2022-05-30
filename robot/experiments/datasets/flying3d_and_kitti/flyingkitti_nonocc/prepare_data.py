import os
import glob
import numpy as np
from robot.datasets.data_utils import get_file_name, save_json

# Get list of filenames / directories
root_dir = "/playpen-raid1/Data/FlyingThings3D_subset_processed_35m"
output_dir = "/playpen-raid1/zyshen/data/flyingkitti_nonocc"
for mode in ["train", "val", "test"]:
    if mode == "train" or mode == "val":
        pattern = "train/0*"
    elif mode == "test":
        pattern = "val/0*"
    else:
        raise ValueError("Mode " + str(mode) + " unknown.")
    file_path_list = glob.glob(os.path.join(root_dir, pattern))

    # Train / val / test split
    if mode == "train" or mode == "val":
        assert len(file_path_list) == 19640, "Problem with size of training set"
        ind_val = set(np.linspace(0, 19639, 2000).astype("int"))
        ind_all = set(np.arange(19640).astype("int"))
        ind_train = ind_all - ind_val
        assert (
            len(ind_train.intersection(ind_val)) == 0
        ), "Train / Val not split properly"
        file_path_list = np.sort(file_path_list)
        if mode == "train":
            file_path_list = file_path_list[list(ind_train)]
        elif mode == "val":
            file_path_list = file_path_list[list(ind_val)]
    else:
        assert len(file_path_list) == 3824, "Problem with size of test set"
    output_dict = {}
    for file_path in file_path_list:
        pair_name = get_file_name(file_path)
        output_dict[pair_name] = {}
        output_dict[pair_name]["source"] = {}
        output_dict[pair_name]["target"] = {}
        output_dict[pair_name]["source"]["name"] = pair_name + "_source"
        output_dict[pair_name]["target"]["name"] = pair_name + "_target"
        output_dict[pair_name]["source"]["data_path"] = os.path.join(
            file_path, "pc1.npy"
        )
        output_dict[pair_name]["target"]["data_path"] = os.path.join(
            file_path, "pc2.npy"
        )
    os.makedirs(os.path.join(output_dir, mode), exist_ok=True)
    save_json(os.path.join(output_dir, mode, "pair_data.json"), output_dict)
    if mode == "train":
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)
        save_json(os.path.join(output_dir, "debug", "pair_data.json"), output_dict)
