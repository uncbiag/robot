import os
import numpy as np
from robot.datasets.data_utils import read_json_into_list, save_json
from robot.experiments.datasets.flying3d_and_kitti.flyingkitti_nonocc.dataset_utils import (
    flyingkitti_nonocc_reader,
)

DEPTH_THRESHOLD = 35.0
kitti_json_path = "/playpen-raid1/zyshen/data/flying3d_nonocc_test_on_kitti/model_eval/test/pair_data.json"
kitti_larger_json_path = "/playpen-raid1/zyshen/data/flying3d_nonocc_larger_kitti"
pair_name_list, pair_info_list = read_json_into_list(kitti_json_path)
count = 0
filterred_data_dict = {}
for pair_name, pair_info in zip(pair_name_list, pair_info_list):
    source_dict = flyingkitti_nonocc_reader(flying3d=False)(pair_info["source"])
    target_dict = flyingkitti_nonocc_reader(flying3d=False)(pair_info["target"])
    npoints = source_dict["points"].shape[0]
    source_dict["extra_info"]["gt_flow"] = target_dict["points"] - source_dict["points"]
    target_dict["extra_info"]["gt_flow"] = target_dict["points"] - source_dict["points"]
    near_mask = np.logical_and(
        source_dict["points"][:, 2] < DEPTH_THRESHOLD,
        target_dict["points"][:, 2] < DEPTH_THRESHOLD,
    )
    is_ground = np.logical_and(
        source_dict["points"][:, 1] < -1.4, target_dict["points"][:, 1] < -1.4
    )
    not_ground = np.logical_not(is_ground)
    near_mask = np.logical_and(near_mask, not_ground)
    indices = np.where(near_mask)[0]
    print(
        "{} points deeper= than {} has been removed, {} remained".format(
            npoints - len(indices), DEPTH_THRESHOLD, len(indices)
        )
    )
    if len(indices) > 30000:
        filterred_data_dict[pair_name] = pair_info
        count += 1

print(count)
test_folder = os.path.join(os.path.join(kitti_larger_json_path, "test"))
os.makedirs(test_folder, exist_ok=True)
save_json(os.path.join(test_folder, "pair_data.json"), filterred_data_dict)
