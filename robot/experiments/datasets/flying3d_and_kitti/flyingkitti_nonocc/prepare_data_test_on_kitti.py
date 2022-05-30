import os
import json


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def get_file_name(file_path):
    name = os.path.split(file_path)[1].rsplit(".", 1)[0]
    return name


# data_root_path = "/playpen-raid1/Data/KITTI_processed_occ_final"
# output_path = "/playpen-raid1/zyshen/data/flying3d_nonocc_test_on_kitti"

def process(data_root_path, output_path):
    all_paths = sorted(os.walk(data_root_path))
    useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
    assert len(useful_paths) == 200, "Problem with size of kitti dataset"

    # Mapping / Filtering of scans as in HPLFlowNet code
    mapping_path = os.path.join(os.path.dirname(__file__), "../KITTI_mapping.txt")
    with open(mapping_path) as fd:
        lines = fd.readlines()
        lines = [line.strip() for line in lines]
    file_path_list = [
        path for path in useful_paths if lines[int(os.path.split(path)[-1])] != ""
    ]

    output_dict = {}
    for file_path in file_path_list:
        pair_name = get_file_name(file_path)
        output_dict[pair_name] = {}
        output_dict[pair_name]["source"] = {}
        output_dict[pair_name]["target"] = {}
        output_dict[pair_name]["source"]["name"] = pair_name + "_source"
        output_dict[pair_name]["target"]["name"] = pair_name + "_target"
        output_dict[pair_name]["source"]["data_path"] = os.path.join(file_path, "pc1.npy")
        output_dict[pair_name]["target"]["data_path"] = os.path.join(file_path, "pc2.npy")
    os.makedirs(output_path, exist_ok=True)
    save_json(os.path.join(output_path, "pair_data.json"), output_dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="preprocessing the kitti data"
    )
    parser.add_argument(
        "--data_root_path",
        required=True,
        type=str,
        default=None,
        help="the path of kitti data folder",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        default=None,
        help="the path of output folder",
    )
    args = parser.parse_args()
    print(args)
    process(args.data_root_path, args.output_path)
