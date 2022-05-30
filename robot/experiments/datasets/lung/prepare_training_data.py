import os, sys
sys.path.insert(0, os.path.abspath("../../../.."))
from robot.datasets.data_utils import save_json, write_list_into_txt
from robot.datasets.prepare_reg_data import GeneralDataSet

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="split the lung vessel tree dataset into train/val/test/debug, *debug refers to a sub-train set",
    )
    parser.add_argument("--val_dirlab", action="store_true", help="use the dirlab as the validation set")
    parser.add_argument(
        "-cf",
        "--copd_data_folder",
        type=str,
        default="",
        help="data folder including COPD lung vessel tree vtk data",
    )
    parser.add_argument(
        "-of",
        "--output_folder",
        type=str,
        default="",
        help="output folder, the experiment splits will be saved into this output folder",
    )
    args = parser.parse_args()
    print(args)
    insp_key = "*_INSP*"
    copd_vessel_data_folder = args.copd_data_folder    #"/playpen-raid1/Data/UNC_vesselParticles_cleaned"
    lung_expri_path = args.output_folder #"/playpen-raid1/zyshen/release_debug/lung_expri"
    use_dirlab_as_validation_set = True
    os.makedirs(lung_expri_path, exist_ok=True)
    NUM_CASE = 1000
    insp_name_list = ["copd_" + "{:06d}_INSP".format(i) for i in range(11, NUM_CASE+11)]
    exp_name_list = ["copd_" + "{:06d}_EXP".format(i) for i in range(11, NUM_CASE+11)]
    insp_path_list = [os.path.join(copd_vessel_data_folder, insp_name+".vtk") for insp_name in insp_name_list]
    exp_path_list = [os.path.join(copd_vessel_data_folder, exp_name+".vtk") for exp_name in exp_name_list]
    insp_list = [
        {"name": insp_name, "data_path": insp_path}
        for insp_name, insp_path in zip(insp_name_list, insp_path_list)
    ]
    exp_list = [
        {"name": exp_name, "data_path": exp_path}
        for exp_name, exp_path in zip(exp_name_list, exp_path_list)
    ]
    dataset = GeneralDataSet()
    dataset.set_output_path(lung_expri_path)
    dataset.set_coupled_pair_list([exp_list, insp_list])
    if not use_dirlab_as_validation_set:
        dataset.set_divided_ratio((0.8, 0.2, 0.0))  # train, val, test
    else:
        dataset.set_divided_ratio((1.0, 0.0, 0.0))  # train, val, test
    dataset.prepare_data()


    # now let's processing dirlab testing case, by default we put dirlab cases into the test folder

    insp_name_list = ["copd_" + "{:06d}_INSP".format(i) for i in range(1,11)]
    exp_name_list = ["copd_" + "{:06d}_EXP".format(i) for i in range(1,11)]
    insp_path_list = [os.path.join(copd_vessel_data_folder, insp_name+".vtk") for insp_name in insp_name_list]
    exp_path_list = [os.path.join(copd_vessel_data_folder, exp_name+".vtk") for exp_name in exp_name_list]
    insp_list = [
        {"name": insp_name, "data_path": insp_path}
        for insp_name, insp_path in zip(insp_name_list, insp_path_list)
    ]
    exp_list = [
        {"name": exp_name, "data_path": exp_path}
        for exp_name, exp_path in zip(exp_name_list, exp_path_list)
    ]
    pair_name_list = [exp_name + "_" + insp_name for exp_name, insp_name in
                      zip(exp_name_list, insp_name_list)]
    pair_path_list = [[exp_path, insp_path] for exp_path, insp_path in zip(exp_path_list, insp_path_list)]
    output_dict = {
        pair_name: {"source": exp, "target": insp}
        for pair_name, insp, exp in zip(pair_name_list, insp_list, exp_list)
    }

    expri_dirlab_test_folder = os.path.join(lung_expri_path,'test')
    save_json(os.path.join(expri_dirlab_test_folder, "pair_data.json"), output_dict)
    write_list_into_txt(os.path.join(expri_dirlab_test_folder, "pair_name_list.txt"), pair_name_list)
    write_list_into_txt(os.path.join(expri_dirlab_test_folder, "pair_path_list.txt"), pair_path_list)
    if use_dirlab_as_validation_set:
        expri_dirlab_val_folder = os.path.join(lung_expri_path, 'val')
        save_json(os.path.join(expri_dirlab_val_folder, "pair_data.json"), output_dict)
        write_list_into_txt(os.path.join(expri_dirlab_val_folder, "pair_name_list.txt"), pair_name_list)
        write_list_into_txt(os.path.join(expri_dirlab_val_folder, "pair_path_list.txt"), pair_path_list)