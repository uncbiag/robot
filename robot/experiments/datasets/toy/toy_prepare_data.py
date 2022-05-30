import os
from robot.datasets.data_utils import save_json

data_folder_path = "toy_synth"
source_path_list = [os.path.join(data_folder_path, "divide_3d_sphere_level1.vtk")]
target_path_list = [os.path.join(data_folder_path, "divide_3d_cube_level4.vtk")]
for path in target_path_list:
    assert os.path.isfile(path), "the file {} is not exist".format(path)
pair_name_list = ["sphere_cube"]
pair_sess_list = ["test"]
source_name_list = ["sphere"]
target_name_list = ["cube"]
source_list = [
    {"name": name, "data_path": os.path.abspath(path)}
    for name, path in zip(source_name_list, source_path_list)
]
target_list = [
    {"name": name, "data_path": os.path.abspath(path)}
    for name, path in zip(target_name_list, target_path_list)
]
output_folder = "./toy_synth/test"
os.makedirs(output_folder, exist_ok=True)
output_dict = {
    pair_name: {"source": source, "target": target}
    for pair_name, source, target in zip(pair_name_list, source_list, target_list)
}
save_json(os.path.join(output_folder, "pair_data.json"), output_dict)
