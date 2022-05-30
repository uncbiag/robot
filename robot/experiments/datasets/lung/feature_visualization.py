"""
the feature comes from deep feature learning, and tsne projection (3 dim)
"""
import os
from robot.datasets.vtk_utils import read_vtk
from robot.global_variable import Shape, ROBOT_PATH
from robot.utils.visualizer import visualize_point_fea, visualize_point_pair, default_plot


def reader(file_path):
    raw_data_dict = read_vtk(file_path)
    points = raw_data_dict["points"][:, None]
    weights = raw_data_dict["weights"][:, None]
    pointfea = raw_data_dict["pointfea"][:, None]
    shape = Shape().set_data(points=points, weights=weights, pointfea=pointfea)
    return shape


# folder_path = "/home/zyshen/remote/llr11_mount/zyshen/data/lung_expri/model_eval/deep_feature_pointconv_dirlab_complex_aniso_15dim_normalized_60000/records/fea_visual"
folder_path = os.path.join(ROBOT_PATH,"demos/data/lung_data/lung_deep_feature_visual")
gif_folder_path = os.path.join(ROBOT_PATH,"demos/output/lung_demo/lung_deep_feature_visual")
os.makedirs(gif_folder_path,exist_ok=True)
gif_path = os.path.join(gif_folder_path,"output.gif")

case_name = "copd1"
source_path = os.path.join(folder_path, "{}_source.vtk".format(case_name))
target_path = os.path.join(folder_path, "{}_target.vtk".format(case_name))
source = reader(source_path)
target = reader(target_path)
camera_pos = [
    [(2.576607272393312, 7.604950051542313, 1.5688047371066083),
 (0.0, 0.0, 0.0),
 (-0.3947804584146587, -0.05543223798587872, 0.9171017700592389)],
[(-6.645152022006699, 4.6303031874159055, 1.1565317214828066),
 (0.0, 0.0, 0.0),
 (-0.08966640328220832, -0.3605772093151145, 0.9284093990503095)]
    ]

visualize_point_pair(
    source.points.squeeze(),
    target.points.squeeze(),
    source.pointfea.squeeze(),
    target.pointfea.squeeze(),
    "Source Feature",
    "Target Feature",
    camera_pos=camera_pos,
    source_plot_func=default_plot(rgb=True),
    target_plot_func=default_plot(rgb=True),
    col_adaptive=False,
    saving_gif_path=gif_path
)
