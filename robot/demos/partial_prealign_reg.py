"""
this script provides a demo on partial registration
"""

import os, sys

sys.path.insert(0, os.path.abspath("../.."))

from robot.utils.utils import memory_sort

import copy
import numpy as np
import open3d as o3
from probreg import cpd, filterreg
import transformations as trans
from probreg import features
from robot.utils.visualizer import visualize_source_flowed_target_overlap, get_slerp_cam_pos
import torch

from robot.global_variable import *
from robot.models_reg.multiscale_optimization import *
from robot.utils.module_parameters import ParameterDict
from robot.datasets.data_utils import compute_interval
from robot.demos.demo_utils import get_omt_mapping
from robot.utils.visualizer import *


def rigid_deform(points, rotation, translation):
    from scipy.spatial.transform import Rotation as R

    translation = translation
    r = R.from_euler("zyx", rotation, degrees=True)
    r_matrix = r.as_matrix()
    r_matrix = torch.tensor(r_matrix, dtype=torch.float, device=points.device)
    deformed_points = points @ r_matrix + torch.tensor(
        translation, device=points.device
    )
    return deformed_points

expri_settings = {
    "bunny":{"data_path":"./data/toy_demo_data/bunny.pcd","FPFH_settings":{"radius_normal":0.02, "radius_feature":0.05}},
    "dragon":{"data_path":"./data/toy_demo_data/dragon_10k.ply","FPFH_settings":{"radius_normal":0.02, "radius_feature":0.05}},
    "armadillo":{"data_path":"./data/toy_demo_data/armadillo_10k.ply","FPFH_settings":{"radius_normal":0.02, "radius_feature":0.05}}
                  }


# prepare data
expri_name = "bunny"
assert expri_name in expri_settings
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # "cuda:0" if torch.cuda.is_available() else "cpu"
totensor = lambda x: torch.tensor(np.asarray(x.points).astype(np.float32))
source = o3.io.read_point_cloud(expri_settings[expri_name]["data_path"])
# source = o3.io.read_point_cloud('./data/toy_demo_data/bunny_high.ply')

source = source.voxel_down_sample(voxel_size=0.005)
source_points = totensor(source).to(device)
source_points, _ = memory_sort(source_points, 0.005)
target_points = rigid_deform(
    source_points, rotation=[120, 10, 10], translation=[0.01, 0.02, -0.01]
)
source_points, target_points = source_points[None], target_points[None]
nsource = source_points.shape[1]
ntarget = target_points.shape[1]
source_points = source_points[:, int(3 * nsource / 10) : -1]
target_points = target_points[:, 0 : int(5 * ntarget / 8)]


# prepare data for probreg
source = o3.geometry.PointCloud()
source.points = o3.utility.Vector3dVector(source_points.squeeze().cpu().numpy())
target = o3.geometry.PointCloud()
target.points = o3.utility.Vector3dVector(target_points.squeeze().cpu().numpy())
cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
# source_fea = features.FPFH()(cv(source)).astype(np.float32)[None]
# target_fea = features.FPFH()(cv(source)).astype(np.float32)[None]
source_fea = None
target_fea = None

#
# """ Experiment 1  run cpd registration from probreg package """
# tf_param, _, _ = cpd.registration_cpd(
#     source, target, tf_type_name="rigid", w=0.7, maxiter=200, tol=0.0001
# )
# result = copy.deepcopy(source)
# result.points = tf_param.transform(result.points)
# source.paint_uniform_color([1, 0, 0])
# target.paint_uniform_color([0, 1, 0])
# result.paint_uniform_color([0, 0, 1])
# # o3.visualization.draw_geometries([source, target, result])
# visualize_source_flowed_target_overlap(
#     np.asarray(source.points),
#     np.asarray(result.points),
#     np.asarray(target.points),
#     np.asarray(source.points),
#     np.asarray(source.points),
#     np.asarray(target.points),
#     "CPD: source",
#     "prealigned",
#     "target",
#     show=True,
#     add_bg_contrast=False,
# )
#

""" Experiment 2  run filterreg registration from probreg package """
cbs = []  # [callbacks.Open3dVisualizerCallback(source, target)]
objective_type = "pt2pt"
tf_param, _, _ = filterreg.registration_filterreg(
    source,
    target,
    objective_type=objective_type,
    update_sigma2=True,
    w=0.9,
    tol=1e-9,
    sigma2=None,
    feature_fn=features.FPFH(**expri_settings[expri_name]["FPFH_settings"]),
    callbacks=cbs,
)
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
# o3.visualization.draw_geometries([source, target, result])
visualize_source_flowed_target_overlap(
    np.asarray(source.points),
    np.asarray(result.points),
    np.asarray(target.points),
    np.asarray(source.points),
    np.asarray(result.points),
    np.asarray(target.points),
    "Filterreg: source",
    "prealigned",
    "target",
    source_plot_func=default_plot(cmap="viridis",rgb=True),
    flowed_plot_func=default_plot(cmap="viridis",rgb=True),
    target_plot_func=default_plot(cmap="magma",rgb=True),
    show=True,
    add_bg_contrast=False,
)


""" Experiment 3  run Robust optimal transport """
source = Shape().set_data(points=source_points, pointfea=None)
target = Shape().set_data(points=target_points, pointfea=None)
compute_interval(source.points[0].cpu().numpy())
shape_pair = create_shape_pair(source, target)

task_name = "prealign_opt"
solver_opt = ParameterDict()
record_path = "./output/prealign_demo/{}".format(task_name)
os.makedirs(record_path, exist_ok=True)
solver_opt = ParameterDict()
record_path = "./output/prealign_demo/{}".format(task_name)
os.makedirs(record_path, exist_ok=True)
solver_opt["record_path"] = record_path
solver_opt["save_2d_capture_every_n_iter"] = 1
solver_opt["capture_plot_obj"] = "visualizer.capture_plotter()"
model_name = "prealign_opt"
model_opt = ParameterDict()
model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
model_opt["module_type"] = "gradflow_prealign"
model_opt[("gradflow_prealign", {}, "settings for gradflow_prealign")]
blur = 0.1  # 0.05
model_opt["gradflow_prealign"]["method_name"] = "rigid"  # affine
model_opt["gradflow_prealign"]["gradflow_mode"] = "ot_mapping"
model_opt["gradflow_prealign"]["niter"] = 10
model_opt["gradflow_prealign"]["plot"] = True
model_opt["gradflow_prealign"]["use_barycenter_weight"] = True
model_opt["gradflow_prealign"]["search_init_transform"] = False
model_opt["gradflow_prealign"][("geomloss", {}, "settings for geomloss")]
model_opt["gradflow_prealign"]["geomloss"]["mode"] = "soft"
model_opt["gradflow_prealign"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur={},reach=2, scaling=0.9,debias=False)".format(
    blur
)
model_opt["gradflow_prealign"][
    "pair_feature_extractor_obj"
] = "local_feature_extractor.pair_feature_FPFH_extractor(radius_normal={}, radius_feature={})"\
    .format(
    expri_settings[expri_name]["FPFH_settings"]["radius_normal"], # it only works for the same scale
    expri_settings[expri_name]["FPFH_settings"]["radius_feature"])  # it only works for the same scale

model_opt["sim_loss"]["loss_list"] = ["geomloss"]
model_opt["sim_loss"][("geomloss", {}, "settings for geomloss")]
model_opt["sim_loss"]["geomloss"]["attr"] = "pointfea"
model_opt["sim_loss"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur={},reach=1, scaling=0.8,debias=False)".format(
    blur
)
model = MODEL_POOL[model_name](model_opt)
solver = build_single_scale_model_embedded_solver(solver_opt, model)
model.init_reg_param(shape_pair)
shape_pair = solver(shape_pair)
visualize_source_flowed_target_overlap(
    shape_pair.source.points,
    shape_pair.flowed.points,
    shape_pair.target.points,
    shape_pair.source.points,
    shape_pair.source.points,
    shape_pair.target.points,
    "OT: source",
    "prealigned",
    "target",
    show=True,
    add_bg_contrast=False,
)


from robot.utils.generate_animation import FlowModel, visualize_animation, generate_gif
stage_name = "rigid"
model_type = "affine_interp"
flow_opt = ParameterDict()
flow_opt["model_type"] = model_type
flow_opt["t_list"] = list(np.linspace(0, 1.0, num=20))
target_list = [target] * len(flow_opt["t_list"])
flow_model = FlowModel(flow_opt)
# shape_pair = create_shape_pair(source, target, pair_name="partial", n_control_points=-1)
# shape_pair.reg_param = prealign_reg_param
flowed_list = flow_model(shape_pair)
camera_setting= {"dragon":
                     {"camera_pos_start":[(1.6266550924236722, -2.099841657967842, -6.593959765170743),
                        (0.0, 0.0, 0.0),
                         (0.7551972577330768, -0.5473888228098531, 0.3606141685725674)],
                        "camera_pos_end":[(-3.0655411776452, -4.8116479294033745, -4.24100797624424),
                        (0.0, 0.0, 0.0),
                        (0.851354951228469, -0.5241825060822709, -0.02067479954150231)]},
                "armadillo":
                     {"camera_pos_start":[(1.6266550924236722, -2.099841657967842, -6.593959765170743),
                        (0.0, 0.0, 0.0),
                         (0.7551972577330768, -0.5473888228098531, 0.3606141685725674)],
                        "camera_pos_end":[(-3.0655411776452, -4.8116479294033745, -4.24100797624424),
                        (0.0, 0.0, 0.0),
                        (0.851354951228469, -0.5241825060822709, -0.02067479954150231)]},
                "bunny":
                     {"camera_pos_start":[(0.8431791512037158, 0.2101021826733043, 0.08686872665705404),
                         (0.0, 0.0, 0.0),
                         (0.26027809752959574, -0.883768292340572, -0.3888559082742651)],
                        "camera_pos_end":[(0.6165830947396432, 0.1284806320433733, 0.6049447894897156),
                         (0.0, 0.0, 0.0),
                         (0.44590473117812224, -0.8522803083860243, -0.2734725701974333)]
}

                 }

pos_interp_list = [get_slerp_cam_pos(camera_setting[expri_name]["camera_pos_start"], camera_setting[expri_name]["camera_pos_end"], t) for t in flow_opt["t_list"]]

#pos_interp_list = [camera_pos_start]*len(flow_opt["t_list"])
gif_output_folder = os.path.join(record_path,"gif")
os.makedirs(gif_output_folder,exist_ok=True)
saving_capture_path_list = [os.path.join(gif_output_folder, "t_{:.2f}.png").format(t) for t in flow_opt["t_list"]]
title1_list = [stage_name] * len(flow_opt["t_list"])
title2_list = ["target"] * len(flow_opt["t_list"])
for _flowed in flowed_list:
    _flowed.weights = source.points

for _target in target_list:
    _target.weights = target.points

visualize_animation(flowed_list,target_list,title1_list,title2_list,saving_capture_path_list=saving_capture_path_list,camera_pos_list=pos_interp_list,light_mode="light_kit",show=False)
saving_capture_path_list += [saving_capture_path_list[-1]]*10
gif_path = os.path.join(record_path,"gif", "{}.gif".format(expri_name))
generate_gif(saving_capture_path_list,gif_path)