import torch

import os, sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
import numpy as np
import torch
from robot.utils.module_parameters import ParameterDict
from robot.datasets.data_utils import get_file_name, generate_pair_name, get_obj
from robot.shape.shape_pair_utils import create_shape_pair
from robot.models_reg.multiscale_optimization import (
    build_single_scale_model_embedded_solver,
    build_multi_scale_solver,
)
from robot.global_variable import MODEL_POOL, Shape, shape_type
from robot.utils.utils import get_grid_wrap_points
from robot.utils.visualizer import (
    visualize_point_fea,
    visualize_point_pair,
    visualize_multi_point, visualize_source_flowed_target_overlap, default_plot,
)
from robot.demos.demo_utils import *
from robot.utils.utils import timming

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
expri_settings = {
    "bunny": {"data_path": "./data/toy_demo_data/bunny_30k.ply"},
    "dragon": {"data_path": "./data/toy_demo_data/dragon_30k.ply"},
    "armadillo": {"data_path": "./data/toy_demo_data/armadillo_30k.ply"}
}
output_path = "./output/ot_fun"
bunny_path = expri_settings["bunny"]["data_path"]
dragon_path = expri_settings["dragon"]["data_path"]
armadillo_path = expri_settings["armadillo"]["data_path"]
os.makedirs(output_path, exist_ok=True)

####################  prepare data ###########################
reader_obj = "toy_dataset_utils.toy_reader()"
sampler_obj = "toy_dataset_utils.toy_sampler()"
normalizer_obj = "toy_dataset_utils.toy_normalizer()"
get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device)
bunny_obj, bunny_interval = get_obj_func(bunny_path)
points = bunny_obj["points"]

source_path = "./data/toy_demo_data/divide_3d_sphere_level1.vtk"
target_path = expri_settings["bunny"]["data_path"]


####################  prepare data ###########################
pair_name = "generate_pair_name([source_path, target_path])"
reader_obj = "toy_dataset_utils.toy_reader()"
sampler_obj = "toy_dataset_utils.toy_sampler()"
source_normalizer_obj = "toy_dataset_utils.toy_normalizer(scale=0.2)"
target_normalizer_obj = "toy_dataset_utils.toy_normalizer()"
get_obj_source_func = get_obj(reader_obj, source_normalizer_obj, sampler_obj, device)
get_obj_target_func = get_obj(reader_obj, target_normalizer_obj, sampler_obj, device)
source_obj, source_interval = get_obj_source_func(source_path)
target_obj, target_interval = get_obj_target_func(target_path)
min_interval = min(source_interval, target_interval)
input_data = {"source": source_obj, "target": target_obj}
create_shape_pair_from_data_dict = obj_factory(
    "shape_pair_utils.create_source_and_target_shape()"
)
source, target = create_shape_pair_from_data_dict(input_data)

shape_pair = create_shape_pair(source, target)
shape_pair.pair_name = "toy"


""" Experiment 1:  Robust optimal transport """
task_name = "gradient_flow"
solver_opt = ParameterDict()
record_path =  "./output/toy_reg/{}".format(task_name)
os.makedirs(record_path, exist_ok=True)
solver_opt["record_path"] = record_path
model_name = "gradient_flow_opt"
model_opt = ParameterDict()
model_opt[
    "interpolator_obj"
] = "point_interpolator.nadwat_kernel_interpolator(scale=0.1, exp_order=2)"
model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
model_opt["sim_loss"]["loss_list"] = ["geomloss"]
model_opt["sim_loss"][("geomloss", {}, "settings for geomloss")]
model_opt["sim_loss"]["geomloss"]["attr"] = "points"
blur = 0.0005
reach = None  # 0.1  # change the value to explore behavior of the OT
model_opt["sim_loss"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.9,debias=False,reach={})".format(
    blur, reach
)
model = MODEL_POOL[model_name](model_opt)
solver = build_single_scale_model_embedded_solver(solver_opt, model)
model.init_reg_param(shape_pair)
shape_pair = timming(solver)(shape_pair)

visualize_multi_point(
    [shape_pair.source.points, shape_pair.flowed.points, shape_pair.target.points],
    [shape_pair.source.points, shape_pair.flowed.points, shape_pair.target.points],
    ["source", "gradient_flow", "target"],
    plot_func_list = [default_plot(cmap="magma"),default_plot(cmap="magma"),default_plot(cmap="viridis")],
    col_adaptive = False,
    saving_gif_path=None,
)

