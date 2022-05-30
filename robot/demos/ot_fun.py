import os, sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))

import torch
import numpy as np
from robot.utils.generate_animation import generate_gif


# os.environ["DISPLAY"] = ":99.0"
# os.environ["PYVISTA_OFF_SCREEN"] = "true"
# os.environ["PYVISTA_USE_IPYVTK"] = "true"
# bashCommand = "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & sleep 3"
# process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
# process.wait()


from robot.utils.module_parameters import ParameterDict
from robot.datasets.data_utils import get_file_name, generate_pair_name, get_obj
from robot.shape.shape_pair_utils import create_shape_pair
from robot.models_reg.multiscale_optimization import (
    build_single_scale_model_embedded_solver,
)
from robot.global_variable import MODEL_POOL, Shape, shape_type
from robot.utils.visualizer import (
     visualize_point_fea
)
from robot.demos.demo_utils import *
from robot.utils.utils import timming



camera_pos = [(1.867353567083761, 0.47007407616247987, -3.4740998944712214),
 (0.0, 0.0, 0.0),
 (-0.153094694952026, 0.9868820080743704, 0.05124369733581401)]

def robust_ot(input_data, task_name):
    create_shape_pair_from_data_dict = obj_factory(
        "shape_pair_utils.create_source_and_target_shape()"
    )
    source, target = create_shape_pair_from_data_dict(input_data)
    shape_pair = create_shape_pair(source, target)
    shape_pair.pair_name = "toy"

    solver_opt = ParameterDict()
    record_path = "./output/toy_reg/{}".format(task_name)
    os.makedirs(record_path, exist_ok=True)
    solver_opt["record_path"] = record_path
    solver_opt["save_res"] = False
    model_name = "gradient_flow_opt"
    model_opt = ParameterDict()
    model_opt[
        "interpolator_obj"
    ] = "point_interpolator.nadwat_kernel_interpolator(scale=0.1, exp_order=2)"
    model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
    model_opt["sim_loss"]["loss_list"] = ["geomloss"]
    model_opt["sim_loss"][("geomloss", {}, "settings for geomloss")]
    model_opt["sim_loss"]["geomloss"]["attr"] = "points"
    blur = 0.1
    reach = 1000  # 0.1  # change the value to explore behavior of the OT
    model_opt["sim_loss"]["geomloss"][
        "geom_obj"
    ] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.9,debias=False, backend='online')".format(
        blur, reach
    )

    model = MODEL_POOL[model_name](model_opt)
    solver = build_single_scale_model_embedded_solver(solver_opt, model)
    model.init_reg_param(shape_pair)
    shape_pair = timming(solver)(shape_pair)
    return shape_pair.flowed

def create_animation(
    shape_list,
    saving_capture_path_list=None,
    saving_gif_path=None,
    camera_pos=None,
):
    for shape, saving_capture_path in zip(
        shape_list,
        saving_capture_path_list
    ):
        visualize_point_fea(
            shape.points,
            shape.weights,
            title="ot_fun",
            opacity=1,
            #plot_func = toy_plot(color="target",point_size=20),
            saving_capture_path=saving_capture_path,
            #light_mode="none",
            camera_pos=camera_pos,
            show=False
        )
    generate_gif(saving_capture_path_list,saving_gif_path)



assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
expri_settings = {
    "bunny":{"data_path":"./data/toy_demo_data/bunny_30k.ply"},
    "dragon":{"data_path":"./data/toy_demo_data/dragon_30k.ply"},
    "armadillo":{"data_path":"./data/toy_demo_data/armadillo_30k.ply"}
                  }
output_path = "./output/ot_fun"
bunny_path = expri_settings["bunny"]["data_path"]
dragon_path = expri_settings["dragon"]["data_path"]
armadillo_path = expri_settings["armadillo"]["data_path"]
os.makedirs(output_path,exist_ok=True)


####################  prepare data ###########################
reader_obj = "toy_dataset_utils.toy_reader()"
sampler_obj = "toy_dataset_utils.toy_sampler()"
normalizer_obj = "toy_dataset_utils.toy_normalizer()"
get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device)
bunny_obj, bunny_interval = get_obj_func(bunny_path)
dragon_obj, dragon_interval = get_obj_func(dragon_path)
armadillo_obj, armadillo_interval = get_obj_func(armadillo_path)


####################  run deformation ###########################
input_data = {"source": bunny_obj, "target": dragon_obj}
deformed_bunny = robust_ot(input_data,"bunny_to_dragon")
input_data= {"source":{"points":deformed_bunny.points},"target":armadillo_obj}
deformed_dragon = robust_ot(input_data,"dragon_to_armadillo")
input_data= {"source":{"points":deformed_dragon.points},"target":bunny_obj}
deformed_armadillo = robust_ot(input_data,"armadillo_to_bunny")



def linear_interp_shape(flowed_points, toflow_points, weights):
    interp_shape_list = []
    for t in t_list:
        interp_points = (flowed_points - toflow_points) * t + toflow_points
        interp_shape = Shape()
        interp_shape.set_data(points=interp_points, weights=weights)
        interp_shape_list.append(interp_shape)
    return interp_shape_list


t_list = list(np.linspace(0, 1.0, num=15))
interp_shape_list = []
interp_shape_list += linear_interp_shape(deformed_bunny.points, bunny_obj["points"],bunny_obj["points"])
interp_shape_list += [interp_shape_list[-1]]*5
interp_shape_list += linear_interp_shape(deformed_dragon.points, deformed_bunny.points,bunny_obj["points"])
interp_shape_list += [interp_shape_list[-1]]*5
interp_shape_list += linear_interp_shape(deformed_armadillo.points, deformed_dragon.points, bunny_obj["points"])
interp_shape_list += [interp_shape_list[-1]]*5
saving_capture_path_list = [os.path.join(output_path,"capture_{}.png".format(i)) for i in range(len(interp_shape_list))]
create_animation(interp_shape_list,saving_capture_path_list,saving_gif_path=os.path.join(output_path,"ot_animation_blur_{}.gif".format(0.1)), camera_pos=camera_pos)

