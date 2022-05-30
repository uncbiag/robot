import os, sys

from robot.experiments.datasets.flying3d_and_kitti.visualizer import kitti_plot
from robot.utils.generate_animation import generate_gif
from robot.utils.linked_slerp import get_slerp_cam_pos

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
from robot.utils.module_parameters import ParameterDict
from robot.datasets.data_utils import get_pair_obj
from robot.shape.shape_pair_utils import create_shape_pair
from robot.models_reg.multiscale_optimization import (
    build_single_scale_model_embedded_solver,
)
from robot.global_variable import MODEL_POOL, Shape, shape_type
from robot.utils.visualizer import visualize_source_flowed_target_overlap, visualize_point_fea, \
    visualize_point_pair_overlap, visualize_point_overlap, default_plot
from robot.demos.demo_utils import *
from robot.experiments.datasets.toy.toy_utils import *

# import pykeops
# pykeops.clean_pykeops()

assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cpu")  # cuda:0  cpu
reader_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_reader(flying3d=False)"
normalizer_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_normalizer()"
sampler_obj = "flyingkitti_nonocc_utils.flyingkitti_nonocc_sampler(num_sample=8192)"
pair_postprocess_obj = (
    "flyingkitti_nonocc_utils.flyingkitti_nonocc_pair_postprocess(flying3d=False)"
)
pair_postprocess = obj_factory(pair_postprocess_obj)

assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
case_id = "000146"
server_path = "./"  # "/playpen-raid1/"#"/home/zyshen/remote/llr11_mount/"
source_path = server_path + "data/kitti_data/{}/pc1.npy".format(case_id)
target_path = server_path + "data/kitti_data/{}/pc2.npy".format(case_id)
get_obj_func = get_pair_obj(
    reader_obj,
    normalizer_obj,
    sampler_obj,
    pair_postprocess_obj,
    place_sampler_in_postprocess= True,
    device = device,
    expand_bch_dim=True,
)
source_obj, target_obj, source_interval, target_interval = get_obj_func(
    source_path, target_path
)
min_interval = min(source_interval, target_interval)
print(
    "the source and the target min interval is {},{}".format(
        source_interval, target_interval
    )
)
input_data = {"source": source_obj, "target": target_obj}
create_shape_pair_from_data_dict = obj_factory(
    "shape_pair_utils.create_source_and_target_shape()"
)
source, target = create_shape_pair_from_data_dict(input_data)
shape_pair = create_shape_pair(source, target)

##############  do registration ###########################s############

""" Experiment 1:  gradient flow """
task_name = "gradient_flow"
solver_opt = ParameterDict()
record_path = server_path + "output/flyingkitti_reg/{}".format(task_name)
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
blur = 0.1
model_opt["sim_loss"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8,debias=False)".format(
    blur
)
model = MODEL_POOL[model_name](model_opt)
solver = build_single_scale_model_embedded_solver(solver_opt, model)
model.init_reg_param(shape_pair)
shape_pair = solver(shape_pair)
print("the registration complete")
gif_folder = os.path.join(record_path, "gif")
os.makedirs(gif_folder, exist_ok=True)
saving_gif_path = os.path.join(gif_folder, task_name + ".gif")
fea_to_map = shape_pair.source.points[0]
# mapped_fea = get_omt_mapping(
#     model_opt["sim_loss"]["geomloss"],
#     source,
#     target,
#     fea_to_map,
#     p=2,
#     mode="hard",
#     confid=0.0,
# )



flow_points = shape_pair.flowed.points - shape_pair.source.points
visualize_source_flowed_target_overlap(
    shape_pair.source.points,
    shape_pair.flowed.points,
    shape_pair.target.points,
    shape_pair.source.points,
    shape_pair.source.points,
    shape_pair.target.points,
    "source",
    "gradient_flow",
    "target",
    flow=shape_pair.flowed.points - shape_pair.source.points,
    saving_gif_path=None,
    col_adaptive=True
)

def linear_interp_shape(flowed_points, toflow_points, weights,t_list):
    interp_shape_list = []
    for t in t_list:
        interp_points = (flowed_points - toflow_points) * t + toflow_points
        interp_shape = Shape()
        interp_shape.set_data(points=interp_points, weights=weights)
        interp_shape_list.append(interp_shape)
    return interp_shape_list

def visualize_animation(
    shape1_list,
    shape2_list,
    title_list,
    saving_capture_path_list=None,
    camera_pos_list=None,
    shape1_plot_func = kitti_plot(color="source",rgb=False),
    shape2_plot_func = kitti_plot(color="target",rgb=False),
    light_mode="none",
    show=False,
):
    from robot.utils.visualizer import format_input

    for shape1, shape2, saving_capture_path, title, camera_pos in zip(
        shape1_list,
        shape2_list,
        saving_capture_path_list,
        title_list,
        camera_pos_list,
    ):
        points1, points2 = shape1.points, shape2.points
        feas1, feas2 = shape1.weights, shape2.weights
        points1 = format_input(points1)
        points2 = format_input(points2)
        feas1 = format_input(feas1)
        feas2 = format_input(feas2)

        visualize_point_overlap(
            points1,
            points2,
            feas1,
            feas2,
            title,
            source_plot_func=shape1_plot_func,
            target_plot_func=shape2_plot_func,
            saving_capture_path=saving_capture_path,
            light_mode=light_mode,
            camera_pos=camera_pos,
            show=show
        )




print("Now Generating gif....")
output_gif_path = os.path.join(record_path,"gif")
os.makedirs(output_gif_path,exist_ok=True)
t_list = list(np.linspace(0, 1.0, num=40))
camera_pos_start =[(-28.292797572097605, 29.04880002585719, 75.4639264991298),
 (0.0, 0.0, 0.0),
 (0.17502876601426093, 0.9390600122012284, -0.2958567635734463)]
camera_pos_end =[(27.40660591245471, 29.441147895914067, 75.63872118825323),
 (0.0, 0.0, 0.0),
 (-0.06122065070146193, 0.9374467035923124, -0.3427035305502367)]
# pos_interp_list = [camera_pos_start]*len(t_list)
pos_interp_list = [get_slerp_cam_pos(camera_pos_start, camera_pos_end, t) for t in t_list]
interp_shape_list= linear_interp_shape(shape_pair.flowed.points, shape_pair.source.points,  shape_pair.source.points,t_list)
title_list = ["flowed_t_{:.2f}".format(t) for t in t_list]
interp_shape_list += [interp_shape_list[-1]]*5
pos_interp_list += [pos_interp_list[-1]]*5
title_list +=[title_list[-1]]*5
shape_pair.target.weights = shape_pair.target.points
target_shape_list = [shape_pair.target]*len(interp_shape_list)
saving_capture_path_list = [os.path.join(output_gif_path,"capture_{}.png".format(i)) for i in range(len(interp_shape_list))]
saving_gif_path=os.path.join(output_gif_path,"ot_animation.gif")
visualize_animation(interp_shape_list,target_shape_list,title_list,saving_capture_path_list, camera_pos_list=pos_interp_list,light_mode="light_kit",show=False)
generate_gif(saving_capture_path_list,saving_gif_path)


