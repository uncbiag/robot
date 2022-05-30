import os, sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
from robot.utils.module_parameters import ParameterDict
from robot.datasets.data_utils import (
    get_file_name,
    generate_pair_name,
    compute_interval,
)
from robot.shape.shape_pair_utils import create_shape_pair
from robot.models_reg.multiscale_optimization import (
    build_single_scale_model_embedded_solver,
    build_multi_scale_solver,
)
from robot.global_variable import MODEL_POOL, Shape, shape_type
from robot.utils.visualizer import (
    visualize_point_fea,
    visualize_point_pair,
    visualize_source_flowed_target_overlap,
)
from robot.demos.demo_utils import *
from robot.experiments.datasets.toy.toy_utils import *
from robot.utils.utils import memory_sort, add_zero_last_dim, sigmoid_decay

# import pykeops
# pykeops.clean_pykeops()
# set shape_type = "pointcloud"  in global_variable.py

assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
server_path = "./"  # "/playpen-raid1/"#"/home/zyshen/remote/llr11_mount/"
source_path = (
    server_path + "data/toy_demo_data/tree_2d_source.png"
)  # tree_2d_source3 tree_2d_source3_3
target_path = server_path + "data/toy_demo_data/tree_2d_target.png"  #


def get_points(path, npoint=1000, dtype=torch.FloatTensor):
    A = load_image(path)
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]), np.linspace(0, 1, A.shape[1]))
    grid = list(zip(xg.ravel(), yg.ravel()))
    grid = np.array(grid)
    A = A.transpose().ravel()
    dots = grid[A > 0.9]
    dots = np.concatenate([dots[:, 1:], dots[:, 0:1]], 1)
    # dots += (.5 / A.shape[0]) * np.random.standard_normal(dots.shape)
    rand_state = np.random.RandomState(0)
    index = rand_state.choice(np.arange(len(dots)), npoint, replace=False)
    dots = dots[index]
    dots, _ = memory_sort(dots, eps=0.0001)
    weights = np.ones([npoint, 1])
    return torch.from_numpy(dots).type(dtype), torch.from_numpy(weights).type(dtype)


####################  prepare data ###########################
pair_name = generate_pair_name([source_path, target_path])
source_points, source_weights = get_points(source_path, npoint=1000)
target_points, target_weights = get_points(target_path, npoint=1200)
source_min_interval = compute_interval(source_points)
target_min_interval = compute_interval(target_points)
min_interval = min(source_min_interval, target_min_interval) / 2
print(
    "the source and the target min interval is {},{}".format(
        source_min_interval, target_min_interval
    )
)
source_obj = {"points": source_points[None].to(device), "weights": source_weights[None].to(device)}
target_obj = {"points": target_points[None].to(device), "weights": target_weights[None].to(device)}
input_data = {"source": source_obj, "target": target_obj}
create_shape_pair_from_data_dict = obj_factory(
    "shape_pair_utils.create_source_and_target_shape()"
)
source, target = create_shape_pair_from_data_dict(input_data)
shape_pair = create_shape_pair(source, target)

# # ###############  do registration ###########################s############


# optimal transport
""" Experiment 1:  gradient flow """
task_name = "gradient_flow"
solver_opt = ParameterDict()
record_path = server_path + "output/toy_demo/{}".format(task_name)
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
blur = min_interval
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
mapped_fea = get_omt_mapping(
    model_opt["sim_loss"]["geomloss"],
    source,
    target,
    fea_to_map,
    p=2,
    mode="hard",
    confid=0.1,
)
flow_points = shape_pair.flowed.points - shape_pair.source.points
visualize_source_flowed_target_overlap(
    add_zero_last_dim(shape_pair.source.points[0]),
    add_zero_last_dim(shape_pair.flowed.points[0]),
    add_zero_last_dim(shape_pair.target.points[0]),
    add_zero_last_dim(fea_to_map),
    add_zero_last_dim(fea_to_map),
    add_zero_last_dim(mapped_fea),
    "source",
    "gradient_flow",
    "target",
    flow=add_zero_last_dim(flow_points[0]),
    saving_gif_path=None,
)


def update_sigma(sigma, iter):
    sigma = float(max(sigmoid_decay(iter, static=10, k=8) * 1.0, 0.01))
    return sigma


# Experiment 4:  optimization based discrete flow (spline)
task_name = "discrete_flow"
gradient_flow_mode = True
solver_opt = ParameterDict()
record_path = server_path + "output/toy_demo/{}".format(task_name)
solver_opt["record_path"] = record_path
solver_opt["save_2d_capture_every_n_iter"] = -1
solver_opt["point_grid_scales"] = [-1]
solver_opt["iter_per_scale"] = [200] if not gradient_flow_mode else [2]
solver_opt["rel_ftol_per_scale"] = [1e-9, 1e-9, 1e-9]
solver_opt["init_lr_per_scale"] = [1e-1, 1e-1, 1e-1]
solver_opt["save_3d_shape_every_n_iter"] = 10
solver_opt["shape_sampler_type"] = "point_grid"
solver_opt["stragtegy"] = (
    "use_optimizer_defined_here"
    if not gradient_flow_mode
    else "use_optimizer_defined_from_model"
)
solver_opt[("optim", {}, "setting for the optimizer")]
solver_opt[("scheduler", {}, "setting for the scheduler")]
solver_opt["optim"]["type"] = "lbgfs"  # lbgfs
solver_opt["scheduler"]["type"] = "step_lr"
solver_opt["scheduler"][("step_lr", {}, "settings for step_lr")]
solver_opt["scheduler"]["step_lr"]["gamma"] = 0.5
solver_opt["scheduler"]["step_lr"]["step_size"] = 30
model_name = "discrete_flow_opt"
model_opt = ParameterDict()
model_opt["drift_every_n_iter"] = 50
use_aniso_kernel = True
model_opt["use_aniso_kernel"] = use_aniso_kernel
model_opt["fix_anistropic_kernel_using_initial_shape"] = False and use_aniso_kernel
model_opt["fix_feature_using_initial_shape"] = True
kernel_size = 0.08  # iso 0.08
# spline_param="cov_sigma_scale=0.01,aniso_kernel_scale={},principle_weight=(3.,1.,1.),eigenvalue_min=0.1,iter_twice=True".format(kernel_size)
spline_param = "cov_sigma_scale=0.04,aniso_kernel_scale=[0.03, 0.05,0.07],aniso_kernel_weight=[0.2,0.3,0.5],eigenvalue_min=0.1,iter_twice=True, fixed={}, leaf_decay=False, self_center=False".format(
    model_opt["fix_anistropic_kernel_using_initial_shape"]
)
if not use_aniso_kernel:
    model_opt[
        "spline_kernel_obj"
    ] = "point_interpolator.NadWatIsoSpline(kernel_scale=[0.05,0.08, 0.1],kernel_weight=[0.1,0.3,0.6], exp_order=2)".format(
        kernel_size
    )
else:
    model_opt[
        "spline_kernel_obj"
    ] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)
model_opt[
    "interp_kernel_obj"
] = "point_interpolator.nadwat_kernel_interpolator(exp_order=2)"  # only used for multi-scale registration
# model_opt["pair_feature_extractor_obj"] ="2d_toy_local_feature_extractor.pair_feature_extractor(fea_type_list=['eigenvalue_prod'],weight_list=[1], std_normalize=False, radius=0.05,include_pos=True)"

model_opt["gradient_flow_mode"] = gradient_flow_mode
model_opt["running_result_visualize"] = True
model_opt[("gradflow_guided", {}, "settings for gradflow guidance")]
model_opt["gradflow_guided"]["gradflow_blur_init"] = 0.0005
model_opt["gradflow_guided"]["update_gradflow_blur_by_raito"] = 0.5
model_opt["gradflow_guided"]["gradflow_blur_min"] = 0.0005
model_opt["gradflow_guided"][("geomloss", {}, "settings for geomloss")]
model_opt["gradflow_guided"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur=blurplaceholder, reach=5,scaling=0.8,debias=False)"

# model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
# model_opt['sim_loss']['loss_list'] = ["gmm"]
# model_opt['sim_loss'][("gmm", {}, "settings for geomloss")]
# model_opt['sim_loss']['gmm']["attr"] = "points"
# model_opt['sim_loss']['gmm']["sigma"] = 0.001
# model_opt['sim_loss']['gmm']["w_noise"] = 0.0
# model_opt['sim_loss']['gmm']["mode"] = "sym_neglog_likelihood"  # sym_neglog_likelihood   neglog_likelihood
#
# #

model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
model_opt["sim_loss"]["loss_list"] = ["geomloss"]
model_opt["sim_loss"][("geomloss", {}, "settings for geomloss")]
model_opt["sim_loss"]["geomloss"][
    "attr"
] = "points"  # todo  the pointfea will be  more generalized choice
blur = 0.005
model_opt["sim_loss"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8,reach=None, debias=False)".format(
    blur
)
model = MODEL_POOL[model_name](model_opt)
solver = build_multi_scale_solver(solver_opt, model)
model.init_reg_param(shape_pair)
shape_pair = solver(shape_pair)
print("the registration complete")

#
# model_opt["running_result_visualize"] = True
#
# use_aniso_kernel = True
# model_opt["use_aniso_kernel"] = use_aniso_kernel
# model_opt["fix_anistropic_kernel_using_initial_shape"] = True and use_aniso_kernel
# model_opt["fix_feature_using_initial_shape"] = True
# kernel_size = 0.04  # iso 0.08
# spline_param = "cov_sigma_scale=0.01,aniso_kernel_scale={},principle_weight=(5.,1.),eigenvalue_min=0.1,iter_twice=True,leaf_decay=False,self_center=False".format(
#     kernel_size
# )
# # spline_param="cov_sigma_scale=0.04,aniso_kernel_scale={},eigenvalue_min=0.1,iter_twice=True, fixed={}, leaf_decay=False, self_center=False".format(kernel_size,model_opt["fix_anistropic_kernel_using_initial_shape"] )
# if not use_aniso_kernel:
#     model_opt[
#         "spline_kernel_obj"
#     ] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(
#         kernel_size
#     )
# else:
#     model_opt[
#         "spline_kernel_obj"
#     ] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)
# model_opt[
#     "interp_kernel_obj"
# ] = "point_interpolator.nadwat_kernel_interpolator(exp_order=2)"  # only used for multi-scale registration
# model_opt["gradient_flow_mode"] = gradient_flow_mode
# shape_pair.source = shape_pair.flowed
# shape_pair.control_points = shape_pair.flowed_control_points
# model = MODEL_POOL[model_name](model_opt)
# solver = build_multi_scale_solver(solver_opt, model)
# model.init_reg_param(shape_pair)
# shape_pair = solver(shape_pair)
# print("the registration complete")
