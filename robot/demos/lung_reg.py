"""
this script provides lung examples on pure point based (no deep feature) Robust optimal transport/spline projection/LDDMM /LDDMM projection/ Discrete flow(point drift)
(Though we list these solutions here, pure point based methods and pure optimization approaches doesn't work well on the lung vessel dataset
To better understand the behaviors of deformation models, we recommend to work on toy_reg.py first)
"""

import os, sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))

from robot.experiments.datasets.lung.visualizer import lung_plot, camera_pos


from robot.utils.module_parameters import ParameterDict
from robot.datasets.data_utils import get_file_name, generate_pair_name, get_obj
from robot.models_reg.multiscale_optimization import (
    build_single_scale_model_embedded_solver,
    build_multi_scale_solver,
)
from robot.global_variable import MODEL_POOL
from robot.utils.utils import get_grid_wrap_points
from robot.utils.visualizer import *
from robot.demos.demo_utils import *
from robot.experiments.datasets.lung.lung_data_analysis import *
from robot.experiments.datasets.lung.lung_shape_pair import create_shape_pair



def analysis(
    shape_pair,
    fea_to_map,
    mapped_fea,
    compute_on_half_lung=True,
    method_name="flowed",
    saving_path=None,
):
    source = shape_pair.source
    target = shape_pair.target
    flowed = shape_pair.flowed

    visualize_multi_point(
        points_list=[source.points, flowed.points, target.points],
        feas_list=[fea_to_map, fea_to_map, mapped_fea],
        titles_list=["source", method_name, "target"],
        saving_gif_path=None
        if not saving_path
        else os.path.join(saving_path, "s_f_t_full.gif"),
        saving_capture_path=None
        if not saving_path
        else os.path.join(saving_path, "s_f_t_full.png"),
        plot_func_list=[default_plot(cmap="magma"), default_plot(cmap="magma"), default_plot(cmap="viridis")],
        camera_pos=camera_pos,
        col_adaptive=False
    )

    # #
    # #if the computation has already based on half computation, here we don't need to refilter again
    source_half = get_half_lung(source) if not compute_on_half_lung else source
    target_half = get_half_lung(target) if not compute_on_half_lung else target
    flowed_half = get_half_lung(flowed) if not compute_on_half_lung else flowed
    visualize_multi_point(
        points_list=[source_half.points, flowed_half.points, target_half.points],
        feas_list=[
            source_weight_transform(source_half.weights, compute_on_half_lung),
            flowed_weight_transform(flowed_half.weights, compute_on_half_lung),
            target_weight_transform(target_half.weights, compute_on_half_lung),
        ],
        titles_list=["source", method_name, "target"],
        saving_gif_path=None
        if not saving_path
        else os.path.join(saving_path, "s_f_t_main.gif"),
        saving_capture_path=None
        if not saving_path
        else os.path.join(saving_path, "s_f_t_main.png"),
        plot_func_list=[lung_plot(color="source"), lung_plot(color="source"), lung_plot(color="target")],
        opacity_list=[1, 1, 1],
        light_mode="none",
        camera_pos=camera_pos
    )

    visualize_point_pair_overlap(
        source_half.points,
        target_half.points,
        source_weight_transform(source_half.weights, compute_on_half_lung),
        target_weight_transform(target_half.weights, compute_on_half_lung),
        "source",
        "target",
        lung_plot(color="source"),
        lung_plot(color="target"),
        saving_gif_path=None
        if not saving_path
        else os.path.join(saving_path, "s_t_overlap.gif"),
        saving_capture_path=None
        if not saving_path
        else os.path.join(saving_path, "s_t_overlap.png"),
        opacity=[1, 1, 1],
        light_mode="none",
        camera_pos=camera_pos
    )
    visualize_point_pair_overlap(
        flowed_half.points,
        target_half.points,
        flowed_weight_transform(flowed_half.weights, compute_on_half_lung),
        target_weight_transform(target_half.weights, compute_on_half_lung),
        "flowed",
        "target",
        lung_plot(color="source"),
        lung_plot(color="target"),
        saving_gif_path=None
        if not saving_path
        else os.path.join(saving_path, "ft_overlap.gif"),
        saving_capture_path=None
        if not saving_path
        else os.path.join(saving_path, "f_t_overlap.png"),
        opacity=[1, 1, 1],
        light_mode="none",
        camera_pos=camera_pos
    )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set shape_type = "pointcloud"  in global_variable.py
assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
server_path = "./"  # "/playpen-raid1/"#"/home/zyshen/remote/llr11_mount/"
source_path = server_path + "data/lung_data/lung_vessel_demo_data/copd_000002_EXP.vtk"  # 10031R 10005Q
target_path = server_path + "data/lung_data/lung_vessel_demo_data/copd_000002_INSP.vtk"
compute_on_half_lung = True

####################  prepare data ###########################
pair_name = generate_pair_name([source_path, target_path])
reader_obj = "lung_dataloader_utils.lung_reader()"
scale = (
    -1
)  # an estimation of the physical diameter of the lung, set -1 for auto rescaling   #[99.90687, 65.66011, 78.61013]
# normalizer_obj = "lung_dataloader_utils.lung_normalizer(scale={})".format(scale)
# sampler_obj = "lung_dataloader_utils.lung_sampler(method='voxelgrid',scale=0.001)"
normalizer_obj = (
    "lung_dataloader_utils.lung_normalizer(weight_scale=30000,scale=[100,100,100])"
)
sampler_obj = "lung_dataloader_utils.lung_sampler( method='combined',scale=0.0003,num_sample=60000,sampled_by_weight=True)"

get_obj_func = get_obj(
    reader_obj, normalizer_obj, sampler_obj, device
)  # reader-> normalized-> pair_post-> sampler
source_obj, source_interval = get_obj_func(source_path)
target_obj, target_interval = get_obj_func(target_path)
min_interval = min(source_interval, target_interval)
input_data = {"source": source_obj, "target": target_obj}
create_shape_pair_from_data_dict = obj_factory(
    "shape_pair_utils.create_source_and_target_shape()"
)
source, target = create_shape_pair_from_data_dict(input_data)
# source, target = matching_shape_radius(source, target,sampled_by_radius=False, show=False)
# source, _ = matching_shape_radius(source, atlas,sampled_by_radius=False, show=False)
# target, _ = matching_shape_radius(target, atlas,sampled_by_radius=False, show=False)

# source = get_half_lung(source,normalize_weight=False) if compute_on_half_lung else source
# target = get_half_lung(target,normalize_weight=False) if compute_on_half_lung else target
# source = get_key_vessel(source,thre=2e-5)
# target = get_key_vessel(target,thre=1.4e-5)
# source, target = matching_shape_radius(source, target,sampled_by_radius=False, show=False)


# shape_pair = create_shape_pair(source, target,pair_name=pair_name, n_control_points=2048)
shape_pair = create_shape_pair(source, target, pair_name=pair_name)


################  do registration ###########################s############


# task_name = "prealign_opt"
# solver_opt = ParameterDict()
# record_path = "./output/lung_demo/{}".format(task_name)
# os.makedirs(record_path,exist_ok=True)
# solver_opt = ParameterDict()
# record_path = "./output/lung_demo/{}".format(task_name)
# os.makedirs(record_path,exist_ok=True)
# solver_opt["record_path"] = record_path
# solver_opt["save_2d_capture_every_n_iter"] = 1
# solver_opt["capture_plot_obj"] = "visualizer.capture_plotter()"
# model_name = "prealign_opt"
# model_opt =ParameterDict()
# model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
# model_opt["module_type"] = "gradflow_prealign"
# model_opt[("gradflow_prealign", {}, "settings for gradflow_prealign")]
# blur = 0.001
# model_opt["gradflow_prealign"]["method_name"]="affine"
# model_opt["gradflow_prealign"]["gradflow_mode"]="grad_forward"
# model_opt["gradflow_prealign"]["niter"] = 3
# model_opt["gradflow_prealign"]["search_init_transform"]=False
# model_opt["gradflow_prealign"][("geomloss", {}, "settings for geomloss")]
# model_opt["gradflow_prealign"]['geomloss']["mode"] = "flow"
# model_opt["gradflow_prealign"]['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur={},reach=1, scaling=0.8,debias=False, backend='online')".format(blur)
# #model_opt["gradflow_prealign"]["pair_feature_extractor_obj"] ="local_feature_extractor.pair_feature_extractor(fea_type_list=['eigenvalue'],weight_list=[1.0], radius=0.08,include_pos=True)"
#
# model_opt['sim_loss']['loss_list'] =  ["geomloss"]
# model_opt['sim_loss'][("geomloss", {}, "settings for geomloss")]
# model_opt['sim_loss']['geomloss']["attr"] = "pointfea"
# model_opt['sim_loss']['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur={},reach=1, scaling=0.8,debias=False, backend='online')".format(blur)
# model = MODEL_POOL[model_name](model_opt)
# solver = build_single_scale_model_embedded_solver(solver_opt,model)
#
#
#
# model.init_reg_param(shape_pair)
# shape_pair = solver(shape_pair)
# print("the registration complete")
# gif_folder = os.path.join(record_path,"gif")
# os.makedirs(gif_folder,exist_ok=True)
# saving_gif_path = os.path.join(gif_folder,task_name+".gif")
# fea_to_map =  shape_pair.source.points[0]
# shape_pair.source.pointfea, shape_pair.target.pointfea = shape_pair.source.points, shape_pair.target.points
# mapped_fea = get_omt_mapping(model_opt['sim_loss']['geomloss'], shape_pair.source, shape_pair.target,fea_to_map ,p=2,mode="hard",confid=0.0)
# #analysis(shape_pair, fea_to_map, mapped_fea, compute_on_half_lung=False,method_name="prealigned")
# shape_pair.source.points = shape_pair.flowed.points.detach()
# shape_pair.control_points = shape_pair.flowed_control_points.detach()
# shape_pair.flowed = None
# shape_pair.reg_param = None


""" Experiment 1:  gradient flow """
# here we use  the dense mode
task_name = "gradient_flow"
solver_opt = ParameterDict()
record_path = server_path + "output/lung_demo/{}".format(task_name)
os.makedirs(record_path, exist_ok=True)
solver_opt["record_path"] = record_path
solver_opt["save_2d_capture_every_n_iter"] = 1
# solver_opt["capture_plot_obj"] = "visualizer.capture_plotter(render_by_weight=True,add_bg_contrast=False,camera_pos=[(-4.924379645467042, 2.17374925796456, 1.5003730890759344),(0.0, 0.0, 0.0),(0.40133888001174545, 0.31574165540339943, 0.8597873634998591)])"
model_name = "gradient_flow_opt"
model_opt = ParameterDict()
model_opt[
    "interpolator_obj"
] = "point_interpolator.nadwat_kernel_interpolator(scale=0.01, exp_order=2)"
model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
model_opt["sim_loss"]["loss_list"] = ["geomloss"]
model_opt["sim_loss"][("geomloss", {}, "settings for geomloss")]
model_opt["sim_loss"]["geomloss"]["attr"] = "pointfea"
model_opt["running_result_visualize"] = True

blur = 0.01
model_opt["sim_loss"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8,reach=1,debias=False,backend='online')".format(
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
shape_pair.source, shape_pair.target = model.extract_fea(
    shape_pair.source, shape_pair.target
)
mapped_fea = get_omt_mapping(
    model_opt["sim_loss"]["geomloss"],
    shape_pair.source,
    shape_pair.target,
    fea_to_map,
    p=2,
    mode="hard",
    confid=0.0,
)
analysis(shape_pair, fea_to_map, mapped_fea, compute_on_half_lung=True)


# """ Experiment 2: lddmm flow  too slow !!!, and likely to experience numerical underflow, see expri 3 for a workaround"""
# task_name = "lddmm"
# solver_opt = ParameterDict()
# record_path = server_path+"output/lung_demo/{}".format(task_name)
# os.makedirs(record_path,exist_ok=True)
# solver_opt["record_path"] = record_path
# solver_opt["point_grid_scales"] =  [0.08, -1]
# solver_opt["iter_per_scale"] = [50, 200]
# solver_opt["rel_ftol_per_scale"] = [ 1e-9,1e-9,]
# solver_opt["init_lr_per_scale"] = [5e-1,5e-2]
# solver_opt["save_3d_shape_every_n_iter"] = 20
# solver_opt["shape_sampler_type"] = "point_grid"
# solver_opt["stragtegy"] = "use_optimizer_defined_here"
# solver_opt[("optim", {}, "setting for the optimizer")]
# solver_opt[("scheduler", {}, "setting for the scheduler")]
# solver_opt["optim"]["type"] = "sgd" #lbgfs
# solver_opt["scheduler"]["type"] = "step_lr"
# solver_opt["scheduler"][("step_lr",{},"settings for step_lr")]
# solver_opt["scheduler"]["step_lr"]["gamma"] = 0.5
# solver_opt["scheduler"]["step_lr"]["step_size"] = 80
# model_name = "lddmm_opt"
# model_opt =ParameterDict()
# model_opt["module"] ="hamiltonian"
# model_opt[("hamiltonian", {}, "settings for hamiltonian")]
# model_opt['hamiltonian']['kernel'] =  "keops_kernels.LazyKeopsKernel(kernel_type='multi_gauss', sigma_list=[0.008,0.02],weight_list=[0.5, 0.5])"
# model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
# model_opt['sim_loss']['loss_list'] =  ["geomloss"]
# model_opt['sim_loss'][("geomloss", {}, "settings for geomloss")]
# model_opt['sim_loss']['geomloss']["attr"] = "points"
# model_opt['sim_loss']['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur=0.0005, scaling=0.8, debias=True)"
# model = MODEL_POOL[model_name](model_opt)
# solver = build_multi_scale_solver(solver_opt,model)
# model.init_reg_param(shape_pair)
# solver(shape_pair)
# print("the registration complete")
#


# #
# """ Experiment 3: lddmm guide by gradient flow """
# task_name = "gradient_flow_guided_by_lddmm"
# solver_opt = ParameterDict()
# record_path = server_path+"output/lung_demo/{}".format(task_name)
# os.makedirs(record_path,exist_ok=True)
# solver_opt["record_path"] = record_path
# solver_opt["save_2d_capture_every_n_iter"] = 5
# solver_opt["capture_plot_obj"] = "visualizer.capture_plotter(render_by_weight=True,add_bg_contrast=False,camera_pos=[(-4.924379645467042, 2.17374925796456, 1.5003730890759344),(0.0, 0.0, 0.0),(0.40133888001174545, 0.31574165540339943, 0.8597873634998591)])"
# solver_opt["save_3d_shape_every_n_iter"] = 5
# solver_opt[("optim", {}, "setting for the optimizer")]
# solver_opt[("scheduler", {}, "setting for the scheduler")]
# solver_opt["optim"]["type"] = "sgd" #lbgfs
# solver_opt["scheduler"]["type"] = "step_lr"
# solver_opt["scheduler"][("step_lr",{},"settings for step_lr")]
# solver_opt["scheduler"]["step_lr"]["gamma"] = 0.5
# solver_opt["scheduler"]["step_lr"]["step_size"] = 30
#
# model_name = "lddmm_opt"
# model_opt =ParameterDict()
# module_name = "hamiltonian" #hamiltonian variational
# model_opt["module"] =module_name
# model_opt[(module_name, {}, "settings for lddmm")]
# model_opt[module_name]['kernel'] =  "keops_kernels.LazyKeopsKernel(kernel_type='multi_gauss', sigma_list=[0.06,0.1, 0.2],weight_list=[0.2,0.3, 0.5])"
# model_opt["use_gradflow_guided"] = True
# model_opt[("gradflow_guided", {}, "settings for gradflow guidance")]
# model_opt["gradflow_guided"] ['update_gradflow_every_n_step']=15
# model_opt["gradflow_guided"] ['gradflow_blur_init']= 0.01
# model_opt["gradflow_guided"] ['update_gradflow_blur_by_raito']= 0.5
# model_opt["gradflow_guided"] ['gradflow_blur_min']= 0.005
# model_opt["gradflow_guided"] ['gradflow_reach_init']= 2.0
# model_opt["gradflow_guided"] ['update_gradflow_reach_by_raito']= 0.8
# model_opt["gradflow_guided"] ['gradflow_reach_min']= 1.0
# model_opt["gradflow_guided"] ['post_kernel_obj']="point_interpolator.NadWatAnisoSpline(exp_order=2, cov_sigma_scale=0.02,aniso_kernel_scale=0.1,eigenvalue_min=0.2,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True, self_center=False)"
# #model_opt["gradflow_guided"] ['post_kernel_obj']="point_interpolator.NadWatIsoSpline(kernel_scale=0.08, exp_order=2)"
#
# model_opt["gradflow_guided"] ['pair_shape_transformer_obj']="lung_data_analysis.pair_shape_transformer( init_thres= 4e-5, nstep=8)"
# model_opt["gradflow_guided"] [("geomloss", {}, "settings for geomloss")]
# model_opt["gradflow_guided"]["geomloss"]["attr"] = "points" #todo  the pointfea will be  more generalized choice
# model_opt["gradflow_guided"]["geomloss"]["mode"] = "flow" #todo  the pointfea will be  more generalized choice
# model_opt["gradflow_guided"]["geomloss"]["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur=blurplaceholder, scaling=0.8,reach=reachplaceholder,debias=False)"
#
# model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
# model_opt['sim_loss']['loss_list'] =  ["l2"]
# model_opt['sim_loss']['l2']["attr"] = "points"
# model_opt['sim_loss'][("geomloss", {}, "settings for geomloss")]
# model_opt['sim_loss']['geomloss']["attr"] = "points"
# model_opt['sim_loss']['geomloss']["geom_obj"] = "geomloss.SamplesLoss(loss='sinkhorn',blur=0.001, scaling=0.8,reach=1,debias=False)"
#
# model = MODEL_POOL[model_name](model_opt)
# model.init_reg_param(shape_pair)
# solver = build_single_scale_general_solver(solver_opt,model,num_iter=100, scale=-1, lr=1e-2, rel_ftol=1e-9, patient=5)
# shape_pair=solver(shape_pair)
# print("the registration complete")
# gif_folder = os.path.join(record_path,"gif")
# os.makedirs(gif_folder,exist_ok=True)
# saving_gif_path = os.path.join(gif_folder,task_name+".gif")
# fea_to_map =  shape_pair.source.points[0]
# blur = 0.001
# mapped_fea = get_omt_mapping(model_opt['sim_loss']['geomloss'], source, target,fea_to_map , p=2,mode="hard",confid=0.1)
# visualize_multi_point([shape_pair.source.points[0],shape_pair.flowed.points[0],shape_pair.target.points[0]],
#                      [fea_to_map,fea_to_map, mapped_fea],
#                      ["source", "gradient_flow","target"],
#                         [True, True, True],
#                       saving_gif_path=None)
#
#


"""Experiment 4: discrete flow """

task_name = "discrete_flow_anisotropic_withouteignfea"
gradient_flow_mode = True
use_aniso_kernel = True
solver_opt = ParameterDict()
record_path = server_path + "output/lung_demo/{}".format(task_name)
solver_opt["record_path"] = record_path
solver_opt["save_2d_capture_every_n_iter"] = 1
solver_opt[
    "capture_plot_obj"
] = "visualizer.capture_plotter(render_by_weight=True,add_bg_contrast=False,camera_pos=[(-4.924379645467042, 2.17374925796456, 1.5003730890759344),(0.0, 0.0, 0.0),(0.40133888001174545, 0.31574165540339943, 0.8597873634998591)])"
solver_opt["point_grid_scales"] = [-1]
solver_opt["iter_per_scale"] = [50] if not gradient_flow_mode else [5]
solver_opt["rel_ftol_per_scale"] = [1e-9, 1e-9, 1e-9]
solver_opt["init_lr_per_scale"] = [5e-1, 1e-1, 1e-1]
solver_opt["save_3d_shape_every_n_iter"] = 10
solver_opt["shape_sampler_type"] = "point_grid"
solver_opt["stragtegy"] = (
    "use_optimizer_defined_here"
    if not gradient_flow_mode
    else "use_optimizer_defined_from_model"
)
solver_opt[("optim", {}, "setting for the optimizer")]
solver_opt[("scheduler", {}, "setting for the scheduler")]
solver_opt["optim"]["type"] = "sgd"  # lbgfs
solver_opt["scheduler"]["type"] = "step_lr"
solver_opt["scheduler"][("step_lr", {}, "settings for step_lr")]
solver_opt["scheduler"]["step_lr"]["gamma"] = 0.5
solver_opt["scheduler"]["step_lr"]["step_size"] = 30
model_name = "discrete_flow_opt"
model_opt = ParameterDict()
model_opt["running_result_visualize"] = True
model_opt["saving_running_result_visualize"] = True
model_opt["drift_every_n_iter"] = 10
model_opt["use_aniso_kernel"] = use_aniso_kernel
model_opt["fix_anistropic_kernel_using_initial_shape"] = True and use_aniso_kernel
model_opt["fix_feature_using_initial_shape"] = True
kernel_size = 0.1  # iso 0.08
spline_param = "cov_sigma_scale=0.02,aniso_kernel_scale={},eigenvalue_min=0.2,iter_twice=True".format(
    kernel_size
)
# spline_param="cov_sigma_scale=0.02,aniso_kernel_scale={},eigenvalue_min=0.3,iter_twice=True, fixed={}, leaf_decay=False".format(kernel_size,model_opt["fix_anistropic_kernel_using_initial_shape"] )
if not use_aniso_kernel:
    model_opt[
        "spline_kernel_obj"
    ] = "point_interpolator.NadWatIsoSpline(kernel_scale={}, exp_order=2)".format(
        kernel_size
    )
else:
    model_opt[
        "spline_kernel_obj"
    ] = "point_interpolator.NadWatAnisoSpline(exp_order=2,{})".format(spline_param)
model_opt[
    "interp_kernel_obj"
] = "point_interpolator.nadwat_kernel_interpolator(exp_order=2)"  # only used for multi-scale registration
feature_kernel_size = 0.03
fixed_feature = True
get_anistropic_gamma_obj = "'local_feature_extractor.compute_anisotropic_gamma_from_points(cov_sigma_scale=0.02,aniso_kernel_scale={},principle_weight=None,eigenvalue_min=0.1,iter_twice=True)'".format(
    feature_kernel_size
)
model_opt[
    "pair_feature_extractor_obj"
] = "lung_feature_extractor.LungFeatureExtractor(fea_type_list=['eigenvalue_prod'],weight_list=[0.2], radius=0.03,get_anistropic_gamma_obj={},std_normalize=True, include_pos=True, fixed={})".format(
    get_anistropic_gamma_obj, fixed_feature
)
model_opt["gradient_flow_mode"] = gradient_flow_mode
model_opt[("gradflow_guided", {}, "settings for gradflow guidance")]
model_opt["gradflow_guided"]["gradflow_blur_init"] = 0.01
model_opt["gradflow_guided"]["update_gradflow_blur_by_raito"] = 0.5
model_opt["gradflow_guided"]["gradflow_blur_min"] = 0.005
model_opt["gradflow_guided"]["gradflow_reach_init"] = 2.0
model_opt["gradflow_guided"]["update_gradflow_reach_by_raito"] = 0.8
model_opt["gradflow_guided"]["gradflow_reach_min"] = 1.0
model_opt["gradflow_guided"][
    "pair_shape_transformer_obj"
] = "lung_data_analysis.pair_shape_transformer( init_thres= 4.0e-5, nstep=8)"
model_opt["gradflow_guided"][("geomloss", {}, "settings for geomloss")]
model_opt["gradflow_guided"]["geomloss"][
    "attr"
] = "points"  # todo  the pointfea will be  more generalized choice
model_opt["gradflow_guided"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur=blurplaceholder, scaling=0.8,reach=reachplaceholder,debias=False, backend='online')"
model_opt["gradflow_guided"][
    "post_kernel_obj"
] = "point_interpolator.NadWatAnisoSpline(exp_order=2, cov_sigma_scale=0.02,aniso_kernel_scale=0.1,eigenvalue_min=0.2,iter_twice=True, fixed=False, leaf_decay=False, is_interp=True, self_center=False)"

model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
model_opt["sim_loss"]["loss_list"] = ["geomloss"]
model_opt["sim_loss"][("geomloss", {}, "settings for geomloss")]
model_opt["sim_loss"]["geomloss"][
    "attr"
] = "points"  # todo  the pointfea will be  more generalized choice
blur = 0.001
model_opt["sim_loss"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8,reach=1.0, debias=False, backend='online')".format(
    blur
)

model = MODEL_POOL[model_name](model_opt)
solver = build_multi_scale_solver(solver_opt, model)
model.init_reg_param(shape_pair)
shape_pair = solver(shape_pair)
print("the registration complete")

solver_opt.write_ext_JSON(os.path.join(record_path, "solver_settings.json"))
model_opt.write_ext_JSON(os.path.join(record_path, "model_settings.json"))


######################### folding detections ##########################################
source_grid_spacing = np.array([0.05] * 3).astype(
    np.float32
)  # max(source_interval*20, 0.01)
source_wrap_grid, grid_size = get_grid_wrap_points(
    source_obj["points"][0], source_grid_spacing
)
source_wrap_grid = source_wrap_grid[None]
toflow = Shape()
toflow.set_data(points=source_wrap_grid)
shape_pair.set_toflow(toflow)
shape_pair.control_weights = (
    torch.ones_like(shape_pair.control_weights) / shape_pair.control_weights.shape[1]
)
model.flow(shape_pair)
detect_folding(
    shape_pair.flowed.points, grid_size, source_grid_spacing, record_path, pair_name
)
