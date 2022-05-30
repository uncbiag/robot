"""
this script provides lung examples on Robust optimal transport, the goal of this script is to explore the behavior of the OT on lung dataset
"""

import os, sys

from robot.experiments.datasets.lung.visualizer import lung_plot, camera_pos

sys.path.insert(0, os.path.abspath("../.."))
from robot.utils.module_parameters import ParameterDict
from robot.utils.visualizer import *
from robot.demos.demo_utils import *
from robot.experiments.datasets.lung.lung_data_analysis import *

# import pykeops
# pykeops.clean_pykeops()


####################  Prepare Data ###########################
compute_on_half_lung = True
assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
source_path = "data/lung_data/lung_vessel_demo_data/copd_000002_EXP.vtk"
target_path = "data/lung_data/lung_vessel_demo_data/copd_000002_INSP.vtk"
reader_obj = "lung_dataloader_utils.lung_reader()"
scale = (
    -1
)  # an estimation of the physical diameter of the lung, set -1 for auto rescaling
normalizer_obj = "lung_dataloader_utils.lung_normalizer(scale={})".format(scale)
sampler_obj = "lung_dataloader_utils.lung_sampler(method='voxelgrid',scale=0.001)"
get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device)
source_obj, source_interval = get_obj_func(source_path)
target_obj, target_interval = get_obj_func(target_path)
min_interval = min(source_interval, target_interval)
input_data = {"source": source_obj, "target": target_obj}
create_shape_pair_from_data_dict = obj_factory(
    "shape_pair_utils.create_source_and_target_shape()"
)
source, target = create_shape_pair_from_data_dict(input_data)
source = (
    get_half_lung(source, normalize_weight=True) if compute_on_half_lung else source
)
target = (
    get_half_lung(target, normalize_weight=True) if compute_on_half_lung else target
)


################  Perform Registration ###########################s############
from torch.autograd import grad

toflow_points = source.points.clone()
toflow_weights = source.weights
toflow_points.requires_grad_()
blur = 0.005
geomloss_setting = ParameterDict()
geomloss_setting["geom_obj"] = \
    "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.8,reach=0.1,debias=False, backend='online')".format(
    blur
)
geomloss_fn = obj_factory(geomloss_setting["geom_obj"])
sim_loss = geomloss_fn(
    toflow_weights[..., 0], toflow_points, target.weights[..., 0], target.points
)
print(" geom loss is {}".format(sim_loss.item()))
grad_toflow = grad(sim_loss, toflow_points)[0]
flowed_points = toflow_points - grad_toflow / (toflow_weights)
flowed_points.detach_()
flowed = Shape()
flowed.set_data_with_refer_to(flowed_points, source)


#########################################  Result Analysis ###################################################
def analysis(
    source,
    flowed,
    target,
    fea_to_map,
    mapped_fea,
    compute_on_half_lung=True,
    saving_path=None,
):
    visualize_multi_point(
        points_list=[source.points, flowed.points, target.points],
        feas_list=[fea_to_map, fea_to_map, mapped_fea],
        titles_list=["source", "gradient_flow", "target"],
        saving_gif_path=None
        if not saving_path
        else os.path.join(saving_path, "s_f_t_full.gif"),
        saving_capture_path=None
        if not saving_path
        else os.path.join(saving_path, "s_f_t_full.png"),
        plot_func_list=[default_plot(cmap="magma"),default_plot(cmap="magma"),default_plot(cmap="viridis")],
        camera_pos=camera_pos,
        col_adaptive = False
    )

    # #
    # #if the previous computation has already based on half lung, here we don't need to get half lung again
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
        titles_list=["source", "gradient_flow", "target"],
        saving_gif_path=None
        if not saving_path
        else os.path.join(saving_path, "s_f_t_main.gif"),
        saving_capture_path=None
        if not saving_path
        else os.path.join(saving_path, "s_f_t_main.png"),
        plot_func_list=[default_plot(cmap="magma"),default_plot(cmap="magma"),default_plot(cmap="viridis")],
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


# color maps from the source to the target, here we use position vector as rbg value
fea_to_map = source.points[0]
mapped_fea = get_omt_mapping(
    geomloss_setting, source, target, fea_to_map, p=2, mode="hard", confid=0.0
)
analysis(source, flowed, target, fea_to_map, mapped_fea, compute_on_half_lung=True)
