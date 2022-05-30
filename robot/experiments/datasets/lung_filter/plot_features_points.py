import os
import pyvista as pv
import numpy as np
import torch
from torch.autograd import grad
from robot.experiments.datasets.lung_filter.key_point_extractor import kpts_world
from robot.experiments.datasets.lung.visualizer import lung_plot
from robot.shape.point_interpolator import NadWatIsoSpline
from robot.utils.img_visual_utils import save_3D_img_from_numpy
from robot.utils.utils import identity_map
from robot.utils.visualizer import visualize_landmark_pair
import SimpleITK as sitk

def read_vtk(path):
    data = pv.read(path)
    data_dict = {}
    data_dict["points"] = data.points.astype(np.float32)
    data_dict["faces"] = data.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    for name in data.array_names:
        try:
            data_dict[name] = data[name]
        except:
            pass
    return data_dict

case_id = "10291R"
source_img_file = "/playpen-raid1/zyshen/debug/debug_lung_filter/lung_filter/preprocessed/{}_target.npy".format(case_id)
points_file = "/playpen-raid1/zyshen/debug/debug_lung_filter/lung_filter/preprocessed/{}_target_pts.npy".format(case_id)
origin_file = "/playpen-raid1/zyshen/debug/debug_lung_filter/lung_filter/preprocessed/{}_target_origin.npy".format(case_id)
vessel_file = "/playpen-raid1/Data/UNC_vesselParticles/{}_EXP_STD_BWH_COPD_wholeLungVesselParticles.vtk".format(case_id)
source = read_vtk(vessel_file)
source_vessel =source["points"]
source_weights =source["radius"]
source_orign = np.load(origin_file)
pts = kpts_world(torch.from_numpy(np.load(points_file)), (350, 350, 350))[0].numpy()
source_pts = np.flip(pts, -1) + np.flipud(source_orign)
target_img_file = "/playpen-raid1/zyshen/debug/debug_lung_filter/lung_filter/preprocessed/{}_source.npy".format(case_id)
points_file = "/playpen-raid1/zyshen/debug/debug_lung_filter/lung_filter/preprocessed/{}_source_pts.npy".format(case_id)
origin_file = "/playpen-raid1/zyshen/debug/debug_lung_filter/lung_filter/preprocessed/{}_source_origin.npy".format(case_id)
vessel_file = "/playpen-raid1/Data/UNC_vesselParticles/{}_INSP_STD_BWH_COPD_wholeLungVesselParticles.vtk".format(case_id)
target = read_vtk(vessel_file)
target_vessel =target["points"]
target_weights =target["radius"]
target_orign = np.load(origin_file)
pts = kpts_world(torch.from_numpy(np.load(points_file)), (350, 350, 350))[0].numpy()
target_pts = np.flip(pts, -1) + np.flipud(target_orign)
source_pts_weights = np.ones_like(source_pts)
target_pts_weights = np.ones_like(target_pts)
# visualize_landmark_pair(
#             source_vessel,
#             source_pts,
#             target_vessel,
#             target_pts,
#             source_weights,
#             source_pts_weights,
#             target_weights,
#             target_pts_weights,
#             "exp",
#             "insp",
#             point_plot_func=lung_plot(color='source'),
#             opacity=(1, 1),
#             light_mode="none",
#         )

import geomloss
source_pts = torch.Tensor(source_pts)
source_pts.requires_grad=True
target_pts = torch.Tensor(target_pts)
source_pts_weights = torch.ones(source_pts.shape[:1])/source_pts.shape[0]
target_pts_weights = torch.ones(target_pts.shape[:1])/target_pts.shape[0]

geomloss_fn = geomloss.SamplesLoss(loss='sinkhorn',blur=1, scaling=0.8,reach=None,debias=False, backend='online')
sim_loss = geomloss_fn(
    source_pts_weights, source_pts, target_pts_weights, target_pts
)
print(" geom loss is {}".format(sim_loss.item()))
grad_toflow = grad(sim_loss, source_pts)[0]
flowed_points = source_pts - grad_toflow / (source_pts_weights[...,None])

# visualize_landmark_pair(
#             target_vessel,
#             flowed_points,
#             target_vessel,
#             target_pts,
#             target_weights,
#             source_pts_weights,
#             target_weights,
#             target_pts_weights,
#             "exp",
#             "insp",
#             point_plot_func=lung_plot(color='source'),
#             opacity=(1, 1),
#             light_mode="none",
#         )

source_img = np.load(source_img_file)
target_img = np.load(target_img_file)
grid_size = source_img.shape
spacing = np.array([1,1,1]).astype(np.float32)
grid_spacing = 1/(np.array(grid_size)-1) .astype(np.float32)
id_map = identity_map(grid_size, spacing) # +1 is due to dirlab convention
inv_grid_ = id_map.transpose(1,2,3,0).reshape(-1,3)
inv_grid = np.zeros_like(inv_grid_)
inv_grid[...,0] = inv_grid_[...,2]
inv_grid[...,1] = inv_grid_[...,1]
inv_grid[...,2] = inv_grid_[...,0]
inv_grid = inv_grid + np.flipud(source_orign)
interp_kernel = NadWatIsoSpline(exp_order=2, kernel_scale=20)
inv_grid = torch.Tensor(inv_grid)
flow_inv_grid = interp_kernel(
        inv_grid[None],
        source_pts[None],
        flowed_points[None] - source_pts[None],
        source_pts_weights[None][...,None],
    )



flow_inv_grid = flow_inv_grid.detach().numpy().squeeze()
flowed_inv_grid = flow_inv_grid+inv_grid.numpy()



flowed_inv_grid  = (flowed_inv_grid-np.flipud(target_orign))/spacing
flowed_points = flowed_points.detach().numpy()
flowed_points_index = (flowed_points - np.flipud(target_orign))/spacing #
target_cp = target_img.copy()
for coord in flowed_points_index:
    coord_int = [int(c) for c in coord]
    target_cp[coord_int[2], coord_int[1], coord_int[0]] = 5000.
save_3D_img_from_numpy(target_cp, "/playpen-raid1/zyshen/debug/debug_filter/{}_debug.nii.gz".format("target_with_keypoint"))


flowed_inv_grid = flowed_inv_grid*grid_spacing
flowed_inv_grid = flowed_inv_grid*2-1
flowed_inv_grid= flowed_inv_grid.reshape(list(grid_size)+[3])
inv_warped = torch.nn.functional.grid_sample(torch.Tensor(target_img)[None][None], torch.Tensor(flowed_inv_grid)[None], mode="bilinear", padding_mode="border",align_corners=True)
inv_warped = inv_warped.cpu().numpy().squeeze()
# inv_warped = inv_warped.transpose([2, 1, 0])
inv_warped = sitk.GetImageFromArray(inv_warped)
os.makedirs("/playpen-raid1/zyshen/debug/debug_filter",exist_ok=True)
sitk.WriteImage(sitk.GetImageFromArray(source_img),"/playpen-raid1/zyshen/debug/debug_filter/source.nii.gz")
sitk.WriteImage(inv_warped,"/playpen-raid1/zyshen/debug/debug_filter/inv_warped.nii.gz")
sitk.WriteImage(sitk.GetImageFromArray(target_img),"/playpen-raid1/zyshen/debug/debug_filter/target.nii.gz")