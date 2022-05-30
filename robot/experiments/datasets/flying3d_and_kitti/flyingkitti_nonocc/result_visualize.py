from robot.utils.visualizer import visualize_source_flowed_target_overlap
import torch

case_id = "000024" # 000060  000061  000062  000146
expri_output_folder = "/home/zyshen/data/kitti_task/methods/30000/robot/"  # drobot_pwc drobot_spline  pwc flot
source_path = expri_output_folder+"{}_source.vtk".format(case_id)
target_path = expri_output_folder+"{}_target.vtk".format(case_id)
flowed_path = expri_output_folder+ "{}_flowed.vtk".format(case_id)
gt_flowed_path = expri_output_folder+"{}_gt_flowed.vtk".format(case_id)

from robot.datasets.vtk_utils import read_vtk
source_dict = read_vtk(source_path)
target_dict = read_vtk(target_path)
flowed_dict = read_vtk(flowed_path)
gt_flowed_dict = read_vtk(gt_flowed_path)
source_points = torch.Tensor(source_dict["points"])
target_points = torch.Tensor(target_dict["points"])
flowed_points = torch.Tensor(flowed_dict["points"])
gt_flowed_points = torch.Tensor(gt_flowed_dict["points"])

pred_sf, gt_sf = flowed_points - source_points, gt_flowed_points - source_points
l2_norm = torch.norm(gt_sf - pred_sf, p=2, dim=-1)
sf_norm = torch.norm(gt_sf, p=2, dim=-1)
relative_err = l2_norm / (sf_norm + 1e-4)
print(relative_err.mean())
visualize_source_flowed_target_overlap(
    source_points,
    flowed_points,
    target_points,
    source_points,
    relative_err,
    target_points,
    "source",
    "gradient_flow",
    "target",
    flow = flowed_points - source_points,
    saving_gif_path=None,
    col_adaptive=True
)
