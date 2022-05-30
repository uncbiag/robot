import os, sys
import pykeops
import subprocess
# sys.path.insert(0, os.path.abspath('../../../..'))
# cache_path ="/playpen/zyshen/keops_cachev2"
# os.makedirs(cache_path,exist_ok=True)
# pykeops.set_bin_folder(cache_path)  # change the build folder
# os.environ['DISPLAY'] = ':99.0'
# os.environ['PYVISTA_OFF_SCREEN'] = 'true'
# os.environ['PYVISTA_USE_IPYVTK'] = 'true'
# bashCommand ="Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & sleep 3"
# process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
# process.wait()
import torch
from robot.datasets.data_utils import read_json_into_list, get_pair_obj
from robot.experiments.datasets.lung.visualizer import lung_plot, camera_pos
from robot.global_variable import shape_type, obj_factory, ROBOT_PATH
from robot.experiments.datasets.lung.global_variable import lung_expri_path
import pointnet2.lib.pointnet2_utils as pointutils
from robot.utils.visualizer import visualize_point_pair_overlap, visualize_point_overlap, visualize_landmark_overlap, \
    visualize_point_pair, default_plot
from robot.shape.point_sampler import (
    uniform_sampler,
    grid_sampler,
)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = torch.zeros(B, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids




if __name__ == "__main__":
    assert (
        shape_type == "pointcloud"
    ), "set shape_type = 'pointcloud'  in global_variable.py"
    device = torch.device("cpu")  # cuda:0  cpu
    reader_obj = "lung_dataloader_utils.lung_reader()"
    normalizer_obj = (
        "lung_dataloader_utils.lung_normalizer(weight_scale=60000,scale=[100,100,100])"
    )
    sampler_obj = "lung_dataloader_utils.lung_sampler( method='combined',scale=0.0003,num_sample=30000,sampled_by_weight=True)"
    pair_postprocess_obj = "lung_dataloader_utils.lung_pair_postprocess()"
    use_local_mount = True
    task_name = "voxel_grid"  # farthest_sampling voxel_grid
    grid_spacing = 0.9
    dataset_json_path = os.path.join(ROBOT_PATH,"demos/data/lung_data/lung_dataset_splits/train/pair_data.json")
    saving_output_path = os.path.join(lung_expri_path, "output/{}".format(task_name))
    path_transfer = lambda x: x.replace('./',ROBOT_PATH+"/")

    pair_name_list, pair_info_list = read_json_into_list(dataset_json_path)

    pair_path_list = [
        [path_transfer(pair_info["source"]["data_path"]), path_transfer(pair_info["target"]["data_path"]) ]
        for pair_info in pair_info_list
    ]
    save_res = False
    n_sampled_points = 100
    for pair_id in range(len(pair_name_list)):
        pair_path = pair_path_list[pair_id]
        get_pair = get_pair_obj(
            reader_obj=reader_obj,
            normalizer_obj=normalizer_obj,
            sampler_obj=sampler_obj,
            pair_postprocess_obj=pair_postprocess_obj,
            expand_bch_dim=True,
        )
        source_dict, target_dict, _, _ = get_pair(*pair_path)
        input_data = {"source": source_dict, "target": target_dict}
        create_shape_pair_from_data_dict = obj_factory(
            "shape_pair_utils.create_source_and_target_shape()"
        )
        source, target = create_shape_pair_from_data_dict(input_data)
        source_points = source.points.cuda()
        source_weights = source.weights.cuda()
        if task_name == "farthest_sampling":
            try:
                fps_idx = pointutils.furthest_point_sample(source_points, n_sampled_points)
            except:
                fps_idx = farthest_point_sample(source_points, n_sampled_points)
            sample_index = fps_idx.squeeze().long()
            sampled_points = source_points[:, sample_index]
            sampled_weights = source_weights[:, sample_index]
        else:
            sampler = grid_sampler(grid_spacing)
            sampled_points, sampled_weights, _ = sampler(
                source_points[0], source_weights[0]
            )

        shape_name = pair_info_list[pair_id]["source"]["name"]
        saving_capture_path = os.path.join(saving_output_path, shape_name)
        if save_res:
            os.makedirs(saving_capture_path, exist_ok=True)
            saving_capture_path = os.path.join(
                saving_capture_path, "{}_synth.png".format(shape_name)
            )
        else:
            saving_capture_path = None

        constant_kwargs = {
            "light_mode": "none",
            "show": True,
            "camera_pos": camera_pos,
        }
        # visualize_point_overlap(
        #     source_points,
        #     sampled_points,
        #     source_weights,
        #     torch.ones_like(sampled_weights),
        #     "control point",
        #     rgb_on=False,
        #     opacity=(0.05, 0.9),
        #     saving_capture_path=saving_capture_path,
        #     camera_pos=camera_pos,
        #     show=True,
        # )
        visualize_point_overlap(
            source_points,
            sampled_points,
            source_weights,
            torch.ones_like(sampled_weights),
            "control point",
            opacity=(0.1, 1),
            saving_capture_path=saving_capture_path,
            source_plot_func=lung_plot("source"),
            target_plot_func = default_plot(point_size=25),
            **constant_kwargs,
        )
