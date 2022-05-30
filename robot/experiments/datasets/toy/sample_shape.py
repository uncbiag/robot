import os
from robot.datasets.vtk_utils import read_vtk
import numpy as np
import torch
from torch_scatter import scatter
from robot.shape.point_sampler import grid_sampler, uniform_sampler
from robot.utils.visualizer import visualize_point_fea, visualize_point_pair_overlap
from robot.utils.shape_visual_utils import make_sphere

import pyvista as pv
def shape_reader():
    """
    :return:
    """
    reader = read_vtk
    def read(path):
        raw_data_dict = reader(path)
        data_dict = {}
        data_dict["points"] = raw_data_dict["points"]
        data_dict["weights"] = np.ones([len( data_dict["points"]),1])
        return data_dict
    return read



def combined_sampler(method="uniform", sampled_by_weight=True, **args):
    """

    :param num_sample: num of points after sampling
    :param method: 'uniform' / 'voxelgrid'
    :param args:
    :return:
    """
    def uniform_sample(data_dict,ind=None, fixed_random_seed=True):
        num_sample = args["num_sample"]
        index = None
        if num_sample !=-1:
            points = data_dict["points"]
            weights = data_dict["weights"]
            sampler= uniform_sampler(num_sample, fixed_random_seed, sampled_by_weight=sampled_by_weight)
            sampled_points, sampled_weights, index = sampler(torch.tensor(points),torch.tensor(weights))

            index = index.numpy()
            data_dict["points"] = sampled_points.numpy()
            data_dict["weights"] = sampled_weights.numpy()
        return data_dict, index

    def voxelgrid_sample(data_dict,ind=None, fixed_random_seed=None):
        scale = args["scale"]
        index = None
        if scale != -1:
            points = torch.Tensor(data_dict["points"])
            weights = torch.Tensor(data_dict["weights"])
            sampler = grid_sampler(scale)
            points, cluster_weights, index = sampler(points,weights)
            # todo complete random sample code by unique sampling from index
            data_dict["points"] = points.numpy()
            data_dict["weights"] = cluster_weights.numpy()
        return data_dict, index

    def combine_sample(data_dict,ind=None, fixed_random_seed=True):

        data_dict, _ = voxelgrid_sample(data_dict,ind,fixed_random_seed)
        return uniform_sample(data_dict,ind, fixed_random_seed)


    assert method in ["uniform", "voxelgrid","combined"], "Not in supported sampler: 'uniform' / 'voxelgrid' / 'combined' "
    if method == "uniform":
        sampler = uniform_sample
    elif method == "voxelgrid":
        sampler = voxelgrid_sample
    else:
        sampler = combine_sample
    def sample(data_dict, ind=None, fixed_random_seed=True):
        return sampler(data_dict, ind=ind, fixed_random_seed=fixed_random_seed)
    return sample



if __name__=="__main__":
    shape_path = "./toy_synth/Armadillo.ply"
    reader = shape_reader()
    #sampler = combined_sampler(method="combined",scale=1,num_sample=10000)
    sampler = combined_sampler(method="uniform",num_sample=30000)
    data_dict = reader(shape_path)
    down_sampled_data_dict =sampler(data_dict)
    data_points = data_dict["points"]/100
    # data_points = data_dict["points"]*5
    data_points = data_points- data_points.mean(0)
    # data_points[:,-1] =data_points[:,-1]*-1
    data_weights = data_dict["weights"]/(data_dict["weights"].sum())
    data = pv.PolyData(data_points)
    data.point_arrays["weights"] =data_weights
    fpath = os.path.join("./toy_synth/armadillo_30k.ply")
    data.save(fpath)

    #visualize_point_pair_overlap(data_points,sphere_points,data_points,sphere_points,"dragon","sphere")
