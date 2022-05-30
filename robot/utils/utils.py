""" General utilities. """

import os
import shutil
from functools import partial
import torch
import random
import warnings
import requests
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import numpy as np
from pykeops.numpy.cluster import (
    grid_cluster,
    cluster_ranges_centroids,
    sort_clusters,
    from_matrix,
)


def init_weights(m, init_type="normal", gain=0.02):
    """Randomly initialize a module's weights.

    Args:
        m (nn.Module): The module to initialize its weights
        init_type (str): Initialization type: 'normal', 'xavier', 'kaiming', or 'orthogonal'
        gain (float): Standard deviation of the normal distribution
    """
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (
        classname.find("Conv") != -1 or classname.find("Linear") != -1
    ):
        if init_type == "normal":
            init.normal_(m.weight.data, 0.0, gain)
        elif init_type == "xavier":
            init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == "kaiming":
            init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError(
                "initialization method [%s] is not implemented" % init_type
            )
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("BatchNorm3d") != -1:
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)


def set_device(gpus=None, use_cuda=True):
    """Sets computing device. Either the CPU or any of the available GPUs.

    Args:
        gpus (list of int, optional): The GPU ids to use. If not specified, all available GPUs will be used
        use_cuda (bool, optional): If True, CUDA enabled GPUs will be used, else the CPU will be used

    Returns:
        torch.device: The selected computing device.
    """
    use_cuda = torch.cuda.is_available() if use_cuda else use_cuda
    if use_cuda:
        gpus = list(range(torch.cuda.device_count())) if not gpus else gpus
        print("=> using GPU devices: {}".format(", ".join(map(str, gpus))))
    else:
        gpus = None
        print("=> using CPU device")
    device = torch.device("cuda:{}".format(gpus[0])) if gpus else torch.device("cpu")

    return device, gpus


def set_seed(seed):
    """Sets computing device. Either the CPU or any of the available GPUs.

    Args:
        gpus (list of int, optional): The GPU ids to use. If not specified, all available GPUs will be used
        use_cuda (bool, optional): If True, CUDA enabled GPUs will be used, else the CPU will be used

    Returns:
        torch.device: The selected computing device.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
    else:
        cudnn.deterministic = False


def save_checkpoint(exp_dir, base_name, state, is_best=False):
    """Saves a model's checkpoint.

    Args:
        exp_dir (str): Experiment directory to save the checkpoint into.
        base_name (str): The output file name will be <base_name>_latest.pth and optionally <base_name>_best.pth
        state (dict): The model state to save.
        is_best (bool): If True, <base_name>_best.pth will be saved as well.
    """
    filename = os.path.join(exp_dir, base_name + "_latest.pth")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(exp_dir, base_name + "_best.pth"))


mag_map = {"K": 3, "M": 6, "B": 9}


def str2int(s):
    """ Converts a string containing a number with 'K', 'M', or 'B' to an integer. """
    if isinstance(s, (list, tuple)):
        return [str2int(o) for o in s]
    if not isinstance(s, str):
        return s
    return (
        int(float(s[:-1]) * 10 ** mag_map[s[-1].upper()])
        if s[-1].upper() in mag_map
        else int(s)
    )


def sigmoid_explode(ep, static=5, k=5):
    """
    factor  increase with epoch, factor = (k + exp(ep / k))/k
    :param ep: cur epoch
    :param static: at the first #  epoch, the factor keep unchanged
    :param k: the explode factor
    :return:
    """
    static = static
    if ep < static:
        return 1.0
    else:
        ep = ep - static
        factor = (k + np.exp(ep / k)) / k
        return float(factor)


def sigmoid_decay(ep, static=5, k=5):
    """
    factor  decease with epoch, factor = k/(k + exp(ep / k))
    :param ep: cur epoch
    :param static: at the first #  epoch, the factor keep unchanged
    :param k: the decay factor
    :return:
    """
    static = static
    if ep < static:
        return float(1.0)
    else:
        ep = ep - static
        factor = k / (k + np.exp(ep / k))
        return float(factor)


def t2np(v):
    """
    Takes a torch array and returns it as a numpy array on the cpu

    :param v: torch array
    :return: numpy array
    """

    if type(v) == torch.Tensor:
        return v.detach().cpu().numpy()
    else:
        try:
            return v.cpu().numpy()
        except:
            return v


def to_tensor(data, device):
    if isinstance(data, dict):
        return {key: to_tensor(item, device) for key, item in data.items()}
    else:
        return torch.from_numpy(data).to(device)


# Adapted from: https://github.com/Sudy/coling2018/blob/master/torchtext/utils.py
def download_from_url(url, output_path):
    """Download file from url including Google Drive.

    Args:
        url (str): File URL
        output_path (str): Output path to write the file to
    """

    def process_response(r):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get("Content-length", 0))
        with open(output_path, "wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=1,
                desc=os.path.split(output_path)[1],
            ) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))

    if "drive.google.com" not in url:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        process_response(response)
        return

    # print('downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    process_response(response)


def identity_map(sz, spacing, dtype="float32"):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0 : sz[0]]
    elif dim == 2:
        id = np.mgrid[0 : sz[0], 0 : sz[1]]
    elif dim == 3:
        id = np.mgrid[0 : sz[0], 0 : sz[1], 0 : sz[2]]
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    # now get it into range [0,(sz-1)*spacing]^d
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index

    for d in range(dim):
        id[d] *= spacing[d]

        # id[d]*=2./(sz[d]-1)
        # id[d]-=1.

    # and now store it in a dim+1 array
    if dim == 1:
        idnp = np.zeros([1, sz[0]], dtype=dtype)
        idnp[0, :] = id[0]
    elif dim == 2:
        idnp = np.zeros([2, sz[0], sz[1]], dtype=dtype)
        idnp[0, :, :] = id[0]
        idnp[1, :, :] = id[1]
    elif dim == 3:
        idnp = np.zeros([3, sz[0], sz[1], sz[2]], dtype=dtype)
        idnp[0, :, :, :] = id[0]
        idnp[1, :, :, :] = id[1]
        idnp[2, :, :, :] = id[2]
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    return idnp


def identity_map_multiN(sz, spacing, dtype="float32"):
    """
    Create an identity map

    :param sz: size of an image in BxCxXxYxZ format
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map
    """
    dim = len(sz) - 2
    nrOfI = int(sz[0])

    if dim == 1:
        id = np.zeros([nrOfI, 1, sz[2]], dtype=dtype)
    elif dim == 2:
        id = np.zeros([nrOfI, 2, sz[2], sz[3]], dtype=dtype)
    elif dim == 3:
        id = np.zeros([nrOfI, 3, sz[2], sz[3], sz[4]], dtype=dtype)
    else:
        raise ValueError(
            "Only dimensions 1-3 are currently supported for the identity map"
        )

    for n in range(nrOfI):
        id[n, ...] = identity_map(sz[2::], spacing, dtype=dtype)

    return id


def point_to_grid(points, grid_size, return_np=False):
    """

    :param points: prod(grid_size)xC
    :param grid_size: 3d: [X,Y,Z], 2d:[X,Y]
    :return: C xgrid_size
    """
    dim = len(grid_size)
    ch = points.shape[-1]
    if isinstance(points, np.ndarray):
        points = torch.Tensor(points)
    grid = points.reshape(grid_size + [ch])
    if dim == 2:
        grid = grid.permute(2, 0, 1)
    if dim == 3:
        grid = grid.permute(3, 0, 1, 2)
    if not return_np:
        return grid
    else:
        return grid.cpu().numpy()


def get_grid_wrap_points(points, spacing, pad_size=10, return_np=False):
    """

    :param points: N*D
    :param spacing:
    :return: prod(grid_size)*D, where grid size is computed from points range and spacing
    """
    device = None
    if isinstance(points, torch.Tensor):
        device = points.device
        points = points.cpu().numpy()
    dim = points.shape[-1]
    low_bound = np.min(points, 0)
    up_bound = np.max(points, 0)
    grid_size = (up_bound - low_bound) / spacing + 2 * pad_size
    grid_size = list(map(int, grid_size))
    id_map = identity_map(grid_size, spacing)
    id_map = id_map.reshape(dim, -1).transpose() + low_bound[None] - pad_size * spacing
    if not return_np:
        return torch.tensor(id_map).contiguous().to(device), grid_size
    else:
        return id_map, grid_size


def detect_folding(
    warped_grid_points, grid_size, spacing, saving_path=None, file_name=None
):
    from robot.utils.img_visual_utils import compute_jacobi_map

    warped_grid = point_to_grid(warped_grid_points, grid_size)
    compute_jacobi_map(warped_grid[None], spacing, saving_path, [file_name])


def compute_jacobi_of_pointcloud():
    def compute_jacobi(shape_pair, pair_name, model, record_path):
        from robot.global_variable import Shape

        source_grid_spacing = np.array([0.05] * 3).astype(
            np.float32
        )  # max(source_interval*20, 0.01)
        source_wrap_grid, grid_size = get_grid_wrap_points(
            shape_pair.source.points[0], source_grid_spacing
        )
        source_wrap_grid = source_wrap_grid[None]
        toflow = Shape()
        toflow.set_data(points=source_wrap_grid)
        shape_pair.set_toflow(toflow)
        shape_pair.control_weights = (
            torch.ones_like(shape_pair.control_weights)
            / shape_pair.control_weights.shape[1]
        )
        model.flow(shape_pair)
        detect_folding(
            shape_pair.flowed.points,
            grid_size,
            source_grid_spacing,
            record_path,
            pair_name,
        )

    return compute_jacobi


def shrink_by_factor(param, factor):
    if isinstance(param, list):
        param = [int(_param / factor) for _param in param]
    else:
        param = int(param / factor)
    return param


def enlarge_by_factor(param, factor):
    if isinstance(param, list):
        param = [int(_param * factor) for _param in param]
    else:
        param = int(param * factor)
    return param


def memory_sort(points, eps=0.0):
    """
    sort neighboring points close to each other in memory
    :param points: BxNxD or NxD  tenosr /array
    :return:
    """

    def _sort(points, eps):
        x_labels = grid_cluster(points, eps)
        # Compute the memory footprint and centroid of each of those non-empty "cubic" clusters:
        points, x_labels_sorted = sort_clusters(points, x_labels)
        return points, x_labels

    is_tensor = isinstance(points, torch.Tensor)
    has_batch = len(points.shape) == 3
    if is_tensor:
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points
    if has_batch:
        points_np_list = []
        ind_np_list = []
        for _points_np in points_np:
            _points_np, _ind = _sort(_points_np, eps)
            points_np_list.append(_points_np)
            ind_np_list.append(_ind)
        points_np = np.stack(points_np_list, 0)
        ind_np = np.stack(ind_np_list, 0)
    else:
        points_np, ind_np = _sort(points_np, eps)
    if is_tensor:
        points = torch.tensor(points_np).to(points.device)
        ind = torch.tensor(ind_np).to(points.device)
        return points, ind
    else:
        return points_np, ind_np


def memory_sort_helper(x, x_labels):
    is_tensor = isinstance(x, torch.Tensor)
    has_batch = len(x.shape) == 3
    if is_tensor:
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    if has_batch:
        x_np_list = []
        for _x in x_np:
            _x, _ = sort_clusters(_x, x_labels)
            x_np_list.append(_x)
        x_np = np.concatenate(x_np_list, 0)
    else:
        x_np, _ = sort_clusters(x_np, x_labels)
    if is_tensor:
        x = torch.tensor(x_np).to(x.device)
        return x
    else:
        return x_np


def add_zero_last_dim(points):
    device = points.device
    if isinstance(points, torch.Tensor):
        shape = list(points.shape)
        shape[-1] = 1
        zero_dim = torch.zeros(shape).to(device)
        return torch.cat([points, zero_dim], -1)
    else:
        shape = list(points.shape)
        shape[-1] = 1
        zero_dim = np.zeros(shape)
        return np.concatenate([points, zero_dim], -1)


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def timming(func, message=""):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def time_diff(*args, **kwargs):
        try:
            start.record()
            res = func(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            print("{}, it takes {} ms".format(message, start.elapsed_time(end)))
        except:
            res = func(*args, **kwargs)
        return res

    return time_diff


def timming(func, message="", return_t=False):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def time_diff(*args, **kwargs):
        try:
            start.record()
            res = func(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            t = start.elapsed_time(end)
            print("{}, it takes {} ms".format(message, t))
        except:
            res = func(*args, **kwargs)
        if not return_t:
            return res
        else:
            return res, t

    return time_diff
