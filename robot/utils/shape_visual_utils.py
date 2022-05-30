import os
import numpy as np
import torch
import pyvista as pv
from robot.datasets.vtk_utils import convert_faces_into_file_format


def save_shape_into_file(folder_path, alias, pair_name, ftype="vtk", **args):
    for key, item in args.items():
        if isinstance(item, torch.Tensor):
            args[key] = item.cpu().detach().numpy()
    points = args["points"]
    if len(points.shape) == 3:
        nbatch, _, _ = points.shape
    elif len(points.shape) == 2:
        points = points[None]
        for key, item in args.items():
            args[key] = args[key][None]
        nbatch = 1
    else:
        raise ValueError("shape not supported")
    os.makedirs(folder_path, exist_ok=True)
    faces = args["faces"] if "faces" in args else None
    for b in range(nbatch):
        if faces is not None:
            face = convert_faces_into_file_format(faces[b])
            data = pv.PolyData(points[b], face)
        else:
            data = pv.PolyData(points[b])
        for key, item in args.items():
            if key not in ["points", "faces"]:
                data.point_arrays[key] = item[b]
        fpath = os.path.join(
            folder_path, pair_name[b] + "_" + alias + ".{}".format(ftype)
        )
        data.save(fpath)


def save_shape_into_files(folder_path, alias, name, shape):

    attri_dict_to_save = {"points": shape.points, "weights": shape.weights}
    attri_dict_to_save["faces"] = shape.faces if not shape.points_mode_on else None
    if shape.pointfea is not None:
        attri_dict_to_save["pointfea"] = torch.norm(shape.pointfea, 2, dim=2)
    save_shape_into_file(folder_path, alias, name, **attri_dict_to_save)


def save_shape_pair_into_files(folder_path, stage_name, pair_name, shape_pair):
    if shape_pair.dimension != 3:
        return
    folder_path = os.path.join(folder_path, stage_name)
    save_shape_into_files(folder_path, "source", pair_name, shape_pair.source)
    save_shape_into_files(folder_path, "target", pair_name, shape_pair.target)
    if shape_pair.flowed is not None:
        save_shape_into_files(folder_path, "flowed", pair_name, shape_pair.flowed)
        save_shape_into_files(folder_path, "toflow", pair_name, shape_pair.toflow)
    if shape_pair.control_points is not None:
        save_shape_into_file(
            folder_path,
            "control",
            pair_name,
            **{
                "points": shape_pair.control_points,
                "weights": shape_pair.control_weights,
            }
        )
    if shape_pair.flowed_control_points is not None:
        save_shape_into_file(
            folder_path,
            "flowed_control",
            pair_name,
            **{
                "points": shape_pair.flowed_control_points,
                "weights": shape_pair.control_weights,
            }
        )
    if shape_pair.reg_param is not None:
        if shape_pair.reg_param.shape[1] == shape_pair.control_points.shape[1]:
            reg_param_norm = shape_pair.reg_param.norm(p=2, dim=2, keepdim=True)
            save_shape_into_file(
                folder_path,
                "reg_param",
                pair_name,
                **{
                    "points": shape_pair.control_points,
                    "reg_param_norm": reg_param_norm,
                    "reg_param_vector": shape_pair.reg_param,
                }
            )
        else:
            reg_param = shape_pair.reg_param.detach().cpu().numpy()
            np.save(os.path.join(folder_path, "reg_param_prealigned.npy"), reg_param)


def make_sphere(npoints=6000, ndim=3, radius=None, center=None):
    if radius is None:
        radius = np.array([1.0] * ndim)
    if center is None:
        center = np.array([0.0] * ndim)
    if not isinstance(radius, np.ndarray):
        radius = np.array(radius)
    if not isinstance(center, np.ndarray):
        center = np.array(center)
    radius = radius.reshape(ndim, 1)
    center = center.reshape(ndim, 1)
    points = np.random.randn(ndim, npoints)
    points /= np.linalg.norm(points, axis=0)
    points *= radius
    points += center
    return points.transpose([1, 0])


def make_ellipsoid(npoints=6000, ndim=3, radius=None, center=None, rotation=None):
    points = make_sphere(npoints, ndim, radius)
    if rotation is not None:
        points = np.matmul(points, rotation.transpose())
    points += center
    return points
