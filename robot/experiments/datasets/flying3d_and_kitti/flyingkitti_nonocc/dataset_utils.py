import os
import torch
import numpy as np
from robot.utils.utils import memory_sort, memory_sort_helper
from robot.shape.point_sampler import uniform_sampler

DEPTH_THRESHOLD = 35.0


def flyingkitti_nonocc_reader(flying3d=True):
    def read_flying3d(file_info):
        path = file_info["data_path"]
        pc = np.load(path)
        pc[..., 0] *= -1
        pc[..., -1] *= -1
        data_dict = {}
        data_dict["points"] = pc
        npoint_source = pc.shape[0]
        data_dict["extra_info"] = {}
        data_dict["extra_info"]["mask"] = np.ones([npoint_source, 1]).astype(bool)
        return data_dict

    def read_kitti(file_info):
        path = file_info["data_path"]
        pc = np.load(path)
        data_dict = {}
        data_dict["points"] = pc
        npoint_source = pc.shape[0]
        data_dict["extra_info"] = {}
        data_dict["extra_info"]["mask"] = np.ones([npoint_source, 1]).astype(bool)
        return data_dict

    return read_flying3d if flying3d else read_kitti


def flyingkitti_nonocc_normalizer():
    def normalize(data_dict):
        return data_dict

    return normalize


def flyingkitti_nonocc_sampler(eps=0, **args):
    def uniform_sample(data_dict, ind=None, fixed_random_seed=True):
        num_sample = args["num_sample"]
        if num_sample != -1:
            points = data_dict["points"]
            # mask = data_dict["extra_info"]["mask"].astype(bool)
            if ind is None:
                sampler = uniform_sampler(
                    num_sample, fixed_random_seed, sampled_by_weight=False
                )
                sampled_points, _, ind = sampler(torch.tensor(points))
                sampled_points = sampled_points.numpy()
                ind = ind.numpy()
            else:
                sampled_points = points[ind]
            data_dict["points"] = sampled_points
            data_dict["extra_info"] = {
                key: item[ind] for key, item in data_dict["extra_info"].items()
            }
            data_dict["weights"] = (
                np.ones([num_sample, 1], dtype=np.float32) / num_sample
            )
            if eps > 0:
                data_dict["points"], m_ind = memory_sort(sampled_points, eps)
                data_dict["extra_info"]["mask"] = memory_sort_helper(
                    data_dict["extra_info"]["mask"], m_ind
                )
        return data_dict, ind

    return uniform_sample


def flyingkitti_nonocc_pair_postprocess(
    flying3d=True, corr_sampled_source_target=False
):
    def flying3d_postprocess(source_dict, target_dict, sampler=None, phase=None):
        source_dict["extra_info"]["gt_flow"] = (
            target_dict["points"] - source_dict["points"]
        )
        target_dict["extra_info"]["gt_flow"] = (
            target_dict["points"] - source_dict["points"]
        )
        npoints = source_dict["points"].shape[0]
        near_mask = np.logical_and(
            source_dict["points"][:, 2] < DEPTH_THRESHOLD,
            target_dict["points"][:, 2] < DEPTH_THRESHOLD,
        )

        indices = np.where(near_mask)[0]
        if len(indices) < npoints:
            print(
                "{} points deeper than {} has been removed".format(
                    npoints - len(indices), DEPTH_THRESHOLD
                )
            )
        extra_info = {
            key: item[indices] for key, item in source_dict["extra_info"].items()
        }
        source_dict = {
            key: item[indices]
            for key, item in source_dict.items()
            if key != "extra_info"
        }
        source_dict.update({"extra_info": extra_info})

        extra_info = {
            key: item[indices] for key, item in target_dict["extra_info"].items()
        }
        target_dict = {
            key: item[indices]
            for key, item in target_dict.items()
            if key != "extra_info"
        }
        target_dict.update({"extra_info": extra_info})
        if sampler is not None:
            if phase == "train":
                " here we will use hybird data generator for further data augmentation"
                source_dict, ind = sampler(
                    source_dict, ind=None, fixed_random_seed=False
                )
                target_dict, _ = sampler(target_dict, ind=ind, fixed_random_seed=False)
            else:
                if not corr_sampled_source_target:  # for test
                    source_dict, _ = sampler(
                        source_dict, ind=None, fixed_random_seed=False
                    )
                    target_dict, _ = sampler(
                        target_dict, ind=None, fixed_random_seed=False
                    )
                else:  # for val and debug
                    source_dict, ind = sampler(
                        source_dict, ind=None, fixed_random_seed=False
                    )
                    target_dict, _ = sampler(
                        target_dict, ind=ind, fixed_random_seed=False
                    )
        target_dict["extra_info"]["gt_flow"] = source_dict["extra_info"]["gt_flow"]
        return source_dict, target_dict

    def kitti_postprocess(source_dict, target_dict, sampler=None, phase=None):
        npoints = source_dict["points"].shape[0]
        source_dict["extra_info"]["gt_flow"] = (
            target_dict["points"] - source_dict["points"]
        )
        target_dict["extra_info"]["gt_flow"] = (
            target_dict["points"] - source_dict["points"]
        )
        near_mask = np.logical_and(
            source_dict["points"][:, 2] < DEPTH_THRESHOLD,
            target_dict["points"][:, 2] < DEPTH_THRESHOLD,
        )
        is_ground = np.logical_and(
            source_dict["points"][:, 1] < -1.4, target_dict["points"][:, 1] < -1.4
        )
        not_ground = np.logical_not(is_ground)
        near_mask = np.logical_and(near_mask, not_ground)
        indices = np.where(near_mask)[0]
        print(
            "{} points deeper= than {} has been removed, {} remained".format(
                npoints - len(indices), DEPTH_THRESHOLD, len(indices)
            )
        )
        #
        # if len(indices) <npoints :
        #     print('{} points deeper= than {} has been removed, {} remained'.format(npoints-len(indices),DEPTH_THRESHOLD, len(indices)))
        extra_info = {
            key: item[indices] for key, item in source_dict["extra_info"].items()
        }
        source_dict = {
            key: item[indices]
            for key, item in source_dict.items()
            if key != "extra_info"
        }
        source_dict.update({"extra_info": extra_info})

        extra_info = {
            key: item[indices] for key, item in target_dict["extra_info"].items()
        }
        target_dict = {
            key: item[indices]
            for key, item in target_dict.items()
            if key != "extra_info"
        }
        target_dict.update({"extra_info": extra_info})
        if sampler is not None:
            if phase == "train":
                raise NotImplemented  # no train on kitti yet
            else:
                source_dict, _ = sampler(source_dict, ind=None, fixed_random_seed=False)
                target_dict, _ = sampler(target_dict, ind=None, fixed_random_seed=False)
        target_dict["extra_info"]["gt_flow"] = source_dict["extra_info"]["gt_flow"]

        return source_dict, target_dict

    postprocess = flying3d_postprocess if flying3d else kitti_postprocess
    return postprocess
