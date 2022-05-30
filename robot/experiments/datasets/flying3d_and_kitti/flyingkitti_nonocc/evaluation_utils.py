import torch
import os
import numpy as np
import robot.experiments.datasets.flying3d_and_kitti.geometry as geometry
from robot.utils.shape_visual_utils import save_shape_into_files
from robot.shape.shape_pair import Shape

def evaluate_3d(sf_pred, sf_gt):
    """
    sf_pred: (B, N, 3)
    sf_gt: (B, N, 3)
    """
    B = sf_pred.shape[0]
    l2_norm = torch.norm(sf_gt - sf_pred, p=2, dim=-1)
    EPE3D = l2_norm.view(B, -1).mean(-1)
    sf_norm = torch.norm(sf_gt, p=2, dim=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)
    acc3d_strict = (
        torch.logical_or(l2_norm < 0.05, relative_err < 0.05)
        .view(B, -1)
        .float()
        .mean(-1)
    )
    acc3d_relax = (
        torch.logical_or(l2_norm < 0.1, relative_err < 0.1).view(B, -1).float().mean(-1)
    )
    outlier = (
        torch.logical_or(l2_norm > 0.3, relative_err > 0.1).view(B, -1).float().mean(-1)
    )
    return EPE3D, acc3d_strict, acc3d_relax, outlier


def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (B, N, 2)
    flow_gt: (B, N, 2)
    """
    B = flow_pred.shape[0]
    epe2d = torch.norm(flow_gt - flow_pred, p=2, dim=-1)
    epe2d_mean = epe2d.view(B, -1).mean(-1)
    flow_gt_norm = torch.norm(flow_gt, p=2, dim=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)
    acc2d = (
        torch.logical_or(epe2d < 3.0, relative_err < 0.05).float().view(B, -1).mean(-1)
    )
    return epe2d_mean, acc2d


def evaluate_res(is_kitti=False):
    def eval(metrics, shape_pair, batch_info, additional_param=None, alias=""):
        has_gt = batch_info["has_gt"]
        if not has_gt:
            return metrics
        gt_flowed = Shape().set_data(points=shape_pair.extra_info["gt_flowed"], weights=shape_pair.source.points)
        sp, tp, fp = (
            shape_pair.source.points,
            shape_pair.extra_info["gt_flowed"],
            shape_pair.flowed.points,
        )
        has_prealign = (
            "prealign_param" in additional_param
            and additional_param["prealign_param"] is not None
        )
        record_path = os.path.join(
            batch_info["record_path"],"3d","{}_epoch_{}".format(batch_info["phase"], batch_info["epoch"]),
        )
        os.makedirs(record_path, exist_ok=True)
        if ("mapped_position" not in additional_param
        ):
            save_shape_into_files(
                record_path,
                (alias + "_gt_flowed") if len(alias) else "gt_flowed",
                batch_info["pair_name"],
                gt_flowed
            )

        if (
            additional_param is not None
            and has_prealign
            and "mapped_position" not in additional_param
        ):
            save_shape_into_files(record_path,
                alias + "_prealigned",batch_info["pair_name"],additional_param["prealigned"],
            )
            reg_param = additional_param["prealign_param"].detach().cpu().numpy()
            for pid, pair_name in enumerate(batch_info["pair_name"]):
                np.save(os.path.join(record_path, pair_name + alias + "_prealigned_reg_param.npy"),
                    reg_param[pid],)

        if additional_param is not None and "mapped_position" in additional_param:
            fp = additional_param["mapped_position"]
            save_shape_into_files(
                record_path,
                (alias + "_flowed") if len(alias) else "flowed",
                batch_info["pair_name"],
                shape_pair.flowed,
            )
        pred_sf, gt_sf = fp - sp, tp - sp
        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, gt_sf)
        # 2D evaluation metrics
        folder_path_list = [
            os.path.split(path)[0] for path in batch_info["source_info"]["data_path"]
        ]
        tnp = lambda x: x.detach().cpu().numpy()
        flow_pred_2d_np, flow_gt_2d_np = geometry.get_batch_2d_flow(
            tnp(sp), tnp(tp), tnp(fp), folder_path_list, is_kitti
        )
        EPE2D, acc2d = evaluate_2d(
            torch.from_numpy(flow_pred_2d_np), torch.from_numpy(flow_gt_2d_np)
        )
        metrics_update = {
            "EPE3D" + alias: [_EPE3D.item() for _EPE3D in EPE3D],
            "ACC3DS" + alias: [_acc3d_strict.item() for _acc3d_strict in acc3d_strict],
            "ACC3DR" + alias: [_acc3d_relax.item() for _acc3d_relax in acc3d_relax],
            "Outliers3D" + alias: [_outlier.item() for _outlier in outlier],
            "EPE2D" + alias: [_EPE2D.item() for _EPE2D in EPE2D],
            "ACC2D" + alias: [_acc2d.item() for _acc2d in acc2d],
        }
        metrics.update(metrics_update)
        return metrics

    return eval
