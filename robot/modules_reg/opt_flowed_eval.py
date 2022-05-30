from copy import deepcopy
import torch

from robot.metrics.reg_losses import GeomDistance
from robot.modules_reg.module_gradient_flow import (
    wasserstein_barycenter_mapping,
    point_based_gradient_flow_guide,
)


def opt_flow_model_eval(
    shape_pair,
    model,
    batch_info=None,
    geom_loss_opt_for_eval=None,
    use_bary_map=True,
    external_evaluate_metric=None,
):
    """
    for  deep approach, we assume the source points = control points
    :param shape_pair:
    :param batch_info:
    :param geom_loss_opt_for_eval:
    :return:
    """
    corr_source_target = batch_info["corr_source_target"]
    geomloss_setting = deepcopy(geom_loss_opt_for_eval)
    geomloss_setting.print_settings_off()
    geomloss_setting["mode"] = "analysis"
    geomloss_setting["attr"] = "points"

    (
        mapped_target_index,
        mapped_topK_target_index,
        bary_mapped_position,
    ) = wasserstein_barycenter_mapping(
        shape_pair.flowed, shape_pair.target, geomloss_setting
    )  # BxN
    gt_mapped_position, wasserstein_dist = point_based_gradient_flow_guide(
        shape_pair.flowed, shape_pair.target, geomloss_setting
    )
    mapped_position = bary_mapped_position if use_bary_map else gt_mapped_position
    source_points = shape_pair.source.points
    B, N = source_points.shape[0], source_points.shape[1]
    device = source_points.device
    if corr_source_target:
        # compute mapped acc
        gt_index = torch.arange(N, device=device).repeat(B, 1)  # B,N
        acc = (mapped_target_index == gt_index).sum(1) / N
        topk_acc = ((mapped_topK_target_index == (gt_index[..., None])).sum(2) > 0).sum(
            1
        ) / N
        metrics = {
            "score": [_topk_acc.item() for _topk_acc in topk_acc],
            "_acc": [_acc.item() for _acc in acc],
            "topk_acc": [_topk_acc.item() for _topk_acc in topk_acc],
            "ot_dist": [_ot_dist.item() for _ot_dist in wasserstein_dist],
        }
    else:
        metrics = {
            "score": [_ot_dist.item() for _ot_dist in wasserstein_dist],
            "ot_dist": [_ot_dist.item() for _ot_dist in wasserstein_dist],
        }
    if external_evaluate_metric is not None:
        additional_param = {
            "model": model,
            "initial_nonp_control_points": shape_pair.control_points,
        }
        external_evaluate_metric(
            metrics, shape_pair, batch_info, additional_param=additional_param, alias=""
        )
        additional_param.update({"mapped_position": mapped_position})
        external_evaluate_metric(
            metrics, shape_pair, batch_info, additional_param, "_and_gf"
        )

    return metrics, shape_pair
