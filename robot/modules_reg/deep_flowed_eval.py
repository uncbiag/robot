from copy import deepcopy
import torch
from robot.global_variable import Shape
from robot.modules_reg.module_gradient_flow import wasserstein_barycenter_mapping, point_based_gradient_flow_guide
from robot.shape.point_interpolator import NNInterpolater


def deep_flow_model_eval(shape_pair, model,buffer, batch_info=None,geom_loss_opt_for_eval=None,mapping_strategy="barycenter",aniso_post_kernel=None, finetune_iter=1, external_evaluate_metric=None,cur_epoch=-1):
    corr_source_target = batch_info["corr_source_target"]
    geomloss_setting = deepcopy(geom_loss_opt_for_eval)
    geomloss_setting.print_settings_off()
    geomloss_setting["mode"] = "analysis"
    geomloss_setting["attr"] = "points"
    use_bary_map = geomloss_setting[("use_bary_map", True,
                                     "take wasserstein baryceneter if there is little noise or outlier,  otherwise use gradient flow")]
    # in the first epoch, we would output the ot baseline, this is for analysis propose, comment the following two lines if don't needed
    source_points = shape_pair.source.points
    if cur_epoch==0:
        print("In the first epoch, the validation/debugging output is the baseline ot mapping")
        shape_pair.flowed= Shape().set_data_with_refer_to(source_points,shape_pair.source)
    if mapping_strategy=="barycenter":
        mapped_target_index,mapped_topK_target_index, mapped_position = wasserstein_barycenter_mapping(shape_pair.flowed, shape_pair.target, geomloss_setting)  # BxN
        wasserstein_dist = torch.Tensor([-1] * shape_pair.nbatch) # todo return distance via wasserstein_barycenter_mapping
    elif mapping_strategy=="nn":
        mapped_position = NNInterpolater()(shape_pair.flowed.points, shape_pair.target.points, shape_pair.target.points)
        wasserstein_dist = torch.Tensor([-1] * shape_pair.nbatch)
    else:
        raise NotImplemented

    if aniso_post_kernel is not None:
        disp = mapped_position - shape_pair.flowed.points
        flowed_points = shape_pair.flowed.points
        smoothed_disp = aniso_post_kernel(flowed_points, flowed_points, disp, shape_pair.flowed.weights)
        mapped_position = flowed_points + smoothed_disp
        for _ in range(1,finetune_iter):
            cur_flowed = Shape().set_data_with_refer_to(mapped_position, shape_pair.flowed)
            if mapping_strategy != "nn":
                mapped_target_index, mapped_topK_target_index, mapped_position = wasserstein_barycenter_mapping(
                    cur_flowed, shape_pair.target, geomloss_setting)  # BxN
                wasserstein_dist = torch.Tensor(
                    [-1] * shape_pair.nbatch)  # todo return distance via wasserstein_barycenter_mapping
            else:
                mapped_position = NNInterpolater()(cur_flowed.points, shape_pair.target.points, shape_pair.target.points)
                wasserstein_dist = torch.Tensor([-1] * shape_pair.nbatch)
            disp = mapped_position - cur_flowed.points
            flowed_points = cur_flowed.points
            smoothed_disp = aniso_post_kernel(flowed_points, flowed_points, disp, shape_pair.flowed.weights)
            mapped_position = flowed_points + smoothed_disp

    B, N = source_points.shape[0], source_points.shape[1]
    device = source_points.device
    print("the current data is {}".format("synth" if batch_info["is_synth"] else "real"))
    if corr_source_target:
        gt_index = torch.arange(N, device=device).repeat(B, 1)  # B,N
        acc = (mapped_target_index == gt_index).sum(1) / N
        topk_acc = ((mapped_topK_target_index == (gt_index[..., None])).sum(2) > 0).sum(1) / N
        metrics = {"score": [_topk_acc.item() for _topk_acc in topk_acc],
                   "_acc": [_acc.item() for _acc in acc], "topk_acc": [_topk_acc.item() for _topk_acc in topk_acc],
                   "ot_dist": [_ot_dist.item() for _ot_dist in wasserstein_dist]}
    else:
        metrics = {"score": [_sim.item() for _sim in buffer["sim_loss"]],
                   "ot_dist": [_ot_dist.item() for _ot_dist in wasserstein_dist]}
    if external_evaluate_metric is not None:
        additional_param = {"model": model, "initial_nonp_control_points": buffer["initial_nonp_control_points"],
                            "prealign_param": buffer["prealign_param"], "prealigned": buffer["prealigned"]}
        external_evaluate_metric(metrics, shape_pair, batch_info, additional_param=additional_param, alias="")
        additional_param.update({"mapped_position": mapped_position})
        external_evaluate_metric(metrics, shape_pair, batch_info, additional_param, "_gf")
    return metrics, shape_pair