from copy import deepcopy
import numpy as np
import torch
from robot.global_variable import Shape
from robot.utils.obj_factory import obj_factory
from robot.metrics.reg_losses import GeomDistance
from torch.autograd import grad


def point_based_gradient_flow_guide(
    cur_source, target, geomloss_setting, local_iter=-1
):
    geomloss_setting = deepcopy(geomloss_setting)
    geomloss_setting.print_settings_off()
    geomloss_setting["attr"] = "points"
    mode = geomloss_setting[("mode", "flow", "flow/analysis")]
    grad_enable_record = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    geomloss = GeomDistance(geomloss_setting)
    cur_source_points_clone = cur_source.points.detach().clone()
    cur_source_points_clone.requires_grad_()
    cur_source_clone = Shape()
    cur_source_clone.set_data_with_refer_to(
        cur_source_points_clone, cur_source
    )  # shallow copy, only points are cloned, other attr are not
    loss = geomloss(cur_source_clone, target)
    # print("{} th step, before gradient flow, the ot distance between the cur_source and the target is {}".format(
    #     local_iter.item(), loss.item()))
    grad_cur_source_points = grad(loss.sum(), cur_source_points_clone)[0]
    torch.set_grad_enabled(grad_enable_record)
    cur_source_points_clone = (
        cur_source_points_clone - grad_cur_source_points / cur_source_clone.weights
    )
    cur_source_clone.points = cur_source_points_clone.detach()
    # loss = geomloss(cur_source_clone, target)
    # print(
    #     "{} th step, after gradient flow, the ot distance between the gradflowed guided points and the target is {}".format(
    #         local_iter.item(), loss.item()))
    if mode == "flow":
        return cur_source_clone, None
    elif mode == "analysis":
        return cur_source_clone.points, loss


def wasserstein_barycenter_mapping(cur_source, target, gemloss_setting):
    from pykeops.torch import LazyTensor

    grad_enable_record = torch.is_grad_enabled()
    geom_obj = gemloss_setting["geom_obj"].replace(")", ",potentials=True)")
    blur_arg_filtered = filter(lambda x: "blur" in x, geom_obj.split(","))
    blur = eval(list(blur_arg_filtered)[0].replace("blur", "").replace("=", ""))
    # though can be generalized to arbitrary order, here we assume the order is 2
    mode = gemloss_setting[
        ("mode", "soft", "soft, hard, mapped_index,analysis,trans_plan")
    ]
    geomloss = obj_factory(geom_obj)
    attr = gemloss_setting[("attr", "pointfea", "points/pointfea/landmarks")]
    attr1 = getattr(cur_source, attr)
    attr2 = getattr(target, attr)
    points1 = cur_source.points
    points2 = target.points
    device = points1.device
    sqrt_const2 = torch.tensor(np.sqrt(2), dtype=torch.float32, device=device)
    weight1 = cur_source.weights[:, :, 0]  # remove the last dim
    weight2 = target.weights[:, :, 0]  # remove the last dim
    F_i, G_j = geomloss(
        weight1, attr1, weight2, attr2
    )  # todo batch sz of input and output in geomloss is not consistent
    torch.set_grad_enabled(grad_enable_record)

    B, N, M, D = points1.shape[0], points1.shape[1], points2.shape[1], points2.shape[2]
    a_i, x_i = LazyTensor(cur_source.weights.view(B, N, 1, 1)), LazyTensor(
        attr1.view(B, N, 1, -1)
    )
    b_j, y_j = LazyTensor(target.weights.view(B, 1, M, 1)), LazyTensor(
        attr2.view(B, 1, M, -1)
    )
    F_i, G_j = LazyTensor(F_i.view(B, N, 1, 1)), LazyTensor(G_j.view(B, 1, M, 1))
    xx_i = x_i / (sqrt_const2 * blur)
    yy_j = y_j / (sqrt_const2 * blur)
    f_i = a_i.log() + F_i / blur ** 2
    g_j = b_j.log() + G_j / blur ** 2  # Bx1xMx1
    C_ij = ((xx_i - yy_j) ** 2).sum(-1)  # BxNxMx1
    log_P_ij = (
        f_i + g_j - C_ij
    )  # BxNxMx1 P_ij = A_i * B_j * exp((F_i + G_j - .5 * |x_i-y_j|^2) / blur**2)
    log_prob_i = log_P_ij - a_i.log()  # BxNxM
    if mode == "soft":
        position_to_map = LazyTensor(points2.view(B, 1, M, -1))  # Bx1xMxD
        mapped_position = log_P_ij.sumsoftmaxweight(position_to_map, dim=2)
        mapped_mass_ratio = log_P_ij.exp().sum(2) / cur_source.weights
    elif mode == "hard":
        P_i_index = log_P_ij.argmax(dim=2).long().view(B, N)  #  over M,  return (B*N)
        for i in range(B):  # todo not test yet
            P_i_index[i] += int(N * i)
        P_i_index = P_i_index.view(-1)
        points2_flatten = points2.view(-1, D)
        mapped_position = points2_flatten[P_i_index]
        mapped_position = mapped_position.view(B, N, D)
        mapped_mass_ratio = log_P_ij.exp().sum(2) / cur_source.weights
    elif mode == "mapped_index":
        P_i_index = log_P_ij.argmax(dim=2).long().view(B, N)  # over M,  return (B,N)
        return P_i_index
    elif mode == "analysis":
        K = 5
        P_i_index = log_P_ij.argmax(dim=2).long().view(B, N)  # over M,  return (B,N)
        P_Ki_index = (
            (-log_P_ij).argKmin(K=K, dim=2).long().view(B, N, K)
        )  # over M,  return (B,N,K)
        position_to_map = LazyTensor(points2.view(B, 1, M, -1))  # Bx1xMxD
        mapped_position = log_P_ij.sumsoftmaxweight(position_to_map, dim=2)

        return P_i_index, P_Ki_index, mapped_position
    elif mode == "trans_plan":
        return log_P_ij.exp(), log_P_ij
    elif mode == "prob":
        return log_prob_i.exp(), log_prob_i
    else:
        raise ValueError(
            "mode {} not defined, support: soft/ hard/ confid".format(mode)
        )
    # print("OT based forward mapping complete")
    mapped_shape = Shape()
    mapped_shape.set_data_with_refer_to(mapped_position, cur_source)
    return mapped_shape, mapped_mass_ratio


# def wasserstein_barycenter_mapping(cur_source, target,gemloss_setting,local_iter=None):
#     from pykeops.torch import LazyTensor
#     geom_obj = gemloss_setting["geom_obj"].replace(")", ",potentials=True)")
#     blur_arg_filtered = filter(lambda x: "blur" in x, geom_obj.split(","))
#     blur = eval(list(blur_arg_filtered)[0].replace("blur", "").replace("=", ""))
#     p = gemloss_setting[("p", 2,"cost order")]
#     mode = gemloss_setting[("mode", 'hard',"soft, hard")]
#     confid = gemloss_setting[("confid", 0.0,"cost order")]
#     geomloss = obj_factory(geom_obj)
#     attr = "pointfea"
#     attr1 = getattr(cur_source, attr).detach()
#     attr2 = getattr(target, attr).detach()
#     points1 = cur_source.points
#     points2 = target.points
#     weight1 = cur_source.weights[:, :, 0]  # remove the last dim
#     weight2 = target.weights[:, :, 0]  # remove the last dim
#     F_i, G_j = geomloss(weight1, attr1, weight2,
#                         attr2)  # todo batch sz of input and output in geomloss is not consistent
#
#     B, N, M, D = points1.shape[0], points1.shape[1], points2.shape[1], points2.shape[2]
#     a_i, x_i = LazyTensor(cur_source.weights.view(B,N, 1, 1)), LazyTensor(attr1.view(B,N, 1, -1))
#     b_j, y_j = LazyTensor(target.weights.view(B,1, M, 1)), LazyTensor(attr2.view(B,1, M, -1))
#     F_i, G_j = LazyTensor(F_i.view(B,N, 1, 1)), LazyTensor(G_j.view(B,1, M, 1))
#     C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)  # (B,N,M,1) cost matrix
#     eps = blur ** p  # temperature epsilon
#     P_i = ((F_i + G_j - C_ij) / eps).exp() * (b_j)  # (B, N,M,1) transport plan
#     if mode=="soft":
#         position_to_map = LazyTensor(points2.view(B,1, M, -1))  # B,1xMxD
#         P_sum_over_j = P_i.sum_reduction(2) # B,N,1
#         mapped_position = (P_i*position_to_map).sum_reduction(2) #(B,N,M,D)-> (B,N,D)
#         mapped_position = mapped_position/P_sum_over_j
#     elif mode == "hard":
#         P_i_max, P_i_index = P_i.max_argmax(2) #  over M,  return (B,N)
#         P_i_max, P_i_index = P_i_max.view(-1), P_i_index.view(-1)
#         points1_flatten = points1.view(-1, D)
#         points2_flatten = points2.view(-1, D)
#         mapped_position = points2_flatten[P_i_index]
#         low_index = P_i_max<confid
#         mapped_position[low_index] = points1_flatten[low_index]
#         mapped_position = mapped_position.view(B,N,D)
#     else:
#         raise ValueError("mode {} not defined, support: soft/ hard/ confid".format(mode))
#     print("OT based forward mapping complete")
#     mapped_shape = Shape()
#     mapped_shape.set_data_with_refer_to(mapped_position,cur_source)
#     return mapped_shape


def gradient_flow_guide(mode="grad_forward"):
    postion_based = mode == "grad_forward"

    def guide(cur_source, target, geomloss_setting, local_iter=None):
        if postion_based:
            return point_based_gradient_flow_guide(
                cur_source, target, geomloss_setting, local_iter
            )
        else:
            return wasserstein_barycenter_mapping(cur_source, target, geomloss_setting)

    return guide
