import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor
import geomloss
import numpy as np
from robot.modules_reg.networks.dcp_util import quat2mat, npmat2euler
from robot.utils.module_parameters import ParameterDict
from robot.utils.obj_factory import obj_factory

def transform_point_cloud(point_cloud, rotation, translation):

    return torch.matmul(rotation, point_cloud) + translation.unsqueeze(2)

def evaluate(source, target,rotation_pred, trans_pred):
    """
    sf_pred: (B, N, 3)
    sf_gt: (B, N, 3)
    """
    rotation_ab_pred, translation_ab_pred = rotation_pred, trans_pred
    source_points, target_points = source.points.transpose(2,1), target.points.transpose(2,1)
    source_extra_info, target_extra_info = source.extra_info, target.extra_info
    rotation_ab = target_extra_info["rotation_ab"]
    translation_ab = target_extra_info["translation_ab"]
    euler_ab = target_extra_info["euler_ab"]
    rotation_ba = target_extra_info["rotation_ba"]
    translation_ba = target_extra_info["translation_ba"]
    euler_ba = target_extra_info["euler_ba"]

    rotation_ba_pred = rotation_ab_pred.transpose(2, 1).contiguous()
    translation_ba_pred = -torch.matmul(rotation_ba_pred, translation_ab_pred.unsqueeze(2)).squeeze(2)


    transformed_src = transform_point_cloud(source_points, rotation_ab_pred, translation_ab_pred)
    transformed_target = transform_point_cloud(target_points, rotation_ba_pred, translation_ba_pred)
    batch_size = source_points.shape[0]

    ###########################
    identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
    loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
           + F.mse_loss(translation_ab_pred, translation_ab)

    mse_ab = torch.mean((transformed_src - target_points) ** 2, dim=[1, 2]).detach().cpu().numpy()
    mae_ab = torch.mean(torch.abs(transformed_src - target_points), dim=[1, 2]).detach().cpu().numpy()

    mse_ba = torch.mean((transformed_target - source_points) ** 2, dim=[1, 2]).detach().cpu().numpy()
    mae_ba = torch.mean(torch.abs(transformed_target - source_points), dim=[1, 2]).detach().cpu().numpy()

    rotation_ab_pred = rotation_ab_pred.detach().cpu().numpy()
    euler_ab = euler_ab.detach().cpu().numpy()
    translation_ab_pred = translation_ab_pred.detach().cpu().numpy()
    translation_ab = translation_ab.detach().cpu().numpy()
    rotation_ab_pred_euler = npmat2euler(rotation_ab_pred)
    test_r_mse_ab = np.mean((rotation_ab_pred_euler - np.degrees(euler_ab)) ** 2, axis=1)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(rotation_ab_pred_euler - np.degrees(euler_ab)), axis=1)
    test_t_mse_ab = np.mean((translation_ab - translation_ab_pred) ** 2, axis=1)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(translation_ab - translation_ab_pred), axis=1)

    rotation_ba_pred = rotation_ba_pred.detach().cpu().numpy()
    euler_ba = euler_ba.detach().cpu().numpy()
    translation_ba_pred = translation_ba_pred.detach().cpu().numpy()
    translation_ba = translation_ba.detach().cpu().numpy()
    rotation_ba_pred_euler = npmat2euler(rotation_ba_pred, 'xyz')
    test_r_mse_ba = np.mean((rotation_ba_pred_euler - np.degrees(euler_ba)) ** 2, axis=1)
    test_r_rmse_ba = np.sqrt(test_r_mse_ba)
    test_r_mae_ba = np.mean(np.abs(rotation_ba_pred_euler - np.degrees(euler_ba)), axis=1)
    test_t_mse_ba = np.mean((translation_ba - translation_ba_pred) ** 2, axis=1)
    test_t_rmse_ba = np.sqrt(test_t_mse_ba)
    test_t_mae_ba = np.mean(np.abs(translation_ba - translation_ba_pred), axis=1)
    metric = {"loss":[loss.item()]*batch_size, "mse_ab":mse_ab.tolist(), "mae_ab":mae_ab.tolist(), "mse_ba":mse_ba.tolist(), "mae_ba":mae_ba.tolist(),
     "test_r_mse_ab":test_r_mse_ab.tolist(), "test_r_rmse_ab":test_r_rmse_ab.tolist(), "test_r_mae_ab":test_r_mae_ab.tolist(),
     "test_t_mse_ab":test_t_mse_ab.tolist(), "test_t_rmse_ab":test_t_rmse_ab.tolist(), "test_t_mae_ab":test_t_mae_ab.tolist(),
     "test_r_mse_ba":test_r_mse_ba.tolist(), "test_r_rmse_ba":test_r_rmse_ba.tolist(), "test_r_mae_ba":test_r_mae_ba.tolist(),
     "test_t_mse_ba":test_t_mse_ba.tolist(), "test_t_rmse_ba":test_t_rmse_ba.tolist(), "test_t_mae_ba":test_t_mae_ba.tolist()
     }
    return metric



class OTHead(torch.nn.Module):
    def __init__(self, geomloss_obj):
        super(OTHead, self).__init__()
        self.geomloss_fn = obj_factory(geomloss_obj)
        self.eval_scale_for_rigid = False

    def solve_rigid(self, x, y, w):
        """

        :param x: BxNxD
        :param y: BxNxD
        :param w: BxNx1
        :return:
        """
        B, N, D = x.shape[0], x.shape[1], x.shape[2]
        device = x.device
        sum_w = w.sum(1, keepdim=True)
        mu_x = (x * w).sum(1, keepdim=True) / sum_w
        mu_y = (y * w).sum(1, keepdim=True) / sum_w
        x_hat = x - mu_x
        wx_hat = x_hat * w
        y_hat = y - mu_y
        wy_hat = y_hat * w
        a = wy_hat.transpose(2, 1) @ wx_hat  # BxDxN @ BxNxD  BxDxD
        u, s, v = torch.svd(a)
        c = torch.ones(B, D).to(device)
        c[:, -1] = torch.det(u @ v)  #
        r = (u * (c[..., None])) @ v.transpose(2, 1)
        tr_atr = torch.diagonal(a.transpose(2, 1) @ r, dim1=-2, dim2=-1).sum(-1)
        tr_xtwx = torch.diagonal(wx_hat.transpose(2, 1) @ wx_hat, dim1=-2, dim2=-1).sum(
            -1
        )
        s = (
            (tr_atr / tr_xtwx)[..., None][..., None]
            if self.eval_scale_for_rigid
            else 1.0
        )
        t = mu_y - s * (r @ mu_x.transpose(2, 1)).transpose(2, 1)
        # A = torch.cat([r.transpose(2, 1) * s, t], 1)
        # X = torch.cat((x, torch.ones_like(x[:, :, :1])), dim=2)  # (B,N, D+1)
        return r * s, t.view(B, 3)

    def forward(self, source, target):

        src = source.points
        tgt = target.points
        src_embedding = source.pointfea
        tgt_embedding = target.pointfea
        batch_size = src.size(0)
        weight1 = torch.ones(src.shape[0],src.shape[1],1).cuda()  # remove the last dim
        weight2 = torch.ones(tgt.shape[0],tgt.shape[1],1).cuda()  # remove the last dim
        grad_enable_record = torch.is_grad_enabled()
        device = src.device
        sqrt_const2 = torch.tensor(np.sqrt(2), dtype=torch.float32, device=device)
        blur = 0.01
        F_i, G_j = self.geomloss_fn( weight1[:,:,0], src_embedding, weight2[:,:,0], tgt_embedding)  # todo batch sz of input and output in geomloss is not consistent
        torch.set_grad_enabled(grad_enable_record)
        points1, points2 = src, tgt
        B, N, M, D = points1.shape[0], points1.shape[1], points2.shape[1], points2.shape[2]
        a_i, x_i = LazyTensor(weight1.view(B, N, 1, 1)), LazyTensor(
            src_embedding.view(B, N, 1, -1)
        )
        b_j, y_j = LazyTensor(weight2.view(B, 1, M, 1)), LazyTensor(
            tgt_embedding.view(B, 1, M, -1)
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
        position_to_map = LazyTensor(points2.view(B, 1, M, -1))  # Bx1xMxD
        mapped_position = log_P_ij.sumsoftmaxweight(position_to_map, dim=2)
        mapped_mass_ratio = log_P_ij.exp().sum(2) / weight1
        # mapped_mass_ratio = mapped_mass_ratio/(mapped_mass_ratio.sum(1,keepdim=True))
        rotation, trans = self.solve_rigid(src, mapped_position,mapped_mass_ratio)
        return rotation, trans



def evaluate_res():
    def eval(metrics, shape_pair, batch_info, additional_param=None, alias=""):
        if "mapped_position" not in additional_param:#10
            geomloss_obj = "geomloss.SamplesLoss(loss='sinkhorn',blur=0.01, scaling=0.8,reach=10, debias=False, backend='online',potentials=True)"
            source, target = shape_pair.source, shape_pair.target
            source.pointfea = shape_pair.flowed.pointfea
            rotation, trans = OTHead(geomloss_obj)(source, target)
            metrics_update = evaluate(source, target,rotation,trans)
            metrics.update(metrics_update)
        return metrics

    return eval
