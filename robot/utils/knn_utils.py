from pykeops.torch import LazyTensor

from robot.utils.obj_factory import obj_factory
from robot.utils.local_feature_extractor import compute_anisotropic_gamma_from_points
from functools import partial

def NN(return_value=True, return_pos=False):
    def compute(pc1, pc2):
        from robot.modules_reg.networks.pointconv_util import index_points_group

        B,N = pc1.shape[0], pc1.shape[1]
        pc_i = LazyTensor(pc1[:,:,None])
        pc_j = LazyTensor(pc2[:,None])
        dist2 = pc_i.sqdist(pc_j)
        if return_value:
            K_min, index = dist2.min_argmin(dim=2)
            return K_min.view(B, N, 1), index.long().view(B, N, 1)
        elif return_pos:
            K_min, index = dist2.min_argmin(dim=2)
            Kmin_pc3 = index_points_group(pc2,index)
            return Kmin_pc3[:,:,0].contiguous(), index.long().view(B, N, 1)
        else:
            return dist2.argmin(dim=2).long().view(B, N, 1)
    return compute

def KNN(return_value=True):
    def compute(pc1, pc2, K):
        from robot.modules_reg.networks.pointconv_util import index_points_group
        B, N = pc1.shape[0], pc1.shape[1]
        pc_i = LazyTensor(pc1[:, :, None])
        pc_j = LazyTensor(pc2[:, None])
        dist2 = pc_i.sqdist(pc_j)
        index = dist2.argKmin(K, dim=2)
        if return_value:
            Kmin_pc3 = index_points_group(pc2, index)
            K_min = (pc1.unsqueeze(2) - Kmin_pc3).norm(p=2, dim=3)
            return K_min, index.long().view(B, N, K)
        else:
            return index.long().view(B, N, K)

    return compute



def AnisoKNN(
    cov_sigma_scale=0.02,
    aniso_kernel_scale=0.08,
    principle_weight=None,
    eigenvalue_min=0.3,
    iter_twice=True,
    leaf_decay=True,
    mass_thres=2.5,
    return_value=True,
    self_center=False,
):
    compute_gamma = partial(
        compute_anisotropic_gamma_from_points,
        cov_sigma_scale=cov_sigma_scale,
        aniso_kernel_scale=aniso_kernel_scale,
        principle_weight=principle_weight,
        eigenvalue_min=eigenvalue_min,
        iter_twice=iter_twice,
        leaf_decay=leaf_decay,
        mass_thres=mass_thres,
    )

    def compute(pc1, pc2, K):
        from robot.modules_reg.networks.pointconv_util import index_points_group
        B, N = pc1.shape[0], pc1.shape[1]
        if not self_center:
            gamma = compute_gamma(pc2)
            gamma = LazyTensor(
                gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, None]
            )  # Bx1xMxD*D
        else:
            gamma = compute_gamma(pc1)
            gamma = LazyTensor(
                gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, :, None]
            )  # BxNx1xD*D
        pc_i = LazyTensor(pc1[:, :, None])
        pc_j = LazyTensor(pc2[:, None])
        dist2 = (pc_i - pc_j) | gamma.matvecmult(pc_i - pc_j)
        index = dist2.argKmin(K, dim=2)
        if return_value:
            Kmin_pc3 = index_points_group(pc2, index)
            K_min = (pc1.unsqueeze(2) - Kmin_pc3).norm(p=2, dim=3)
            return K_min, index.long().view(B, N, K)
        else:
            return index.long().view(B, N, K)

    return compute

