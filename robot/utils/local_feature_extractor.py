import math
import numpy as np
import torch
from pykeops.torch import LazyTensor
from robot.utils.compute_2d_eigen import compute_2d_eigen
from robot.utils.obj_factory import partial_obj_factory
from robot.utils.visualizer import (
    visualize_point_fea,
    visualize_point_fea_with_arrow,
    visualize_point_overlap,
)
from torchvectorized.vlinalg import vSymEig

# import pykeops
# pykeops.clean_pykeops()


def numpy(x):
    return x.detach().cpu().numpy()


def compute_local_moments(points, radius=1.0):
    # B, N, D = points.shape
    shape_head, D = points.shape[:-1], points.shape[-1]

    scale = 1.41421356237 * radius  # math.sqrt(2) is not super JIT-friendly...
    x = points / scale  # Normalize the kernel size
    # Computation:
    x = torch.cat((torch.ones_like(x[..., :1]), x), dim=-1)  # (B, N, D+1)

    x_i = LazyTensor(x[..., :, None, :])  # (B, N, 1, D+1)
    x_j = LazyTensor(x[..., None, :, :])  # (B, 1, N, D+1)

    D_ij = ((x_i - x_j) ** 2).sum(-1)  # (B, N, N), squared distances
    K_ij = (-D_ij / 2).exp()  # (B, N, N), Gaussian kernel

    C_ij = (K_ij * x_j).tensorprod(x_j)  # (B, N, N, (D+1)*(D+1))
    C_i = C_ij.sum(dim=len(shape_head)).view(
        shape_head + (D + 1, D + 1)
    )  # (B, N, D+1, D+1) : descriptors of order 0, 1 and 2

    w_i = C_i[..., :1, :1]  # (B, N, 1, 1), weights
    m_i = C_i[..., :1, 1:] * scale  # (B, N, 1, D), sum
    c_i = C_i[..., 1:, 1:] * (scale ** 2)  # (B, N, D, D), outer products

    mass_i = w_i.squeeze(-1)  # (B, N)
    dev_i = (m_i / w_i).squeeze(-2) - points  # (B, N, D)
    cov_i = (c_i - (m_i.transpose(-1, -2) * m_i) / w_i) / w_i  # (B, N, D, D)

    return mass_i, dev_i, cov_i


def compute_aniso_local_moments(points, gamma=None):
    # B, N, D = points.shape
    shape_head, D = points.shape[:-1], points.shape[-1]

    x = points  # Normalize the kernel size
    # Computation:
    xp_i = LazyTensor(x[..., :, None, :])
    xp_j = LazyTensor(x[..., None, :, :])
    gamma = LazyTensor(
        gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, None]
    )  # Bx1xMxD*D
    dist2 = (xp_i - xp_j) | gamma.matvecmult(xp_i - xp_j)
    # dist2 = xp_i.weightedsqdist(xp_j, gamma)
    K_ij = (-dist2).exp()  # BxNxN

    x = torch.cat((torch.ones_like(x[..., :1]), x), dim=-1)  # (B, N, D+1)
    x_j = LazyTensor(x[..., None, :, :])  # (B, 1, N, D+1)

    C_ij = (K_ij * x_j).tensorprod(x_j)  # (B, N, N, (D+1)*(D+1))
    # if  dim= 1,  self-centered mode  if set dim=2, then is the interpolate mode
    C_i = C_ij.sum(dim=2).view(shape_head + (D + 1, D + 1))  # (B, N, D+1, D+1)

    w_i = C_i[..., :1, :1]  # (B, N, 1, 1), weights
    m_i = C_i[..., :1, 1:]  # (B, N, 1, D), sum
    c_i = C_i[..., 1:, 1:]  # (B, N, D, D), outer products

    mass_i = w_i.squeeze(-1)  # (B, N)
    dev_i = (m_i / w_i).squeeze(-2) - points  # (B, N, D)
    cov_i = (c_i - (m_i.transpose(-1, -2) * m_i) / w_i) / w_i  # (B, N, D, D)

    return mass_i, dev_i, cov_i


def compute_local_fea_from_moments(fea_type, weights, mass, dev, cov):
    fea = None
    B, N, D = cov.shape[0], cov.shape[1], cov.shape[-1]
    if fea_type == "weights":
        fea = weights
    elif fea_type == "mass":
        fea = mass.view(B, N, -1)
    elif fea_type == "dev":
        fea = (dev ** 2).sum(-1, keepdim=True)
    elif fea_type == "cov":
        fea = cov.view(B, N, -1)
    elif "eigen" in fea_type:
        compute_eigen_vector = fea_type in ["eigenvector", "eigenvector_main"]
        if D == 2:
            cov = cov.view(B, N, -1, 1).permute(0, 2, 1, 3)
            vals, vectors = compute_2d_eigen(
                cov, eigenvectors=compute_eigen_vector, flatten_output=True
            )
        elif D == 3:
            cov = cov.view(B, N, -1, 1, 1).permute(0, 2, 1, 3, 4)
            vals, vectors = vSymEig(
                cov,
                eigenvectors=compute_eigen_vector,
                flatten_output=True,
                descending_eigenvals=True,
            )
        else:
            raise NotImplemented(
                "high dimension >3 eigen decomposition is not implemented"
            )
        vals = vals.view(B, N, -1)
        if vectors is not None:
            vectors = vectors / (
                torch.norm(vectors, p=2, dim=1, keepdim=True) + 1e-9
            )  # B*NxDxD
            assert not torch.any(torch.isnan(vectors)) and not torch.any(
                torch.isinf(vectors)
            )
        if fea_type == "eigenvalue_prod":
            if D == 2:
                fea = mass * vals[..., 1:2]
            elif D == 3:
                fea = mass * vals[..., 1:2] * vals[..., 2:3]
        elif fea_type == "eigenvalue":
            fea = vals
        elif fea_type == "eigenvector_main":
            vectors = vectors.view(B, N, D, D)
            fea = vectors[..., 0]
        elif fea_type == "eigenvector":
            fea = vectors.view(B, N, -1)

    else:
        raise ValueError("not implemented")
    return fea


def extract_point_fea(flowed, target):
    flowed.pointfea = flowed.points
    target.pointfea = target.points
    return flowed, target


def feature_extractor(fea_type_list, radius=1.0, std_normalize=True, include_pos=False):
    def _compute_fea(
        points,
        weights,
        weight_list=None,
        gamma=None,
        return_stats=False,
        mean=None,
        std=None,
    ):
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)
        if weight_list is None:
            weight_list = [1.0] * len(fea_type_list)
        if gamma is None:
            mass, dev, cov = compute_local_moments(
                points, radius=radius
            )  # (N,), (N, D), (N, D, D)
        else:
            mass, dev, cov = compute_aniso_local_moments(
                points, gamma=gamma
            )  # (N,), (N, D), (N, D, D)
        fea_list = [
            compute_local_fea_from_moments(fea_type, weights, mass, dev, cov)
            for fea_type in fea_type_list
        ]
        fea_dim_list = [fea.shape[-1] for fea in fea_list]
        weights = []
        for i, dim in enumerate(fea_dim_list):
            weights += [weight_list[i] / math.sqrt(dim)] * dim
        fea_combined = None
        if fea_dim_list:
            weights = (
                torch.tensor(weights).view(1, 1, sum(fea_dim_list)).to(points.device)
            )
            fea_combined = torch.cat(fea_list, -1)
            if std_normalize:
                if mean is None and std is None:
                    mean = fea_combined.mean(1, keepdim=True)
                    std = fea_combined.std(1, keepdim=True)
                fea_combined = (fea_combined - mean) / std
                fea_combined = fea_combined.clamp(-1, 1)
                fea_combined = fea_combined * weights
        if include_pos:
            fea_combined = (
                torch.cat([points, fea_combined], -1)
                if fea_combined is not None
                else points
            )
        if return_stats:
            return fea_combined, mean, std, mass
        else:
            return fea_combined, mass

    return _compute_fea


def pair_feature_extractor(
    fea_type_list, weight_list=None, radius=0.01, std_normalize=True, include_pos=False
):
    fea_extractor = feature_extractor(fea_type_list, radius, std_normalize, include_pos)

    def extract(source, target, iter=None, flowed_gamma=None, target_gamma=None):
        source.pointfea, mean, std, _ = fea_extractor(
            source.points,
            source.weights,
            weight_list=weight_list,
            gamma=flowed_gamma,
            return_stats=True,
        )
        target.pointfea, _ = fea_extractor(
            target.points,
            target.weights,
            weight_list=weight_list,
            gamma=target_gamma,
            mean=mean,
            std=std,
        )
        return source, target

    return extract



def default_local_pair_feature_extractor():
    def extract(source, target):
        source.pointfea = (
            source.points.clone()
        )
        target.pointfea = (
            target.points.clone()
        )
        return source, target
    return extract



# def pair_feature_preextracted_extractor():
#     def extract(flowed, target):
#         return flowed, target
#     return extract


def FPFH_extractor(radius_normal, radius_feature):
    """ call the FPFH extractor of open3d"""
    from probreg import features

    fn = features.FPFH(radius_normal, radius_feature)

    def extract(input):
        points = input.points
        device = points.device
        B = points.shape[0]
        points = points.detach().cpu().numpy()
        fea_list = []
        for b in range(B):
            fea_list.append(fn(points[b]).astype(np.float32))
        input_fea = torch.tensor(np.stack(fea_list)).to(device)
        input.pointfea = input_fea
        return input

    return extract


def pair_feature_FPFH_extractor(radius_normal=0.1, radius_feature=0.5):
    fn = FPFH_extractor(radius_normal, radius_feature)

    def extract(flowed, target, iter=None, flowed_gamma=None, target_gamma=None):
        flowed = fn(flowed)
        target = fn(target)
        return flowed, target

    return extract


def get_Gamma(
    sigma_scale,
    principle_weight=None,
    eigenvector=None,
):
    """

    :param sigma_scale: scale the sigma
    :param principle_weight: BxNxD
    :param eigenvector: torch.Tensor, BXNxDxD, denote as U
    :return: Gamma: torch.Tensor, BxNxDxD  U{\Lambda}^{-2}U^T
    """
    principle_weight = principle_weight * sigma_scale
    principle_weight_inv_sq = 1 / (principle_weight ** 2)
    principle_diag = torch.diag_embed(principle_weight_inv_sq)
    Gamma = eigenvector @ principle_diag @ (eigenvector.permute(0, 1, 3, 2))
    return Gamma, principle_weight


def compute_anisotropic_gamma_from_points(
    points,
    weights=None,
    cov_sigma_scale=0.05,
    aniso_kernel_scale=None,
    principle_weight=None,
    eigenvalue_min=0.1,
    iter_twice=False,
    leaf_decay=False,
    mass_thres=2.5,
    return_details=False,
):
    """
    compute inverse covariance matrix for anisotropic kernel
    this function doesn't support auto-grad
    :param input_shape: torch.Tensor, BxNxD
    :param cov_sigma_scale: float, the sigma used for computing the local covariance matrix for eigenvalue, eigenvector extraction
    :param aniso_kernel_scale: float,  anisotropic kernel scale
    :param principle_weight: list of size D, weight of directions in anistropic kernel, don't have to be norm to 1, will normalized later, if not given, use the eigenvalue instead
    :param eigenvalue_min: float, if the principal vector is not given, then the norm2 normalized eigenvalue will be used for compute the weight of each principle direction,
     this value is to control the weight of the eigenvector, to avoid extreme narraw direction (typically happens when eigenvalue close to zero)
    :return: Gamma, torch.Tensor, BxNxDxD  U{\Lambda}^{-2}U^T,  where U is the eigenvector of the local covariance matrix
    """
    aniso_kernel_scale = (
        aniso_kernel_scale if aniso_kernel_scale is not None else cov_sigma_scale
    )
    B, N, D, device = points.shape[0], points.shape[1], points.shape[2], points.device
    fea_type_list = ["eigenvalue", "eigenvector"]
    fea_extractor = feature_extractor(
        fea_type_list, radius=cov_sigma_scale, std_normalize=False, include_pos=False
    )
    combined_fea, mass = fea_extractor(points, weights)
    eigenvalue, eigenvector = combined_fea[:, :, :D], combined_fea[:, :, D:]
    eigenvector = eigenvector.view(eigenvector.shape[0], eigenvector.shape[1], D, D)
    # print("detect there is {} eigenvalue smaller or equal to 0, set to 1e-7".format(torch.sum(eigenvalue <= 0)))
    use_predefined_principle_weight = principle_weight is not None
    if not use_predefined_principle_weight:
        eigenvalue[eigenvalue <= 0.0] = 1e-9
        eigenvalue = eigenvalue / torch.norm(eigenvalue, p=2, dim=2, keepdim=True)
        eigenvalue[eigenvalue < eigenvalue_min] = eigenvalue_min
        principle_weight = (
            eigenvalue / (torch.norm(eigenvalue, p=2, dim=2, keepdim=True)) * (D ** 0.5)
        )
    else:
        principle_weight = np.array(principle_weight) / np.linalg.norm(principle_weight)
        principle_weight = principle_weight.astype(np.float32)
        principle_weight = torch.from_numpy(principle_weight).to(device).repeat(
            B, N, 1
        ) * (D ** 0.5)
    Gamma, principle_weight_ouput = get_Gamma(
        aniso_kernel_scale, principle_weight=principle_weight, eigenvector=eigenvector
    )
    if iter_twice:
        mass, dev, cov = compute_aniso_local_moments(points, Gamma)
        eigenvector = compute_local_fea_from_moments(
            "eigenvector", weights, mass, dev, cov
        )
        eigenvalue = compute_local_fea_from_moments(
            "eigenvalue", weights, mass, dev, cov
        )
        eigenvector = eigenvector.view(eigenvector.shape[0], eigenvector.shape[1], D, D)
        if not use_predefined_principle_weight:
            eigenvalue[eigenvalue <= 0.0] = 1e-9
            eigenvalue = eigenvalue / (
                torch.norm(eigenvalue, p=2, dim=2, keepdim=True) + 1e-9
            )
            eigenvalue[eigenvalue < eigenvalue_min] = eigenvalue_min
            principle_weight = (
                eigenvalue
                / (torch.norm(eigenvalue, p=2, dim=2, keepdim=True))
                * (D ** 0.5)
            )
        Gamma, principle_weight_ouput = get_Gamma(
            aniso_kernel_scale,
            principle_weight=principle_weight,
            eigenvector=eigenvector,
        )
    if leaf_decay:
        mass_thres = mass_thres
        decay_factor = 3
        mass_falttern = mass.view(-1)
        gamma_mask = torch.ones_like(mass_falttern)
        principle_mask = torch.ones_like(mass_falttern)
        low_index = mass_falttern < mass_thres
        gamma_mask[low_index] = decay_factor ** 2
        principle_mask[low_index] = 1 / decay_factor
        Gamma = Gamma.view(-1, Gamma.shape[2], Gamma.shape[3]) * gamma_mask.view(
            -1, 1, 1
        )
        principle_weight_ouput = principle_weight_ouput.view(
            -1, D
        ) * principle_mask.view(-1, 1)
    Gamma = Gamma.view(B, N, D, D)
    principle_weight_ouput = principle_weight_ouput.view(B, N, D)

    if not return_details:
        return Gamma  # , principle_weight
    else:
        return Gamma, principle_weight_ouput, eigenvector, mass
