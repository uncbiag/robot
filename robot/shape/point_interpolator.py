import torch
from pykeops.torch import LazyTensor
from robot.utils.local_feature_extractor import compute_anisotropic_gamma_from_points
from robot.utils.obj_factory import obj_factory
from robot.utils.knn_utils import NN, KNN


def compute_nadwat_kernel(scale=0.1, exp_order=2, iso=True, self_center=False):
    """
    Nadaraya-Watson kernel interpolation

    :param scale: kernel width of isotropic kernel, disabled if the iso is False
    :param exp_order: float, 1,2,0.5
    :param iso: bool, use isotropic kernel, sigma equals to scale
    """
    # todo write plot-test for this function

    assert exp_order in [1, 2, 0.5]

    def interp(points, control_points, control_weights, gamma=None):
        """

        :param points: BxNxD Tensor
        :param control_points: BxMxD Tensor
        :param control_weights: BxMx1 Tensor
        :param gamma: optional BxMxDxD Tensor, anisotropic inverse kernel
        :return: BxNxMxd Tensor
        """

        if iso:
            points, control_points = points / scale, control_points / scale
            points_i = LazyTensor(points[:, :, None, :])  # (B,N, 1, D)  "column"
            control_points_j = LazyTensor(
                control_points[:, None, :, :]
            )  # (B,1, M, D)  "line"
            dist2 = ((points_i - control_points_j) ** 2).sum(
                -1
            )  # (N, M) squared distances (B,N,M,1)
        else:
            # assert scale ==1, "debugging, make sure this is only triggered when you are playing on multi-scale anisotropic"
            if scale != 1:
                gamma = gamma * (1 / scale ** 2)
            points_i = LazyTensor(points[:, :, None, :])  # (B,N, 1, D)  "column"
            control_points_j = LazyTensor(
                control_points[:, None, :, :]
            )  # (B,1, M, D)  "line"
            if not self_center:
                gamma = LazyTensor(
                    gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, None]
                )  # Bx1xMxD*D
            else:
                gamma = LazyTensor(
                    gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, :, None]
                )  # BxNx1xD*D
            dist2 = (points_i - control_points_j) | gamma.matvecmult(
                points_i - control_points_j
            )

        if exp_order == 1:
            C_ij = dist2.sqrt()  # + D_ij
        elif exp_order == 2:
            C_ij = dist2 / 2
        elif exp_order == 1 / 2:
            C_ij = 2 * dist2.sqrt().sqrt()

        # For the sake of numerical stability, we perform the weight normalization
        # in the log-domain --------------------------------------------------------
        logw_j = LazyTensor(control_weights.log()[:, None, :, :])  # (B,1,M,1)

        scores = -(logw_j - C_ij).logsumexp(dim=2)  # (B,N, 1)
        scores_i = LazyTensor(scores[:, :, None, :])  # (B,N, 1, 1)

        value_weight_j = LazyTensor((control_weights)[:, None, :, :])  # (B,1, M, D)

        kernel = (scores_i - C_ij).exp() * value_weight_j  # LazyTenosr  (B,N,M,D)
        return kernel

    return interp


def nadwat_kernel_interpolator(
    scale=0.1, weight=1.0, exp_order=2, iso=True, self_center=False
):
    """
    Nadaraya-Watson kernel interpolation

    :param scale: kernel width of isotropic kernel, if iso, scale can be viewed as sigma,
     if aniso, scale is use to scale the gamma, the equvialient kernel size is aniso_kernel_scale*scale
    :param exp_order: float, 1,2,0.5
    :param iso: bool, use isotropic kernel, sigma equals to scale
    """
    # todo write plot-test on this function
    multi_scale = isinstance(scale, list)
    if not multi_scale:
        compute_kernel = compute_nadwat_kernel(
            scale=scale, exp_order=exp_order, iso=iso, self_center=self_center
        )
    else:
        assert sum(weight) == 1, "sum of weight should be 1"
        assert len(weight) == len(
            scale
        ), "weight and scale list should be of the same length"
        compute_kernel_list = [
            compute_nadwat_kernel(
                scale=_scale, exp_order=exp_order, iso=iso, self_center=self_center
            )
            for _scale in scale
        ]

    def interp(points, control_points, control_value, control_weights, gamma=None):
        points = points.contiguous()
        control_points = control_points.contiguous()
        control_value = control_value.contiguous()
        control_weights = control_weights.contiguous()

        if not multi_scale:
            kernel = compute_kernel(
                points, control_points, control_weights, gamma=gamma
            )
        else:
            kernel = sum(
                [
                    _weight
                    * _compute_kernel(
                        points, control_points, control_weights, gamma=gamma
                    )
                    for _weight, _compute_kernel in zip(weight, compute_kernel_list)
                ]
            )
        value_weight_j = LazyTensor((control_value)[:, None, :, :])  # (B,1, M, D)
        points_value = (kernel * value_weight_j).sum(dim=2)  # BxNxd
        return points_value

    return interp


def _spline_intepolator(scale=0.1, kernel="gauss", iso=True):
    """
    Performs a ridge kernel regression.

    :param scale: kernel width
    :param kernel: kernel_type, "TPS", "cauchy", "gauss"
    :return:
    """

    # todo write plot-test on this function

    def kernel_matrix(x, y, gamma=None):
        """Implement your favorite SPD kernel matrix here."""
        if kernel != "TPS" and iso:
            x, y = x / scale, y / scale

        if iso:
            x_i = LazyTensor(x[:, None, :])  # (N, 1, D)  "column"
            y_j = LazyTensor(y[None, :, :])  # (1, N, D)  "line"
            D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, N) squared distances
        else:
            x_i = LazyTensor(x[:, None, :])  # (N, 1, D)  "column"
            y_j = LazyTensor(y[None, :, :])  # (1, N, D)  "line"
            gamma = LazyTensor(gamma.view(gamma.shape[0], -1)[None])  # 1xNxD*D
            D_ij = (x_i - y_j) | gamma.matvecmult(x_i - y_j)  # (N, N) squared distances

        if kernel == "TPS":  # Thin plate spline in 3D
            K_ij = -D_ij.sqrt()
        elif kernel == "cauchy":
            K_ij = 1 / (1 + D_ij)
        else:  # Gaussian kernel
            K_ij = (-D_ij / 2).exp()  # (N, N)  kernel matrix

        return K_ij  # (N, N) kernel matrix

    def interp(points, control_points, control_weights, control_value, gamma=None):
        """

        :param points: NxD Tensor
        :param control_points: MxD Tensor
        :param control_value: Mxd Tensor
        :param control_weights: Mx1 Tensor
        :return: Nxd Tensor
        """
        # Sinv(y) = (Id + diag(w) * K_xx)^-1 @ y
        assert iso == True, "aniso is not fully supported yet"
        K_xx = kernel_matrix(control_points, control_points)
        wK_xx = LazyTensor(control_weights[:, :, None]) * K_xx
        Sinv = lambda y: wK_xx.solve(y, alpha=1)
        momentum = Sinv(control_value * control_weights)
        # Apply the spline deformation on the full point cloud:
        K_px = kernel_matrix(points, control_points)
        return K_px @ momentum

    return interp


def ridge_kernel_intepolator(scale=0.1, kernel="gauss"):
    """
    Performs a ridge kernel regression,

    :param scale: kernel width
    :param kernel: kernel_type, "TPS", "cauchy", "gauss"
    :return:
    """
    # todo rewrite in batch form
    _interp = _spline_intepolator(scale=scale, kernel=kernel)

    def interp(points, control_points, control_weights, control_value, gamma=None):
        interpolated_list = []
        gamma = gamma if gamma is not None else [None] * points.shape[0]
        for _points, _control_points, _control_weights, _control_value, _gamma in zip(
            points, control_points, control_weights, control_value, gamma
        ):
            output = _interp(
                _points, _control_points, _control_weights, _control_value, _gamma
            )
            interpolated_list.append(output)
        return torch.stack(interpolated_list, 1)

    return interp


def nadwat_interpolator_with_aniso_kernel_extractor_embedded(
    exp_order=2,
    cov_sigma_scale=0.05,
    aniso_kernel_scale=0.05,
    principle_weight=None,
    eigenvalue_min=0.1,
    iter_twice=False,
    self_center=False,
    leaf_decay=False,
    mass_thres=2.5,
):
    interp = nadwat_kernel_interpolator(
        scale=1.0, exp_order=exp_order, iso=False, self_center=self_center
    )

    def compute(points, control_points, control_value, control_weights, gamma=None):
        Gamma_control_points = gamma
        if Gamma_control_points is None:
            Gamma_control_points = compute_anisotropic_gamma_from_points(
                points,
                cov_sigma_scale=cov_sigma_scale,
                aniso_kernel_scale=aniso_kernel_scale,
                principle_weight=principle_weight,
                eigenvalue_min=eigenvalue_min,
                iter_twice=iter_twice,
                leaf_decay=leaf_decay,
                mass_thres=mass_thres,
            )

        interp_value = interp(
            points, control_points, control_value, control_weights, Gamma_control_points
        )
        return interp_value

    return compute


class NadWatAnisoSpline(object):
    def __init__(
        self,
        exp_order=2,
        cov_sigma_scale=0.05,
        aniso_kernel_scale=0.05,
        aniso_kernel_weight=1.0,
        principle_weight=None,
        eigenvalue_min=0.1,
        iter_twice=False,
        leaf_decay=False,
        fixed=False,
        is_interp=False,
        mass_thres=2.5,
        self_center=False,
        requires_grad=True,
    ):
        self.exp_order = exp_order
        self.cov_sigma_scale = cov_sigma_scale
        self.aniso_kernel_scale = aniso_kernel_scale
        self.principle_weight = principle_weight
        self.eigenvalue_min = eigenvalue_min
        self.iter_twice = iter_twice
        self.mass_thres = mass_thres
        self.leaf_decay = leaf_decay
        self.self_center = self_center
        self.relative_scale = 1.0
        self.aniso_kernel_weight = aniso_kernel_weight
        if isinstance(aniso_kernel_scale, list):
            self.relative_scale = [
                scale / aniso_kernel_scale[0] for scale in aniso_kernel_scale
            ]
            self.aniso_kernel_scale = aniso_kernel_scale[0]
        self.spline = nadwat_kernel_interpolator(
            scale=self.relative_scale,
            weight=aniso_kernel_weight,
            exp_order=exp_order,
            iso=False,
            self_center=self_center,
        )
        self.fixed = fixed
        self.is_interp = is_interp
        self.iter = 0
        self.requires_grad = requires_grad

    def initialize(self, points, weights=None):
        self.Gamma = compute_anisotropic_gamma_from_points(
            points,
            cov_sigma_scale=self.cov_sigma_scale,
            aniso_kernel_scale=self.aniso_kernel_scale,
            principle_weight=self.principle_weight,
            eigenvalue_min=self.eigenvalue_min,
            iter_twice=self.iter_twice,
            leaf_decay=self.leaf_decay,
            mass_thres=self.mass_thres,
        )
        self.Gamma = self.Gamma if self.requires_grad else self.Gamma.detach()
        if self.fixed and self.iter == 0:
            if not isinstance(self.relative_scale, list):
                compute_kernel = compute_nadwat_kernel(
                    scale=1.0,
                    exp_order=self.exp_order,
                    iso=False,
                    self_center=self.self_center,
                )
                self.kernel = compute_kernel(points, points, weights, gamma=self.Gamma)
            else:
                compute_kernel_list = [
                    compute_nadwat_kernel(
                        scale=_scale,
                        exp_order=self.exp_order,
                        iso=False,
                        self_center=self.self_center,
                    )
                    for _scale in self.relative_scale
                ]
                self.kernel = sum(
                    [
                        _weight
                        * _compute_kernel(points, points, weights, gamma=self.Gamma)
                        for _weight, _compute_kernel in zip(
                            self.aniso_kernel_weight, compute_kernel_list
                        )
                    ]
                )

        return self

    def set_interp(self, is_interp=True):
        self.is_interp = is_interp
        self.iter = -1 if self.is_interp else 0  # to avoid save the kernel

    def set_flow(self, is_interp=True):
        self.is_interp = is_interp

    def reset_kernel(self):
        self.iter = 0

    def get_buffer(self):
        return {"normalized_Gamma": self.Gamma * (self.aniso_kernel_scale ** 2)}

    def __call__(self, points, control_points, control_value, control_weights):
        if not self.fixed or self.is_interp:
            kernel_points = control_points if not self.self_center else points
            self.initialize(kernel_points)
            Gamma = self.Gamma
            spline_value = self.spline(
                points, control_points, control_value, control_weights, Gamma
            )
        else:
            if self.iter == 0:
                self.initialize(
                    control_points, control_weights
                )  # here control points are the same as the points, so self.self_center doesn't affect
            value_weight_j = LazyTensor((control_value)[:, None, :, :])  # (B,1, M, D)
            spline_value = (self.kernel * value_weight_j).sum(dim=2)
        self.iter += 1
        return spline_value


class NadWatIsoSpline(object):
    def __init__(self, exp_order=2, kernel_scale=0.05, kernel_weight=1.0):
        self.exp_order = exp_order
        self.kernel_scale = kernel_scale
        self.spline = nadwat_kernel_interpolator(
            scale=kernel_scale, weight=kernel_weight, exp_order=exp_order, iso=True
        )
        self.is_interp = False
        self.iter = 0

    def set_interp(self, is_interp=True):
        pass

    def set_flow(self, is_interp=True):
        pass

    def reset_kernel(self):
        pass

    def get_buffer(self):
        return {"normalized_Gamma": None}

    def __call__(self, points, control_points, control_value, control_weights):
        spline_value = self.spline(
            points, control_points, control_value, control_weights, None
        )
        return spline_value


class KNNInterpolater(object):
    def __init__(self, initial_radius=1, use_aniso=False, aniso_knn_obj=None):
        super(KNNInterpolater, self).__init__()
        self.initial_radius = initial_radius
        if use_aniso:
            self.knn = obj_factory(aniso_knn_obj)
        else:
            self.knn = KNN()

    def __call__(self, pc1, pc2, pc2_fea, resol_factor=1, K=9):
        from robot.modules_reg.networks.pointconv_util import index_points_group

        sigma = self.initial_radius * resol_factor
        K_dist, index = self.knn(pc1 / sigma, pc2 / sigma, K)
        grouped_pc2_fea = index_points_group(pc2_fea, index)
        K_w = torch.nn.functional.softmax(-K_dist, dim=2)
        pc1_interp_fea = torch.sum(K_w[..., None] * grouped_pc2_fea, dim=2)
        return pc1_interp_fea




class NNInterpolater(object):
    def __init__(self):
        super(NNInterpolater,self).__init__()
        self.nn = NN(return_value=False)

    def __call__(self,pc1, pc2, pc2_fea):
        from robot.modules_reg.networks.pointconv_util import index_points_group
        index = self.nn(pc1, pc2)
        pc1_interp_fea =index_points_group(pc2_fea,index)[:,:,0]
        return pc1_interp_fea
#
# if __name__ == "__main__":
#     a=torch.rand(1,10000,3).cuda()
#     b=torch.rand(1,10000,3).cuda()
#     d = torch.rand(1,10000,200).cuda()
