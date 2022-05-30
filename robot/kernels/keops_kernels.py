import numpy as np
import torch
from pykeops.torch import Vi, Vj, Pm, LazyTensor

##################  Lazy Tensor  #######################


class LazyKeopsKernel(object):
    """
    LazyTensor formulaton in Keops,  support batch
    """

    def __init__(self, kernel_type="gauss", **kernel_args):
        assert kernel_type in [
            "gauss",
            "multi_gauss",
            "normalized_gauss",
            "normalized_multi_gauss",
            "gauss_lin",
            "gauss_grad",
            "multi_gauss_grad",
            "gauss_lin",
            "aniso_gauss",
            "aniso_multi_gauss",
        ]
        self.kernel_type = kernel_type
        self.kernels = {
            "gauss": self.gauss_kernel,
            "normalized_gauss": self.normalized_gauss_kernel,
            "multi_gauss": self.multi_gauss_kernel,
            "normalized_multi_gauss": self.normalized_multi_gauss_kernel,
            "gauss_grad": self.gaussian_gradient,
            "multi_gauss_grad": self.multi_gaussian_gradient,
            "gauss_lin": self.gauss_lin_kernel,
            "aniso_gauss": self.aniso_gauss_kernel,
            "aniso_multi_gauss": self.aniso_multi_gauss_kernel,
        }
        self.kernel = self.kernels[self.kernel_type](**kernel_args)

    @staticmethod
    def gauss_kernel(sigma=0.1):
        """
        :param sigma: scalar
        :return:
        """
        sig2 = sigma * (2 ** (1 / 2))

        def conv(x, y, b):
            """

            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxMxD input position2
            :param b: torch.Tensor, BxMxd, input val
            :return:torch.Tensor, BxNxd, output
            """
            x = LazyTensor(x[:, :, None] / sig2)  # BxNx1xD
            y = LazyTensor(y[:, None] / sig2)  # Bx1xMxD
            b = LazyTensor(b[:, None])  # Bx1xMxd
            dist2 = x.sqdist(y)
            kernel = (-dist2).exp()  # BxNxM
            return (kernel * b).sum_reduction(axis=2)

        return conv

    @staticmethod
    def normalized_gauss_kernel(sigma=0.1):
        """
        :param sigma: scalar
        :return:
        """

        def conv(x, y, b):
            """

            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxMxD input position2
            :param b: torch.Tensor, BxMxd, input val
            :return:torch.Tensor, BxNxd, output
            """
            sig2 = sigma * (2 ** (1 / 2))
            x = LazyTensor(x[:, :, None] / sig2)  # BxNx1xD
            y = LazyTensor(y[:, None] / sig2)  # Bx1xMxD
            b = LazyTensor(b[:, None])  # Bx1xMxd
            dist2 = -x.sqdist(y)
            return dist2.sumsoftmaxweight(b, axis=2)

        return conv

    @staticmethod
    def multi_gauss_kernel(sigma_list=None, weight_list=None):
        """
        :param sigma_list: a list of sigma
        :param weight_list: corresponding list of weight, sum(weight_list)=1
        :return:
        """
        log_weight_list = [float(np.log(weight)) for weight in weight_list]
        gamma_list = [1 / (2 * sigma * sigma) for sigma in sigma_list]

        # def conv(x, y, b):
        #     """
        #
        #     :param x: torch.Tensor, BxNxD,  input position1
        #     :param y: torch.Tensor, BxMxD input position2
        #     :param b: torch.Tensor, BxMxd, input val
        #     :return:torch.Tensor, BxNxd, output
        #     """
        #
        #     kernel = 0
        #     x = LazyTensor(x[:, :, None])
        #     y = LazyTensor(y[:, None])
        #     b = LazyTensor(b[:, None])
        #     for sigma, log_w in zip(sigma_list, log_weight_list):
        #         kernel += (log_w-(x/sigma).sqdist(y/sigma)).exp()
        #     return (kernel * b).sum_reduction(axis=2)

        def conv(x, y, b):
            """

            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxMxD input position2
            :param b: torch.Tensor, BxMxd, input val
            :return:torch.Tensor, BxNxd, output
            """
            B, device = x.shape[0], x.device
            K = len(sigma_list)
            gammas = torch.tensor(gamma_list, device=device).view(1, 1, 1, K)
            log_ws = LazyTensor(
                torch.tensor(log_weight_list, device=device).view(1, 1, 1, K)
            )
            x = LazyTensor(x[:, :, None])
            y = LazyTensor(y[:, None])
            b = LazyTensor(b[:, None])
            dist2 = x.sqdist(y)
            dist2 = dist2 * gammas
            kernel = (log_ws - dist2).exp().sum(3)
            return (kernel * b).sum_reduction(axis=2)

        return conv

    @staticmethod
    def normalized_multi_gauss_kernel(sigma_list=None, weight_list=None):
        """
        :param sigma_list: a list of sigma
        :param weight_list: corresponding list of weight, sum(weight_list)=1
        :return:
        """

        def conv(x, y, b):
            """

            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxMxD input position2
            :param b: torch.Tensor, BxMxd, input val
            :return:torch.Tensor, BxNxd, output
            """
            x = LazyTensor(x[:, :, None])  # BxNx1xD
            y = LazyTensor(y[:, None])  # Bx1xMxD
            b = LazyTensor(b[:, None])  # Bx1xMxd
            res = 0
            D = x.shape[-1]

            for sigma, weight in zip(sigma_list, weight_list):
                sig2 = sigma * (2 ** (1 / D))
                dist2 = -(x / sig2).sqdist(y / sig2)
                res += weight * dist2.sumsoftmaxweight(b, axis=2)
            return res

        return conv

    @staticmethod
    def gaussian_gradient(sigma=0.1):
        def conv(px, x, py=None, y=None):
            """
            :param px: torch.Tensor, BxNxD,  input position1
             :param y: torch.Tensor, BxMxD, input val1
            :param py: torch.Tensor, BxNxD input position2
            :param y: torch.Tensor, BxMxD, input val2
            :return: torch.Tensor, BxNxD, output
            """
            if y is None:
                y = x
            if py is None:
                py = px
            x = LazyTensor(x[:, :, None] / sigma)  # BxNx1xD
            y = LazyTensor(y[:, None] / sigma)  # Bx1xMxD
            px = LazyTensor(px[:, :, None])  # BxNx1xD
            py = LazyTensor(py[:, None, :])  # Bx1xMxD
            dist2 = x.sqdist(y)  # BxNxM
            kernel = (-dist2 * 0.5).exp()
            diff_kernel = (x - y) * kernel  # BxNxMxD
            pyx = (py * px).sum()  # BxNxM
            return (-1 / sigma) * (diff_kernel * pyx).sum_reduction(axis=2)

        return conv

    @staticmethod
    def multi_gaussian_gradient(sigma_list=None, weight_list=None):
        """
        :param sigma_list: a list of sigma
        :param weight_list: corresponding list of weight, sum(weight_list)=1
        :return:
        """
        gamma_list = [1 / (2 * sigma * sigma) for sigma in sigma_list]

        def conv(px, x, py=None, y=None):
            """
            :param px: torch.Tensor, BxNxD,  input position1
            :param x: torch.Tensor, BxNxD input position2
            :param y: torch.Tensor, BxMxD, input val1
            :param py: torch.Tensor, BxMxD, input val2
            :return: torch.Tensor, BxNxD, output
            """
            if y is None:
                y = x
            if py is None:
                py = px
            kernel = 0.0
            x = LazyTensor(x[:, :, None])  # BxNx1xD
            y = LazyTensor(y[:, None])  # Bx1xMxD
            px = LazyTensor(px[:, :, None])  # BxNx1xD
            py = LazyTensor(py[:, None, :])  # Bx1xMxD
            dist2 = x.sqdist(y)  # BxNxM
            for gamma, weight in zip(gamma_list, weight_list):
                kernel += ((-dist2 * gamma).exp()) * gamma * weight  # BxNxMx1
            diff_kernel = (x - y) * kernel  # BxNxMxD
            pyx = (py * px).sum(-1)  # BxNxM
            return (-2) * (diff_kernel * pyx).sum_reduction(axis=2)

        return conv

    @staticmethod
    def gauss_lin_kernel(sigma=0.1):
        """
        :param sigma: scalar
        :return:
        """
        sig2 = sigma * (2 ** (1 / 2))

        def conv(x, y, u, v, b):
            """
            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxMxD input position2
            :param u: torch.Tensor, BxNxD, input val1
            :param v: torch.Tensor, BxMxD, input val2
            :param b: torch.Tensor, BxMxd, input scalar vector
            :return: torch.Tensor, BxNxd, output
            """
            x = LazyTensor(x[:, :, None] / sig2)
            y = LazyTensor(y[:, None] / sig2)
            u = LazyTensor(u[:, :, None])
            v = LazyTensor(v[:, None])
            b = LazyTensor(b[:, None])  # Bx1xMxd
            dist2 = x.sqdist(y)
            kernel = (-dist2).exp() * ((u | v).square())  # BxNxMx1
            return (kernel * b).sum_reduction(axis=2)

        return conv

    def __call__(self, *data_args):
        return self.kernel(*data_args)

    @staticmethod
    def aniso_gauss_kernel(self_center=False):
        """
        anisotropic rbf interpolation kernel,
        """
        assert False, "not test yet"

        def conv(x, y, b, gamma):
            """
            :param x: torch.Tensor, BxNxD,
            :param y: torch.Tensor, BxMxD,
            :param gamma: BxMxDxD  if not self_center  else BxNxDxD
            :param b: torch.Tensor, BxMxd, input val
            :return: torch.Tensor, BxNxd, output
            """

            sqrt2 = 2 ** (1 / 2)
            x_i = LazyTensor(x[:, None, :] / sqrt2)  # (N, 1, D)  "column"
            y_j = LazyTensor(y[None, :, :] / sqrt2)  # (1, M, D)  "line"
            if not self_center:
                gamma = LazyTensor(
                    gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, None]
                )  # Bx1xMxD*D
            else:
                gamma = LazyTensor(
                    gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, :, None]
                )  # BxNx1xD*D
            D_ij = (x_i - y_j) | gamma.matvecmult(x_i - y_j)  # (N, M) squared distances
            K_ij = (-D_ij).exp()  # (N, M)  kernel matrix
            b_j = LazyTensor((b)[:, None, :, :])  # (B,1, M, D)
            return (K_ij * b_j).sum(dim=2)  # BxNxd

        return conv

    @staticmethod
    def aniso_multi_gauss_kernel(relative_scale_list, weight_list, self_center=False):
        """
        anisotropic rbf interpolation kernel,
        """
        assert False, "not test yet"

        def conv(x, y, b, gamma):
            """
            :param x: torch.Tensor, BxNxD,
            :param y: torch.Tensor, BxMxD,
            :param gamma: BxMxDxD  if not self_center  else BxNxDxD
            :param b: torch.Tensor, BxMxd, input val
            :return: torch.Tensor, BxNxd, output
            """
            kernel = 0
            sqrt2 = 2 ** (1 / 2)
            x_i = LazyTensor(x[:, None, :] / sqrt2)  # (N, 1, D)  "column"
            y_j = LazyTensor(y[None, :, :] / sqrt2)  # (1, M, D)  "line"
            b_j = LazyTensor((b)[:, None, :, :])  # (B,1, M, D

            for rscale, weight in zip(relative_scale_list, weight_list):
                gamma = gamma * (1 / rscale ** 2)
                if not self_center:
                    gamma = LazyTensor(
                        gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, None]
                    )  # Bx1xMxD*D
                else:
                    gamma = LazyTensor(
                        gamma.view(gamma.shape[0], gamma.shape[1], -1)[:, :, None]
                    )  # BxNx1xD*D
                D_ij = (x_i - y_j) | gamma.matvecmult(
                    x_i - y_j
                )  # (N, M) squared distances
                K_ij = (-D_ij).exp()  # (N, M)  kernel matrix
                kernel += weight * K_ij
            return (kernel * b_j).sum(dim=2)

        return conv


if __name__ == "__main__":
    from robot.kernels.keops_kernels import *

    batch_sz = 2
    gamma = torch.rand(3, 3).repeat(batch_sz, 1500, 1, 1)
    x = torch.rand(batch_sz, 1500, 3)
    b = torch.rand(batch_sz, 1500, 2)
    kernel1, kernel2 = LazyKeopsKernel.multi_gauss_kernel(
        sigma_list=[0.1, 0.2, 0.3, 0.4, 0.5], weight_list=[0.1, 0.2, 0.2, 0.2, 0.3]
    )
    z1 = kernel1(x, x, b)
    z2 = kernel2(x, x, b)
    print()
