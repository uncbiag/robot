import torch


class TorchKernel(object):
    """
    Torch Kernel,  support batch
    """

    def __init__(self, kernel_type="gauss", **kernel_args):
        assert kernel_type in [
            "gauss",
            "multi_gauss",
            "gauss_lin",
            "gauss_grad",
            "multi_gauss_grad",
            "gauss_lin",
        ]
        self.kernel_type = kernel_type
        self.kernels = {
            "gauss": self.gauss_kernel,
            "multi_gauss": self.multi_gauss_kernel,
            "gauss_grad": self.gaussian_gradient,
            "multi_gauss_grad": self.multi_gaussian_gradient,
            "gauss_lin": self.gauss_lin_kernel,
        }
        self.kernel = self.kernels[self.kernel_type](**kernel_args)

    @staticmethod
    def gauss_kernel(sigma=0.1):
        """
        :param sigma: scalar
        :return:
        """
        sig2 = sigma * (2 ** (1 / 2))

        def reduce(x, y, b):
            """

            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxKxD input position2
            :param b: torch.Tensor, BxKxd, input val
            :return: torch.Tensor, BxNxd, output
            """
            x = x[:, :, None]
            y = y[:, None]
            b = b[:, None]  # Bx1xKxd
            dist2 = ((x / sig2 - y / sig2) ** 2).sum(-1, keepdim=True)  # BxNxKx1
            kernel = (-dist2).exp()
            return (kernel * b).sum(axis=2)

        return reduce

    @staticmethod
    def multi_gauss_kernel(sigma_list=None, weight_list=None):
        """
        :param sigma_list: a list of sigma
        :param weight_list: corresponding list of weight, sum(weight_list)=1
        :return:
        """
        gamma_list = [1 / (2 * sigma * sigma) for sigma in sigma_list]

        def reduce(x, y, b):
            """

            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxKxD input position2
            :param b: torch.Tensor, BxKxd, input val
            :return: torch.Tensor, BxNxd, output
            """
            kernel = 0.0
            x = x[:, :, None]
            y = y[:, None]
            b = b[:, None]  # Bx1xKxd
            dist2 = ((x - y) ** 2).sum(-1, keepdim=True)  # BxNxKx1
            for gamma, weight in zip(gamma_list, weight_list):
                kernel += (-dist2 * gamma).exp() * weight
            return (kernel * b).sum(axis=2)

        return reduce

    @staticmethod
    def gaussian_gradient(sigma=0.1):
        def reduce(px, x, py=None, y=None):
            if y is None:
                y = x
            if py is None:
                py = px
            x = x[:, :, None] / sigma
            y = y[:, None] / sigma
            px = px[:, :, None]  # BxNx1xD
            py = py[:, None]  # Bx1xKxD
            dist2 = ((x - y) ** 2).sum(-1, keepdim=True)  # BxNxKx1
            kernel = (-dist2 * 0.5).exp()  # BxNxKx1
            diff_kernel = (x - y) * kernel  # BxNxKxD
            B, N, K, D = diff_kernel.shape
            pyx = (px * py).sum(-1)  # BxNxK
            return (-1 / sigma) * torch.bmm(
                pyx.view(-1, 1, K), diff_kernel.view(-1, K, D)
            ).view(B, N, D)

        return reduce

    @staticmethod
    def multi_gaussian_gradient(sigma_list=None, weight_list=None):
        gamma_list = [1 / (2 * sigma * sigma) for sigma in sigma_list]

        def reduce(px, x, py=None, y=None):
            if y is None:
                y = x
            if py is None:
                py = px
            kernel = 0.0
            x = x[:, :, None]
            y = y[:, None]
            px = px[:, :, None]  # BxNx1xD
            py = py[:, None]  # Bx1xKxD
            dist2 = ((x - y) ** 2).sum(-1, keepdim=True)  # BxNxKx1
            for gamma, weight in zip(gamma_list, weight_list):
                kernel += ((-dist2 * gamma).exp()) * gamma * weight  # BxNxKx1
            diff_kernel = (x - y) * kernel  # BxNxKxD
            B, N, K, D = diff_kernel.shape
            pyx = (px * py).sum(-1)  # BxNxK
            return (-2) * torch.bmm(
                pyx.view(-1, 1, K), diff_kernel.view(-1, K, D)
            ).view(B, N, D)

        return reduce

    @staticmethod
    def gauss_lin_kernel(sigma=0.1):
        """
        :param sigma: scalar
        :return:
        """
        sig2 = sigma * (2 ** (1 / 2))

        def reduce(x, y, u, v, b):
            """
            :param x: torch.Tensor, BxNxD,  input position1
            :param y: torch.Tensor, BxKxD input position2
            :param u: torch.Tensor, BxNxD, input val1
            :param v: torch.Tensor, BxKxD, input val2
            :param b: torch.Tensor, BxKxd, input scalar vector
            :return: torch.Tensor, BxNxd, output
            """
            x = x[:, :, None] / sig2
            y = y[:, None] / sig2
            u = u[:, :, None]
            v = v[:, None]
            b = b[:, None]  # Bx1xKxD
            dist2 = ((x - y) ** 2).sum(-1, keepdim=True)  # BxNxKx1
            kernel = (-dist2).exp() * ((u * v).sum(-1, keepdim=True) ** 2)  # BxNxKx1
            return (kernel * b).sum(axis=2)

        return reduce

    def __call__(self, *data_args):
        return self.kernel(*data_args)
