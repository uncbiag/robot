import torch

from torchvectorized.vlinalg import vSymEig


def _grad_sym(X):
    return 0.5 * (X + X.transpose(1, 2))


class EigValsFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        V, U = vSymEig(X, eigenvectors=True, flatten_output=True)
        ctx.save_for_backward(V, U, X)

        return V

    @staticmethod
    def backward(ctx, *grad_outputs):
        S, U, X = ctx.saved_tensors
        b, c, d, h, w = X.size()

        grad_X = torch.diag_embed(grad_outputs[0])

        return _grad_sym(torch.bmm(torch.bmm(U, grad_X), U.transpose(1, 2))).reshape(b, d * h * w, 3, 3) \
                   .permute(0, 2, 3, 1).reshape(b, c, d, h, w), None


class EigVals(torch.nn.Module):
    """
    Differentiable neural network layer (:class:`torch.nn.Module`) that performs eigendecomposition on
    every voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW** and return the eigenvalues.

    See **Ionescu et al., Matrix backpropagation for deep networks with structured layers, CVPR 2015** for details on the
    gradients computation
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Takes a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW** and return a volume of their eigenvalues

        :param x: A volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**
        :type x: torch.Tensor
        :return: A tensor with shape **(B*D*H*W)x3** where every voxel's channels are the eigenvalues of the inpur matrix
            at the same spatial location.
        :rtype: torch.Tensor
        """
        return EigValsFunc.apply(x)


class LogmFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        b, c, d, h, w = X.size()
        S, U = vSymEig(X, eigenvectors=True, flatten_output=True)

        ctx.save_for_backward(torch.log(S), S, U, X)

        return U.bmm(torch.diag_embed(torch.log(S))).bmm(U.transpose(1, 2)).reshape(
            b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

    @staticmethod
    def backward(ctx, *grad_outputs):
        S_log, S, U, X = ctx.saved_tensors
        b, c, d, h, w = X.size()

        grad_X = grad_outputs[0].reshape(b, 3, 3, d * h * w).permute(0, 3, 1, 2).reshape(b * d * h * w, 3, 3)

        # Backward Log
        inv_S = torch.diag_embed(1 / S)
        grad_U = 2 * _grad_sym(grad_X).bmm(U.bmm(torch.diag_embed(S_log)))
        grad_S = torch.eye(3).cuda() * (inv_S.bmm(U.transpose(1, 2).bmm(_grad_sym(grad_X).bmm(U))))

        S = S.view(1, -1)
        P = S.view(S.size(1) // 3, 3).unsqueeze(2)
        P = P.expand(P.size(0), P.size(1), 3)
        P = P - P.transpose(1, 2)
        mask_zero = torch.abs(P) == 0
        P = 1 / P
        P[mask_zero] = 0

        return U.bmm(_grad_sym(P.transpose(1, 2) * (U.transpose(1, 2).bmm(grad_U))) + grad_S).bmm(
            U.transpose(1, 2)).reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w), None


class Logm(torch.nn.Module):
    """
    Differentiable neural network layer (:class:`torch.nn.Module`) that performs matrix logarithm on
    every voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

    See **Ionescu et al., Matrix backpropagation for deep networks with structured layers, CVPR 2015** for details on the
    gradients computation
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        r"""
        Compute the matrix exponential :math:`\mathbf{M} = \mathbf{U} log(\mathbf{\Sigma}) \mathbf{U}^{\top}` of
        every voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

        :param x: A volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**
        :type x: torch.Tensor
        :return: A volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.
        :rtype: torch.Tensor
        """
        return LogmFunc.apply(x)


class ExpmFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        b, c, d, h, w = X.size()
        S, U = vSymEig(X, eigenvectors=True, flatten_output=True)

        ctx.save_for_backward(S, torch.exp(S), U, X)

        return U.bmm(torch.diag_embed(torch.exp(S))).bmm(U.transpose(1, 2)).reshape(
            b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

    @staticmethod
    def backward(ctx, *grad_outputs):
        S, S_exp, U, X = ctx.saved_tensors
        b, c, d, h, w = X.size()

        grad_X = grad_outputs[0].reshape(b, 3, 3, d * h * w).permute(0, 3, 1, 2).reshape(b * d * h * w, 3, 3)
        grad_U = 2 * _grad_sym(grad_X).bmm(U.bmm(torch.diag_embed(S_exp)))
        grad_S = torch.eye(3).cuda() * torch.diag_embed(S_exp).bmm(U.transpose(1, 2).bmm(_grad_sym(grad_X).bmm(U)))

        S = S.view(1, -1)
        P = S.view(S.size(1) // 3, 3).unsqueeze(2)
        P = P.expand(P.size(0), P.size(1), 3)
        P = P - P.transpose(1, 2)
        mask_zero = torch.abs(P) == 0
        P = 1 / P
        P[mask_zero] = 0

        return U.bmm(_grad_sym(P.transpose(1, 2) * (U.transpose(1, 2).bmm(grad_U))) + grad_S).bmm(
            U.transpose(1, 2)).reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w), None


class Expm(torch.nn.Module):
    """
    Differentiable neural network layer (:class:`torch.nn.Module`) that performs matrix exponential on
    every voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

    See **Ionescu et al., Matrix backpropagation for deep networks with structured layers, CVPR 2015** for details on the
    gradients computation
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        r"""
        Compute the matrix exponential :math:`\mathbf{M} = \mathbf{U} exp(\mathbf{\Sigma}) \mathbf{U}^{\top}` of
        every voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

        :param x: A volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**
        :type x: torch.Tensor
        :return: A volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.
        :rtype: torch.Tensor
        """
        return ExpmFunc.apply(x)


class ExpmLogmFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        b, c, d, h, w = X.size()
        S_log, U = vSymEig(X, eigenvectors=True, flatten_output=True)

        ctx.save_for_backward(S_log, torch.exp(S_log), U, X)

        return U.bmm(torch.diag_embed(S_log)).bmm(U.transpose(1, 2)).reshape(
            b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

    @staticmethod
    def backward(ctx, *grad_outputs):
        S_log, S_exp, U, X = ctx.saved_tensors
        b, c, d, h, w = X.size()

        grad_X = grad_outputs[0].reshape(b, 3, 3, d * h * w).permute(0, 3, 1, 2).reshape(b * d * h * w, 3, 3)

        # Backward Log
        inv_S = torch.diag_embed(1 / S_exp)
        grad_U = 2 * _grad_sym(grad_X).bmm(U.bmm(torch.diag_embed(S_log)))
        grad_S = torch.eye(3).cuda() * (inv_S.bmm(U.transpose(1, 2).bmm(_grad_sym(grad_X).bmm(U))))

        S = S_exp.view(1, -1)
        P = S.view(S.size(1) // 3, 3).unsqueeze(2)
        P = P.expand(P.size(0), P.size(1), 3)
        P = P - P.transpose(1, 2)
        mask_zero = torch.abs(P) == 0
        P = 1 / P
        P[mask_zero] = 0

        grad_X = U.bmm(_grad_sym(P.transpose(1, 2) * (U.transpose(1, 2).bmm(grad_U))) + grad_S).bmm(U.transpose(1, 2))

        # Backward Exp
        grad_U = 2 * _grad_sym(grad_X).bmm(U.bmm(torch.diag_embed(S_exp)))
        grad_S = torch.eye(3).cuda() * torch.diag_embed(S_exp).bmm(U.transpose(1, 2).bmm(_grad_sym(grad_X).bmm(U)))

        S = S_log.view(1, -1)
        P = S.view(S.size(1) // 3, 3).unsqueeze(2)
        P = P.expand(P.size(0), P.size(1), 3)
        P = P - P.transpose(1, 2)
        mask_zero = torch.abs(P) == 0
        P = 1 / P
        P[mask_zero] = 0

        return U.bmm(_grad_sym(P.transpose(1, 2) * (U.transpose(1, 2).bmm(grad_U))) + grad_S).bmm(
            U.transpose(1, 2)).reshape(
            b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w), None


class ExpmLogm(torch.nn.Module):
    """
    Differentiable neural network layer (:class:`torch.nn.Module`) that performs consecutive matrix exponential
    and logarithm on every voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

    See **Ionescu et al., Matrix backpropagation for deep networks with structured layers, CVPR 2015** for details on the
    gradients computation
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        r"""
        Compute the matrix exponential :math:`\mathbf{M} = \mathbf{U} exp(\mathbf{\Sigma}) \mathbf{U}^{\top}` and
        the matrix logarithm :math:`\mathbf{M} = \mathbf{U} log(\mathbf{\Sigma}) \mathbf{U}^{\top}` of every voxel
        in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

        :param x: A volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**
        :type x: torch.Tensor
        :return: A volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.
        :rtype: torch.Tensor
        """
        return ExpmLogmFunc.apply(x)
