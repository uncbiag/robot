import torch

EPSILON = 1e-15


def overload_diag(inputs: torch.Tensor):
    """
    Add an EPSILON to the diagonal of every 3x3 matrix represented by the 9 channels of an input of shape **Bx9xDxHxW**
    to improve numerical stability

    :param inputs: The input tensor of shape **Bx9xDxHxW**, where the 9 channels represent flattened 3x3 symmetric
        matrices.
    :type inputs: torch.Tensor
    :return: A volume of shape **Bx9xDxHxW** where each voxel represent a flattened 3x3 symmetric matrix.
    :rtype: torch.Tensor

    Example:
        .. code-block:: python

            import torch
            from torchvectorized.utils import sym, overloadd_diag

            b, c, d, h, w = 1, 9, 32, 32, 32
            inputs = overload_diag(sym(torch.rand(b, c, d, h, w)))

    """
    inputs[:, 0, :, :, :] = inputs[:, 0, :, :, :] + EPSILON
    inputs[:, 4, :, :, :] = inputs[:, 4, :, :, :] + EPSILON
    inputs[:, 8, :, :, :] = inputs[:, 8, :, :, :] + EPSILON

    return inputs


def sym(inputs: torch.Tensor):
    r"""
    Symmetrizes every 3x3 matrix represented by the 9 channels of an input of shape **Bx9xDxHxW** by applying
    :math:`\frac{1}{2}(\mathbf{X} + \mathbf{X}^{\top})`.

    :param inputs: The input tensor of shape **Bx9xDxHxW**, where the 9 channels represent flattened 3x3 symmetric matrices.
    :type inputs: torch.Tensor
    :return: A volume of shape **Bx9xDxHxW** where each voxel represent a flattened 3x3 symmetric matrix.
    :rtype: torch.Tensor

    Example:
        .. code-block:: python

            import torch
            from torchvectorized.utils import sym

            b, c, d, h, w = 1, 9, 32, 32, 32
            inputs = sym(torch.rand(b, c, d, h, w))

    """
    return (inputs + inputs[:, [0, 3, 6, 1, 4, 7, 2, 5, 8], :, :, :]) / 2.0
