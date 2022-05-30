from math import pi

import torch

from torchvectorized.utils import EPSILON


def _compute_eigenvalues(input: torch.Tensor):
    b, c, d, h, w = input.size()
    a11 = input[:, 0, :, :, :].double()
    a12 = input[:, 1, :, :, :].double()
    a13 = input[:, 2, :, :, :].double()
    a22 = input[:, 4, :, :, :].double()
    a23 = input[:, 5, :, :, :].double()
    a33 = input[:, 8, :, :, :].double()
    eig_vals = torch.zeros(b, 3, d, h, w).to(input.device).double()

    nd = torch.pow(a12, 2) + torch.pow(a13, 2) + torch.pow(a23, 2)

    if torch.any(nd != 0):
        q = (a11 + a22 + a33) / 3.0
        p = torch.pow((a11 - q), 2) + torch.pow((a22 - q), 2) + torch.pow((a33 - q), 2) + 2.0 * nd
        p = torch.sqrt(p / 6.0)

        r = torch.pow((1.0 / p), 3) * ((a11 - q) * ((a22 - q) * (a33 - q) - a23 * a23) - a12 * (
                a12 * (a33 - q) - a13 * a23) + a13 * (a12 * a23 - a13 * (a22 - q))) / 2.0

        phi = torch.acos(r) / 3.0
        phi[r <= -1] = pi / 3
        phi[r >= 1] = 0

        eig_vals[:, 0, :, :, :] = q + 2 * p * torch.cos(phi)
        eig_vals[:, 2, :, :, :] = q + 2 * p * torch.cos(phi + pi * (2.0 / 3.0))
        eig_vals[:, 1, :, :, :] = 3 * q - eig_vals[:, 0, :, :, :] - eig_vals[:, 2, :, :, :]

    if torch.any(nd == 0):
        diag_matrix_index = torch.where(nd == 0)
        eig_vals[:, 0, :, :, :][diag_matrix_index] = a11[diag_matrix_index]
        eig_vals[:, 1, :, :, :][diag_matrix_index] = a22[diag_matrix_index]
        eig_vals[:, 2, :, :, :][diag_matrix_index] = a33[diag_matrix_index]

    return eig_vals


def _compute_eigenvectors(input: torch.Tensor, eigenvalues: torch.Tensor):
    a11 = input[:, 0, :, :, :].unsqueeze(1).expand(eigenvalues.size()).double()
    a12 = input[:, 1, :, :, :].unsqueeze(1).expand(eigenvalues.size()).double()
    a13 = input[:, 2, :, :, :].unsqueeze(1).expand(eigenvalues.size()).double()
    a22 = input[:, 4, :, :, :].unsqueeze(1).expand(eigenvalues.size()).double()
    a23 = input[:, 5, :, :, :].unsqueeze(1).expand(eigenvalues.size()).double()

    nd = torch.pow(a12[:, 0, ...], 2) + torch.pow(a13[:, 0, ...], 2) + torch.pow(a23[:, 0, ...], 2)

    u0 = a12 * a23 - a13 * (a22 - eigenvalues)
    u1 = a12 * a13 - a23 * (a11 - eigenvalues)
    u2 = (a11 - eigenvalues) * (a22 - eigenvalues) - a12 * a12
    norm = torch.sqrt(torch.pow(u0, 2) + torch.pow(u1, 2) + torch.pow(u2, 2) + EPSILON)
    u0 = u0 / norm
    u1 = u1 / norm
    u2 = u2 / norm

    if torch.any(nd == 0):
        index = torch.where(nd == 0)
        u0[index[0], :, index[1], index[2], index[3]] = torch.tensor([1, 0, 0]).to(input.device).double()
        u1[index[0], :, index[1], index[2], index[3]] = torch.tensor([0, 1, 0]).to(input.device).double()
        u2[index[0], :, index[1], index[2], index[3]] = torch.tensor([0, 0, 1]).to(input.device).double()

    return torch.cat([u0.unsqueeze(1), u1.unsqueeze(1), u2.unsqueeze(1)], dim=1)


def vSymEig(inputs: torch.Tensor, eigenvectors=False, flatten_output=False, descending_eigenvals=False):
    r"""
    Compute the eigendecomposition :math:`\mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{U}^{\top}` of every
    voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

    :param inputs: The input tensor of shape **Bx9xDxHxW**, where the 9 channels represent flattened 3x3 symmetric matrices.
    :type inputs: torch.Tensor
    :param eigenvectors: If ``True``, computes the eigenvectors.
    :type eigenvectors: bool
    :param flatten_output: If ``True`` the eigenvalues are returned as: **(B*D*H*W)x3** and the eigenvectors as **(B*D*H*W)x3x3**
        otherwise they are returned with shapes **Bx3xDxHxW** and **Bx3x3xDxHxW** respectively.
    :type flatten_output: bool
    :param descending_eigenvals: If ``True``, return the eigenvvalues in descending order
    :type descending_eigenvals: bool
    :return: Return the eigenvalues and the eigenvectors as tensors.
    :rtype: tuple[torch.Tensor, None]

    Example:
        .. code-block:: python

            import torch
            from torchvectorized.utils import sym
            from torchvectorized.vlinalg import vSymEig

            b, c, d, h, w = 1, 9, 32, 32, 32
            inputs = sym(torch.rand(b, c, d, h, w))
            eig_vals, eig_vecs = vSymEig(inputs, eigenvectors=True)

    """
    eig_vals = _compute_eigenvalues(inputs)

    if eigenvectors:
        eig_vecs = _compute_eigenvectors(inputs, eig_vals)
    else:
        eig_vecs = None

    eig_vals, sort_idx = torch.sort(eig_vals, dim=1, descending=descending_eigenvals)

    if eigenvectors:
        sort_idx = sort_idx.unsqueeze(1).expand(eig_vecs.size())
        eig_vecs = eig_vecs.gather(dim=2, index=sort_idx)

    if flatten_output:
        b, c, d, h, w = inputs.size()
        eig_vals = eig_vals.permute(0, 2, 3, 4, 1).reshape(b * d * h * w, 3)
        eig_vecs = eig_vecs.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, 3, 3) if eigenvectors else eig_vecs

    return eig_vals.float(), eig_vecs.float() if eig_vecs is not None else None


def vExpm(inputs: torch.Tensor, replace_nans=False):
    r"""
    Compute the matrix exponential :math:`\mathbf{M} = \mathbf{U} exp(\mathbf{\Sigma}) \mathbf{U}^{\top}` of
    every voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

    :param inputs: The input tensor of shape **Bx9xDxHxW**, where the 9 channels represent flattened 3x3 symmetric matrices.
    :type inputs: torch.Tensor
    :param replace_nans: If ``True``, replace nans by 0
    :type replace_nans: bool
    :return: Return a tensor with shape **Bx9xDxHxW** where every voxel is the matrix exponential of the inpur matrix
        at the same spatial location.
    :rtype: torch.Tensor

    Example:
        .. code-block:: python

            import torch
            from torchvectorized.utils import sym
            from torchvectorized.vlinalg import vExpm

            b, c, d, h, w = 1, 9, 32, 32, 32
            inputs = sym(torch.rand(b, c, d, h, w))
            output = vExpm(inputs)

       """
    b, c, d, h, w = inputs.size()
    eig_vals, eig_vecs = vSymEig(inputs, eigenvectors=True, flatten_output=True)

    # UVU^T
    reconstructed_input = eig_vecs.bmm(torch.diag_embed(torch.exp(eig_vals))).bmm(eig_vecs.transpose(1, 2))
    output = reconstructed_input.reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

    if replace_nans:
        output[torch.where(torch.isnan(output))] = 0

    return output


def vLogm(inputs: torch.Tensor, replace_nans=False):
    r"""
    Compute the matrix logarithm :math:`\mathbf{M} = \mathbf{U} log(\mathbf{\Sigma}) \mathbf{U}^{\top}` of
    every voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

    :param inputs: The input tensor of shape **Bx9xDxHxW**, where the 9 channels represent flattened 3x3 symmetric matrices.
    :type inputs: torch.Tensor
    :param replace_nans: If ``True``, replace nans by 0
    :type replace_nans: bool
    :return: Return a tensor with shape **Bx9xDxHxW** where every voxel is the matrix logarithm of the inpur matrix
        at the same spatial location.
    :rtype: torch.Tensor

    Example:
        .. code-block:: python

            import torch
            from torchvectorized.utils import sym
            from torchvectorized.vlinalg import vLogm

            b, c, d, h, w = 1, 9, 32, 32, 32
            inputs = sym(torch.rand(b, c, d, h, w))
            output = vLogm(inputs)

    """
    b, c, d, h, w = inputs.size()
    eig_vals, eig_vecs = vSymEig(inputs, eigenvectors=True, flatten_output=True)

    # UVU^T
    reconstructed_input = eig_vecs.bmm(torch.diag_embed(torch.log(eig_vals))).bmm(eig_vecs.transpose(1, 2))
    output = reconstructed_input.reshape(b, d * h * w, 3, 3).permute(0, 2, 3, 1).reshape(b, c, d, h, w)

    if replace_nans:
        output[torch.where(torch.isnan(output))] = 0

    return output


def vTrace(inputs: torch.Tensor):
    """
    Compute the trace of every voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

    :param inputs: The input tensor of shape **Bx9xDxHxW**, where the 9 channels represent flattened 3x3 symmetric matrices.
    :type inputs: torch.Tensor
    :return: Return a tensor with shape **Bx1xDxHxW** where every voxel is the trace of the inpur matrix at the
        same spatial location.
    :rtype: torch.Tensor

    Example:
        .. code-block:: python

            import torch
            from torchvectorized.utils import sym
            from torchvectorized.vlinalg import vTrace

            b, c, d, h, w = 1, 9, 32, 32, 32
            inputs = sym(torch.rand(b, c, d, h, w))
            output = vTrace(inputs)

    """
    return inputs[:, 0, :, :, :] + inputs[:, 4, :, :, :] + inputs[:, 8, :, :, :]


def vDet(inputs: torch.Tensor):
    """
    Compute the determinant of every voxel in a volume of flattened 3x3 symmetric matrices of shape **Bx9xDxHxW**.

    :param inputs: The input tensor of shape **Bx9xDxHxW**, where the 9 channels represent flattened 3x3 symmetric matrices.
    :type inputs: torch.Tensor
    :return: Return a tensor with shape **Bx1xDxHxW** where every voxel is the determinant of the inpur matrix at the
        same spatial location.
    :rtype: torch.Tensor

    Example:
        .. code-block:: python

            import torch
            from torchvectorized.utils import sym
            from torchvectorized.vlinalg import vDet

            b, c, d, h, w = 1, 9, 32, 32, 32
            inputs = sym(torch.rand(b, c, d, h, w))
            output = vDet(inputs)

    """
    a = inputs[:, 0, :, :, :].double()
    b = inputs[:, 1, :, :, :].double()
    c = inputs[:, 2, :, :, :].double()
    d = inputs[:, 4, :, :, :].double()
    e = inputs[:, 5, :, :, :].double()
    f = inputs[:, 8, :, :, :].double()
    return (a * (d * f - (e ** 2)) + b * (c * e - (b * f)) + c * (b * e - (d * c))).float()
