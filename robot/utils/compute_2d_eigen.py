import torch


def compute_2d_eigen(
    cov, eigenvectors=True, flatten_output=True, descending_eigenvals=True
):
    """

    :param cov: Bx4xHxW
    :param eigenvectors:
    :param flatten_output:
    :param descending_eigenvals:
    :return:
    """
    B, D2, H, W = cov.shape[0], cov.shape[1], cov.shape[2], cov.shape[3]
    assert D2 == 4
    T = cov[:, 0] + cov[:, 3]  # BxHxW
    D = cov[:, 0] * cov[:, 3] - cov[:, 1] * cov[:, 2]  # BxHxW
    Td2 = T / 2
    p = Td2 ** 2 - D
    p[p < 0] = 0.0
    sqp = torch.sqrt(p)
    val1 = Td2 + sqp
    val2 = Td2 - sqp
    vec = None
    val = torch.stack([val1, val2], 1)  # Bx2xHxW
    if eigenvectors:
        vec = torch.stack(
            [(val1 - cov[:, 3]), (val2 - cov[:, 3]), cov[:, 2], cov[:, 2]], 1
        )  # Bx4xHxW  the eigenvectors are not normalized
    if flatten_output:
        val = val.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        vec = (
            vec.permute(0, 2, 3, 1).contiguous().view(-1, 2, 2)
            if vec is not None
            else None
        )
    else:
        val = val.view(B, 2, H, W)
        vec = vec.view(B, 2, 2, H, W) if vec is not None else None
    return val, vec
