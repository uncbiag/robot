import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
from matplotlib import colors as mcolors
import numpy as np
import os
import torch
from torch.autograd import Function
from torch.autograd.functional import jacobian as J
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import memory_summary


########################
# This util script is borrowed from
# https://github.com/multimodallearning/deep-geo-reg
########################

#### misc #####

def parameter_count(model):
    print('# parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))


def gpu_usage():
    print('gpu usage (current/max): {:.2f} / {:.2f} GB'.format(torch.cuda.memory_allocated() * 1e-9,
                                                               torch.cuda.max_memory_allocated() * 1e-9))


#### distance #####

def pdist(x, p=2):
    if p == 1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p == 2:
        xx = (x ** 2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist


def pdist2(x, y, p=2):
    if p == 1:
        dist = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(dim=3)
    elif p == 2:
        xx = (x ** 2).sum(dim=2).unsqueeze(2)
        yy = (y ** 2).sum(dim=2).unsqueeze(1)
        dist = xx + yy - 2.0 * torch.bmm(x, y.permute(0, 2, 1))
    return dist


#### thin plate spline #####

class TPS:
    @staticmethod
    def fit(c, f, lambd=0.):
        device = c.device

        n = c.size(0)
        f_dim = f.size(1)

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n).to(device) * lambd

        P = torch.ones((n, 4)).to(device)
        P[:, 1:] = c

        v = torch.zeros((n + 4, f_dim)).to(device)
        v[:n, :] = f

        A = torch.zeros((n + 4, n + 4)).to(device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta, _ = torch.solve(v, A)
        return theta

    @staticmethod
    def d(a, b):
        ra = (a ** 2).sum(dim=1).view(-1, 1)
        rb = (b ** 2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist = torch.clamp(dist, 0.0, np.inf)
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return (r ** 2) * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-4], theta[-4:]
        b = torch.matmul(U, w)
        return (a[0].unsqueeze(1) + a[1].unsqueeze(1) * x[:, 0] + a[2].unsqueeze(1) * x[:, 1] + a[3].unsqueeze(1) * x[:,
                                                                                                                    2] + b.t()).t()


def thin_plate(x1, y1, x2, lambd=0.):
    tps = TPS()
    theta = tps.fit(x1, y1, lambd)
    y2 = tps.z(x2, x1, theta)
    return y2


def thin_plate_dense(x1, y1, shape, lambd=0., unroll_factor=100):
    D, H, W = shape
    device = x1.device

    grid = torch.stack(torch.meshgrid(torch.linspace(0, D - 1, D),
                                      torch.linspace(0, H - 1, H),
                                      torch.linspace(0, W - 1, W))).permute(1, 2, 3, 0).view(-1, 3).to(device)

    tps = TPS()
    theta = tps.fit(kpts_world(x1, (D, H, W)).squeeze(0),
                    (y1.flip(-1) * (torch.Tensor([D, H, W]).to(device) - 1) / 2).squeeze(0), lambd)

    y2 = torch.zeros(D * H * W, 3).to(device)
    split = np.array_split(np.arange(D * H * W), unroll_factor)
    for i in range(unroll_factor):
        y2[split[i], :] = tps.z(grid[split[i], :], kpts_world(x1, (D, H, W)).squeeze(0), theta)

    return (y2.permute(1, 0).view(1, 3, D, H, W)).flip(1) * 2 / (torch.Tensor([W, H, D]).to(device) - 1).view(1, -1, 1,
                                                                                                              1, 1)


#### pca #####

def pca_train(x, num_components):
    mean = x.mean(dim=1, keepdim=True)
    x = x - mean
    u, s, v = torch.svd(x)
    v = v[:, :, :num_components]
    return v, mean


def pca_fit(x, v, mean):
    x = x - mean
    x = torch.bmm(x, v)
    return x


def pca_reconstruct(x, v, mean):
    x = torch.bmm(x, v.permute(0, 2, 1)) + mean
    return x


#### filter #####

def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(6, )
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N // 2
    padding = padding.long().tolist()

    view = torch.ones(5, )
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(F.pad(img.view(B * C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H,
                                                                                                            W)


def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img


class GaussianSmoothing(nn.Module):
    def __init__(self, sigma):
        super(GaussianSmoothing, self).__init__()

        sigma = torch.tensor([sigma])
        N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

        weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N), 2) / (2 * torch.pow(sigma, 2)))
        weight /= weight.sum()

        self.weight = weight

    def forward(self, x):
        device = x.device

        x = filter1D(x, self.weight.to(device), 0)
        x = filter1D(x, self.weight.to(device), 1)
        x = filter1D(x, self.weight.to(device), 2)

        return x


class ApproxMinConv(nn.Module):
    def __init__(self):
        super(ApproxMinConv, self).__init__()

        self.alpha = nn.Parameter(torch.Tensor([1, .1]))

        self.pad = nn.ReplicationPad3d(3)
        self.max = nn.MaxPool3d(3, stride=1)
        self.avg = nn.AvgPool3d(3, stride=1)

    def forward(self, x):
        return self.avg(self.avg(-self.max(-self.pad(self.alpha[0] * x + self.alpha[1]))))


def minconv(input, l_width):
    disp1d = torch.linspace(-1, 1, l_width).to(input.device)
    regular1d = (disp1d.reshape(1, -1) - disp1d.reshape(-1, 1)) ** 2

    output = torch.min(input.view(-1, l_width, 1, l_width, l_width) + regular1d.view(1, l_width, l_width, 1, 1), 1)[0]
    output = torch.min(output.view(-1, l_width, l_width, 1, l_width) + regular1d.view(1, 1, l_width, l_width, 1), 2)[0]
    output = torch.min(output.view(-1, l_width, l_width, l_width, 1) + regular1d.view(1, 1, 1, l_width, l_width), 3)[0]

    output = output - (torch.min(output.view(-1, l_width ** 3), 1)[0]).view(output.shape[0], 1, 1, 1)

    return output.view_as(input)


#### feature #####

def mindssc(img, delta=1, sigma=0.8):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    device = img.device
    dtype = img.dtype

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    # squared distances
    dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(dtype).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(dtype).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad = nn.ReplicationPad3d(delta)

    # compute patch-ssd
    ssd = smooth(((F.conv3d(rpad(img), mshift1, dilation=delta) - F.conv3d(rpad(img), mshift2, dilation=delta)) ** 2),
                 sigma)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind).to(dtype)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind


#### transform #####

def warp_img(img, disp, mode='bilinear', padding_mode='border'):
    _, _, D, H, W = img.shape
    device = img.device

    identity = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), (1, 1, D, H, W), align_corners=True)

    return F.grid_sample(img, identity + disp, mode=mode, padding_mode=padding_mode, align_corners=True)


#### similarity metrics #####

def ssd(kpts_fixed, feat_fixed, feat_moving, orig_shape, disp_radius=16, disp_step=2, patch_radius=2, alpha=1.5,
        unroll_factor=50):
    _, N, _ = kpts_fixed.shape
    device = kpts_fixed.device
    D, H, W = orig_shape
    C = feat_fixed.shape[1]
    dtype = feat_fixed.dtype

    patch_step = disp_step  # same stride necessary for fast implementation
    patch = torch.stack(torch.meshgrid(torch.arange(0, 2 * patch_radius + 1, patch_step),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step),
                                       torch.arange(0, 2 * patch_radius + 1, patch_step))).permute(1, 2, 3,
                                                                                                   0).contiguous().view(
        1, 1, -1, 1, 3).float() - patch_radius
    patch = (patch.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(dtype).to(device)

    patch_width = round(patch.shape[2] ** (1.0 / 3))

    if patch_width % 2 == 0:
        pad = [(patch_width - 1) // 2, (patch_width - 1) // 2 + 1]
    else:
        pad = [(patch_width - 1) // 2, (patch_width - 1) // 2]

    disp = torch.stack(torch.meshgrid(torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)),
                                                   (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1,
                                                   disp_step),
                                      torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)),
                                                   (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1,
                                                   disp_step),
                                      torch.arange(- disp_step * (disp_radius + ((pad[0] + pad[1]) / 2)),
                                                   (disp_step * (disp_radius + ((pad[0] + pad[1]) / 2))) + 1,
                                                   disp_step))).permute(1, 2, 3, 0).contiguous().view(1, 1, -1, 1,
                                                                                                      3).float()
    disp = (disp.flip(-1) * 2 / (torch.tensor([W, H, D]) - 1)).to(dtype).to(device)

    disp_width = disp_radius * 2 + 1
    ssd = torch.zeros(1, N, disp_width ** 3).to(dtype).to(device)
    split = np.array_split(np.arange(N), unroll_factor)
    for i in range(unroll_factor):
        feat_fixed_patch = F.grid_sample(feat_fixed, kpts_fixed[:, split[i], :].view(1, -1, 1, 1, 3).to(dtype) + patch,
                                         padding_mode='border', align_corners=True)
        feat_moving_disp = F.grid_sample(feat_moving, kpts_fixed[:, split[i], :].view(1, -1, 1, 1, 3).to(dtype) + disp,
                                         padding_mode='border', align_corners=True)
        corr = F.conv3d(feat_moving_disp.view(1, -1, disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1],
                                              disp_width + pad[0] + pad[1]),
                        feat_fixed_patch.view(-1, 1, patch_width, patch_width, patch_width),
                        groups=C * split[i].shape[0]).view(C, split[i].shape[0], -1)
        patch_sum = (feat_fixed_patch ** 2).squeeze(0).squeeze(3).sum(dim=2, keepdims=True)
        disp_sum = (patch_width ** 3) * F.avg_pool3d(
            (feat_moving_disp ** 2).view(C, -1, disp_width + pad[0] + pad[1], disp_width + pad[0] + pad[1],
                                         disp_width + pad[0] + pad[1]), patch_width, stride=1).view(C,
                                                                                                    split[i].shape[0],
                                                                                                    -1)
        ssd[0, split[i], :] = ((- 2 * corr + patch_sum + disp_sum)).sum(0)

    ssd *= (alpha / (patch_width ** 3))

    return ssd


#### keypoints #####

def kpts_pt(kpts_world, shape):
    device = kpts_world.device
    D, H, W = shape
    return (kpts_world.flip(-1) / (torch.tensor([W, H, D]).to(device) - 1)) * 2 - 1


def kpts_world(kpts_pt, shape):
    device = kpts_pt.device
    D, H, W = shape
    return ((kpts_pt.flip(-1) + 1) / 2) * (torch.tensor([D, H, W]).to(device) - 1)


# def farthest_point_sampling(kpts, num_points):
#    _, N, _ = kpts.size()
#    ind = torch.zeros(num_points).long()
#    ind[0] = torch.randint(N, (1,))
#    dist = torch.sum((kpts - kpts[:, ind[0], :]) ** 2, dim=2)
#    for i in range(1, num_points):
#        ind[i] = torch.argmax(dist)
#        dist = torch.min(dist, torch.sum((kpts - kpts[:, ind[i], :]) ** 2, dim=2))
#
#    return kpts[:, ind, :], ind

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def farthest_point_sampling(kpts, num_points):
    _, N, _ = kpts.shape

    ind_rand = torch.randperm(N)
    kpts_rand = kpts[:, ind_rand, :]
    ind = farthest_point_sample(kpts_rand, num_points).long()[0]
    if N < num_points:
        ind = torch.cat([ind[:N], ind[:num_points - N]])

    return kpts_rand[:, ind, :], ind_rand[ind]


def uniform_kpts(mask, d, num_points=None):
    _, _, D, H, W = mask.shape
    device = mask.device

    kpts = torch.nonzero(mask[:, :, ::d, ::d, ::d]).unsqueeze(0).float()[:, :, 2:]
    kpts *= d

    if not num_points is None:
        kpts = farthest_point_sampling(kpts, num_points)[0]

    return kpts_pt(kpts, (D, H, W))


def random_kpts(mask, num_points=None):
    _, _, D, H, W = mask.shape
    device = mask.device

    kpts = torch.nonzero(mask[:, :, :, :, :]).unsqueeze(0).float()[:, :, 2:]

    if not num_points is None:
        kpts = farthest_point_sampling(kpts, num_points)[0]

    return kpts_pt(kpts, (D, H, W))


def structure_tensor(img, sigma):
    B, C, D, H, W = img.shape
    device = img.device

    struct = []
    for i in range(C):
        for j in range(i, C):
            struct.append(smooth((img[:, i, ...] * img[:, j, ...]).unsqueeze(1), sigma))

    return torch.cat(struct, dim=1)


def invert_structure_tensor(struct):
    a = struct[:, 0, ...]
    b = struct[:, 1, ...]
    c = struct[:, 2, ...]
    e = struct[:, 3, ...]
    f = struct[:, 4, ...]
    i = struct[:, 5, ...]

    A = e * i - f * f
    B = - b * i + c * f
    C = b * f - c * e
    E = a * i - c * c
    F = - a * f + b * c
    I = a * e - b * b

    det = (a * A + b * B + c * C).unsqueeze(1)
    trace = (a + e + i).unsqueeze(1)

    mask = det != 0
    b, _, d, w, h = det.shape

    w = torch.zeros((b, 1, d, w, h), device=struct.device)
    q = torch.zeros_like(w)

    w[mask] = det[mask] / trace[mask]
    q[mask] = 4 * det[mask] / trace[mask] ** 2

    # struct_inv = torch.zeros((b, 6, d, w, h), device=struct.device)

    # struct_inv[mask.expand(-1, 6, -1,-1,-1)] = (1./det[mask]) * torch.stack([A[mask], B[mask], C[mask], E[mask], F[mask], I[mask]], dim=1)

    return w, q


def foerstner_kpts(img, mask, sigma=1.4, d=9, roundness_thresh=0.0001, accuracy_thresh=1e-8, num_points=None):
    _, _, D, H, W = img.shape
    device = img.device
    dtype = img.dtype

    filt = torch.tensor([1.0 / 12.0, -8.0 / 12.0, 0.0, 8.0 / 12.0, -1.0 / 12.0]).to(dtype).to(device)
    grad = torch.cat([filter1D(img, filt, 0),
                      filter1D(img, filt, 1),
                      filter1D(img, filt, 2)], dim=1)

    w, q = invert_structure_tensor(structure_tensor(grad, sigma))
    distinctiveness = (q > roundness_thresh) * (w > accuracy_thresh) * w

    # distinctiveness = 1. / (struct_inv[:, 0, ...] + struct_inv[:, 3, ...] + struct_inv[:, 5, ...]).unsqueeze(1)

    pad1 = d // 2
    pad2 = d - pad1 - 1

    maxfeat = F.max_pool3d(F.pad(distinctiveness, (pad2, pad1, pad2, pad1, pad2, pad1)), d, stride=1)

    structure_element = torch.tensor([[[0., 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]],
                                      [[0, 1, 0],
                                       [1, 0, 1],
                                       [0, 1, 0]],
                                      [[0, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]]]).to(device)

    mask_eroded = (1 - F.conv3d(1 - mask.to(dtype), structure_element.unsqueeze(0).unsqueeze(0), padding=1).clamp_(0,
                                                                                                                   1)).bool()

    kpts = torch.nonzero(mask_eroded & (maxfeat == distinctiveness) & (distinctiveness != 0)).unsqueeze(0).to(dtype)[:,
           :, 2:]

    if not num_points is None:
        kpts = farthest_point_sampling(kpts, num_points)[0]

    return kpts_pt(kpts, (D, H, W))


def kpts_to_img(kpts, shape, radius, sigma):
    _, N, _ = kpts.shape
    D, H, W = shape
    device = kpts.device

    kpts = kpts.squeeze(0)
    val = torch.arange(N).to(device) + 1
    unique_kpts, counts = torch.unique(kpts, dim=0, return_counts=True)
    identical_kpts = unique_kpts[torch.nonzero(counts == 2).squeeze(1)]
    for kpt in identical_kpts:
        ind = torch.nonzero((kpts == kpt).all(dim=1)).squeeze(1) + 1
        val_new = ind[0] * N + ind[1]
        val[ind - 1] = val_new

    grid = (torch.stack(torch.meshgrid(torch.linspace(-radius, radius, radius * 2 + 1),
                                       torch.linspace(-radius, radius, radius * 2 + 1),
                                       torch.linspace(-radius, radius, radius * 2 + 1)), dim=-1).to(
        device) + kpts.round().view(-1, 1, 1, 1, 3)).view(-1, 3).long()
    grid[:, 0].clamp_(0, D - 1)
    grid[:, 1].clamp_(0, H - 1)
    grid[:, 2].clamp_(0, W - 1)

    disp = kpts - kpts.round()
    grid_disp = (torch.stack(torch.meshgrid(torch.linspace(- radius, radius, radius * 2 + 1),
                                            torch.linspace(- radius, radius, radius * 2 + 1),
                                            torch.linspace(- radius, radius, radius * 2 + 1)), dim=-1).to(
        device) - disp.view(-1, 1, 1, 1, 3)).view(N, -1, 3)

    hm = torch.exp(-(torch.pow(grid_disp[:, :, 0] / sigma, 2) / 2 + \
                     torch.pow(grid_disp[:, :, 1] / sigma, 2) / 2 + \
                     torch.pow(grid_disp[:, :, 2] / sigma, 2) / 2))
    hm /= hm.sum(1, keepdim=True)
    hm += val.unsqueeze(1)
    hm = hm.view(-1)

    img = torch.zeros(1, 1, D, H, W).to(device)
    perm = torch.randperm(grid.shape[0])
    img[0, 0, grid[perm, 0], grid[perm, 1], grid[perm, 2]] = hm[perm]

    return img


def img_to_kpts(img, N):
    _, _, D, H, W = img.shape
    device = img.device

    img = img.squeeze(0).squeeze(0)
    grid = torch.stack(torch.meshgrid(torch.linspace(0, D - 1, D),
                                      torch.linspace(0, H - 1, H),
                                      torch.linspace(0, W - 1, W))).permute(1, 2, 3, 0).to(device)
    ind = img.nonzero(as_tuple=True)
    val = img[ind]
    grid = grid[ind]

    kpts = torch.zeros(1, N, 3).to(device)
    for i in val.floor().unique():
        p = val[i == val.floor()] - i
        p /= p.sum()
        p = p.unsqueeze(1)
        grid_i = grid[i == val.floor()]
        if i > N:
            i1 = i // N
            i2 = i % N
            if i2 == 0:
                i2 = torch.tensor(N - 1.)
            kpts[0, i1.long() - 1, :] = (p * grid_i).sum(dim=0)
            kpts[0, i2.long() - 1, :] = (p * grid_i).sum(dim=0)
        else:
            kpts[0, i.long() - 1, :] = (p * grid_i).sum(dim=0)

    return kpts


#### graph ###

def knn_graph(kpts, k, include_self=False):
    B, N, D = kpts.shape
    device = kpts.device

    dist = pdist(kpts)
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self):]
    A = torch.zeros(B, N, N).to(device)
    A[:, torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1

    return ind, dist * A, A


def laplacian(kpts, k, lambd, sigma=0):
    _, dist, A = knn_graph(kpts, k)
    W = lambd * A.squeeze(0)
    if sigma > 0:
        W = W * torch.exp(- dist.squeeze(0) / (sigma ** 2))
    return (torch.diag(W.sum(1) + 1) - W).unsqueeze(0), W.unsqueeze(0)


#### evaluation #####

def copdgene_error(path, idx, lms_inh_displacement):
    voxel_spacing = torch.load(os.path.join(path, 'voxel_spacing.pth'))[idx]
    crop_inh = torch.load(os.path.join(path, 'crops_inh.pth'))[idx]
    crop_exh = torch.load(os.path.join(path, 'crops_exh.pth'))[idx]

    lms_inh_path = os.path.join(path, 'copd' + str(idx + 1) + '_300_iBH_xyz_r1.txt')
    lms_inh = torch.cat((torch.from_numpy(np.loadtxt(lms_inh_path).astype('float32')) - 1, torch.ones(300, 1)), dim=1)
    lms_exh_path = os.path.join(path, 'copd' + str(idx + 1) + '_300_eBH_xyz_r1.txt')
    lms_exh = torch.cat((torch.from_numpy(np.loadtxt(lms_exh_path).astype('float32')) - 1, torch.ones(300, 1)), dim=1)

    T1 = torch.eye(4)
    T1[3, :3] = crop_inh[0, :] - 1
    S1 = torch.diag(torch.cat((1 / voxel_spacing, torch.ones(1)), 0))
    A1 = S1.matmul(T1)
    new_size = torch.round((crop_inh[1, :] - crop_inh[0, :] + 1) * voxel_spacing)

    T2 = torch.eye(4)
    T2[3, :3] = crop_exh[0, :] - 1
    voxel_spacing_exh = new_size / (crop_exh[1, :] - crop_exh[0, :] + 1)
    S2 = torch.diag(torch.cat((1 / voxel_spacing_exh, torch.ones(1)), 0))
    A2 = S2.matmul(T2)

    lms_inh_conv = lms_inh.matmul(torch.inverse(A1))
    lms_exh_conv = lms_exh.matmul(torch.inverse(A2))

    lms_inh_est = torch.cat((lms_inh_conv[:, :3] + lms_inh_displacement, torch.ones(300, 1)), dim=1).matmul(A2)

    dists = torch.sqrt(
        torch.sum((voxel_spacing.unsqueeze(0) * torch.round(lms_inh_est[:, :3] - lms_exh[:, :3])) ** 2, dim=1))

    return dists


def dct4_error(path, idx, lms_inh_displacement):
    voxel_spacing = torch.load(os.path.join(path, 'voxel_spacing.pth'))[idx]
    crop_inh = torch.load(os.path.join(path, 'crops_inh.pth'))[idx]
    crop_exh = torch.load(os.path.join(path, 'crops_exh.pth'))[idx]

    lms_inh_path = os.path.join(path, 'case' + str(idx + 1) + '_300_T00_xyz.txt')
    lms_inh = torch.cat((torch.from_numpy(np.loadtxt(lms_inh_path).astype('float32')) - 1, torch.ones(300, 1)), dim=1)
    lms_exh_path = os.path.join(path, 'case' + str(idx + 1) + '_300_T50_xyz.txt')
    lms_exh = torch.cat((torch.from_numpy(np.loadtxt(lms_exh_path).astype('float32')) - 1, torch.ones(300, 1)), dim=1)

    T1 = torch.eye(4)
    T1[3, :3] = crop_inh[0, :] - 1
    S1 = torch.diag(torch.cat((1 / voxel_spacing, torch.ones(1)), 0))
    A1 = S1.matmul(T1)
    new_size = torch.round((crop_inh[1, :] - crop_inh[0, :] + 1) * voxel_spacing)

    T2 = torch.eye(4)
    T2[3, :3] = crop_exh[0, :] - 1
    voxel_spacing_exh = new_size / (crop_exh[1, :] - crop_exh[0, :] + 1)
    S2 = torch.diag(torch.cat((1 / voxel_spacing_exh, torch.ones(1)), 0))
    A2 = S2.matmul(T2)

    lms_inh_conv = lms_inh.matmul(torch.inverse(A1))
    lms_exh_conv = lms_exh.matmul(torch.inverse(A2))

    lms_inh_est = torch.cat((lms_inh_conv[:, :3] + lms_inh_displacement, torch.ones(300, 1)), dim=1).matmul(A2)

    dists = torch.sqrt(
        torch.sum((voxel_spacing.unsqueeze(0) * torch.round(lms_inh_est[:, :3] - lms_exh[:, :3])) ** 2, dim=1))

    return dists


#### inverse grid sampling #####

class InverseGridSample(Function):

    @staticmethod
    def forward(ctx, input, grid, shape, mode='bilinear', padding_mode='zeros', align_corners=None):
        B, C, N = input.shape
        D = grid.shape[-1]
        device = input.device
        dtype = input.dtype

        ctx.save_for_backward(input, grid)

        if D == 2:
            input_view = [B, C, -1, 1]
            grid_view = [B, -1, 1, 2]
        elif D == 3:
            input_view = [B, C, -1, 1, 1]
            grid_view = [B, -1, 1, 1, 3]

        ctx.grid_view = grid_view
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners

        with torch.enable_grad():
            output = J(lambda x: InverseGridSample.sample(input.view(*input_view), grid.view(*grid_view), x, mode,
                                                          padding_mode, align_corners),
                       (torch.zeros(B, C, *shape).to(dtype).to(device)))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grid_view = ctx.grid_view
        mode = ctx.mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners

        grad_input = F.grid_sample(grad_output, grid.view(*grid_view), mode, padding_mode, align_corners)

        return grad_input.view(*input.shape), None, None, None, None, None

    @staticmethod
    def sample(input, grid, accu, mode='bilinear', padding_mode='zeros', align_corners=None):
        sampled = F.grid_sample(accu, grid, mode, padding_mode, align_corners)
        return -0.5 * ((input - sampled) ** 2).sum()


def inverse_grid_sample(input, grid, shape, mode='bilinear', padding_mode='zeros', align_corners=None):
    return InverseGridSample.apply(input, grid, shape, mode, padding_mode, align_corners)


def densify(kpts, kpts_disp, shape, smooth_iter=3, kernel_size=5, eps=0.0001):
    B, N, _ = kpts.shape
    device = kpts.device
    D, H, W = shape

    grid = inverse_grid_sample(kpts_disp.permute(0, 2, 1), kpts, shape, padding_mode='border', align_corners=True)
    grid_norm = inverse_grid_sample(torch.ones(B, 1, N).to(device), kpts, shape, padding_mode='border',
                                    align_corners=True)

    avg_pool = nn.AvgPool3d(kernel_size, stride=1, padding=kernel_size // 2).to(device)
    for i in range(smooth_iter):
        grid = avg_pool(grid)
        grid_norm = avg_pool(grid_norm)

    grid = grid / (grid_norm + eps)

    return grid


#### plot #####

def overlay_segment(img, seg):
    H, W = seg.squeeze().shape
    device = img.device

    colors = torch.FloatTensor(
        [0, 0, 0, 199, 67, 66, 225, 140, 154, 78, 129, 170, 45, 170, 170, 240, 110, 38, 111, 163, 91, 235, 175, 86, 202,
         255, 52, 162, 0, 183]).view(-1, 3).to(device) / 255.0
    max_label = seg.long().max().item()
    seg_one_hot = F.one_hot(seg.long(), max_label + 1).float()

    seg_color = torch.mm(seg_one_hot.view(-1, max_label + 1), colors[:max_label + 1, :]).view(H, W, 3)
    alpha = torch.clamp(1.0 - 0.5 * (seg > 0).float(), 0, 1.0)

    overlay = (img * alpha).unsqueeze(2) + seg_color * (1.0 - alpha).unsqueeze(2)
    return overlay


def overlay_parula(img, heatmap):
    x = np.linspace(0.0, 1.0, 256)
    rgb_jet = mpl_color_map.get_cmap(plt.get_cmap('jet'))(x)[:, :3]
    rgb_gray = mpl_color_map.get_cmap(plt.get_cmap('gray'))(x)[:, :3]
    rgb_heat = rgb_jet[(heatmap * 255).numpy().astype('uint8'), :]
    rgb_img = rgb_gray[(img * 255).numpy().astype('uint8'), :]

    rgb0 = (rgb_heat * 127.5 + rgb_img * 127.5).astype('uint8')
    weight = torch.tanh((heatmap - 0.5) * 3) * 0.5 + 0.5
    alpha = torch.clamp(1.0 - 0.5 * weight, 0, 1.0)
    overlay = rgb_img * alpha.unsqueeze(2).numpy() + rgb_heat * (1.0 - alpha).unsqueeze(2).numpy()
    return overlay


def show_flow(disp):
    _, _, H, W = disp.shape

    x = disp.squeeze().numpy()[0, :, :]
    y = disp.squeeze().numpy()[1, :, :]

    rho = np.sqrt(x * x + y * y)
    rho = np.clip(rho / np.percentile(rho, 99), 0, 1)

    theta = np.arctan2(x, -y)
    theta = (-theta + np.pi) / (2.0 * np.pi);

    hsv = np.stack((theta, rho, np.ones((H, W))), axis=2)
    flow = mcolors.hsv_to_rgb(hsv)

    return flow