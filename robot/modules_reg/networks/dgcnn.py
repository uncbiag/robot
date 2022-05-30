from functools import partial

import torch
import torch.nn.functional as F
import torch.nn as nn
from pykeops.torch import LazyTensor
from robot.utils.utils import shrink_by_factor


def knn(x, k):
    # inner = -2*torch.matmul(x.transpose(2, 1), x)
    # xx = torch.sum(x**2, dim=1, keepdim=True)
    # pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    ############
    y = x.transpose(2, 1).contiguous()
    ay = LazyTensor(y[:, None])
    by = LazyTensor(y[:, :, None, :])
    d_ab = ((ay - by) ** 2).sum(-1)
    idx = d_ab.argKmin(k, dim=2)
    ###########
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()

    device = (
        x.device
    )  # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


def get_graph_for_first_layer_feature(xyz, point_fea, k=20):
    # x = x.squeeze()
    idx = knn(xyz, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()

    device = (
        xyz.device
    )  # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = xyz.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    xyz = xyz.transpose(2, 1).contiguous()
    point_fea = point_fea.transpose(2, 1).contiguous()
    xyz_neigh = xyz.view(batch_size * num_points, -1)[idx, :]
    xyz_neigh = xyz_neigh.view(batch_size, num_points, k, num_dims)
    point_fea_neigh = point_fea.view(batch_size * num_points, -1)[idx, :]
    point_fea_neigh = point_fea_neigh.view(batch_size, num_points, k, -1)
    xyz_center = xyz.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    point_fea_center = point_fea.view(batch_size, num_points, 1, -1).repeat(1, 1, k, 1)
    feature = torch.cat(
        (xyz_neigh, point_fea_neigh, xyz_center, point_fea_center), dim=3
    ).permute(0, 3, 1, 2)

    return feature


class DGCNN(nn.Module):
    def __init__(
        self,
        input_channel=3,
        output_channels=16,
        emb_dims=512,
        k=20,
        param_shrink_factor=1.0,
        use_dropout=False,
    ):
        super(DGCNN, self).__init__()
        sbf = partial(shrink_by_factor, factor=param_shrink_factor)
        self.k = k

        self.bn1 = nn.BatchNorm2d(sbf(64))
        self.bn2 = nn.BatchNorm2d(sbf(64))
        self.bn3 = nn.BatchNorm2d(sbf(128))
        self.bn4 = nn.BatchNorm2d(sbf(256))
        self.bn5 = nn.BatchNorm1d(sbf(emb_dims))

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel * 2, sbf(64), kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(sbf(64 * 2), sbf(64), kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(sbf(64 * 2), sbf(128), kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(sbf(128 * 2), sbf(256), kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(sbf(512), sbf(emb_dims), kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear1 = nn.Linear(sbf(emb_dims * 2), sbf(512), bias=False)
        self.bn6 = nn.BatchNorm1d(sbf(512))
        self.dp1 = nn.Dropout(p=use_dropout)
        self.linear2 = nn.Linear(sbf(512), sbf(256))
        self.bn7 = nn.BatchNorm1d(sbf(256))
        self.dp2 = nn.Dropout(p=use_dropout)
        self.linear3 = nn.Linear(sbf(256), output_channels)

    def forward(self, xyz, point_fea):
        batch_size = xyz.size(0)
        xyz = xyz.transpose(2, 1).contiguous()  # BxDxN
        point_fea = point_fea.transpose(2, 1).contiguous()  # BxCxN
        x = get_graph_for_first_layer_feature(
            xyz, point_fea, k=self.k
        )  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(
            x
        )  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x1, k=self.k
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x2, k=self.k
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(
            x3, k=self.k
        )  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(
            x
        )  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat(
            (x1, x2, x3, x4), dim=1
        )  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(
            x
        )  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(
            batch_size, -1
        )  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(
            batch_size, -1
        )  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(
            self.bn6(self.linear1(x)), negative_slope=0.2
        )  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(
            self.bn7(self.linear2(x)), negative_slope=0.2
        )  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x


if __name__ == "__main__":
    # Test the code.
    x = torch.rand((10, 1024, 3))

    dgcnn = DGCNN()
    y = dgcnn(x)
    print("\nInput Shape of DGCNN: ", x.shape, "\nOutput Shape of DGCNN: ", y.shape)
