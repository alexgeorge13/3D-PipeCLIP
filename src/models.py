# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(data, k):
    inner = -2 * torch.matmul(data.transpose(2, 1), data)
    xx = torch.sum(data**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(data, k=20, idx=None):
    batch_size = data.size(0)
    num_points = data.size(2)
    data = data.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(data, k=k)

    device = torch.device('cuda' if data.is_cuda else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = data.size()
    data = data.transpose(2, 1).contiguous()
    feature = data.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    data = data.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - data, data), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class EdgeConv(nn.Module):
    def __init__(self, k, channels_in, channels_out):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, data):
        data = get_graph_feature(data, k=self.k)
        data = self.conv1(data)
        return data.max(dim=-1, keepdim=False)[0]


class DGCNN_CLIP(nn.Module):
    def __init__(self, clip_dim=512, k=20):
        """
        Modified DGCNN mapping a 3D point cloud to a multimodal embedding space.
        """
        super(DGCNN_CLIP, self).__init__()
        self.k = k
        self.emb_dims = 1024

        self.edge_conv1 = EdgeConv(k, channels_in=6, channels_out=64)
        self.edge_conv2 = EdgeConv(k, channels_in=64 * 2, channels_out=64)
        self.edge_conv3 = EdgeConv(k, channels_in=64 * 2, channels_out=128)
        self.edge_conv4 = EdgeConv(k, channels_in=128 * 2, channels_out=256)

        self.conv1 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.maxpool = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten(1))
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(1))

        self.projection_head = nn.Sequential(
            nn.Linear(self.emb_dims * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, clip_dim)
        )

    def forward(self, x):
        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        x4 = self.edge_conv4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x)

        x5 = self.maxpool(x)
        x6 = self.avgpool(x)
        scene_embedding = torch.cat((x5, x6), dim=1)

        aligned_vector = self.projection_head(scene_embedding)
        normalized_vector = F.normalize(aligned_vector, p=2, dim=1)

        return normalized_vector