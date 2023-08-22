import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.relu()
        return x


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        y = torch.matmul(self.Coefficient, x)
        return y


class GCNCluster(torch.nn.Module):
    def __init__(self, features, hidden_channels, num_sample):
        super(GCNCluster, self).__init__()
        self.gcnconv1 = GCN(features[0].shape[-1], hidden_channels)
        self.gcnconv2 = GCN(features[1].shape[-1], hidden_channels)
        self.gcnconv3 = GCN(features[2].shape[-1], hidden_channels)

        self.lin1 = nn.Linear(features[0].shape[-1], 384)
        self.lin2 = nn.Linear(features[1].shape[-1], 384)
        self.lin3 = nn.Linear(features[2].shape[-1], 384)

        self.content_expression = SelfExpression(num_sample)
        self.structure_expression = SelfExpression(num_sample)

        self.W = nn.Linear(2 * num_sample, 2)


    def forward(self, datas):
        x1 = self.gcnconv1(datas[0].x, datas[0].edge_index)
        x2 = self.gcnconv2(datas[1].x, datas[1].edge_index)
        x3 = self.gcnconv3(datas[2].x, datas[2].edge_index)

        structure_features = self.lin1(x1) + self.lin2(x2) + self.lin3(x3)
        content_features = datas[-1].x

        fusion_expression = self.content_expression.Coefficient + self.structure_expression.Coefficient

        return fusion_expression, content_features, structure_features, self.content_expression.Coefficient, self.structure_expression.Coefficient