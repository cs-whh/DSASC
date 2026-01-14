import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """Basic GCN module with same input/output dimensions"""
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


class GCNDecoderModule(torch.nn.Module):
    """GCN decoder module that maps from hidden_dim to output_dim"""
    def __init__(self, hidden_dim, output_dim, gcn_hidden):
        super().__init__()
        self.conv1 = GCNConv(hidden_dim, gcn_hidden)
        self.conv2 = GCNConv(gcn_hidden, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        return x


class MLPEncoder(nn.Module):
    """MLP Encoder for content features (DINO)"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        intermediate_dim = max(input_dim, hidden_dim) // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class MLPDecoder(nn.Module):
    """MLP Decoder for content features (DINO)"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        intermediate_dim = max(hidden_dim, output_dim) // 2
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class GCNEncoder(nn.Module):
    """GCN Encoder for structure features (3 layers)"""
    def __init__(self, layer_dims, hidden_dim, gcn_hidden):
        super().__init__()
        # Individual GCN for each layer
        self.gcn1 = GCN(layer_dims[0], gcn_hidden)
        self.gcn2 = GCN(layer_dims[1], gcn_hidden)
        self.gcn3 = GCN(layer_dims[2], gcn_hidden)

        # Project to unified hidden dimension
        self.proj1 = nn.Linear(layer_dims[0], hidden_dim)
        self.proj2 = nn.Linear(layer_dims[1], hidden_dim)
        self.proj3 = nn.Linear(layer_dims[2], hidden_dim)

    def forward(self, x1, x2, x3, edge_index):
        # Process each layer through GCN
        h1 = self.gcn1(x1, edge_index)
        h2 = self.gcn2(x2, edge_index)
        h3 = self.gcn3(x3, edge_index)

        # Project and fuse
        h_fused = self.proj1(h1) + self.proj2(h2) + self.proj3(h3)
        return h_fused


class GCNDecoder(nn.Module):
    """GCN Decoder for structure features (3 layers)"""
    def __init__(self, hidden_dim, layer_dims, gcn_hidden):
        super().__init__()
        # Individual GCN decoder for each layer
        # Map from hidden_dim to layer_dims[i]
        self.gcn_dec1 = GCNDecoderModule(hidden_dim, layer_dims[0], gcn_hidden)
        self.gcn_dec2 = GCNDecoderModule(hidden_dim, layer_dims[1], gcn_hidden)
        self.gcn_dec3 = GCNDecoderModule(hidden_dim, layer_dims[2], gcn_hidden)

    def forward(self, h, edge_index):
        # Reconstruct each layer
        x1_recon = self.gcn_dec1(h, edge_index)
        x2_recon = self.gcn_dec2(h, edge_index)
        x3_recon = self.gcn_dec3(h, edge_index)
        return x1_recon, x2_recon, x3_recon


class SharedSelfExpression(nn.Module):
    """Shared self-expression coefficient matrix"""
    def __init__(self, num_samples):
        super().__init__()
        self.C = nn.Parameter(
            1.0e-8 * torch.ones(num_samples, num_samples, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        return torch.matmul(self.C, x)


class CoupledSelfRepresentation(nn.Module):
    """
    Coupled Self-Representation Deep Clustering Model

    Key innovation: Content and structure features share a single self-expression
    coefficient matrix C, enforcing consistent sample affinity across modalities.
    """
    def __init__(self, num_samples, dino_dim, layer_dims, hidden_dim=384, gcn_hidden=16):
        """
        Args:
            num_samples: Number of samples (n)
            dino_dim: DINO feature dimension
            layer_dims: List of 3 layer feature dimensions [d1, d2, d3]
            hidden_dim: Shared hidden space dimension
            gcn_hidden: GCN intermediate layer dimension
        """
        super().__init__()

        # Content branch: DINO features
        self.content_encoder = MLPEncoder(dino_dim, hidden_dim)
        self.content_decoder = MLPDecoder(hidden_dim, dino_dim)

        # Structure branch: 3-layer features
        self.structure_encoder = GCNEncoder(layer_dims, hidden_dim, gcn_hidden)
        self.structure_decoder = GCNDecoder(hidden_dim, layer_dims, gcn_hidden)

        # Shared self-expression layer
        self.self_expression = SharedSelfExpression(num_samples)

    def forward(self, dino_features, layer_features, edge_index):
        """
        Args:
            dino_features: (n, dino_dim) DINO features
            layer_features: [(n, d1), (n, d2), (n, d3)] 3-layer features
            edge_index: (2, num_edges) KNN graph edge index

        Returns:
            C: (n, n) Shared self-expression coefficient matrix
            recon_dino: (n, dino_dim) Reconstructed DINO features
            recon_layers: [(n, d1), (n, d2), (n, d3)] Reconstructed layer features
            H_c: (n, hidden_dim) Content hidden representation
            H_s: (n, hidden_dim) Structure hidden representation
            H_c_recon: (n, hidden_dim) Content hidden representation after self-expression
            H_s_recon: (n, hidden_dim) Structure hidden representation after self-expression
        """
        # Encode to shared hidden space
        H_c = self.content_encoder(dino_features)  # (n, hidden_dim)
        H_s = self.structure_encoder(layer_features[0], layer_features[1],
                                     layer_features[2], edge_index)  # (n, hidden_dim)

        # Shared self-expression (key step!)
        H_c_recon = self.self_expression(H_c)  # C @ H_c
        H_s_recon = self.self_expression(H_s)  # C @ H_s (same C!)

        # Decode back to original space
        recon_dino = self.content_decoder(H_c_recon)
        recon_layers = self.structure_decoder(H_s_recon, edge_index)

        return self.self_expression.C, recon_dino, recon_layers, H_c, H_s, H_c_recon, H_s_recon
