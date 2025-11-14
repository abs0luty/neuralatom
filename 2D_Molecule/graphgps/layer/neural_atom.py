import math
from typing import Optional, Tuple, Type

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear
from torch_geometric.utils import to_dense_batch


class StraightThroughRound(torch.autograd.Function):
    """Straight-through estimator for rounding operation.
    Forward: rounds the input
    Backward: passes gradient through as identity
    """
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ClusterCountPredictor(torch.nn.Module):
    """Predicts the number of clusters/neural atoms based on graph STRUCTURE.

    Uses graph-level pooling and structural features (density, degrees, etc.)
    to predict cluster count directly, rather than a ratio multiplied by size.

    This enables true unsupervised clustering based on graph complexity.
    """
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        min_clusters: int = 3,
        max_clusters: int = 50,
        use_structural_features: bool = True,
    ):
        super().__init__()
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.use_structural_features = use_structural_features

        # MLP to predict cluster count from graph features
        # Input: [mean, max, std] of node features + structural features
        # Structural: log_size, density, avg_degree, max_degree, degree_std (5 features)
        num_structural = 5 if use_structural_features else 1  # Just log_size if disabled
        self.mlp = torch.nn.Sequential(
            Linear(3 * node_dim + num_structural, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid()  # Output between 0 and 1
        )

    def _compute_structural_features(
        self,
        x: Tensor,
        valid_mask: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None
    ) -> Tensor:
        """
        Compute structural features that capture graph complexity.

        Args:
            x: Node features [batch, num_nodes, dim]
            valid_mask: Boolean mask [batch, num_nodes]
            graph: Optional (x_graph, edge_index, batch) for computing edge-based metrics

        Returns:
            structural_features: [batch, num_features] where features are:
                - log_size: log(num_nodes) to reduce size dominance
                - density: edge_count / max_possible_edges
                - avg_degree: mean node degree
                - max_degree: maximum node degree (normalized by size)
                - degree_std: std deviation of degrees
        """
        batch_size = x.size(0)
        features_list = []

        for b in range(batch_size):
            node_count = valid_mask[b].sum().item()

            # Feature 1: Log of size (reduces size dominance)
            log_size = torch.log(torch.tensor(node_count + 1.0, device=x.device)) / 5.0

            # Features 2-5: Edge-based structural metrics
            if self.use_structural_features and graph is not None:
                x_graph, edge_index, batch_vec = graph

                # Get edges for this graph
                if batch_vec is not None:
                    edge_mask = (batch_vec[edge_index[0]] == b) & (batch_vec[edge_index[1]] == b)
                    graph_edges = edge_index[:, edge_mask]
                    node_indices = torch.where(batch_vec == b)[0]
                else:
                    # Single graph in batch
                    graph_edges = edge_index
                    node_indices = torch.arange(node_count, device=x.device)

                if graph_edges.size(1) > 0 and node_count > 1:
                    # Density
                    num_edges = graph_edges.size(1) // 2  # Undirected
                    max_edges = node_count * (node_count - 1) / 2
                    density = torch.tensor(num_edges / max(max_edges, 1), device=x.device)

                    # Degree statistics
                    degrees = torch.zeros(len(node_indices), device=x.device)
                    for i, node_idx in enumerate(node_indices):
                        degrees[i] = (graph_edges[0] == node_idx).sum()

                    avg_degree = degrees.mean() / 10.0  # Normalize
                    max_degree = degrees.max() / node_count  # Normalized by size
                    degree_std = degrees.std() / 10.0 if len(degrees) > 1 else torch.tensor(0.0, device=x.device)
                else:
                    # No edges or single node
                    density = torch.tensor(0.0, device=x.device)
                    avg_degree = torch.tensor(0.0, device=x.device)
                    max_degree = torch.tensor(0.0, device=x.device)
                    degree_std = torch.tensor(0.0, device=x.device)
            else:
                # Fallback: use defaults (will just use log_size)
                density = torch.tensor(0.1, device=x.device)
                avg_degree = torch.tensor(0.2, device=x.device)
                max_degree = torch.tensor(0.4, device=x.device)
                degree_std = torch.tensor(0.1, device=x.device)

            # Combine all structural features
            if self.use_structural_features:
                graph_structural = torch.stack([log_size, density, avg_degree, max_degree, degree_std])
            else:
                graph_structural = torch.stack([log_size])

            features_list.append(graph_structural)

        return torch.stack(features_list)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None
    ) -> Tuple[int, Tensor]:
        """
        Predict cluster count based on graph structure.

        Args:
            x: Node features [batch, num_nodes, dim]
            mask: Mask for valid nodes [batch, 1, num_nodes]
            graph: Optional (x_graph, edge_index, batch) for structural features

        Returns:
            num_clusters: Predicted number of clusters (int)
            cluster_score: Raw prediction score (for logging/analysis)
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)

        # Handle masking for graph-level pooling
        if mask is not None:
            # mask is typically [batch, 1, num_nodes] with -1e9 for invalid nodes
            # Convert to boolean mask [batch, num_nodes]
            valid_mask = (mask.squeeze(1) > -1e8)
        else:
            valid_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=x.device)

        # Count valid nodes per graph for later clamping
        node_counts = valid_mask.sum(dim=1).to(x.dtype)
        safe_node_counts = torch.clamp(node_counts, min=1.0)
        max_clusters_tensor = torch.full_like(safe_node_counts, float(self.max_clusters))
        min_clusters_tensor = torch.full_like(safe_node_counts, float(self.min_clusters))
        max_allowed = torch.minimum(safe_node_counts, max_clusters_tensor)
        min_allowed = torch.minimum(max_allowed, min_clusters_tensor)

        # Compute graph-level features via pooling
        # Mean pooling
        x_mean = []
        for i in range(batch_size):
            if valid_mask[i].any():
                x_mean.append(x[i][valid_mask[i]].mean(dim=0))
            else:
                x_mean.append(torch.zeros(x.size(2), device=x.device))
        x_mean = torch.stack(x_mean)

        # Max pooling
        x_max = []
        for i in range(batch_size):
            if valid_mask[i].any():
                x_max.append(x[i][valid_mask[i]].max(dim=0)[0])
            else:
                x_max.append(torch.zeros(x.size(2), device=x.device))
        x_max = torch.stack(x_max)

        # Std pooling
        x_std = []
        for i in range(batch_size):
            if valid_mask[i].any() and valid_mask[i].sum() > 1:
                x_std.append(x[i][valid_mask[i]].std(dim=0))
            else:
                x_std.append(torch.zeros(x.size(2), device=x.device))
        x_std = torch.stack(x_std)

        # Compute structural features (density, degrees, etc.)
        structural_features = self._compute_structural_features(x, valid_mask, graph)

        # Concatenate node features and structural features
        graph_features = torch.cat([x_mean, x_max, x_std, structural_features], dim=1)

        # Predict cluster count DIRECTLY (not ratio!)
        cluster_score = self.mlp(graph_features).squeeze(-1)  # [batch]

        # Scale from [0, 1] to [min_clusters, max_clusters]
        num_clusters_continuous = (
            self.min_clusters +
            cluster_score * (self.max_clusters - self.min_clusters)
        )

        # Respect per-graph limits (can't exceed valid nodes)
        num_clusters_continuous = torch.minimum(num_clusters_continuous, max_allowed)
        num_clusters_continuous = torch.maximum(num_clusters_continuous, min_allowed)

        # Use straight-through estimator for rounding
        num_clusters_rounded = StraightThroughRound.apply(num_clusters_continuous)

        # For batched processing, take mean across batch
        num_clusters_final = int(num_clusters_rounded.mean().item())
        max_batch_clusters = int(max_allowed.min().item())
        num_clusters_final = max(1, min(max_batch_clusters, num_clusters_final))

        # Ratio of predicted clusters to available nodes (for logging/training)
        cluster_ratio = (num_clusters_continuous / safe_node_counts).mean()

        return num_clusters_final, cluster_ratio


class BaseMHA(torch.nn.Module):
    def __init__(
        self,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        num_heads: int,
        Conv: Optional[Type] = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.fc_q = Linear(dim_Q, dim_V)

        if Conv is None:
            self.layer_k = Linear(dim_K, dim_V)
            self.layer_v = Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V)
            self.layer_v = Conv(dim_K, dim_V)

        if layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        self.fc_o = Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.layer_k.reset_parameters()
        self.layer_v.reset_parameters()
        if self.layer_norm:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()
        self.fc_o.reset_parameters()
        pass

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        Q = self.fc_q(Q)

        if graph is not None:
            x, edge_index, batch = graph
            K, V = self.layer_k(x, edge_index), self.layer_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2))
            attention_score = attention_score / math.sqrt(self.dim_V)
            A = torch.softmax(mask + attention_score, 1)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) /
                              math.sqrt(self.dim_V), 1)

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if self.layer_norm:
            out = self.ln0(out)

        out = out + self.fc_o(out).relu()

        if self.layer_norm:
            out = self.ln1(out)

        return out, A

class Exchanging(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.mab = BaseMHA(in_channels, in_channels, out_channels, num_heads,
                       Conv=Conv, layer_norm=layer_norm)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(x, x, graph, mask)

class Porjecting(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_seeds: int,
        Conv: Optional[Type] = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = BaseMHA(
            channels, channels, channels, num_heads, Conv=Conv, layer_norm=layer_norm
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(
            self.S.repeat(x.size(0), 1, 1), x, graph, mask
        )  # * S-> Q; x -> K, V


class DynamicPorjecting(torch.nn.Module):
    """Dynamic version of Porjecting that predicts the number of clusters.

    Uses ClusterCountPredictor to determine the number of neural atoms,
    then uses only that many seed embeddings from a larger pool.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        max_seeds: int = 50,
        min_seeds: int = 3,
        Conv: Optional[Type] = None,
        layer_norm: bool = False,
        predictor_hidden_dim: int = 64,
    ):
        super().__init__()
        self.max_seeds = max_seeds
        self.min_seeds = min_seeds
        self.channels = channels

        # Initialize with maximum number of seeds
        self.S = torch.nn.Parameter(torch.Tensor(1, max_seeds, channels))

        # Cluster count predictor (structure-based)
        self.cluster_predictor = ClusterCountPredictor(
            node_dim=channels,
            hidden_dim=predictor_hidden_dim,
            min_clusters=min_seeds,
            max_clusters=max_seeds,
            use_structural_features=True,
        )

        self.mab = BaseMHA(
            channels, channels, channels, num_heads, Conv=Conv, layer_norm=layer_norm
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, int, Tensor]:
        """
        Args:
            x: Node features [batch, num_nodes, dim]
            graph: Optional graph connectivity (x_graph, edge_index, batch)
            mask: Mask for valid nodes

        Returns:
            output: Neural atom embeddings and attention matrix (from mab)
            num_clusters: Predicted number of clusters used
            cluster_score: Predicted score (for logging)
        """
        # Predict number of clusters based on graph structure
        num_clusters, cluster_score = self.cluster_predictor(x, mask, graph)

        # Use only the first num_clusters seeds
        S_dynamic = self.S[:, :num_clusters, :]

        # Apply multi-head attention with dynamic seeds
        output = self.mab(
            S_dynamic.repeat(x.size(0), 1, 1), x, graph, mask
        )

        return (*output, num_clusters, cluster_score)


# Aliases for compatibility
PMA = Porjecting  # Pooling by Multi-head Attention
SAB = Exchanging  # Set Attention Block
DynamicPMA = DynamicPorjecting  # Dynamic Pooling by Multi-head Attention
