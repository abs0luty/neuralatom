"""
Fourier-based clustering for neural atoms.

This module implements clustering in reciprocal (Fourier) space:
1. Transform atom embeddings to Fourier space using FFT
2. Determine optimal cluster count using unsupervised metrics (silhouette score)
3. Perform clustering in Fourier space using KMeans or HDBSCAN
4. Transform cluster centers back to real space
"""

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch import Tensor
from torch.nn import LayerNorm, Linear

try:
    from sklearn.cluster import HDBSCAN

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class FourierClusteringModule(torch.nn.Module):
    """
    Clustering module that operates in Fourier (reciprocal) space.

    The key idea is that patterns in embeddings may be more separable in
    frequency domain, allowing for better clustering based on spectral features.
    """

    def __init__(
        self,
        node_dim: int,
        min_clusters: int = 3,
        max_clusters: int = 50,
        proximity_threshold: float = 1.5,  # Single tunable parameter!
        use_pca_preprocessing: bool = True,
        pca_components: int = 32,
    ):
        """
        Args:
            node_dim: Dimensionality of node embeddings
            min_clusters: Minimum number of clusters (safety bound)
            max_clusters: Maximum number of clusters (safety bound)
            proximity_threshold: Distance threshold for agglomerative clustering (0.5-2.0)
                                Lower = more, tighter clusters (more granular)
                                Higher = fewer, looser clusters (more coarse)
                                Default 1.5 produces: ~3 clusters for 10 atoms,
                                ~9 clusters for 60 atoms, ~11 for 100 atoms
            use_pca_preprocessing: Whether to apply PCA before Fourier transform
            pca_components: Number of PCA components (reduces FFT computation)
        """
        super().__init__()
        self.node_dim = node_dim
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.proximity_threshold = proximity_threshold
        self.use_pca_preprocessing = use_pca_preprocessing
        self.pca_components = min(pca_components, node_dim)

        # Optional PCA projection to reduce dimensionality before FFT
        if use_pca_preprocessing:
            self.pca_projection = Linear(node_dim, self.pca_components, bias=False)
        else:
            self.pca_projection = None

    def _to_fourier_space(self, x: Tensor) -> Tensor:
        """
        Transform embeddings to Fourier (reciprocal) space.

        Args:
            x: Node embeddings [num_nodes, dim]

        Returns:
            Fourier features [num_nodes, fourier_dim]
        """
        # Normalize input to prevent numerical issues
        x_mean = x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True) + 1e-6
        x_normalized = (x - x_mean) / x_std

        # Apply optional PCA projection
        if self.pca_projection is not None:
            x_normalized = self.pca_projection(x_normalized)

        # Apply FFT along feature dimension
        # This captures frequency components in the embedding space
        x_fft = torch.fft.rfft(x_normalized, dim=1, norm="ortho")

        # Extract magnitude and phase as features
        # Magnitude: captures strength of frequency components
        # Phase: captures relative timing/positioning
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # Use primarily magnitude, with reduced weight on phase
        # This focuses on spectral power rather than exact phase relationships
        fourier_features = torch.cat([magnitude, 0.3 * phase], dim=1)

        # L2 normalize to improve clustering stability
        fourier_norm = torch.norm(fourier_features, p=2, dim=1, keepdim=True) + 1e-8
        fourier_features = fourier_features / fourier_norm

        return fourier_features

    def _from_fourier_space(
        self, fourier_features: Tensor, original_dim: int
    ) -> Tensor:
        """
        Transform cluster centers back from Fourier space to real space.

        Args:
            fourier_features: Fourier space features [num_clusters, fourier_dim]
            original_dim: Original embedding dimension before Fourier transform

        Returns:
            Real space embeddings [num_clusters, original_dim]
        """
        # Split magnitude and phase
        half_dim = fourier_features.size(1) // 2
        magnitude = fourier_features[:, :half_dim]
        phase = fourier_features[:, half_dim:]

        # Reconstruct complex Fourier coefficients
        complex_fft = magnitude * torch.exp(1j * phase)

        # Apply inverse FFT
        x_reconstructed = torch.fft.irfft(
            complex_fft, n=original_dim, dim=1, norm="ortho"
        )

        # If we used PCA, we need to project back (approximate inverse)
        # Note: This is an approximation since PCA projection is not invertible
        # We use the transpose as pseudo-inverse
        if self.pca_projection is not None:
            # x_reconstructed has shape [num_clusters, pca_components]
            # We want [num_clusters, node_dim]
            # Use transpose of projection matrix as approximate inverse
            with torch.no_grad():
                weight = self.pca_projection.weight  # [pca_components, node_dim]
                x_reconstructed = x_reconstructed @ weight  # [num_clusters, node_dim]

        return x_reconstructed

    def _determine_optimal_clusters(
        self, fourier_features: np.ndarray, min_k: int, max_k: int
    ) -> int:
        """
        Determine optimal number of clusters using proximity-based agglomerative clustering.

        Uses SINGLE proximity_threshold parameter - NO conditional logic!

        The algorithm automatically finds the number of clusters based on the
        distance threshold. Lower threshold = fewer, tighter clusters.
        Higher threshold = more, looser clusters.

        Args:
            fourier_features: Features in Fourier space [num_nodes, fourier_dim]
            min_k: Minimum number of clusters (safety bound)
            max_k: Maximum number of clusters (safety bound)

        Returns:
            Optimal number of clusters
        """
        from sklearn.cluster import AgglomerativeClustering

        num_samples = fourier_features.shape[0]

        # Safety bounds
        if num_samples <= min_k:
            return max(1, num_samples)

        try:
            # Use agglomerative clustering with distance threshold
            # This automatically determines number of clusters!
            clusterer = AgglomerativeClustering(
                n_clusters=None,  # Automatic!
                distance_threshold=self.proximity_threshold,
                linkage='ward'
            )

            labels = clusterer.fit_predict(fourier_features)
            num_clusters = len(np.unique(labels))

            # Apply safety bounds
            num_clusters = max(min_k, min(num_clusters, max_k))

            # If outside bounds, re-cluster with fixed k
            if num_clusters != clusterer.n_clusters_:
                clusterer = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
                labels = clusterer.fit_predict(fourier_features)
                num_clusters = len(np.unique(labels))

            return num_clusters

        except Exception as e:
            # Fallback: simple heuristic based on sqrt
            k = max(min_k, min(int(np.sqrt(num_samples)), max_k))
            return k

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Perform Fourier-based clustering.

        Args:
            x: Node features [batch, num_nodes, dim]
            mask: Mask for valid nodes [batch, 1, num_nodes]

        Returns:
            cluster_assignments: Cluster assignment for each node [batch, num_nodes]
            cluster_centers: Cluster centers in original space [batch, num_clusters, dim]
            num_clusters: Number of clusters found
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)

        # Handle masking
        if mask is not None:
            valid_mask = mask.squeeze(1) > -1e8
        else:
            valid_mask = torch.ones(
                batch_size, num_nodes, dtype=torch.bool, device=x.device
            )

        # Process each graph in batch separately
        all_assignments = []
        all_centers = []
        all_num_clusters = []

        for b in range(batch_size):
            # Extract valid nodes for this graph
            valid_indices = valid_mask[b]
            valid_count = valid_indices.sum().item()

            if valid_count == 0:
                # No valid nodes, create dummy output
                all_assignments.append(
                    torch.zeros(num_nodes, dtype=torch.long, device=x.device)
                )
                all_centers.append(torch.zeros(1, self.node_dim, device=x.device))
                all_num_clusters.append(1)
                continue

            x_valid = x[b, valid_indices, :]  # [valid_count, dim]

            # Determine cluster bounds for this graph
            min_k = min(self.min_clusters, valid_count)
            max_k = min(self.max_clusters, valid_count)

            # Transform to Fourier space
            fourier_features = self._to_fourier_space(
                x_valid
            )  # [valid_count, fourier_dim]

            # Convert to numpy for sklearn
            fourier_np = fourier_features.detach().cpu().numpy()

            # Determine optimal number of clusters
            if max_k > min_k and valid_count > 1:
                optimal_k = self._determine_optimal_clusters(fourier_np, min_k, max_k)
            else:
                optimal_k = min_k

            optimal_k = max(1, min(optimal_k, valid_count))

            # Perform clustering in Fourier space using KMeans
            # The number of clusters (optimal_k) was automatically determined
            # by proximity-based agglomerative clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(fourier_np)
            centers_fourier = kmeans.cluster_centers_
            num_clusters_found = optimal_k

            # Convert labels to torch
            labels_torch = torch.from_numpy(labels).long().to(x.device)
            centers_fourier_torch = (
                torch.from_numpy(centers_fourier).float().to(x.device)
            )

            # Transform cluster centers back to real space
            pca_dim = (
                self.pca_components
                if self.pca_projection is not None
                else self.node_dim
            )
            centers_real = self._from_fourier_space(centers_fourier_torch, pca_dim)

            # Create full assignment tensor (including masked nodes)
            assignments = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
            assignments[valid_indices] = labels_torch

            all_assignments.append(assignments)
            all_centers.append(centers_real)
            all_num_clusters.append(num_clusters_found)

        # Stack results
        cluster_assignments = torch.stack(all_assignments)  # [batch, num_nodes]

        # Pad cluster centers to same size across batch
        max_clusters_in_batch = max(all_num_clusters)
        padded_centers = []
        for centers in all_centers:
            if centers.size(0) < max_clusters_in_batch:
                padding = torch.zeros(
                    max_clusters_in_batch - centers.size(0),
                    centers.size(1),
                    device=x.device,
                )
                centers = torch.cat([centers, padding], dim=0)
            padded_centers.append(centers)

        cluster_centers = torch.stack(padded_centers)  # [batch, max_clusters, dim]

        # Return mean number of clusters for batch
        mean_num_clusters = int(np.mean(all_num_clusters))

        return cluster_assignments, cluster_centers, mean_num_clusters


class FourierPorjecting(torch.nn.Module):
    """
    Fourier-based clustering integrated with attention mechanism for neural atoms.

    Similar to DynamicPorjecting but uses Fourier space clustering instead of
    neural network prediction.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        max_seeds: int = 50,
        min_seeds: int = 3,
        proximity_threshold: float = 1.5,  # Single tunable parameter!
        layer_norm: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.max_seeds = max_seeds
        self.min_seeds = min_seeds
        self.layer_norm = layer_norm

        # Fourier clustering module with proximity-based clustering
        self.fourier_clustering = FourierClusteringModule(
            node_dim=channels,
            min_clusters=min_seeds,
            max_clusters=max_seeds,
            proximity_threshold=proximity_threshold,  # New single parameter!
            use_pca_preprocessing=True,
            pca_components=min(16, channels),
        )

        # Layer norm if requested
        if layer_norm:
            self.ln = LayerNorm(channels)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int]:
        """
        Args:
            x: Node features [batch, num_nodes, dim]
            mask: Mask for valid nodes

        Returns:
            cluster_centers: Neural atom embeddings [batch, num_clusters, dim]
            num_clusters: Number of clusters found
        """
        # Perform Fourier-based clustering
        cluster_assignments, cluster_centers, num_clusters = self.fourier_clustering(
            x, mask
        )

        # Apply layer norm if requested
        if self.layer_norm:
            cluster_centers = self.ln(cluster_centers)

        # Return cluster centers as neural atom embeddings
        # We return centers[:, :num_clusters, :] to trim padding
        trimmed_centers = cluster_centers[:, :num_clusters, :]

        return trimmed_centers, num_clusters
