"""
Create 3D rotating GIF visualizations for molecular clustering.

This script creates:
1. Rotating 3D GIFs of molecular graphs with cluster coloring
2. Side-by-side Fourier space and real space visualizations
"""

import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3d projection)
import networkx as nx

from molecule_viz_utils import (
    prepare_atom_visuals,
    nodes_by_cluster,
    cluster_composition_label,
    element_label_color,
)


# Load modules
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


print("Loading implementations...")
# Add parent directory to path for imports
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, base_dir)

from core.fourier_clustering import FourierPorjecting, FourierClusteringModule
neural_atom_path = os.path.join(
    base_dir, "2D_Molecule/graphgps/layer/neural_atom.py"
)
neural_atom = load_module("neural_atom", neural_atom_path)

DynamicPorjecting = neural_atom.DynamicPorjecting


def create_molecular_graph(size, seed=None):
    """Create a random molecular graph structure."""
    if seed is not None:
        np.random.seed(seed)

    G = nx.Graph()

    # Add nodes
    for i in range(size):
        G.add_node(i)

    # Create a connected graph (molecular backbone)
    for i in range(size - 1):
        G.add_edge(i, i + 1)

    # Add some branches
    num_branches = max(1, size // 10)
    for _ in range(num_branches):
        node_from = np.random.randint(0, size)
        node_to = np.random.randint(0, size)
        if node_from != node_to and not G.has_edge(node_from, node_to):
            G.add_edge(node_from, node_to)

    # Add some cycles (rings)
    if size > 10:
        num_cycles = size // 20
        for _ in range(num_cycles):
            cycle_size = np.random.randint(5, 8)
            start_node = np.random.randint(0, max(1, size - cycle_size))
            for i in range(cycle_size):
                node_from = start_node + i
                node_to = start_node + ((i + 1) % cycle_size)
                if node_from < size and node_to < size:
                    G.add_edge(node_from, node_to)

    return G


def create_3d_rotating_gif_simple(
    graph_size, output_path, method='fourier', num_frames=36, duration=100
):
    """
    Create a single 3D rotating GIF of molecular clustering.

    Args:
        graph_size: Number of atoms in molecule
        output_path: Path to save GIF
        method: 'fourier' or 'dynamic'
        num_frames: Number of frames in rotation
        duration: Duration between frames in ms
    """
    print(f"\nCreating {method} clustering GIF for {graph_size} atoms...")

    # Create molecular graph
    G = create_molecular_graph(graph_size, seed=graph_size)
    atom_labels, node_sizes = prepare_atom_visuals(G, seed=graph_size, size_scale=75.0)
    node_sizes = np.array(node_sizes)

    # Create features with periodic patterns for better Fourier analysis
    base = torch.randn(1, 1, 64)
    noise = torch.randn(1, graph_size, 64) * 0.3
    features = base + noise

    # Add periodic patterns
    if graph_size > 10:
        period = graph_size // 4
        for i in range(graph_size):
            phase = 2 * np.pi * i / period
            features[:, i, :16] += 0.5 * torch.sin(torch.tensor(phase)) * torch.randn(1, 16)

    # Pad to max size
    max_size = 200
    padded = torch.zeros(1, max_size, 64)
    padded[:, :graph_size, :] = features

    mask = torch.zeros(1, 1, max_size) - 1e9
    mask[:, :, :graph_size] = 0

    # Perform clustering
    if method == 'fourier':
        fourier_layer = FourierPorjecting(
            channels=64,
            num_heads=2,
            max_seeds=50,
            min_seeds=3,
            proximity_threshold=1.5,  # Automatically determines cluster count
            layer_norm=False,
        )
        with torch.no_grad():
            # Get cluster assignments
            cluster_assignments, cluster_centers, num_clusters = \
                fourier_layer.fourier_clustering(padded, mask)
            cluster_labels = cluster_assignments[0, :graph_size].cpu().numpy()
    else:  # dynamic NN
        dynamic_layer = DynamicPorjecting(
            channels=64, num_heads=2, max_seeds=50, min_seeds=3, layer_norm=False
        )
        with torch.no_grad():
            output, attn, num_clusters, ratio = dynamic_layer(padded, mask=mask)
            # Get cluster assignments from attention
            attn_reshaped = attn.view(1, 2, num_clusters, -1)
            attn_mean = attn_reshaped.mean(dim=1)[0]
            cluster_labels = attn_mean.argmax(dim=0)[:graph_size].cpu().numpy()

    # Create 3D layout
    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(graph_size), dim=3)
    coords = np.array([pos[i] for i in range(graph_size)])

    # Generate colors for clusters
    if num_clusters <= 20:
        cmap = plt.cm.get_cmap('tab20', num_clusters)
        colors = cmap(np.arange(num_clusters))
    else:
        cmap = plt.cm.get_cmap('hsv', num_clusters)
        colors = cmap(np.linspace(0, 0.95, num_clusters))

    node_colors = colors[cluster_labels]

    cluster_groups = nodes_by_cluster(cluster_labels, graph_size)
    show_cluster_labels = method != 'fourier'

    # Create figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update_view(frame):
        ax.clear()

        # Rotate view
        angle = frame * (360 / num_frames)
        ax.view_init(elev=20, azim=angle)

        # Plot edges
        for u, v in G.edges():
            xs, ys, zs = coords[[u, v], 0], coords[[u, v], 1], coords[[u, v], 2]
            ax.plot(xs, ys, zs, color='gray', alpha=0.4, linewidth=1.5)

        # Plot nodes
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=node_colors,
            s=node_sizes,
            edgecolors='black',
            linewidths=1.5,
            alpha=0.9
        )

        for idx in range(graph_size):
            elem = G.nodes[idx]['element']
            ax.text(
                coords[idx, 0],
                coords[idx, 1],
                coords[idx, 2],
                atom_labels[idx],
                fontsize=8,
                color=element_label_color(elem),
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, edgecolor='none'),
            )

        if show_cluster_labels:
            for cluster_id, members in cluster_groups.items():
                member_coords = coords[members]
                center = member_coords.mean(axis=0)
                composition = cluster_composition_label(atom_labels, members, max_chars=30)
                ax.text(
                    center[0],
                    center[1],
                    center[2],
                    composition,
                    fontsize=8,
                    ha='center',
                    va='center',
                    color='navy',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.65, edgecolor='navy', linewidth=0.8),
                )

        # Title
        ax.set_title(
            f'{method.capitalize()} Clustering: {num_clusters} clusters\n{graph_size} atoms',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Set axis limits for consistent view
        margin = 0.1
        for dim in range(3):
            ax_min, ax_max = coords[:, dim].min(), coords[:, dim].max()
            ax_range = ax_max - ax_min
            if dim == 0:
                ax.set_xlim(ax_min - margin * ax_range, ax_max + margin * ax_range)
            elif dim == 1:
                ax.set_ylim(ax_min - margin * ax_range, ax_max + margin * ax_range)
            else:
                ax.set_zlim(ax_min - margin * ax_range, ax_max + margin * ax_range)

    # Create animation
    anim = FuncAnimation(fig, update_view, frames=num_frames, interval=duration)

    # Save as GIF
    writer = PillowWriter(fps=1000//duration)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"✓ Saved to: {output_path}")
    return num_clusters


def create_fourier_real_space_gif(
    graph_size, output_path, num_frames=36, duration=100
):
    """
    Create side-by-side 3D GIF showing Fourier space and real space.

    Args:
        graph_size: Number of atoms
        output_path: Path to save GIF
        num_frames: Number of frames
        duration: Duration between frames in ms
    """
    print(f"\nCreating Fourier/Real space dual GIF for {graph_size} atoms...")

    # Create molecular graph
    G = create_molecular_graph(graph_size, seed=graph_size)

    # Create features with periodic patterns
    base = torch.randn(1, 1, 64)
    noise = torch.randn(1, graph_size, 64) * 0.3
    features = base + noise

    if graph_size > 10:
        period = graph_size // 4
        for i in range(graph_size):
            phase = 2 * np.pi * i / period
            features[:, i, :16] += 0.5 * torch.sin(torch.tensor(phase)) * torch.randn(1, 16)

    # Pad
    max_size = 200
    padded = torch.zeros(1, max_size, 64)
    padded[:, :graph_size, :] = features

    mask = torch.zeros(1, 1, max_size) - 1e9
    mask[:, :, :graph_size] = 0

    # Get clustering
    fourier_clustering = FourierClusteringModule(
        node_dim=64,
        min_clusters=3,
        max_clusters=50,
        proximity_threshold=1.5,  # Automatically determines cluster count
        use_pca_preprocessing=True,
        pca_components=16,
    )

    with torch.no_grad():
        # Get valid features
        valid_features = features[0, :graph_size, :]  # [graph_size, 64]

        # Get Fourier features
        fourier_features = fourier_clustering._to_fourier_space(valid_features)

        # Get cluster assignments
        cluster_assignments, cluster_centers, num_clusters = \
            fourier_clustering(padded, mask)
        cluster_labels = cluster_assignments[0, :graph_size].cpu().numpy()

    # Convert to numpy for visualization
    fourier_np = fourier_features.cpu().numpy()

    # Use PCA to reduce Fourier features to 3D for visualization
    from sklearn.decomposition import PCA
    pca_fourier = PCA(n_components=3)
    fourier_3d = pca_fourier.fit_transform(fourier_np)

    # Real space layout
    pos_real = nx.spring_layout(G, seed=42, k=1/np.sqrt(graph_size), dim=3)
    coords_real = np.array([pos_real[i] for i in range(graph_size)])

    # Generate cluster colors
    if num_clusters <= 20:
        cmap = plt.cm.get_cmap('tab20', num_clusters)
        colors = cmap(np.arange(num_clusters))
    else:
        cmap = plt.cm.get_cmap('hsv', num_clusters)
        colors = cmap(np.linspace(0, 0.95, num_clusters))

    node_colors = colors[cluster_labels]

    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 8))

    def update_dual_view(frame):
        fig.clear()

        angle = frame * (360 / num_frames)

        # Left plot: Fourier space
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.view_init(elev=20, azim=angle)

        ax1.scatter(
            fourier_3d[:, 0],
            fourier_3d[:, 1],
            fourier_3d[:, 2],
            c=node_colors,
            s=150,
            edgecolors='black',
            linewidths=1.2,
            alpha=0.8
        )

        ax1.set_title(
            f'Fourier (Reciprocal) Space\n{num_clusters} clusters',
            fontsize=13,
            fontweight='bold',
            pad=15
        )
        ax1.set_xlabel('FFT Component 1', fontsize=10)
        ax1.set_ylabel('FFT Component 2', fontsize=10)
        ax1.set_zlabel('FFT Component 3', fontsize=10)

        # Right plot: Real space
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.view_init(elev=20, azim=angle)

        # Plot edges in real space
        for u, v in G.edges():
            xs = coords_real[[u, v], 0]
            ys = coords_real[[u, v], 1]
            zs = coords_real[[u, v], 2]
            ax2.plot(xs, ys, zs, color='gray', alpha=0.4, linewidth=1.5)

        # Plot nodes in real space
        ax2.scatter(
            coords_real[:, 0],
            coords_real[:, 1],
            coords_real[:, 2],
            c=node_colors,
            s=150,
            edgecolors='black',
            linewidths=1.2,
            alpha=0.8
        )

        ax2.set_title(
            f'Real (Molecular) Space\n{graph_size} atoms',
            fontsize=13,
            fontweight='bold',
            pad=15
        )
        ax2.set_xlabel('X', fontsize=10)
        ax2.set_ylabel('Y', fontsize=10)
        ax2.set_zlabel('Z', fontsize=10)

        plt.suptitle(
            'Fourier-Based Clustering: Reciprocal ↔ Real Space Correspondence',
            fontsize=16,
            fontweight='bold',
            y=0.98
        )

    # Create animation
    anim = FuncAnimation(fig, update_dual_view, frames=num_frames, interval=duration)

    # Save as GIF
    writer = PillowWriter(fps=1000//duration)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"✓ Saved to: {output_path}")
    return num_clusters


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("3D Rotating GIF Visualizations for Neural Atom Clustering")
    print("=" * 80)

    # Test sizes
    test_sizes = [25, 50, 100]

    print("\n" + "=" * 80)
    print("Creating Fourier/Real Space Dual GIFs")
    print("=" * 80)

    for size in test_sizes:
        output = os.path.join(base_dir, f"visualizations/gifs_3d/clustering_dual_space_{size}atoms.gif")
        create_fourier_real_space_gif(
            size, output, num_frames=36, duration=100
        )

    print("\n" + "=" * 80)
    print("✓ All 3D GIF visualizations created successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    for size in test_sizes:
        print(f"  - clustering_dual_space_{size}atoms.gif")
