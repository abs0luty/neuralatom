"""
Visualize individual molecular graphs with cluster assignments.

Shows how dynamic clustering groups atoms into neural atoms.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
import importlib.util
import os

from molecule_viz_utils import (
    prepare_atom_visuals,
    nodes_by_cluster,
    cluster_composition_label,
    element_label_color,
)

# Load implementation
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

neural_atom_path = os.path.join(os.path.dirname(__file__), '2D_Molecule/graphgps/layer/neural_atom.py')
neural_atom = load_module("neural_atom", neural_atom_path)

DynamicPorjecting = neural_atom.DynamicPorjecting
Porjecting = neural_atom.Porjecting


def create_molecular_graph(size, seed=None):
    """Create a random molecular graph structure."""
    if seed is not None:
        np.random.seed(seed)

    G = nx.Graph()

    # Add nodes
    for i in range(size):
        G.add_node(i)

    # Create a connected graph (molecular backbone)
    # Start with a path (chain)
    for i in range(size - 1):
        G.add_edge(i, i + 1)

    # Add some branches (like side chains)
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


def get_cluster_assignments(features, mask, layer):
    """Get cluster assignments from attention matrix."""
    with torch.no_grad():
        output, attn, num_clusters, ratio = layer(features, mask=mask)

    # attn shape: [batch*heads, num_clusters, num_nodes]
    # Average over heads
    batch_size = features.size(0)
    num_heads = 2
    attn_reshaped = attn.view(batch_size, num_heads, num_clusters, -1)
    attn_mean = attn_reshaped.mean(dim=1)[0]  # [num_clusters, num_nodes]

    # Get cluster assignment for each node (argmax)
    node_to_cluster = attn_mean.argmax(dim=0).numpy()

    # Get attention strength for visualization
    cluster_strengths = attn_mean.max(dim=0)[0].numpy()

    return node_to_cluster, cluster_strengths, num_clusters


def visualize_molecule_clustering(graph_size, ax_dyn, ax_stat, title_prefix=""):
    """Visualize a molecule with dynamic and static clustering."""

    # Create molecular graph
    G = create_molecular_graph(graph_size, seed=graph_size)
    atom_labels, node_sizes = prepare_atom_visuals(G, seed=graph_size, size_scale=65.0)
    node_sizes = np.array(node_sizes)

    # Create features
    features = torch.randn(1, graph_size, 64)
    # Add structure
    base = torch.randn(1, 1, 64)
    noise = torch.randn(1, graph_size, 64) * 0.3
    features = base + noise

    # Pad to max size
    max_size = 200
    padded = torch.zeros(1, max_size, 64)
    padded[:, :graph_size, :] = features

    mask = torch.zeros(1, 1, max_size) - 1e9
    mask[:, :, :graph_size] = 0

    # Dynamic clustering
    dynamic_layer = DynamicPorjecting(
        channels=64,
        num_heads=2,
        max_seeds=50,
        min_seeds=3,
        layer_norm=False
    )

    node_to_cluster_dyn, strengths_dyn, num_clusters_dyn = \
        get_cluster_assignments(padded, mask, dynamic_layer)

    # Static clustering
    min_static_clusters = 8
    max_static_clusters = 10
    if graph_size >= min_static_clusters:
        num_static_clusters = min(graph_size, max_static_clusters)
    else:
        num_static_clusters = graph_size

    static_layer = Porjecting(
        channels=64,
        num_heads=2,
        num_seeds=num_static_clusters,
        layer_norm=False
    )

    with torch.no_grad():
        output_stat, attn_stat = static_layer(padded, mask=mask)

    attn_stat_mean = attn_stat.view(1, 2, num_static_clusters, -1).mean(dim=1)[0]
    node_to_cluster_stat = attn_stat_mean.argmax(dim=0).numpy()
    strengths_stat = attn_stat_mean.max(dim=0)[0].numpy()

    # Create 3D layout
    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(graph_size), dim=3)

    def plot_clusters(ax, node_to_cluster, strengths, total_clusters, title):
        coords = np.array([pos[i] for i in range(graph_size)])
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

        # Generate distinct colors for any number of clusters
        # Use HSV colormap which can generate arbitrary number of distinct colors
        if total_clusters <= 20:
            cmap = plt.cm.get_cmap('tab20', total_clusters)
            palette = cmap(np.arange(total_clusters))
        else:
            # For more clusters, use HSV which provides better color distribution
            cmap = plt.cm.get_cmap('hsv', total_clusters)
            palette = cmap(np.linspace(0, 0.95, total_clusters))  # Avoid wrapping to red

        node_colors = palette[node_to_cluster[:graph_size]]

        for u, v in G.edges():
            ax.plot([xs[u], xs[v]],
                    [ys[u], ys[v]],
                    [zs[u], zs[v]],
                    color='gray',
                    alpha=0.35,
                    linewidth=1.2)

        strengths_clip = strengths[:graph_size] if strengths is not None else np.ones(graph_size)
        if strengths_clip.max() - strengths_clip.min() < 1e-6:
            size_scale = np.ones_like(strengths_clip)
        else:
            size_scale = (strengths_clip - strengths_clip.min()) / (
                strengths_clip.max() - strengths_clip.min()
            )
            size_scale = 0.4 * size_scale + 0.8

        scatter_sizes = node_sizes * size_scale

        ax.scatter(xs, ys, zs,
                   c=node_colors,
                   s=scatter_sizes,
                   edgecolors='black',
                   linewidths=0.8,
                   alpha=0.95)

        for node in range(graph_size):
            elem = G.nodes[node]['element']
            ax.text(xs[node], ys[node], zs[node],
                    atom_labels[node],
                    fontsize=7,
                    color=element_label_color(elem),
                    ha='center',
                    va='center',
                    zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

        cluster_groups = nodes_by_cluster(node_to_cluster[:graph_size], graph_size)
        for cluster_id, members in cluster_groups.items():
            coords_cluster = coords[members]
            center = coords_cluster.mean(axis=0)
            composition = cluster_composition_label(atom_labels, members, max_chars=28)
            ax.text(center[0], center[1], center[2],
                    composition,
                    fontsize=7,
                    ha='center',
                    va='center',
                    color='navy',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.6, edgecolor='navy', linewidth=0.7))

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=18, azim=35)

    plot_clusters(
        ax_dyn,
        node_to_cluster_dyn,
        strengths_dyn,
        num_clusters_dyn,
        f'{title_prefix}Dynamic: {num_clusters_dyn} clusters\n({graph_size} atoms)'
    )

    plot_clusters(
        ax_stat,
        node_to_cluster_stat,
        strengths_stat,
        num_static_clusters,
        f'{title_prefix}Static: {num_static_clusters} clusters\n({graph_size} atoms)'
    )

    return num_clusters_dyn, num_static_clusters


print("Creating molecular graph visualizations...")

# Create figure
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25)

# Visualize molecules of different sizes
molecule_sizes = [10, 25, 50, 75, 100, 150]
cluster_counts = []

for idx, size in enumerate(molecule_sizes):
    row = idx // 2
    col = (idx % 2) * 2

    ax_dyn = fig.add_subplot(gs[row, col], projection='3d')
    ax_stat = fig.add_subplot(gs[row, col + 1], projection='3d')

    num_dyn, num_static = visualize_molecule_clustering(
        size, ax_dyn, ax_stat, title_prefix=f"{chr(65+idx)}. ")
    cluster_counts.append((size, num_dyn, num_static))

# Add title
fig.suptitle('Neural Atom Clustering: Molecular Graph Examples\nDynamic vs Static Clustering',
             fontsize=16, fontweight='bold', y=0.98)

# Add legend explaining the visualization
legend_text = "Each color represents a different cluster (neural atom)\n"
legend_text += "Dynamic adapts cluster count to molecule size\n"
legend_text += "Static uses 8-10 clusters (or as many atoms as available)"

fig.text(0.5, 0.01, legend_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_path = 'molecular_graphs_clustering.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved to: {output_path}")

# Print cluster count summary
print("\nCluster Count Summary:")
print("Size | Dynamic | Static | Ratio")
print("-" * 40)
for size, dyn_clusters, static_clusters in cluster_counts:
    ratio = dyn_clusters / size
    print(f"{size:4d} | {dyn_clusters:7d} | {static_clusters:6d} | {ratio:.3f}")

print(f"\n✓ Visualization complete!")
