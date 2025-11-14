"""
Visualize and compare Fourier-based clustering with existing approaches.

This script demonstrates:
1. Fourier-based clustering using spectral features
2. Comparison with neural network-based dynamic clustering
3. Analysis of cluster quality and adaptivity
"""

import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams["font.size"] = 10


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

from core.fourier_clustering import FourierPorjecting
neural_atom_path = os.path.join(
    base_dir, "2D_Molecule/graphgps/layer/neural_atom.py"
)
neural_atom = load_module("neural_atom", neural_atom_path)

DynamicPorjecting = neural_atom.DynamicPorjecting
Porjecting = neural_atom.Porjecting


# =============================================================================
# Generate Synthetic Molecular Graphs
# =============================================================================


class SyntheticMolecularDataset:
    """Generate synthetic molecular graphs with realistic properties."""

    def __init__(self, n_graphs=100, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.n_graphs = n_graphs

        # Generate graph sizes following a realistic distribution
        sizes = []
        for _ in range(n_graphs):
            if np.random.random() < 0.6:  # Small molecules (60%)
                size = int(np.random.normal(25, 8))
            elif np.random.random() < 0.3:  # Medium molecules (30%)
                size = int(np.random.normal(60, 15))
            else:  # Large molecules (10%)
                size = int(np.random.normal(120, 30))
            sizes.append(max(5, min(200, size)))

        self.graph_sizes = sorted(sizes)

        # Generate features with structure
        self.graphs = []
        for size in self.graph_sizes:
            # Add structure to simulate chemical similarity within molecule
            base_pattern = torch.randn(1, 1, 64)
            noise = torch.randn(1, size, 64) * 0.3
            node_features = base_pattern + noise

            # Add some periodic patterns (to make Fourier clustering more effective)
            # Simulate repeating structural motifs in molecules
            if size > 10:
                period = size // 4
                for i in range(size):
                    phase = 2 * np.pi * i / period
                    node_features[:, i, :16] += 0.5 * torch.sin(
                        torch.tensor(phase)
                    ) * torch.randn(1, 16)

            # Pad to max size
            max_size = max(self.graph_sizes)
            padded_features = torch.zeros(1, max_size, 64)
            padded_features[:, :size, :] = node_features

            mask = torch.zeros(1, 1, max_size) - 1e9
            mask[:, :, :size] = 0

            self.graphs.append(
                {"features": padded_features, "mask": mask, "true_size": size}
            )

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


print("\nGenerating synthetic molecular dataset...")
dataset = SyntheticMolecularDataset(n_graphs=100)
print(f"Created {len(dataset)} synthetic molecular graphs")
print(f"Size range: {min(dataset.graph_sizes)} - {max(dataset.graph_sizes)} atoms")
print(f"Mean size: {np.mean(dataset.graph_sizes):.1f} atoms")


# =============================================================================
# Test Different Clustering Approaches
# =============================================================================

print("\n" + "=" * 80)
print("Testing Clustering Approaches")
print("=" * 80)

# Initialize models
print("\nInitializing clustering models...")
fourier_proximity = FourierPorjecting(
    channels=64,
    num_heads=2,
    max_seeds=50,
    min_seeds=3,
    proximity_threshold=1.5,  # Automatically determines cluster count
    layer_norm=False,
)

dynamic_layer = DynamicPorjecting(
    channels=64, num_heads=2, max_seeds=50, min_seeds=3, layer_norm=False
)

static_layer = Porjecting(
    channels=64,
    num_heads=2,
    num_seeds=13,
    layer_norm=False,
)

# Test on all graphs
print("\nRunning clustering on all graphs...")
results = {
    'fourier_proximity': [],
    'dynamic_nn': [],
    'static': [],
}

for idx, graph in enumerate(dataset):
    features = graph["features"]
    mask = graph["mask"]
    size = graph["true_size"]

    # Fourier clustering with proximity-based algorithm
    with torch.no_grad():
        centers_fourier, num_clusters_fourier = fourier_proximity(features, mask=mask)

    # Dynamic NN clustering
    with torch.no_grad():
        output_dyn, _, num_clusters_dyn, ratio_dyn = dynamic_layer(features, mask=mask)

    # Static clustering
    with torch.no_grad():
        output_stat, _ = static_layer(features, mask=mask)
        num_clusters_stat = 13

    results['fourier_proximity'].append({
        'size': size,
        'clusters': num_clusters_fourier,
        'ratio': num_clusters_fourier / size
    })

    results['dynamic_nn'].append({
        'size': size,
        'clusters': num_clusters_dyn,
        'ratio': ratio_dyn.item()
    })

    results['static'].append({
        'size': size,
        'clusters': num_clusters_stat,
        'ratio': num_clusters_stat / size
    })

    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{len(dataset)} graphs...")

print("✓ Clustering complete")


# =============================================================================
# Create Comprehensive Visualizations
# =============================================================================

print("\n" + "=" * 80)
print("Creating Visualizations")
print("=" * 80)

fig = plt.figure(figsize=(22, 14))
gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# Color palette
colors = sns.color_palette("husl", 5)
color_fourier_sil = colors[0]
color_fourier_elbow = colors[1]
color_dynamic = colors[2]
color_static = colors[3]


# =============================================================================
# Plot 1: Cluster Count vs Graph Size - Main Comparison
# =============================================================================

ax1 = fig.add_subplot(gs[0, :2])

for method_name, color, marker, label in [
    ('fourier_kmeans_sil', color_fourier_sil, 'o', 'Fourier (Silhouette)'),
    ('fourier_kmeans_elbow', color_fourier_elbow, 's', 'Fourier (Elbow)'),
    ('dynamic_nn', color_dynamic, '^', 'Dynamic NN'),
    ('static', color_static, 'x', 'Static'),
]:
    sizes = [r['size'] for r in results[method_name]]
    clusters = [r['clusters'] for r in results[method_name]]

    ax1.scatter(
        sizes,
        clusters,
        alpha=0.6,
        s=60,
        label=label,
        color=color,
        marker=marker,
        edgecolors='black',
        linewidth=0.5,
    )

    # Fit trend line (except for static)
    if method_name != 'static':
        z = np.polyfit(sizes, clusters, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(sizes), max(sizes), 100)
        ax1.plot(x_smooth, p(x_smooth), "--", color=color, alpha=0.7, linewidth=2)
    else:
        ax1.axhline(y=13, color=color, linestyle="--", alpha=0.7, linewidth=2)

ax1.set_xlabel("Graph Size (Number of Nodes)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Number of Clusters", fontsize=12, fontweight="bold")
ax1.set_title(
    "Clustering Adaptivity: Fourier vs Neural Network vs Static",
    fontsize=14,
    fontweight="bold",
)
ax1.legend(fontsize=10, framealpha=0.9, loc='upper left')
ax1.grid(True, alpha=0.3)


# =============================================================================
# Plot 2: Cluster Ratio Distribution
# =============================================================================

ax2 = fig.add_subplot(gs[0, 2:])

ratios_data = []
labels_data = []
colors_data = []

for method_name, color, label in [
    ('fourier_kmeans_sil', color_fourier_sil, 'Fourier (Sil)'),
    ('fourier_kmeans_elbow', color_fourier_elbow, 'Fourier (Elbow)'),
    ('dynamic_nn', color_dynamic, 'Dynamic NN'),
    ('static', color_static, 'Static'),
]:
    ratios = [r['ratio'] for r in results[method_name]]
    ratios_data.append(ratios)
    labels_data.append(label)
    colors_data.append(color)

bp = ax2.boxplot(
    ratios_data,
    labels=labels_data,
    patch_artist=True,
    widths=0.6,
)

for patch, color in zip(bp["boxes"], colors_data):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor("black")

for element in ["whiskers", "fliers", "means", "medians", "caps"]:
    plt.setp(bp[element], color="black", linewidth=1.5)

ax2.axhline(y=0.2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.2)')
ax2.set_ylabel("Pooling Ratio (Clusters/Nodes)", fontsize=11, fontweight="bold")
ax2.set_title("Ratio Distribution Comparison", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')


# =============================================================================
# Plot 3: Cluster Count by Size Category
# =============================================================================

ax3 = fig.add_subplot(gs[1, :2])

bins = [0, 30, 60, 100, 200]
bin_labels = ["Small\n(5-30)", "Medium\n(30-60)", "Large\n(60-100)", "XLarge\n(100+)"]

method_means = {name: [[] for _ in range(len(bins) - 1)] for name in results.keys()}

for method_name in results.keys():
    for r in results[method_name]:
        size = r['size']
        clusters = r['clusters']
        for j in range(len(bins) - 1):
            if bins[j] <= size < bins[j + 1]:
                method_means[method_name][j].append(clusters)
                break

x_pos = np.arange(len(bin_labels))
width = 0.2

for idx, (method_name, color, label) in enumerate([
    ('fourier_kmeans_sil', color_fourier_sil, 'Fourier (Sil)'),
    ('fourier_kmeans_elbow', color_fourier_elbow, 'Fourier (Elbow)'),
    ('dynamic_nn', color_dynamic, 'Dynamic NN'),
    ('static', color_static, 'Static'),
]):
    means = [np.mean(b) if b else 0 for b in method_means[method_name]]
    offset = (idx - 1.5) * width
    ax3.bar(
        x_pos + offset,
        means,
        width,
        label=label,
        color=color,
        alpha=0.7,
        edgecolor="black",
    )

ax3.set_xlabel("Graph Size Category", fontsize=11, fontweight="bold")
ax3.set_ylabel("Mean Number of Clusters", fontsize=11, fontweight="bold")
ax3.set_title("Cluster Count by Molecule Size", fontsize=12, fontweight="bold")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(bin_labels, fontsize=9)
ax3.legend(fontsize=9, ncol=2)
ax3.grid(True, alpha=0.3, axis="y")


# =============================================================================
# Plot 4: Method Comparison - Adaptivity Range
# =============================================================================

ax4 = fig.add_subplot(gs[1, 2])

adaptivity_data = []
for method_name in ['fourier_kmeans_sil', 'fourier_kmeans_elbow', 'dynamic_nn', 'static']:
    clusters = [r['clusters'] for r in results[method_name]]
    cluster_range = max(clusters) - min(clusters)
    adaptivity_data.append(cluster_range)

bars = ax4.bar(
    ['Fourier\n(Sil)', 'Fourier\n(Elbow)', 'Dynamic\nNN', 'Static'],
    adaptivity_data,
    color=[color_fourier_sil, color_fourier_elbow, color_dynamic, color_static],
    alpha=0.7,
    edgecolor='black',
    linewidth=1.5,
)

ax4.set_ylabel("Cluster Count Range", fontsize=11, fontweight="bold")
ax4.set_title("Adaptivity Comparison", fontsize=12, fontweight="bold")
ax4.grid(True, alpha=0.3, axis="y")


# =============================================================================
# Plot 5: Efficiency Analysis
# =============================================================================

ax5 = fig.add_subplot(gs[1, 3])

efficiency_means = []
for method_name in ['fourier_kmeans_sil', 'fourier_kmeans_elbow', 'dynamic_nn', 'static']:
    ratios = [r['ratio'] for r in results[method_name]]
    efficiency_means.append(np.mean(ratios))

bars = ax5.bar(
    ['Fourier\n(Sil)', 'Fourier\n(Elbow)', 'Dynamic\nNN', 'Static'],
    efficiency_means,
    color=[color_fourier_sil, color_fourier_elbow, color_dynamic, color_static],
    alpha=0.7,
    edgecolor='black',
    linewidth=1.5,
)

ax5.axhline(y=0.2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target')
ax5.set_ylabel("Mean Ratio", fontsize=11, fontweight="bold")
ax5.set_title("Mean Efficiency", fontsize=12, fontweight="bold")
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis="y")


# =============================================================================
# Plot 6: Scatter - Size vs Ratio
# =============================================================================

ax6 = fig.add_subplot(gs[2, :2])

for method_name, color, marker, label in [
    ('fourier_kmeans_sil', color_fourier_sil, 'o', 'Fourier (Sil)'),
    ('fourier_kmeans_elbow', color_fourier_elbow, 's', 'Fourier (Elbow)'),
    ('dynamic_nn', color_dynamic, '^', 'Dynamic NN'),
]:
    sizes = [r['size'] for r in results[method_name]]
    ratios = [r['ratio'] for r in results[method_name]]

    ax6.scatter(
        sizes,
        ratios,
        alpha=0.5,
        s=60,
        label=label,
        color=color,
        marker=marker,
        edgecolors='black',
        linewidth=0.5,
    )

ax6.axhline(y=0.2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.2)')
ax6.set_xlabel("Graph Size", fontsize=11, fontweight="bold")
ax6.set_ylabel("Pooling Ratio", fontsize=11, fontweight="bold")
ax6.set_title("Size vs Efficiency", fontsize=12, fontweight="bold")
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)


# =============================================================================
# Plot 7 & 8: Statistical Summary Tables
# =============================================================================

ax7 = fig.add_subplot(gs[2, 2:])
ax7.axis('tight')
ax7.axis('off')

table_data = []
for method_name, label in [
    ('fourier_kmeans_sil', 'Fourier (Silhouette)'),
    ('fourier_kmeans_elbow', 'Fourier (Elbow)'),
    ('dynamic_nn', 'Dynamic NN'),
    ('static', 'Static'),
]:
    clusters = [r['clusters'] for r in results[method_name]]
    ratios = [r['ratio'] for r in results[method_name]]

    table_data.append([
        label,
        f"{np.mean(clusters):.1f}",
        f"{np.std(clusters):.1f}",
        f"{min(clusters)}-{max(clusters)}",
        f"{np.mean(ratios):.3f}",
        f"{np.std(ratios):.3f}",
    ])

table = ax7.table(
    cellText=table_data,
    colLabels=['Method', 'Mean\nClusters', 'Std\nClusters', 'Range', 'Mean\nRatio', 'Std\nRatio'],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code the rows
for i in range(1, 5):
    for j in range(6):
        cell = table[(i, j)]
        cell.set_facecolor([color_fourier_sil, color_fourier_elbow, color_dynamic, color_static][i-1])
        cell.set_alpha(0.3)

ax7.set_title("Statistical Summary", fontsize=12, fontweight="bold", pad=20)


# =============================================================================
# Add Overall Title and Save
# =============================================================================

fig.suptitle(
    "Fourier-Based Clustering vs Neural Network Clustering\n"
    "Unsupervised Spectral Clustering in Reciprocal Space",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

plt.tight_layout()
output_path = os.path.join(base_dir, "visualizations/png_comparisons/fourier_clustering_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n✓ Visualization saved to: {output_path}")


# =============================================================================
# Print Summary
# =============================================================================

print("\n" + "=" * 80)
print("Summary Statistics")
print("=" * 80)

for method_name, label in [
    ('fourier_kmeans_sil', 'Fourier Clustering (Silhouette)'),
    ('fourier_kmeans_elbow', 'Fourier Clustering (Elbow)'),
    ('dynamic_nn', 'Dynamic NN Clustering'),
    ('static', 'Static Clustering'),
]:
    clusters = [r['clusters'] for r in results[method_name]]
    ratios = [r['ratio'] for r in results[method_name]]

    print(f"\n{label}:")
    print(f"  Mean clusters: {np.mean(clusters):.2f} ± {np.std(clusters):.2f}")
    print(f"  Range: {min(clusters)} - {max(clusters)}")
    print(f"  Mean ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    print(f"  Adaptivity range: {max(clusters) - min(clusters)} clusters")

print("\n" + "=" * 80)
print("✓ Analysis Complete!")
print("=" * 80)
print(f"\nVisualization saved to: {output_path}")
