"""
Comprehensive visualization of dynamic clustering on molecular-like data.

This script:
1. Creates synthetic molecular graphs with realistic properties
2. Tests dynamic vs static clustering
3. Creates visualizations showing adaptivity and performance
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
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 10


# Load our implementation
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


print("Loading neural atom implementation...")
neural_atom_path = os.path.join(
    os.path.dirname(__file__), "2D_Molecule/graphgps/layer/neural_atom.py"
)
neural_atom = load_module("neural_atom", neural_atom_path)

ClusterCountPredictor = neural_atom.ClusterCountPredictor
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
        # Most molecules have 10-100 atoms, with some larger ones
        sizes = []
        for _ in range(n_graphs):
            if np.random.random() < 0.6:  # Small molecules (60%)
                size = int(np.random.normal(25, 8))
            elif np.random.random() < 0.3:  # Medium molecules (30%)
                size = int(np.random.normal(60, 15))
            else:  # Large molecules (10%)
                size = int(np.random.normal(120, 30))
            sizes.append(max(5, min(200, size)))  # Clip to reasonable range

        self.graph_sizes = sorted(sizes)

        # Generate features with some structure
        # (simulating chemical properties)
        self.graphs = []
        for size in self.graph_sizes:
            # Node features: simulate atom types, charges, etc.
            node_features = torch.randn(1, size, 64)

            # Add some structure (simulate chemical similarity within molecule)
            base_pattern = torch.randn(1, 1, 64)
            noise = torch.randn(1, size, 64) * 0.3
            node_features = base_pattern + noise

            # Create mask
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
# Experiment 1: Adaptivity - Cluster Count vs Graph Size
# =============================================================================

print("\n" + "=" * 80)
print("Experiment 1: Testing Adaptivity to Graph Size")
print("=" * 80)

predictor = ClusterCountPredictor(
    node_dim=64,
    hidden_dim=64,
    # min_ratio=0.1,
    # max_ratio=0.4,
    min_clusters=3,
    max_clusters=50,
)

dynamic_layer = DynamicPorjecting(
    channels=64, num_heads=2, max_seeds=50, min_seeds=3, layer_norm=False
)

# Test on all graphs
results_dynamic = []
results_static = []

static_layer = Porjecting(
    channels=64,
    num_heads=2,
    num_seeds=13,  # Fixed at 13 clusters (typical default)
    layer_norm=False,
)

print("Running clustering on all graphs...")
for graph in dataset:
    features = graph["features"]
    mask = graph["mask"]
    size = graph["true_size"]

    # Dynamic clustering
    with torch.no_grad():
        output_dyn, _, num_clusters_dyn, ratio_dyn = dynamic_layer(features, mask=mask)
        if num_clusters_dyn > size:
            raise ValueError(
                f"Dynamic clustering predicted {num_clusters_dyn} clusters "
                f"for a molecule with only {size} atoms."
            )

    # Static clustering
    with torch.no_grad():
        output_stat, _ = static_layer(features, mask=mask)
        num_clusters_stat = 13  # Fixed

    results_dynamic.append(
        {"size": size, "clusters": num_clusters_dyn, "ratio": ratio_dyn.item()}
    )

    results_static.append(
        {"size": size, "clusters": num_clusters_stat, "ratio": num_clusters_stat / size}
    )

print("✓ Clustering complete")


# =============================================================================
# Experiment 2: Learning Experiment
# =============================================================================

print("\n" + "=" * 80)
print("Experiment 2: Training the Predictor")
print("=" * 80)

# Create a new predictor for training
trainable_predictor = ClusterCountPredictor(
    node_dim=64,
    hidden_dim=64,
    # min_ratio=0.1,
    # max_ratio=0.4,
    min_clusters=3,
    max_clusters=50,
)

trainable_layer = DynamicPorjecting(
    channels=64, num_heads=2, max_seeds=50, min_seeds=3, layer_norm=False
)

optimizer = torch.optim.Adam(
    list(trainable_predictor.parameters()) + list(trainable_layer.parameters()), lr=0.01
)

# Training objective: learn to use ~20% of nodes as clusters
target_ratio = 0.2

print(f"Training to achieve target ratio: {target_ratio}")
print("Epoch | Loss   | Mean Ratio | Mean Clusters")
print("-" * 50)

training_history = {"epoch": [], "loss": [], "mean_ratio": [], "mean_clusters": []}

# Use a subset for training
train_indices = list(range(0, len(dataset), 2))  # Every other graph

for epoch in range(100):
    epoch_losses = []
    epoch_ratios = []
    epoch_clusters = []

    for idx in train_indices:
        graph = dataset[idx]
        features = graph["features"]
        mask = graph["mask"]

        optimizer.zero_grad()

        output, _, num_clusters, ratio = trainable_layer(features, mask=mask)

        # Loss: encourage target ratio
        loss = (ratio - target_ratio) ** 2

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_ratios.append(ratio.item())
        epoch_clusters.append(num_clusters)

    mean_loss = np.mean(epoch_losses)
    mean_ratio = np.mean(epoch_ratios)
    mean_clusters = np.mean(epoch_clusters)

    training_history["epoch"].append(epoch)
    training_history["loss"].append(mean_loss)
    training_history["mean_ratio"].append(mean_ratio)
    training_history["mean_clusters"].append(mean_clusters)

    if epoch % 20 == 0:
        print(
            f"  {epoch:3d}  | {mean_loss:.4f} | {mean_ratio:.4f}     | {mean_clusters:.1f}"
        )

print("✓ Training complete")


# =============================================================================
# Create Comprehensive Visualizations
# =============================================================================

print("\n" + "=" * 80)
print("Creating Visualizations")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Color palette
colors = sns.color_palette("husl", 3)
color_dynamic = colors[0]
color_static = colors[1]
color_target = colors[2]


# =============================================================================
# Plot 1: Cluster Count vs Graph Size (Main Result)
# =============================================================================

ax1 = fig.add_subplot(gs[0, :2])

sizes_dyn = [r["size"] for r in results_dynamic]
clusters_dyn = [r["clusters"] for r in results_dynamic]
sizes_stat = [r["size"] for r in results_static]
clusters_stat = [r["clusters"] for r in results_static]

ax1.scatter(
    sizes_dyn,
    clusters_dyn,
    alpha=0.6,
    s=80,
    label="Dynamic Clustering",
    color=color_dynamic,
    edgecolors="black",
    linewidth=0.5,
)
ax1.scatter(
    sizes_stat,
    clusters_stat,
    alpha=0.6,
    s=80,
    marker="s",
    label="Static Clustering",
    color=color_static,
    edgecolors="black",
    linewidth=0.5,
)

# Fit trend lines
z_dyn = np.polyfit(sizes_dyn, clusters_dyn, 2)
p_dyn = np.poly1d(z_dyn)
x_smooth = np.linspace(min(sizes_dyn), max(sizes_dyn), 100)
ax1.plot(x_smooth, p_dyn(x_smooth), "--", color=color_dynamic, alpha=0.8, linewidth=2)

ax1.axhline(y=13, color=color_static, linestyle="--", alpha=0.8, linewidth=2)

ax1.set_xlabel("Graph Size (Number of Nodes)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Number of Clusters", fontsize=12, fontweight="bold")
ax1.set_title(
    "Adaptivity: Dynamic vs Static Clustering", fontsize=14, fontweight="bold"
)
ax1.legend(fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3)


# =============================================================================
# Plot 2: Cluster Ratio Distribution
# =============================================================================

ax2 = fig.add_subplot(gs[0, 2])

ratios_dyn = [r["ratio"] for r in results_dynamic]
ratios_stat = [r["ratio"] for r in results_static]

ax2.hist(
    ratios_dyn,
    bins=20,
    alpha=0.7,
    label="Dynamic",
    color=color_dynamic,
    edgecolor="black",
)
ax2.hist(
    ratios_stat,
    bins=20,
    alpha=0.7,
    label="Static",
    color=color_static,
    edgecolor="black",
)
ax2.axvline(
    x=0.2, color="red", linestyle="--", linewidth=2, label="Typical Target (0.2)"
)

ax2.set_xlabel("Pooling Ratio", fontsize=11, fontweight="bold")
ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
ax2.set_title("Ratio Distribution", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")


# =============================================================================
# Plot 3: Training Loss Curve
# =============================================================================

ax3 = fig.add_subplot(gs[1, 0])

ax3.plot(
    training_history["epoch"],
    training_history["loss"],
    linewidth=2.5,
    color=color_dynamic,
    marker="o",
    markersize=3,
    markevery=10,
)
ax3.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax3.set_ylabel("Loss (MSE)", fontsize=11, fontweight="bold")
ax3.set_title("Training Convergence", fontsize=12, fontweight="bold")
ax3.grid(True, alpha=0.3)
ax3.set_yscale("log")


# =============================================================================
# Plot 4: Learned Ratio Over Training
# =============================================================================

ax4 = fig.add_subplot(gs[1, 1])

ax4.plot(
    training_history["epoch"],
    training_history["mean_ratio"],
    linewidth=2.5,
    color=color_dynamic,
    marker="o",
    markersize=3,
    markevery=10,
)
ax4.axhline(
    y=target_ratio,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Target ({target_ratio})",
)
ax4.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax4.set_ylabel("Mean Predicted Ratio", fontsize=11, fontweight="bold")
ax4.set_title("Learning to Predict Target Ratio", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)


# =============================================================================
# Plot 5: Mean Cluster Count Over Training
# =============================================================================

ax5 = fig.add_subplot(gs[1, 2])

ax5.plot(
    training_history["epoch"],
    training_history["mean_clusters"],
    linewidth=2.5,
    color=color_dynamic,
    marker="o",
    markersize=3,
    markevery=10,
)
ax5.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax5.set_ylabel("Mean Cluster Count", fontsize=11, fontweight="bold")
ax5.set_title("Cluster Count During Training", fontsize=12, fontweight="bold")
ax5.grid(True, alpha=0.3)


# =============================================================================
# Plot 6: Efficiency Analysis - Clusters per Node
# =============================================================================

ax6 = fig.add_subplot(gs[2, 0])

# Calculate efficiency (clusters / nodes)
efficiency_dyn = [clusters_dyn[i] / sizes_dyn[i] for i in range(len(sizes_dyn))]
efficiency_stat = [clusters_stat[i] / sizes_stat[i] for i in range(len(sizes_stat))]

# Group by size bins
bins = [0, 30, 60, 100, 200]
bin_labels = ["Small\n(5-30)", "Medium\n(30-60)", "Large\n(60-100)", "XLarge\n(100+)"]

eff_dyn_binned = [[] for _ in range(len(bins) - 1)]
eff_stat_binned = [[] for _ in range(len(bins) - 1)]

for i, size in enumerate(sizes_dyn):
    for j in range(len(bins) - 1):
        if bins[j] <= size < bins[j + 1]:
            eff_dyn_binned[j].append(efficiency_dyn[i])
            eff_stat_binned[j].append(efficiency_stat[i])
            break

x_pos = np.arange(len(bin_labels))
width = 0.35

means_dyn = [np.mean(b) if b else 0 for b in eff_dyn_binned]
means_stat = [np.mean(b) if b else 0 for b in eff_stat_binned]

ax6.bar(
    x_pos - width / 2,
    means_dyn,
    width,
    label="Dynamic",
    color=color_dynamic,
    alpha=0.7,
    edgecolor="black",
)
ax6.bar(
    x_pos + width / 2,
    means_stat,
    width,
    label="Static",
    color=color_static,
    alpha=0.7,
    edgecolor="black",
)

ax6.set_xlabel("Graph Size Category", fontsize=11, fontweight="bold")
ax6.set_ylabel("Clusters / Nodes Ratio", fontsize=11, fontweight="bold")
ax6.set_title("Efficiency by Graph Size", fontsize=12, fontweight="bold")
ax6.set_xticks(x_pos)
ax6.set_xticklabels(bin_labels, fontsize=9)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis="y")


# =============================================================================
# Plot 7: Size Distribution of Dataset
# =============================================================================

ax7 = fig.add_subplot(gs[2, 1])

ax7.hist(
    dataset.graph_sizes, bins=30, color=color_dynamic, alpha=0.7, edgecolor="black"
)
ax7.set_xlabel("Graph Size (Nodes)", fontsize=11, fontweight="bold")
ax7.set_ylabel("Frequency", fontsize=11, fontweight="bold")
ax7.set_title("Dataset Size Distribution", fontsize=12, fontweight="bold")
ax7.axvline(
    x=np.mean(dataset.graph_sizes),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {np.mean(dataset.graph_sizes):.1f}",
)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis="y")


# =============================================================================
# Plot 8: Statistical Comparison
# =============================================================================

ax8 = fig.add_subplot(gs[2, 2])

# Calculate statistics
stats_data = [clusters_dyn, clusters_stat]

bp = ax8.boxplot(
    stats_data, labels=["Dynamic", "Static"], patch_artist=True, widths=0.6
)

for patch, color in zip(bp["boxes"], [color_dynamic, color_static]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor("black")

for element in ["whiskers", "fliers", "means", "medians", "caps"]:
    plt.setp(bp[element], color="black", linewidth=1.5)

ax8.set_ylabel("Number of Clusters", fontsize=11, fontweight="bold")
ax8.set_title("Cluster Count Distribution", fontsize=12, fontweight="bold")
ax8.grid(True, alpha=0.3, axis="y")

# Add statistics text
dyn_mean = np.mean(clusters_dyn)
dyn_std = np.std(clusters_dyn)
stat_mean = np.mean(clusters_stat)
stat_std = np.std(clusters_stat)

stats_text = f"Dynamic: μ={dyn_mean:.1f}, σ={dyn_std:.1f}\n"
stats_text += f"Static: μ={stat_mean:.1f}, σ={stat_std:.1f}"
ax8.text(
    0.05,
    0.95,
    stats_text,
    transform=ax8.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)


# =============================================================================
# Add Overall Title and Save
# =============================================================================

fig.suptitle(
    "Dynamic Neural Atom Clustering: Comprehensive Analysis on Molecular-Like Graphs",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

plt.tight_layout()
output_path = "dynamic_clustering_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n✓ Visualization saved to: {output_path}")


# =============================================================================
# Print Summary Statistics
# =============================================================================

print("\n" + "=" * 80)
print("Summary Statistics")
print("=" * 80)

print("\nCluster Count Statistics:")
print(f"  Dynamic Clustering:")
print(f"    Mean: {dyn_mean:.2f} clusters")
print(f"    Std:  {dyn_std:.2f} clusters")
print(f"    Min:  {min(clusters_dyn)} clusters")
print(f"    Max:  {max(clusters_dyn)} clusters")

print(f"\n  Static Clustering:")
print(f"    Mean: {stat_mean:.2f} clusters (fixed)")
print(f"    Std:  {stat_std:.2f} clusters")

print(f"\nAdaptivity Metrics:")
dyn_range = max(clusters_dyn) - min(clusters_dyn)
stat_range = max(clusters_stat) - min(clusters_stat)
print(f"  Dynamic range: {dyn_range} clusters")
print(f"  Static range: {stat_range} clusters")
if stat_range > 0:
    print(f"  Dynamic adapts {dyn_range / stat_range:.1f}x more")
else:
    print(f"  Dynamic is fully adaptive, static is fixed!")

print(f"\nEfficiency (Clusters/Nodes ratio):")
print(f"  Dynamic mean: {np.mean(efficiency_dyn):.3f}")
print(f"  Static mean:  {np.mean(efficiency_stat):.3f}")

print(f"\nTraining Results:")
print(f"  Initial loss: {training_history['loss'][0]:.6f}")
print(f"  Final loss:   {training_history['loss'][-1]:.6f}")
print(
    f"  Reduction:    {(1 - training_history['loss'][-1] / training_history['loss'][0]) * 100:.1f}%"
)
print(f"  Initial ratio: {training_history['mean_ratio'][0]:.4f}")
print(f"  Final ratio:   {training_history['mean_ratio'][-1]:.4f}")
print(f"  Target ratio:  {target_ratio:.4f}")
print(f"  Error:         {abs(training_history['mean_ratio'][-1] - target_ratio):.4f}")

print("\n" + "=" * 80)
print("✓ Analysis Complete!")
print("=" * 80)
print(f"\nVisualization saved to: {output_path}")
print("Open the image to see comprehensive analysis of dynamic clustering.")
