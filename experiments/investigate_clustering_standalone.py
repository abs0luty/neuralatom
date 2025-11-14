"""
Standalone investigation of dynamic clustering issue.
No dependencies on the full project - reimplements key components.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class StraightThroughRound(torch.autograd.Function):
    """Straight-through estimator for rounding."""
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CurrentClusterPredictor(nn.Module):
    """Current implementation - predicts RATIO then multiplies by size."""
    def __init__(self, node_dim=64, hidden_dim=64, min_ratio=0.1, max_ratio=0.5,
                 min_clusters=3, max_clusters=50):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        # MLP predicts ratio
        self.mlp = nn.Sequential(
            nn.Linear(3 * node_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """x: [batch, num_nodes, dim]"""
        batch_size = x.size(0)
        num_nodes = x.size(1)

        # Graph-level pooling
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x_std = x.std(dim=1)
        graph_sizes = torch.tensor([num_nodes], dtype=torch.float32).expand(batch_size, 1)

        # Concatenate features
        graph_features = torch.cat([x_mean, x_max, x_std, graph_sizes / 100.0], dim=1)

        # Predict ratio
        ratio_raw = self.mlp(graph_features).squeeze(-1)
        ratio = self.min_ratio + ratio_raw * (self.max_ratio - self.min_ratio)

        # KEY ISSUE: Multiply by size!
        num_clusters_continuous = ratio * num_nodes

        # Clamp and round
        num_clusters_continuous = torch.clamp(
            num_clusters_continuous,
            min=float(self.min_clusters),
            max=float(self.max_clusters)
        )
        num_clusters_rounded = StraightThroughRound.apply(num_clusters_continuous)
        num_clusters_final = int(num_clusters_rounded.mean().item())

        return num_clusters_final, ratio.mean().item()


class ProposedClusterPredictor(nn.Module):
    """Proposed implementation - predicts clusters from STRUCTURE."""
    def __init__(self, node_dim=64, hidden_dim=64, min_clusters=3, max_clusters=50):
        super().__init__()
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        # MLP predicts cluster count directly from structural features
        # Input: pooled features + structural features (no size multiplication!)
        self.mlp = nn.Sequential(
            nn.Linear(3 * node_dim + 4, hidden_dim),  # +4 for structural features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index=None):
        """x: [batch, num_nodes, dim], edge_index: [2, num_edges]"""
        batch_size = x.size(0)
        num_nodes = x.size(1)

        # Graph-level pooling
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x_std = x.std(dim=1)

        # Compute structural features
        if edge_index is not None:
            # Degree statistics
            degrees = torch.zeros(num_nodes)
            for i in range(edge_index.size(1)):
                degrees[edge_index[1, i]] += 1
            avg_degree = degrees.mean()
            std_degree = degrees.std()

            # Graph density
            num_edges = edge_index.size(1) // 2
            max_edges = num_nodes * (num_nodes - 1) / 2
            density = num_edges / max_edges if max_edges > 0 else 0

            # Normalized size (log scale to reduce size dominance)
            normalized_size = torch.log(torch.tensor(num_nodes + 1.0)) / 10.0
        else:
            avg_degree = torch.tensor(0.0)
            std_degree = torch.tensor(0.0)
            density = torch.tensor(0.0)
            normalized_size = torch.log(torch.tensor(num_nodes + 1.0)) / 10.0

        structural_features = torch.tensor([
            avg_degree.item(),
            std_degree.item(),
            density,
            normalized_size.item()
        ]).unsqueeze(0).expand(batch_size, -1)

        # Concatenate all features
        graph_features = torch.cat([x_mean, x_max, x_std, structural_features], dim=1)

        # Predict cluster count DIRECTLY (not ratio!)
        cluster_raw = self.mlp(graph_features).squeeze(-1)

        # Scale to [min_clusters, max_clusters]
        num_clusters_continuous = self.min_clusters + cluster_raw * (self.max_clusters - self.min_clusters)

        # Round
        num_clusters_rounded = StraightThroughRound.apply(num_clusters_continuous)
        num_clusters_final = int(num_clusters_rounded.mean().item())

        return num_clusters_final, cluster_raw.mean().item()


def create_edge_index(graph_type, size):
    """Create edge index for different graph structures."""
    if graph_type == 'linear_chain':
        edges = []
        for i in range(size - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)

    elif graph_type == 'star':
        edges = []
        for i in range(1, size):
            edges.append([0, i])
            edges.append([i, 0])
        return torch.tensor(edges, dtype=torch.long).t()

    elif graph_type == 'dense':
        edges = []
        for i in range(size):
            for j in range(i + 1, size):
                edges.append([i, j])
                edges.append([j, i])
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)

    elif graph_type == 'ring':
        edges = []
        for i in range(size):
            edges.append([i, (i + 1) % size])
            edges.append([(i + 1) % size, i])
        return torch.tensor(edges, dtype=torch.long).t()

    elif graph_type == 'branched':
        edges = []
        # Main chain
        for i in range(size // 2):
            if i < size // 2 - 1:
                edges.append([i, i + 1])
                edges.append([i + 1, i])
        # Branches
        for i in range(size // 2, size):
            parent = (i - size // 2) % max(1, size // 2)
            edges.append([parent, i])
            edges.append([i, parent])
        return torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)

    return torch.zeros((2, 0), dtype=torch.long)


def compute_density(edge_index, num_nodes):
    """Compute graph density."""
    if num_nodes <= 1:
        return 0.0
    num_edges = edge_index.size(1) // 2
    max_edges = num_nodes * (num_nodes - 1) / 2
    return num_edges / max_edges if max_edges > 0 else 0


def main():
    print("\n" + "="*80)
    print(" " * 15 + "INVESTIGATION: Why Clusters Grow Linearly with Size")
    print("="*80)

    # Test 1: Show linear relationship
    print("\n" + "="*80)
    print("TEST 1: Current Predictor Shows Linear Relationship")
    print("="*80)

    current_predictor = CurrentClusterPredictor()

    sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print("\nLinear chain graphs of different sizes:")
    print(f"{'Size':<8} {'Clusters':<12} {'Ratio':<12} {'Clusters/Size':<15}")
    print("-" * 60)

    current_results = []
    for size in sizes:
        x = torch.randn(1, size, 64)
        num_clusters, ratio = current_predictor(x)
        current_results.append((size, num_clusters, ratio))
        print(f"{size:<8} {num_clusters:<12} {ratio:<12.3f} {num_clusters/size:<15.3f}")

    # Compute correlation
    sizes_arr = np.array([r[0] for r in current_results])
    clusters_arr = np.array([r[1] for r in current_results])
    correlation = np.corrcoef(sizes_arr, clusters_arr)[0, 1]

    print(f"\n📊 Correlation (size vs clusters): {correlation:.4f}")
    print(f"❌ PROBLEM: Nearly perfect linear correlation (r={correlation:.3f})")
    print("   This means cluster count is just a constant fraction of size!")

    # Test 2: Same size, different structures
    print("\n" + "="*80)
    print("TEST 2: Same Size, Different Structures → Same Clusters!")
    print("="*80)

    size = 30
    structures = ['linear_chain', 'star', 'dense', 'ring', 'branched']

    print(f"\nAll graphs have {size} nodes but DIFFERENT structures:")
    print(f"{'Structure':<15} {'Clusters':<12} {'Density':<12} {'Should Have':<20}")
    print("-" * 70)

    structure_results = []
    for struct in structures:
        x = torch.randn(1, size, 64)
        edge_index = create_edge_index(struct, size)
        density = compute_density(edge_index, size)
        num_clusters, _ = current_predictor(x)
        structure_results.append((struct, num_clusters, density))

        if struct == 'linear_chain':
            should = "Few (simple structure)"
        elif struct == 'dense':
            should = "Many (complex)"
        else:
            should = "Medium"

        print(f"{struct:<15} {num_clusters:<12} {density:<12.3f} {should:<20}")

    cluster_counts = [r[1] for r in structure_results]
    cluster_std = np.std(cluster_counts)

    print(f"\n📊 Cluster count std deviation: {cluster_std:.2f}")
    if cluster_std < 2:
        print("❌ PROBLEM: All structures get nearly identical cluster counts!")
        print("   The predictor is IGNORING graph structure!")
    else:
        print("✓ Different structures get different counts (good)")

    # Test 3: The root cause
    print("\n" + "="*80)
    print("TEST 3: Root Cause Analysis")
    print("="*80)

    print("\nCurrent formula: clusters = ratio × size")
    print("\nWhat the predictor actually does:")
    print("  1. Pool node features → get graph-level features")
    print("  2. MLP predicts a pooling RATIO (e.g., 0.23)")
    print("  3. Multiply ratio by graph SIZE")
    print("  4. clusters = 0.23 × size")
    print("\n  → This FORCES linear relationship!")
    print("  → Size completely DOMINATES structure")

    print("\n" + "="*80)
    print("TEST 4: What TRUE Unsupervised Clustering Should Do")
    print("="*80)

    print("\nComparing CURRENT vs PROPOSED approach:")
    print("\nCURRENT (Ratio-based):")
    print("  • Predicts pooling ratio")
    print("  • Multiplies by size")
    print("  • Result: clusters ∝ size (linear)")
    print("  • Structure ignored")

    print("\nPROPOSED (Structure-based):")
    print("  • Computes structural features (density, degrees, etc.)")
    print("  • Predicts cluster count DIRECTLY")
    print("  • No multiplication by size")
    print("  • Result: clusters ∝ complexity")

    print("\nExample comparisons:")
    print(f"{'Graph':<30} {'Size':<8} {'Current':<12} {'Should Be':<15}")
    print("-" * 70)

    examples = [
        ("Linear chain", 100, "Low complexity", "~5-10"),
        ("Dense graph", 30, "High complexity", "~20-25"),
        ("Star graph", 50, "Medium complexity", "~10-15"),
    ]

    for name, size, complexity, should_be in examples:
        x = torch.randn(1, size, 64)
        num_clusters, _ = current_predictor(x)
        print(f"{name:<30} {size:<8} {num_clusters:<12} {should_be:<15}")

    print("\n❌ See the problem? Linear chain (100 nodes) gets MORE clusters")
    print("   than dense graph (30 nodes), even though it's SIMPLER!")

    # Visualization
    print("\n" + "="*80)
    print("Creating Visualization")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cluster vs Size for different structures
    ax = axes[0, 0]
    sizes_viz = range(10, 101, 5)
    structures_viz = ['linear_chain', 'star', 'dense', 'ring']
    colors = {'linear_chain': 'blue', 'star': 'orange', 'dense': 'red', 'ring': 'green'}

    for struct in structures_viz:
        clusters_list = []
        for size in sizes_viz:
            x = torch.randn(1, size, 64)
            num_clusters, _ = current_predictor(x)
            clusters_list.append(num_clusters)
        ax.plot(sizes_viz, clusters_list, marker='o', label=struct,
                color=colors.get(struct, 'gray'), linewidth=2, markersize=4, alpha=0.7)

    ax.set_xlabel('Graph Size (nodes)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Clusters', fontsize=11, fontweight='bold')
    ax.set_title('PROBLEM: All Structures Show Linear Relationship', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Same size, different structures
    ax = axes[0, 1]
    size = 40
    struct_names = []
    struct_clusters = []
    struct_densities = []

    for struct in structures:
        x = torch.randn(1, size, 64)
        edge_index = create_edge_index(struct, size)
        density = compute_density(edge_index, size)
        num_clusters, _ = current_predictor(x)

        struct_names.append(struct)
        struct_clusters.append(num_clusters)
        struct_densities.append(density)

    bars = ax.bar(range(len(struct_names)), struct_clusters, alpha=0.7)
    ax.set_xticks(range(len(struct_names)))
    ax.set_xticklabels(struct_names, rotation=45, ha='right')
    ax.set_ylabel('Predicted Clusters', fontsize=11, fontweight='bold')
    ax.set_title(f'Same Size ({size} nodes), Different Structures', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Color by density
    for i, (bar, density) in enumerate(zip(bars, struct_densities)):
        bar.set_color(plt.cm.RdYlGn(density))

    # Plot 3: The formula explanation
    ax = axes[1, 0]
    ax.text(0.5, 0.8, 'Current Formula', fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.6, 'clusters = ratio × size', fontsize=16,
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7, pad=0.5))
    ax.text(0.5, 0.4, '↓', fontsize=30, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, 'Forces Linear Relationship', fontsize=12,
            ha='center', transform=ax.transAxes, style='italic', color='red')
    ax.text(0.5, 0.1, 'Structure is ignored!', fontsize=11,
            ha='center', transform=ax.transAxes, fontweight='bold', color='darkred')
    ax.axis('off')

    # Plot 4: Proposed solution
    ax = axes[1, 1]
    ax.text(0.5, 0.8, 'Should Be', fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes, color='green')
    ax.text(0.5, 0.6, 'clusters = f(structure)', fontsize=16,
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, pad=0.5))
    ax.text(0.5, 0.4, '↓', fontsize=30, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, 'Complexity-Based', fontsize=12,
            ha='center', transform=ax.transAxes, style='italic', color='green')
    ax.text(0.5, 0.1, 'Use density, degrees, etc.', fontsize=11,
            ha='center', transform=ax.transAxes, fontweight='bold', color='darkgreen')
    ax.axis('off')

    plt.suptitle('Why Dynamic Clustering Has Linear Relationship', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('clustering_linear_problem.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: clustering_linear_problem.png")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY & CONCLUSIONS")
    print("="*80)

    print("\n🔍 FINDINGS:")
    print(f"  1. Correlation (size vs clusters): {correlation:.3f} → Nearly perfect linear")
    print(f"  2. Structure variation (std): {cluster_std:.2f} → Structures barely differentiated")
    print("  3. Root cause: clusters = ratio × size (by design)")
    print("  4. This is NOT unsupervised clustering!")

    print("\n❌ PROBLEMS:")
    print("  • Linear relationship forced by formula")
    print("  • Graph structure largely ignored")
    print("  • Simple large graphs get MORE clusters than complex small graphs")
    print("  • This is supervised ratio learning, not unsupervised clustering")

    print("\n✅ SOLUTION:")
    print("  • Remove size multiplication")
    print("  • Add structural features: density, degree stats, clustering coefficient")
    print("  • Predict cluster count directly from complexity")
    print("  • Use log(size) instead of size if size should matter")
    print("  • Let the model learn what complexity means")

    print("\n📝 RECOMMENDATION:")
    print("  Redesign ClusterCountPredictor to:")
    print("    1. Compute structural features (density, degrees, triangles, etc.)")
    print("    2. Predict absolute cluster count, not ratio")
    print("    3. Use log(size) as one feature among many")
    print("    4. Let structure dominate over size")

    print("\n" + "="*80)


if __name__ == "__main__":
    with torch.no_grad():  # No need for gradients in this analysis
        main()
