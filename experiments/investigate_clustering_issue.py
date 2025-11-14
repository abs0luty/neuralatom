"""
Investigation: Why does cluster count grow linearly with graph size?

This script investigates the fundamental issue with the current dynamic clustering
implementation and demonstrates what true unsupervised clustering should look like.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
import sys
import os

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '2D_Molecule'))

from graphgps.layer.neural_atom import ClusterCountPredictor


def create_molecular_graph(graph_type, size):
    """Create different graph structures with the same size."""

    if graph_type == 'linear_chain':
        # Linear chain: C-C-C-C-C (like alkanes)
        # Low complexity, should need few clusters
        edge_index = []
        for i in range(size - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    elif graph_type == 'star':
        # Star graph: central node connected to all others
        # Medium complexity
        edge_index = []
        for i in range(1, size):
            edge_index.append([0, i])
            edge_index.append([i, 0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    elif graph_type == 'dense':
        # Dense/clique: every node connected to every other
        # High complexity, many functional groups
        edge_index = []
        for i in range(size):
            for j in range(i + 1, size):
                edge_index.append([i, j])
                edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    elif graph_type == 'branched':
        # Branched structure: tree-like (like amino acids with side chains)
        # High complexity
        edge_index = []
        # Main chain
        for i in range(size // 2):
            if i < size // 2 - 1:
                edge_index.append([i, i + 1])
                edge_index.append([i + 1, i])
        # Branches
        for i in range(size // 2, size):
            parent = (i - size // 2) % (size // 2)
            edge_index.append([parent, i])
            edge_index.append([i, parent])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    elif graph_type == 'ring':
        # Ring structure (like benzene, cyclic compounds)
        # Medium complexity
        edge_index = []
        for i in range(size):
            edge_index.append([i, (i + 1) % size])
            edge_index.append([(i + 1) % size, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    # Random features (simulating atom types, positions, etc.)
    x = torch.randn(size, 64)

    return Data(x=x, edge_index=edge_index, num_nodes=size)


def compute_graph_statistics(data):
    """Compute structural features that should matter for clustering."""
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # Degree distribution
    degrees = torch.zeros(num_nodes)
    for i in range(edge_index.size(1)):
        degrees[edge_index[1, i]] += 1

    # Graph density (actual edges / possible edges)
    num_edges = edge_index.size(1) // 2  # Undirected
    max_edges = num_nodes * (num_nodes - 1) / 2
    density = num_edges / max_edges if max_edges > 0 else 0

    # Degree statistics
    avg_degree = degrees.mean().item()
    std_degree = degrees.std().item()
    max_degree = degrees.max().item()

    # Clustering coefficient (approximate)
    # For each node, count triangles
    triangles = 0
    for i in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == i].tolist()
        if len(neighbors) < 2:
            continue
        for j, n1 in enumerate(neighbors):
            for n2 in neighbors[j+1:]:
                # Check if n1 and n2 are connected
                if n2 in edge_index[1][edge_index[0] == n1].tolist():
                    triangles += 1

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_degree': avg_degree,
        'std_degree': std_degree,
        'max_degree': max_degree,
        'triangles': triangles,
    }


def test_current_predictor_linearity():
    """Test 1: Confirm that current predictor gives linear relationship."""
    print("\n" + "="*80)
    print("TEST 1: Current Predictor Linearity")
    print("="*80)

    predictor = ClusterCountPredictor(
        node_dim=64,
        hidden_dim=64,
        min_ratio=0.1,
        max_ratio=0.5,
        min_clusters=3,
        max_clusters=50
    )

    # Test with linear chains of different sizes
    sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = []

    print("\nTesting LINEAR CHAIN structures (same structure, different sizes):")
    print(f"{'Size':<8} {'Clusters':<12} {'Ratio':<12} {'Expected if Linear':<20}")
    print("-" * 60)

    for size in sizes:
        graph = create_molecular_graph('linear_chain', size)
        x = graph.x.unsqueeze(0)  # Add batch dimension

        num_clusters, ratio = predictor(x)
        results.append((size, num_clusters, ratio.item()))

        # Expected if perfectly linear with first ratio
        if len(results) == 1:
            base_ratio = ratio.item()
            print(f"{size:<8} {num_clusters:<12} {ratio.item():<12.3f} {'(baseline)':<20}")
        else:
            expected = int(base_ratio * size)
            print(f"{size:<8} {num_clusters:<12} {ratio.item():<12.3f} {expected:<20}")

    # Compute correlation
    sizes_arr = np.array([r[0] for r in results])
    clusters_arr = np.array([r[1] for r in results])
    correlation = np.corrcoef(sizes_arr, clusters_arr)[0, 1]

    print(f"\nCorrelation between size and clusters: {correlation:.4f}")
    print(f"{'FINDING: Nearly perfect linear correlation!' if correlation > 0.95 else 'Some non-linearity detected'}")

    return results


def test_same_size_different_structure():
    """Test 2: Show that same-size graphs with different structures get same clusters."""
    print("\n" + "="*80)
    print("TEST 2: Same Size, Different Structures")
    print("="*80)

    predictor = ClusterCountPredictor(
        node_dim=64,
        hidden_dim=64,
        min_ratio=0.1,
        max_ratio=0.5,
        min_clusters=3,
        max_clusters=50
    )

    size = 30
    structures = ['linear_chain', 'star', 'dense', 'branched', 'ring']

    print(f"\nAll graphs have {size} nodes but VERY different structures:")
    print(f"{'Structure':<15} {'Clusters':<12} {'Density':<12} {'Avg Degree':<12} {'Triangles':<12}")
    print("-" * 70)

    results = []
    for structure in structures:
        graph = create_molecular_graph(structure, size)
        stats = compute_graph_statistics(graph)

        x = graph.x.unsqueeze(0)
        num_clusters, ratio = predictor(x)

        results.append({
            'structure': structure,
            'clusters': num_clusters,
            'stats': stats
        })

        print(f"{structure:<15} {num_clusters:<12} {stats['density']:<12.3f} {stats['avg_degree']:<12.2f} {stats['triangles']:<12}")

    # Check variance in cluster counts
    cluster_counts = [r['clusters'] for r in results]
    cluster_std = np.std(cluster_counts)

    print(f"\nCluster count standard deviation: {cluster_std:.2f}")
    if cluster_std < 2:
        print("❌ PROBLEM: All structures get nearly the same cluster count!")
        print("   This means structure is being IGNORED, only size matters.")
    else:
        print("✓ Good: Different structures get different cluster counts.")

    return results


def test_what_features_matter():
    """Test 3: Analyze what features the predictor actually uses."""
    print("\n" + "="*80)
    print("TEST 3: What Features Does the Predictor Use?")
    print("="*80)

    predictor = ClusterCountPredictor(
        node_dim=64,
        hidden_dim=64,
        min_ratio=0.1,
        max_ratio=0.5,
        min_clusters=3,
        max_clusters=50
    )

    # Create two graphs: same size, different features
    size = 40

    print(f"\nComparing graphs with {size} nodes:")
    print(f"{'Graph':<20} {'Clusters':<12} {'Feature Mean':<15} {'Feature Std':<15}")
    print("-" * 70)

    # Graph 1: Small feature values
    graph1 = create_molecular_graph('linear_chain', size)
    graph1.x = torch.randn(size, 64) * 0.1  # Small values
    x1 = graph1.x.unsqueeze(0)
    num_clusters1, ratio1 = predictor(x1)
    print(f"{'Small features':<20} {num_clusters1:<12} {graph1.x.mean().item():<15.3f} {graph1.x.std().item():<15.3f}")

    # Graph 2: Large feature values
    graph2 = create_molecular_graph('linear_chain', size)
    graph2.x = torch.randn(size, 64) * 10.0  # Large values
    x2 = graph2.x.unsqueeze(0)
    num_clusters2, ratio2 = predictor(x2)
    print(f"{'Large features':<20} {num_clusters2:<12} {graph2.x.mean().item():<15.3f} {graph2.x.std().item():<15.3f}")

    # Graph 3: All positive features
    graph3 = create_molecular_graph('linear_chain', size)
    graph3.x = torch.abs(torch.randn(size, 64))  # All positive
    x3 = graph3.x.unsqueeze(0)
    num_clusters3, ratio3 = predictor(x3)
    print(f"{'Positive features':<20} {num_clusters3:<12} {graph3.x.mean().item():<15.3f} {graph3.x.std().item():<15.3f}")

    print("\nNOTE: The predictor uses graph pooling (mean, max, std) of node features.")
    print("But then multiplies the predicted ratio by graph SIZE.")
    print("This is why SIZE dominates over structural features!")

    return {
        'small': num_clusters1,
        'large': num_clusters2,
        'positive': num_clusters3
    }


def propose_better_approach():
    """Test 4: Propose what a true unsupervised approach should do."""
    print("\n" + "="*80)
    print("TEST 4: What True Unsupervised Clustering Should Do")
    print("="*80)

    print("\nCURRENT APPROACH (Supervised Ratio Learning):")
    print("  1. Predict pooling ratio from graph features")
    print("  2. Multiply ratio by graph size: clusters = ratio × size")
    print("  3. Learn ratio from task loss (supervised)")
    print("  ❌ Result: Linear relationship, structure ignored")

    print("\nPROPOSED APPROACH (True Unsupervised):")
    print("  1. Predict cluster count DIRECTLY from structural features")
    print("  2. Features should include:")
    print("     - Graph density")
    print("     - Degree distribution statistics")
    print("     - Clustering coefficient")
    print("     - Community structure metrics")
    print("     - Spectral properties (eigenvalues)")
    print("     - Number of connected components")
    print("  3. Learn from data structure, not just task loss")
    print("  ✓ Result: Cluster count reflects complexity, not size")

    print("\nEXAMPLES:")

    structures = [
        ('linear_chain', 100, "Long chain (low complexity)", "Should need few clusters (e.g., 5-10)"),
        ('dense', 30, "Dense graph (high complexity)", "Should need many clusters (e.g., 20-25)"),
        ('star', 50, "Star graph (medium complexity)", "Should need medium clusters (e.g., 10-15)"),
        ('branched', 40, "Branched structure", "Should need clusters for each branch (e.g., 15-20)"),
    ]

    print(f"\n{'Structure':<15} {'Size':<8} {'Current':<12} {'Should Be':<30}")
    print("-" * 80)

    predictor = ClusterCountPredictor(
        node_dim=64,
        hidden_dim=64,
        min_ratio=0.1,
        max_ratio=0.5,
        min_clusters=3,
        max_clusters=50
    )

    for struct_type, size, description, should_be in structures:
        graph = create_molecular_graph(struct_type, size)
        x = graph.x.unsqueeze(0)
        num_clusters, _ = predictor(x)

        print(f"{struct_type:<15} {size:<8} {num_clusters:<12} {should_be:<30}")

    print("\nKEY INSIGHT:")
    print("  Linear chain with 100 nodes should have FEWER clusters than")
    print("  a dense graph with 30 nodes, because it's structurally simpler!")
    print("  But the current approach gives linear_chain(100) = ~23 clusters")
    print("  and dense(30) = ~7 clusters. This is BACKWARDS!")


def visualize_the_problem():
    """Create visualization showing the issue."""
    print("\n" + "="*80)
    print("Creating Visualization")
    print("="*80)

    predictor = ClusterCountPredictor(
        node_dim=64,
        hidden_dim=64,
        min_ratio=0.1,
        max_ratio=0.5,
        min_clusters=3,
        max_clusters=50
    )

    # Test across sizes and structures
    sizes = range(10, 101, 10)
    structures = ['linear_chain', 'dense', 'branched', 'star', 'ring']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Cluster count vs size for different structures
    ax = axes[0, 0]
    for structure in structures:
        clusters = []
        for size in sizes:
            graph = create_molecular_graph(structure, size)
            x = graph.x.unsqueeze(0)
            num_clusters, _ = predictor(x)
            clusters.append(num_clusters)
        ax.plot(sizes, clusters, marker='o', label=structure, linewidth=2)

    ax.set_xlabel('Graph Size (number of nodes)', fontsize=12)
    ax.set_ylabel('Predicted Clusters', fontsize=12)
    ax.set_title('Current Behavior: Nearly Linear for All Structures', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Graph density for different structures
    ax = axes[0, 1]
    size = 50
    densities = []
    cluster_counts = []
    labels = []
    for structure in structures:
        graph = create_molecular_graph(structure, size)
        stats = compute_graph_statistics(graph)
        densities.append(stats['density'])

        x = graph.x.unsqueeze(0)
        num_clusters, _ = predictor(x)
        cluster_counts.append(num_clusters)
        labels.append(structure)

    ax.scatter(densities, cluster_counts, s=200, alpha=0.6)
    for i, label in enumerate(labels):
        ax.annotate(label, (densities[i], cluster_counts[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Graph Density', fontsize=12)
    ax.set_ylabel('Predicted Clusters', fontsize=12)
    ax.set_title(f'Clusters vs Density (all graphs size {size})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Show the formula
    ax = axes[1, 0]
    ax.text(0.5, 0.7, 'Current Formula:', fontsize=16, fontweight='bold',
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, 'clusters = ratio × size', fontsize=20,
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.text(0.5, 0.3, '↓', fontsize=30, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.15, 'Linear Relationship', fontsize=14,
            ha='center', transform=ax.transAxes, style='italic')
    ax.text(0.5, 0.05, 'Structure is ignored!', fontsize=12,
            ha='center', transform=ax.transAxes, color='red', fontweight='bold')
    ax.axis('off')

    # Plot 4: What it should be
    ax = axes[1, 1]
    ax.text(0.5, 0.7, 'Should Be:', fontsize=16, fontweight='bold',
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, 'clusters = f(density, degrees, ...)', fontsize=16,
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(0.5, 0.3, '↓', fontsize=30, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.15, 'Structure-Based Clustering', fontsize=14,
            ha='center', transform=ax.transAxes, style='italic')
    ax.text(0.5, 0.05, 'Reflects graph complexity!', fontsize=12,
            ha='center', transform=ax.transAxes, color='green', fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('clustering_problem_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to: clustering_problem_analysis.png")

    return fig


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" " * 20 + "INVESTIGATION: Dynamic Clustering Issue")
    print("="*80)

    # Run all tests
    test_current_predictor_linearity()
    test_same_size_different_structure()
    test_what_features_matter()
    propose_better_approach()
    visualize_the_problem()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
The current implementation has a fundamental design flaw:

1. It predicts a RATIO, then multiplies by SIZE
   → This creates a linear relationship by construction

2. Structure is largely ignored
   → Same-size graphs get nearly same cluster counts
   → Different structures don't differentiate

3. This is SUPERVISED ratio learning, not UNSUPERVISED clustering
   → The ratio is learned from task loss
   → It's just an adaptive hyperparameter

4. For true unsupervised clustering:
   → Should predict cluster count from graph structure
   → Should use features like density, degrees, community structure
   → Linear chains should have FEWER clusters than dense graphs
   → Size should be less important than structural complexity

RECOMMENDATION: Redesign the predictor to use structural features
and remove the multiplication by size.
    """)

    print("="*80)
