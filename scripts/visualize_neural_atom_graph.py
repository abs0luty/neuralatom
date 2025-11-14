"""
Visualize the Neural Atom Graph - the reduced graph after clustering.

Shows:
1. Original molecular graph with atoms colored by cluster
2. Neural atom graph where clusters become supernodes
3. Side-by-side comparison as 3D rotating GIF
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx

# Add parent directory to path
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from core.fourier_clustering import FourierPorjecting


def create_molecular_graph(size, seed=None):
    """Create a molecular graph structure."""
    if seed is not None:
        np.random.seed(seed)

    G = nx.Graph()
    for i in range(size):
        G.add_node(i)

    # Create connected graph
    for i in range(size - 1):
        G.add_edge(i, i + 1)

    # Add branches
    num_branches = max(1, size // 10)
    for _ in range(num_branches):
        u, v = np.random.randint(0, size, 2)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    # Add rings
    if size > 10:
        num_cycles = size // 20
        for _ in range(num_cycles):
            cycle_size = np.random.randint(5, 8)
            start = np.random.randint(0, max(1, size - cycle_size))
            for i in range(cycle_size):
                u = start + i
                v = start + ((i + 1) % cycle_size)
                if u < size and v < size:
                    G.add_edge(u, v)

    return G


def create_neural_atom_graph(molecular_graph, cluster_labels):
    """
    Create neural atom graph from clustering.

    Neural atoms (supernodes) are connected if any of their constituent atoms
    are connected in the original graph.
    """
    num_clusters = len(np.unique(cluster_labels))

    # Create neural atom graph
    NA_graph = nx.Graph()
    for i in range(num_clusters):
        NA_graph.add_node(i)

    # Add edges between neural atoms if their atoms are connected
    for u, v in molecular_graph.edges():
        cluster_u = cluster_labels[u]
        cluster_v = cluster_labels[v]
        if cluster_u != cluster_v and not NA_graph.has_edge(cluster_u, cluster_v):
            NA_graph.add_edge(cluster_u, cluster_v)

    return NA_graph


def visualize_before_after_gif(molecule_size, output_path, num_frames=36, duration=100):
    """
    Create side-by-side GIF:
    - Left: Original molecular graph with atoms colored by cluster
    - Right: Neural atom graph (reduced graph)
    """
    print(f"\nCreating before/after GIF for {molecule_size} atoms...")

    # Create molecular graph
    G_mol = create_molecular_graph(molecule_size, seed=molecule_size)

    # Create features
    base = torch.randn(1, 1, 64)
    noise = torch.randn(1, molecule_size, 64) * 0.3
    features = base + noise

    # Add periodic patterns
    if molecule_size > 10:
        period = molecule_size // 4
        for i in range(molecule_size):
            phase = 2 * np.pi * i / period
            features[:, i, :16] += 0.5 * torch.sin(torch.tensor(phase)) * torch.randn(1, 16)

    # Pad
    max_size = 200
    padded = torch.zeros(1, max_size, 64)
    padded[:, :molecule_size, :] = features

    mask = torch.zeros(1, 1, max_size) - 1e9
    mask[:, :, :molecule_size] = 0

    # Perform clustering with proximity-based algorithm
    fourier_layer = FourierPorjecting(
        channels=64,
        num_heads=2,
        max_seeds=50,
        min_seeds=3,
        proximity_threshold=1.5,  # Automatically determines cluster count
        layer_norm=False,
    )

    with torch.no_grad():
        cluster_assignments, cluster_centers, num_clusters = \
            fourier_layer.fourier_clustering(padded, mask)
        cluster_labels = cluster_assignments[0, :molecule_size].cpu().numpy()

    # Create neural atom graph
    G_neural = create_neural_atom_graph(G_mol, cluster_labels)

    # 3D layouts
    pos_mol = nx.spring_layout(G_mol, seed=42, k=1/np.sqrt(molecule_size), dim=3)
    coords_mol = np.array([pos_mol[i] for i in range(molecule_size)])

    pos_neural = nx.spring_layout(G_neural, seed=42, k=1/np.sqrt(num_clusters), dim=3)
    coords_neural = np.array([pos_neural[i] for i in range(num_clusters)])

    # Generate colors
    if num_clusters <= 20:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(num_clusters)]
    else:
        cmap = plt.get_cmap('hsv')
        colors = [cmap(i / num_clusters * 0.95) for i in range(num_clusters)]

    node_colors_mol = [colors[cluster_labels[i]] for i in range(molecule_size)]

    # Create figure
    fig = plt.figure(figsize=(16, 7))

    def update_view(frame):
        fig.clear()
        angle = frame * (360 / num_frames)

        # Left: Molecular graph
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.view_init(elev=20, azim=angle)

        # Plot molecular edges
        for u, v in G_mol.edges():
            xs = coords_mol[[u, v], 0]
            ys = coords_mol[[u, v], 1]
            zs = coords_mol[[u, v], 2]
            ax1.plot(xs, ys, zs, color='gray', alpha=0.3, linewidth=1)

        # Plot molecular nodes
        ax1.scatter(
            coords_mol[:, 0],
            coords_mol[:, 1],
            coords_mol[:, 2],
            c=node_colors_mol,
            s=100,
            edgecolors='black',
            linewidths=1,
            alpha=0.8
        )

        ax1.set_title(
            f'Original Molecular Graph\n{molecule_size} atoms in {num_clusters} clusters',
            fontsize=12,
            fontweight='bold',
            pad=15
        )
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])

        # Right: Neural atom graph
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.view_init(elev=20, azim=angle)

        # Plot neural atom edges (thicker)
        for u, v in G_neural.edges():
            xs = coords_neural[[u, v], 0]
            ys = coords_neural[[u, v], 1]
            zs = coords_neural[[u, v], 2]
            ax2.plot(xs, ys, zs, color='darkblue', alpha=0.6, linewidth=3)

        # Plot neural atoms (larger nodes)
        ax2.scatter(
            coords_neural[:, 0],
            coords_neural[:, 1],
            coords_neural[:, 2],
            c=[colors[i] for i in range(num_clusters)],
            s=500,  # Much larger
            edgecolors='black',
            linewidths=2.5,
            alpha=0.9
        )

        ax2.set_title(
            f'Neural Atom Graph (Reduced)\n{num_clusters} neural atoms ({G_neural.number_of_edges()} edges)',
            fontsize=12,
            fontweight='bold',
            pad=15
        )
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])

        plt.suptitle(
            'Molecular Graph → Neural Atom Graph',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )

    # Create animation
    anim = FuncAnimation(fig, update_view, frames=num_frames, interval=duration)

    # Save
    writer = PillowWriter(fps=1000//duration)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"✓ Saved: {output_path}")
    print(f"  Reduced {molecule_size} atoms → {num_clusters} neural atoms")
    print(f"  Edges: {G_mol.number_of_edges()} → {G_neural.number_of_edges()}")

    return num_clusters


# Generate visualizations
print("=" * 80)
print("Neural Atom Graph Visualization")
print("=" * 80)

test_sizes = [25, 50, 100]

for size in test_sizes:
    output = base_dir / f"visualizations/gifs_3d/neural_atom_graph_{size}atoms.gif"
    visualize_before_after_gif(size, str(output), num_frames=36, duration=100)

print("\n" + "=" * 80)
print("✓ All neural atom graph visualizations created!")
print("=" * 80)
