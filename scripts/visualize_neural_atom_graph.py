"""
Combined neural atom visualization (single multi-panel GIF).

Layout per frame:
1. Element-colored molecule with Van der Waals-scaled atoms
2. Fourier-space embedding (PCA projection) colored by neural clusters
3. Molecule colored by neural clusters with composition labels
4. Neural atom (supernode) graph with cluster composition labels
"""

import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from core.fourier_clustering import FourierClusteringModule
from molecule_viz_utils import (
    cluster_composition_label,
    element_fill_color,
    element_label_color,
    nodes_by_cluster,
    prepare_atom_visuals,
)


def create_molecular_graph(size: int, seed: int | None = None) -> nx.Graph:
    """Create a molecular-style graph with chains, branches, and rings."""
    if seed is not None:
        np.random.seed(seed)

    G = nx.Graph()
    for i in range(size):
        G.add_node(i)

    # Backbone
    for i in range(size - 1):
        G.add_edge(i, i + 1)

    # Branches
    num_branches = max(1, size // 10)
    for _ in range(num_branches):
        u, v = np.random.randint(0, size, size=2)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    # Rings
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


def create_neural_atom_graph(molecular_graph: nx.Graph, cluster_labels: np.ndarray) -> nx.Graph:
    """Create neural atom graph by collapsing nodes per cluster id."""
    unique_clusters = np.unique(cluster_labels)
    neural = nx.Graph()
    for cid in unique_clusters:
        neural.add_node(int(cid))

    for u, v in molecular_graph.edges():
        cu = int(cluster_labels[u])
        cv = int(cluster_labels[v])
        if cu != cv:
            neural.add_edge(cu, cv)

    return neural


def _format_axis(ax, title: str):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])


def _draw_edges(ax, coords: np.ndarray, edges: Iterable, color="gray", alpha=0.35, linewidth=1.0):
    for u, v in edges:
        seg = coords[[u, v]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, alpha=alpha, linewidth=linewidth)


def _label_position(point: np.ndarray, cloud: np.ndarray, offset_ratio: float = 0.08) -> np.ndarray:
    """Project label slightly away from point toward exterior for readability."""
    point = np.asarray(point)
    center = cloud.mean(axis=0)
    direction = point - center
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = direction / norm

    span = np.max(cloud, axis=0) - np.min(cloud, axis=0)
    max_span = max(float(span.max()), 1e-3)
    return point + direction * (offset_ratio * max_span)


def _add_atom_labels(ax, coords: np.ndarray, labels: Dict[int, str], colors: List[str], fontsize: int = 7):
    for idx, label in labels.items():
        ax.text(
            coords[idx, 0],
            coords[idx, 1],
            coords[idx, 2],
            label,
            fontsize=fontsize,
            color=colors[idx],
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )


def visualize_combined_neural_atom(molecule_size: int, output_path: str, num_frames: int = 48, duration: int = 100):
    """Generate a GIF showing all neural atom views in a single layout."""
    print(f"\nCreating combined visualization for {molecule_size} atoms...")

    # Graph + elements
    G_mol = create_molecular_graph(molecule_size, seed=molecule_size)
    atom_labels, node_sizes = prepare_atom_visuals(G_mol, seed=molecule_size, size_scale=70.0)
    node_sizes = np.array(node_sizes)
    element_colors = np.array([element_fill_color(G_mol.nodes[i]["element"]) for i in range(molecule_size)])
    label_colors = np.array([element_label_color(G_mol.nodes[i]["element"]) for i in range(molecule_size)])

    # Synthetic features
    base = torch.randn(1, 1, 64)
    noise = torch.randn(1, molecule_size, 64) * 0.3
    features = base + noise

    # Add periodic pattern for richer Fourier space
    if molecule_size > 10:
        period = max(4, molecule_size // 4)
        for i in range(molecule_size):
            phase = 2 * np.pi * i / period
            features[:, i, :16] += 0.5 * torch.sin(torch.tensor(phase)) * torch.randn(1, 16)

    # Pad for clustering module
    max_size = 200
    padded = torch.zeros(1, max_size, 64)
    padded[:, :molecule_size, :] = features
    mask = torch.zeros(1, 1, max_size) - 1e9
    mask[:, :, :molecule_size] = 0

    fourier_module = FourierClusteringModule(
        node_dim=64,
        min_clusters=3,
        max_clusters=50,
        proximity_threshold=1.5,
        use_pca_preprocessing=True,
        pca_components=16,
    )

    with torch.no_grad():
        cluster_assignments, _, _ = fourier_module(padded, mask)
        cluster_labels = cluster_assignments[0, :molecule_size].cpu().numpy()
        valid_features = features[0, :molecule_size, :]
        fourier_features = fourier_module._to_fourier_space(valid_features).cpu().numpy()

    # Reduce Fourier features to 3D
    pca_components = min(3, fourier_features.shape[1])
    fourier_coords = PCA(n_components=pca_components).fit_transform(fourier_features)
    if fourier_coords.shape[1] < 3:
        padding = np.zeros((fourier_coords.shape[0], 3 - fourier_coords.shape[1]))
        fourier_coords = np.hstack([fourier_coords, padding])

    # Cluster visuals
    cluster_groups = nodes_by_cluster(cluster_labels, molecule_size)
    num_clusters = len(cluster_groups)
    cmap_name = "tab20" if num_clusters <= 20 else "hsv"
    cmap = plt.colormaps[cmap_name]
    cluster_palette = np.array(cmap(np.linspace(0, 0.95, max(num_clusters, 1))))
    cluster_node_colors = cluster_palette[cluster_labels]

    cluster_compositions = {
        cid: cluster_composition_label(atom_labels, members, max_chars=40)
        for cid, members in cluster_groups.items()
    }

    # Graph layouts
    pos_mol = nx.spring_layout(G_mol, seed=42, k=1 / np.sqrt(max(molecule_size, 1)), dim=3)
    coords_mol = np.array([pos_mol[i] for i in range(molecule_size)])

    G_neural = create_neural_atom_graph(G_mol, cluster_labels)
    pos_neural = nx.spring_layout(G_neural, seed=24, k=1 / np.sqrt(max(num_clusters, 1)), dim=3)
    coords_neural = np.array([pos_neural[i] for i in range(num_clusters)]) if num_clusters > 0 else np.zeros((1, 3))

    neural_sizes = []
    for cid in range(num_clusters):
        members = cluster_groups.get(cid, [])
        if members:
            neural_sizes.append(max(220.0, 0.7 * node_sizes[members].sum()))
        else:
            neural_sizes.append(220.0)
    neural_sizes = np.array(neural_sizes) if neural_sizes else np.array([220.0])

    # Helper for axis view and animations
    fig = plt.figure(figsize=(22, 6.5))

    def update_view(frame: int):
        fig.clf()
        angle = frame * (360 / num_frames)

        # Panel 1: Element-colored molecule
        ax1 = fig.add_subplot(141, projection="3d")
        ax1.view_init(elev=18, azim=angle)
        _draw_edges(ax1, coords_mol, G_mol.edges(), color="gray", alpha=0.3, linewidth=1.0)
        ax1.scatter(
            coords_mol[:, 0],
            coords_mol[:, 1],
            coords_mol[:, 2],
            c=element_colors,
            s=node_sizes,
            edgecolors="black",
            linewidths=1.0,
            alpha=0.9,
        )
        _add_atom_labels(ax1, coords_mol, atom_labels, label_colors, fontsize=7)
        _format_axis(ax1, f"Element View\n({molecule_size} atoms)")

        # Panel 2: Fourier space clustering
        ax2 = fig.add_subplot(142, projection="3d")
        ax2.view_init(elev=18, azim=angle)
        ax2.scatter(
            fourier_coords[:, 0],
            fourier_coords[:, 1],
            fourier_coords[:, 2],
            c=cluster_node_colors,
            s=node_sizes,
            edgecolors="black",
            linewidths=1.0,
            alpha=0.9,
        )
        _add_atom_labels(ax2, fourier_coords, atom_labels, ["black"] * molecule_size, fontsize=6)
        _format_axis(ax2, "Fourier Space (PCA)")

        # Panel 3: Cluster-colored molecule
        ax3 = fig.add_subplot(143, projection="3d")
        ax3.view_init(elev=18, azim=angle)
        _draw_edges(ax3, coords_mol, G_mol.edges(), color="gray", alpha=0.25, linewidth=1.0)
        ax3.scatter(
            coords_mol[:, 0],
            coords_mol[:, 1],
            coords_mol[:, 2],
            c=cluster_node_colors,
            s=node_sizes,
            edgecolors="black",
            linewidths=1.0,
            alpha=0.95,
        )
        cluster_centers = {cid: coords_mol[members].mean(axis=0) for cid, members in cluster_groups.items()}
        for cid, center in cluster_centers.items():
            label_pos = _label_position(center, coords_mol, offset_ratio=0.06)
            ax3.text(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                cluster_compositions.get(cid, f"Cluster {cid}"),
                fontsize=7,
                ha="center",
                va="center",
                color="navy",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.65, edgecolor="navy", linewidth=0.7),
            )
        _format_axis(ax3, f"Clustered Molecule\n({num_clusters} neural atoms)")

        # Panel 4: Neural atom graph
        ax4 = fig.add_subplot(144, projection="3d")
        ax4.view_init(elev=18, azim=angle)
        _draw_edges(ax4, coords_neural, G_neural.edges(), color="darkblue", alpha=0.6, linewidth=2.0)
        ax4.scatter(
            coords_neural[:, 0],
            coords_neural[:, 1],
            coords_neural[:, 2],
            c=[cluster_palette[cid] for cid in range(num_clusters)],
            s=neural_sizes,
            edgecolors="black",
            linewidths=2.0,
            alpha=0.9,
        )
        if num_clusters:
            for cid in range(num_clusters):
                label_pos = _label_position(coords_neural[cid], coords_neural, offset_ratio=0.1)
                ax4.text(
                    label_pos[0],
                    label_pos[1],
                    label_pos[2],
                    cluster_compositions.get(cid, f"Cluster {cid}"),
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.9, edgecolor="black", linewidth=0.8),
                )
        _format_axis(ax4, "Neural Atom Graph")

        fig.suptitle(
            "Neural Atom Pipeline: Molecule → Fourier Space → Clusters → Neural Graph",
            fontsize=14,
            fontweight="bold",
        )

    anim = FuncAnimation(fig, update_view, frames=num_frames, interval=duration)
    writer = PillowWriter(fps=max(1, 1000 // duration))
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"✓ Saved: {output_path}")
    print(f"  Atoms: {molecule_size} → Neural atoms: {num_clusters}")
    print(f"  Clusters: {cluster_compositions}")

    return num_clusters


if __name__ == "__main__":
    print("=" * 80)
    print("Combined Neural Atom Visualizations")
    print("=" * 80)

    output_root = base_dir / "visualizations/gifs_3d"
    output_root.mkdir(parents=True, exist_ok=True)

    for size in [25, 50, 100]:
        output_file = output_root / f"neural_atom_graph_{size}atoms.gif"
        visualize_combined_neural_atom(size, str(output_file))

    print("\n" + "=" * 80)
    print("✓ All combined visualizations created!")
    print("=" * 80)
