"""
Shared helpers for molecule visualizations.

Provides:
- Element assignment with approximate chemical distributions
- Marker sizes based on Van der Waals radii
- Atom label helpers (C1, C2, ...)
- Cluster/Neural atom labeling utilities
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np

# Approximate Van der Waals radii (Å)
VAN_DER_WAALS_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "other": 1.50,
}

# Used to provide consistent label/text colors
ELEMENT_TEXT_COLORS = {
    "C": "#2c2c2c",
    "N": "#1f4ed8",
    "O": "#b40000",
    "H": "#303030",
    "S": "#7a5d00",
    "P": "#7a3500",
    "F": "#076b48",
    "Cl": "#0b5a29",
    "other": "#1f1f1f",
}

ELEMENT_FILL_COLORS = {
    "C": "#696969",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "H": "#DDDDDD",
    "S": "#FFDB1A",
    "P": "#FF8000",
    "F": "#98FB98",
    "Cl": "#228B22",
    "other": "#7A7A7A",
}

# Rough element distribution for organic molecules
ELEMENT_PROBABILITIES = [
    ("C", 0.55),
    ("H", 0.2),
    ("O", 0.1),
    ("N", 0.07),
    ("S", 0.03),
    ("P", 0.02),
    ("F", 0.02),
    ("Cl", 0.01),
]

_ELEMENT_NAMES = [name for name, _ in ELEMENT_PROBABILITIES]
_ELEMENT_WEIGHTS = np.array([weight for _, weight in ELEMENT_PROBABILITIES], dtype=float)
_ELEMENT_WEIGHTS = _ELEMENT_WEIGHTS / _ELEMENT_WEIGHTS.sum()
_DEFAULT_ELEMENT = "C"


def assign_element_types(count: int, seed: Optional[int] = None) -> List[str]:
    """Sample element types based on a rough organic distribution."""
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    chosen = rng.choice(_ELEMENT_NAMES, size=count, p=_ELEMENT_WEIGHTS)
    return [str(elem) for elem in chosen.tolist()]


def annotate_graph_with_elements(G, seed: Optional[int] = None) -> List[str]:
    """
    Attach element labels to every node in a NetworkX graph (if missing).

    Returns a list of element labels ordered by sorted node index.
    """
    nodes = sorted(G.nodes())
    if not nodes:
        return []

    if any("element" not in G.nodes[node] for node in nodes):
        elements = assign_element_types(len(nodes), seed=seed)
        for node, element in zip(nodes, elements):
            G.nodes[node]["element"] = element

    return [G.nodes[node]["element"] for node in nodes]


def marker_size_for_element(element: str, scale: float = 60.0) -> float:
    """
    Convert Van der Waals radius to a scatter marker size (points^2).

    scale controls visual prominence. Minimum size keeps hydrogens visible.
    """
    radius = VAN_DER_WAALS_RADII.get(element, VAN_DER_WAALS_RADII["other"])
    size = (radius ** 3) * scale  # cube keeps relative separation noticeable
    return max(60.0, size)


def node_marker_sizes(G, scale: float = 60.0) -> List[float]:
    """Return marker sizes for nodes ordered by sorted node index."""
    nodes = sorted(G.nodes())
    return [marker_size_for_element(G.nodes[node].get("element", _DEFAULT_ELEMENT), scale) for node in nodes]


def generate_atom_labels(G) -> Dict[int, str]:
    """Generate C1, C2, O1, ... labels for each node."""
    counts = defaultdict(int)
    labels = {}
    for node in sorted(G.nodes()):
        element = G.nodes[node].get("element", _DEFAULT_ELEMENT)
        counts[element] += 1
        labels[node] = f"{element}{counts[element]}"
    return labels


def nodes_by_cluster(cluster_assignments: Iterable[int], node_count: int) -> Dict[int, List[int]]:
    """Group node indices by cluster id."""
    grouped: Dict[int, List[int]] = defaultdict(list)
    for idx in range(node_count):
        grouped[int(cluster_assignments[idx])].append(idx)
    return grouped


def cluster_composition_label(
    atom_labels: Dict[int, str],
    cluster_nodes: Iterable[int],
    max_chars: Optional[int] = None,
) -> str:
    """Create labels like 'O8C1C2H2H4' by concatenating sorted atom labels."""
    labels = sorted(atom_labels[node] for node in cluster_nodes)
    composition = "".join(labels)
    if max_chars is not None and max_chars > 3 and len(composition) > max_chars:
        return composition[: max_chars - 3] + "..."
    return composition


def prepare_atom_visuals(G, seed: Optional[int] = None, size_scale: float = 60.0):
    """Convenience helper returning both atom labels and marker sizes."""
    annotate_graph_with_elements(G, seed=seed)
    atom_labels = generate_atom_labels(G)
    node_sizes = node_marker_sizes(G, scale=size_scale)
    return atom_labels, node_sizes


def element_label_color(element: str) -> str:
    """Color used for text representing a given element."""
    return ELEMENT_TEXT_COLORS.get(element, "#000000")


def element_fill_color(element: str) -> str:
    """Color used to render atoms based on their element."""
    return ELEMENT_FILL_COLORS.get(element, ELEMENT_FILL_COLORS["other"])
