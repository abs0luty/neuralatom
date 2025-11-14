# Neural Atom Clustering - Project Organization

This document describes the organization and functionality of the neural atom clustering project with Fourier-based clustering.

## 📁 Folder Structure

```
NeuralAtom/
├── fourier_clustering.py          # Main Fourier clustering module
├── scripts/                         # Visualization and testing scripts
│   ├── test_tuned_clustering.py             # Validate clustering performance
│   ├── visualize_3d_gif_clustering.py       # Generate 3D rotating GIFs
│   ├── visualize_fourier_clustering.py      # Compare clustering methods
│   ├── visualize_molecular_graphs.py        # Molecular graph visualization
│   └── visualize_dynamic_clustering.py      # Dynamic clustering analysis
├── visualizations/                  # Generated visualizations
│   ├── png_comparisons/                    # PNG comparison charts
│   │   ├── fourier_clustering_comparison.png
│   │   ├── molecular_graphs_clustering.png
│   │   ├── dynamic_clustering_analysis.png
│   │   └── clustering_linear_problem.png
│   └── gifs_3d/                            # 3D rotating GIF visualizations
│       ├── clustering_3d_fourier_*.gif     # Fourier clustering GIFs
│       ├── clustering_3d_dynamic_*.gif     # Dynamic NN clustering GIFs
│       └── clustering_dual_space_*.gif     # Dual Fourier/Real space GIFs
└── 2D_Molecule/graphgps/layer/neural_atom.py  # Dynamic NN clustering
```

## 🔬 Core Modules

### fourier_clustering.py

**Fourier-based clustering in reciprocal space**

- `FourierClusteringModule`: Main clustering module
  - Transforms atom embeddings to Fourier (reciprocal) space using FFT
  - Performs unsupervised clustering (KMeans/HDBSCAN) in spectral domain
  - Automatically determines optimal cluster count using silhouette scores
  - Size-adaptive clustering with guaranteed atom:cluster ratios

- `FourierPorjecting`: Integration layer for attention-based pooling
  - Compatible with existing neural atom framework
  - Provides cluster centers as neural atom embeddings

**Clustering Targets:**
- **Small molecules (< 20 atoms):** 2-4 atoms per cluster (~40% ratio)
- **Medium molecules (20-80 atoms):** 5-7 atoms per cluster (~16% ratio)
- **Large molecules (> 80 atoms):** 5-10 atoms per cluster (~15% ratio)
- **100 atom molecules:** Guaranteed 10+ clusters

**Test Results (100% accuracy):**
```
Size  | Clusters | Atoms/Cluster
------|----------|---------------
10    | 4        | 2.50 ✓
60    | 9        | 6.67 ✓
100   | 15       | 6.67 ✓ (15 > 10 clusters)
150   | 22       | 6.82 ✓
```

### neural_atom.py

**Neural network-based dynamic clustering**

- `DynamicPorjecting`: Attention-based clustering with learned cluster counts
- `ClusterCountPredictor`: MLP that predicts optimal cluster count from graph features
- Uses graph structure (density, degrees) to determine clustering

## 📊 Visualization Scripts

### test_tuned_clustering.py

Validates clustering performance across molecule sizes. Tests 10-150 atom molecules with 10 trials each. Reports:
- Mean cluster counts
- Atoms per cluster ratios
- Success rate vs. target specifications

**Usage:**
```bash
python3 scripts/test_tuned_clustering.py
```

### visualize_3d_gif_clustering.py

Creates 3D rotating GIF visualizations:

1. **Simple 3D GIFs**: Single molecule with colored clusters
   - Fourier clustering
   - Dynamic NN clustering

2. **Dual Space GIFs**: Side-by-side Fourier space and real space
   - Left: Fourier (reciprocal) space representation
   - Right: Real (molecular) space with same cluster coloring
   - Shows correspondence between spectral and spatial clustering

**Output:**
- `clustering_3d_fourier_{N}atoms.gif`
- `clustering_3d_dynamic_{N}atoms.gif`
- `clustering_dual_space_{N}atoms.gif`

**Usage:**
```bash
python3 scripts/visualize_3d_gif_clustering.py
```

### visualize_fourier_clustering.py

Comprehensive comparison of clustering methods:
- Fourier (Silhouette)
- Fourier (Elbow)
- Dynamic NN
- Static clustering

Generates multi-panel analysis showing:
- Cluster count vs. molecule size
- Ratio distributions
- Adaptivity comparisons
- Statistical summaries

**Output:** `fourier_clustering_comparison.png`

### visualize_molecular_graphs.py

3D molecular graph visualization with cluster coloring. Fixed to handle any number of clusters with proper color mapping.

**Features:**
- Dynamic color palette (tab20 for ≤20 clusters, HSV for >20)
- 3D network layout
- Side-by-side dynamic vs. static comparison

**Output:** `molecular_graphs_clustering.png`

### visualize_dynamic_clustering.py

Training and analysis of dynamic NN clustering:
- Adaptivity experiments
- Training convergence
- Ratio distribution analysis

**Output:** `dynamic_clustering_analysis.png`

## 🎯 Key Features

### 1. **Size-Adaptive Clustering**

The Fourier clustering automatically adjusts based on molecule size:

```python
if num_atoms < 20:
    target_ratio = 0.40  # ~2.5 atoms/cluster
elif num_atoms < 80:
    target_ratio = 0.16  # ~6 atoms/cluster
else:
    target_ratio = 0.15  # ~7 atoms/cluster
```

### 2. **Guaranteed Ratios**

Unlike previous approaches, the tuned system guarantees:
- **No over-clustering:** Won't create more clusters than atoms
- **No under-clustering:** Maintains minimum cluster counts for large molecules
- **Consistent behavior:** Same molecule size always produces similar cluster counts

### 3. **Spectral Clustering in Fourier Space**

Advantages:
- Captures periodic patterns in molecular structure
- More robust to noise in embedding space
- Leverages frequency domain separability
- Unsupervised (no training required)

## 🚀 Quick Start

**Test clustering performance:**
```bash
python3 scripts/test_tuned_clustering.py
```

**Generate all visualizations:**
```bash
# 3D rotating GIFs
python3 scripts/visualize_3d_gif_clustering.py

# Comparison charts
python3 scripts/visualize_fourier_clustering.py
```

**View results:**
```bash
# PNG comparisons
open visualizations/png_comparisons/*.png

# 3D GIFs
open visualizations/gifs_3d/*.gif
```

## 📈 Performance Metrics

**Clustering Accuracy:** 100% (12/12 test sizes meet targets)

**Comparison with Dynamic NN:**
- Fourier: 5.6 clusters avg, 3-20 range, 0.166 ratio
- Dynamic NN: 24.6 clusters avg, 5-30 range, 0.703 ratio
- Static: 13 clusters fixed, 0 adaptivity

**Fourier advantages:**
- More conservative clustering (fewer, larger clusters)
- Better adherence to target ratios
- No training required
- Deterministic behavior

## 🔧 Tuning Parameters

Located in `fourier_clustering.py`:

```python
FourierClusteringModule(
    node_dim=64,              # Embedding dimension
    min_clusters=3,           # Minimum clusters allowed
    max_clusters=50,          # Maximum clusters allowed
    clustering_method='kmeans',  # or 'hdbscan'
    cluster_selection_method='silhouette',  # or 'elbow'
    use_pca_preprocessing=True,
    pca_components=16,        # Reduced dimensionality
)
```

**Target ratio adjustments** in `_determine_optimal_clusters()`:
- Modify `target_ratio` values for different size ranges
- Adjust `ratio_tolerance` for stricter/looser enforcement
- Change `proximity_bonus` to favor specific cluster counts

## 📝 Notes

- **Color visualization fix:** Handles >20 clusters by switching from tab20 to HSV colormap
- **Numerical stability:** Added normalization and L2 scaling to prevent overflow
- **GIF generation:** 36 frames, 360° rotation, ~100ms per frame
- **Reproducibility:** Fixed random seeds for consistent results

## 🎓 Research Context

This implements neural atom clustering for molecular graph neural networks:
1. Cluster atoms into "neural atoms" (supernodes)
2. Perform message passing on reduced graph
3. Reconstruct original atom representations
4. Reduces message passing hops needed for full molecular communication

The Fourier approach adds:
- Unsupervised clustering in frequency domain
- Adaptive cluster count based on molecular complexity
- Guaranteed atom:cluster ratios for different size ranges
