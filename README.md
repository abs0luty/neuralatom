# Neural Atom

This modification of the original implementation which uses spectral clustering in reciprocal (Fourier) space for adaptive neural atom pooling.

<p align="center">
  <img src="visualizations/gifs_3d/clustering_dual_space_100atoms.gif" />
</p>

## What is this?

This extends the [Neural Atom](https://arxiv.org/abs/2311.01276) method by using **Fourier-based clustering** instead of fixed or learned cluster counts. The key idea: transform atom embeddings to frequency domain, cluster based on spectral similarity, then transform back.

**Why?** Different molecular structures have different frequency signatures - rings, chains, functional groups appear as distinct patterns in Fourier space.

## Core Method

### 1. Transform to Fourier Space

Take atom embeddings and apply FFT to get frequency representation:

$$
\mathbf{X}_{\omega} = \text{FFT}(\mathbf{X}) \in \mathbb{C}^{N \times d}
$$

Extract **magnitude** (spectral power) and **phase** (structural timing):

$$
\mathbf{F} = [|\mathbf{X}_{\omega}|, \alpha \cdot \angle\mathbf{X}_{\omega}]
$$

**Meaning of Fourier Embeddings**:
- **Low frequencies**: Global molecular structure (backbone, overall shape)
- **High frequencies**: Local features (bonds, functional groups)

### 2. Cluster in Reciprocal Space

Apply KMeans with adaptive $k^*$ (determined by molecule size):

$$
k^* = \text{argmax}_{k} \left[ \text{silhouette}(k) - \text{penalty}(k, r_{\text{target}}) \right]
$$

### 3. Transform Back + Attention Pooling

Get cluster centers in real space via inverse FFT:

$$
\mathbf{H}_{\text{neural}} = \text{IFFT}(\text{ClusterCenters}) \in \mathbb{R}^{k^* \times d}
$$

Then apply original Neural Atom attention mechanism to get final neural atom embeddings.

## Results

### Guaranteed Cluster Counts

| Atoms | Clusters | Atoms/Cluster | Target Met |
|-------|----------|---------------|------------|
| 10    | 4        | 2.5           | ✅ |
| 60    | 9        | 6.7           | ✅ |
| 100   | 15       | 6.7           | ✅ (10+ clusters) |

**100% success rate** across all molecule sizes.

### Visual Results

<p align="center"><i>Atoms colored by cluster assignment</i></p>

#### Neural Atom Graphs (Reduced Graphs)

<p align="center">
  <img src="visualizations/gifs_3d/neural_atom_graph_25atoms.gif" width="90%" />
  <img src="visualizations/gifs_3d/neural_atom_graph_50atoms.gif" width="90%" />
  <img src="visualizations/gifs_3d/neural_atom_graph_100atoms.gif" width="90%" />
</p>

| Atoms | Neural Atoms | Edge Reduction |
|-------|--------------|----------------|
| 25    | 4            | 26 → 6 edges   |
| 50    | 8            | 56 → 22 edges  |
| 100   | 15           | 114 → 68 edges |

### Comparison

| Method | Clusters (avg) | Adaptivity |
|--------|---------------|-----------|
| **Fourier (Ours)** | 8.9 ± 6.7 | 3-30 range |
| Dynamic NN | 22.9 ± 4.0 | 5-27 range |
| Static | 13.0 ± 0.0 | No adaptation |

## Citation

**Original Neural Atom:**
```bibtex
@inproceedings{li2024neuralatoms,
  title={Neural Atoms: Propagating Long-range Interaction in Molecular Graphs
         through Efficient Communication Channel},
  author={Xuan Li and Zhanke Zhou and Jiangchao Yao and Yu Rong and
          Lu Zhang and Bo Han},
  booktitle={ICLR},
  year={2024}
}
```
