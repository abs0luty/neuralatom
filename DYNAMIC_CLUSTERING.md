# Dynamic Neural Atom Clustering

This document describes the dynamic clustering feature that makes the number of neural atoms a learnable parameter instead of a fixed hyperparameter.

## Overview

The original neural atom implementation uses a fixed number of clusters determined by hyperparameters (`pool_ratio` and `avg_nodes`). This new implementation adds a **learnable cluster count predictor** that dynamically determines the optimal number of neural atoms based on the input graph's characteristics.

## How It Works

### Architecture

1. **ClusterCountPredictor**: A small MLP that predicts the number of clusters
   - Input: Graph-level features (mean, max, std pooling of node features + graph size)
   - Output: Predicted pooling ratio → number of clusters
   - Uses straight-through estimator for gradient flow

2. **DynamicPorjecting**: Enhanced projection layer that uses predicted cluster count
   - Maintains a pool of max_clusters seed embeddings
   - Dynamically selects the first k seeds based on prediction
   - Fully differentiable end-to-end

### Key Features

- **Adaptive**: Different molecules can use different numbers of neural atoms
- **Learnable**: The predictor is trained end-to-end with the rest of the model
- **Efficient**: Uses a pool of embeddings and selects a subset (no dynamic memory allocation)
- **Compatible**: Works as a drop-in replacement for static clustering

## Usage

### 2D Molecules (GraphGPS)

#### Configuration

Add these parameters to your config file or set them programmatically:

```python
# In your config YAML or cfg object
gvm:
  use_dynamic_clustering: True  # Enable dynamic clustering
  max_clusters: 50              # Maximum number of clusters
  min_clusters: 3               # Minimum number of clusters

  # Original static parameters (used when use_dynamic_clustering=False)
  avg_nodes: 50
  pool_ratio: 0.25
  n_pool_heads: 2
```

#### Code Example

The models automatically detect the configuration:

```python
from torch_geometric.graphgym.config import cfg

# GVM Model - automatically uses dynamic clustering if configured
from graphgps.network.gvm_model import GVMModel
model = GVMModel(dim_in=..., dim_out=...)

# CustomGNN - also supports dynamic clustering
from graphgps.network.custom_gnn import CustomGNN
model = CustomGNN(dim_in=..., dim_out=...)
```

#### Accessing Predicted Cluster Counts

During forward pass, predicted cluster information is stored in the batch:

```python
output = model(batch)

# For GVMModel
if hasattr(batch, 'num_predicted_clusters'):
    print(f"Predicted clusters: {batch.num_predicted_clusters}")
    print(f"Predicted ratio: {batch.predicted_cluster_ratio}")

# For CustomGNN (per-layer predictions)
if hasattr(batch, 'num_predicted_clusters_per_layer'):
    print(f"Clusters per layer: {batch.num_predicted_clusters_per_layer}")
```

### 3D Molecules (OCP Models)

#### Code Example

```python
from ocpmodels.models.neural_atom_block import NeuralAtom

# Create with dynamic clustering
na_block = NeuralAtom(
    emb_size_atom=128,
    num_hidden=2,
    activation='silu',
    use_dynamic_clustering=True,
    max_clusters=50,
    min_clusters=3
)

# Or without (static clustering)
na_block_static = NeuralAtom(
    emb_size_atom=128,
    num_hidden=2,
    activation='silu',
    use_dynamic_clustering=False  # Uses default 10 clusters (100 * 0.1)
)
```

#### Accessing Predictions

```python
output = na_block.forward(h, x, num_batch, batch_seg)

# Check predicted counts
if hasattr(na_block, 'last_num_clusters'):
    print(f"Last predicted clusters: {na_block.last_num_clusters}")
    print(f"Last predicted ratio: {na_block.last_cluster_ratio}")
```

## Implementation Details

### Straight-Through Estimator

The cluster count must be an integer, but we need gradients to flow. The implementation uses a straight-through estimator:

- **Forward pass**: Round to integer
- **Backward pass**: Pass gradients through as identity

This allows the predictor MLP to learn despite the discrete output.

### Gradient Flow

```
Input Graph Features
    ↓
Graph Pooling (mean, max, std)
    ↓
MLP Predictor (learnable)
    ↓ [gradients flow here]
Sigmoid → Scale to [min_ratio, max_ratio]
    ↓
Multiply by graph size → num_clusters
    ↓
Straight-Through Round
    ↓ [gradients still flow]
Select first k seed embeddings (learnable)
    ↓
Multi-Head Attention Clustering
```

### Predictor Architecture

```python
ClusterCountPredictor(
    node_dim=feature_dim,
    hidden_dim=64,           # MLP hidden dimension
    min_ratio=0.1,           # Minimum pooling ratio (10%)
    max_ratio=0.5,           # Maximum pooling ratio (50%)
    min_clusters=3,          # Hard minimum
    max_clusters=50          # Hard maximum
)
```

MLP structure:
- Input: `3 * node_dim + 1` (mean, max, std features + normalized graph size)
- Layer 1: Linear(input_dim, 64) + ReLU
- Layer 2: Linear(64, 32) + ReLU
- Output: Linear(32, 1) + Sigmoid

## Files Modified

### 2D Molecule Implementation

- `2D_Molecule/graphgps/layer/neural_atom.py`
  - Added `StraightThroughRound`
  - Added `ClusterCountPredictor`
  - Added `DynamicPorjecting`
  - Added aliases: `PMA`, `SAB`, `DynamicPMA`

- `2D_Molecule/graphgps/layer/gvm_layer.py`
  - Updated `GVMLayer.__init__()` to accept dynamic clustering params
  - Updated forward method to handle different return signatures

- `2D_Molecule/graphgps/network/gvm_model.py`
  - Pass dynamic clustering params from config to GVMLayer

- `2D_Molecule/graphgps/network/custom_gnn.py`
  - Updated to use DynamicPorjecting when configured
  - Modified get_NAs_emb to handle different return signatures

- `2D_Molecule/graphgps/config/gvm_config.py`
  - Added configuration parameters

- `2D_Molecule/graphgps/config/custom_gnn_config.py`
  - Added configuration parameters

### 3D Molecule Implementation

- `3D_Molecule/ocpmodels/models/na_pooling.py`
  - Added `StraightThroughRound`
  - Added `ClusterCountPredictor`
  - Added `DynamicPorjecting`
  - Added aliases

- `3D_Molecule/ocpmodels/models/neural_atom_block.py`
  - Updated `NeuralAtom.__init__()` to accept dynamic clustering params
  - Updated get_NAs_emb to handle different return signatures

## Benefits

1. **Adaptability**: Different molecules can use different numbers of clusters based on their complexity
2. **No Manual Tuning**: The model learns the optimal cluster count during training
3. **Better Performance**: Can potentially improve results by using molecule-specific clustering
4. **Interpretability**: Predicted cluster counts can provide insights into molecular complexity

## Hyperparameters to Tune

When using dynamic clustering, consider tuning:

- `max_clusters`: Maximum allowed clusters (default: 50)
- `min_clusters`: Minimum allowed clusters (default: 3)
- Learning rate for the predictor (uses model's optimizer by default)
- MLP hidden dimension in ClusterCountPredictor (default: 64)

## Comparison: Static vs Dynamic

| Aspect | Static Clustering | Dynamic Clustering |
|--------|------------------|-------------------|
| Cluster count | Fixed per layer | Varies per graph |
| Parameters | pool_ratio × avg_nodes | Learned by MLP |
| Flexibility | Same for all molecules | Adapts to each molecule |
| Training | Simpler | Requires learning predictor |
| Inference | Faster (no prediction) | Slightly slower (MLP forward) |

## Example Training Script

```python
from torch_geometric.graphgym.config import cfg
from graphgps.network.gvm_model import GVMModel
import torch

# Set configuration
cfg.gvm.use_dynamic_clustering = True
cfg.gvm.max_clusters = 50
cfg.gvm.min_clusters = 3

# Create model
model = GVMModel(dim_in=9, dim_out=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, batch.y)
    loss.backward()  # Gradients flow through cluster predictor
    optimizer.step()

    # Optional: Log cluster counts
    if hasattr(batch, 'num_predicted_clusters'):
        print(f"Used {batch.num_predicted_clusters} clusters")
```

## Troubleshooting

### Issue: Cluster count not changing

**Solution**: Ensure `use_dynamic_clustering=True` in config and the predictor MLP is being trained (check gradients).

### Issue: Poor performance with dynamic clustering

**Solution**:
- Try adjusting `min_clusters` and `max_clusters` range
- Increase MLP `hidden_dim` for more capacity
- Ensure sufficient training time for predictor to learn

### Issue: Memory issues

**Solution**: Reduce `max_clusters` parameter to use less memory for seed embeddings pool.

## Future Enhancements

Possible improvements:
1. Per-graph cluster counts (currently averaged across batch)
2. Different predictors for different layers
3. Attention-based selection instead of first-k
4. Curriculum learning for cluster count predictor
5. Regularization to encourage specific cluster count ranges
