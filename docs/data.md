# Data

This document describes the data pipeline, supported datasets, and noise types.

## Data Modules

Two data module classes are available, depending on the experimental protocol.

### SingleGraphDataModule

For single-graph training protocols (Stages 1, 1.5). All splits use the same graph structure; only noise varies across samples.

**Location:** `src/tmgg/data/single_graph_data_module.py`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_type` | str | "sbm" | Graph type (sbm, erdos_renyi, regular, tree, lfr, ring_of_cliques, pyg_*) |
| `n` | int | 50 | Number of nodes |
| `num_train_samples` | int | 1000 | Training samples per epoch (noise realizations) |
| `num_val_samples` | int | 100 | Validation samples |
| `num_test_samples` | int | 100 | Test samples |
| `batch_size` | int | 16 | Batch size |
| `same_graph_all_splits` | bool | False | If True, val/test use identical graph as training |
| `train_seed` | int | 42 | Seed for training graph generation |
| `val_test_seed` | int | 123 | Seed for val/test graphs (when different) |

**Stage protocols:**
- `same_graph_all_splits=True` (Stage 1): All splits use identical graph G, only noise varies
- `same_graph_all_splits=False` (Stage 2+): Validation/test use different graphs G', G''

**Usage:**

```python
from tmgg.data import SingleGraphDataModule

dm = SingleGraphDataModule(
    graph_type="sbm",
    n=50,
    num_train_samples=1000,
    batch_size=16,
    same_graph_all_splits=True,  # Stage 1 protocol
    p_intra=0.7,
    p_inter=0.05,
    num_blocks=3,
)
dm.setup()

for batch in dm.train_dataloader():
    clean = batch  # (batch_size, n, n) - noise applied in LightningModule
```

### GraphDataModule

For multi-graph training protocols (Stages 2+). Multiple graphs with train/val/test splits.

**Location:** `src/tmgg/data/data_module.py`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | required | Dataset type (sbm, nx, pyg, synthetic) |
| `dataset_config` | dict | required | Dataset-specific configuration |
| `num_samples_per_graph` | int | 1000 | Permutations per base graph |
| `batch_size` | int | 100 | Batch size |
| `val_split` | float | 0.2 | Validation set fraction |
| `test_split` | float | 0.2 | Test set fraction |
| `noise_type` | str | "digress" | Noise model (gaussian, rotation, digress, edge_flip, logit) |
| `noise_levels` | list | [0.1] | Noise levels to sample from |

**Usage:**

```python
from tmgg.data import GraphDataModule

data_module = GraphDataModule(
    dataset_name="sbm",
    dataset_config={"num_nodes": 20, "p_intra": 1.0, "p_inter": 0.0},
    noise_type="digress",
    noise_levels=[0.05, 0.1, 0.2],
)
data_module.setup()

for batch in data_module.train_dataloader():
    clean, noisy = batch  # Both (batch_size, n, n)
```

## Supported Datasets

### Stochastic Block Model (SBM)

Random graphs with community structure. Nodes are partitioned into blocks, with different edge probabilities within and between blocks.

**Config:** `exp_configs/data/sbm_default.yaml`

```yaml
dataset_name: sbm
dataset_config:
  num_nodes: 20
  p_intra: 1.0       # Edge probability within blocks
  p_inter: 0.0       # Edge probability between blocks
  min_blocks: 2
  max_blocks: 4
  min_block_size: 2
  max_block_size: 15
  num_train_partitions: 10
  num_test_partitions: 10
```

### NetworkX Graphs

Classical graph structures from NetworkX.

**Configs:** `exp_configs/data/nx_square.yaml`, `exp_configs/data/nx_star.yaml`

```yaml
dataset_name: nx
dataset_config:
  graph_type: "grid_2d"  # or "star", "cycle", etc.
  n: 20
```

### PyG Benchmark Datasets

Datasets from PyTorch Geometric.

**Config:** `exp_configs/data/pyg_enzymes.yaml`

```yaml
dataset_name: pyg
dataset_config:
  name: "ENZYMES"  # or "PROTEINS", "QM9"
```

### LFR Benchmark Graphs

Lancichinetti-Fortunato-Radicchi benchmark with planted community structure.

**Config:** `exp_configs/data/lfr_single_graph.yaml`

```yaml
graph_type: lfr
n: 50
tau1: 3.0          # Exponent for degree distribution
tau2: 1.5          # Exponent for community size distribution
mu: 0.1            # Mixing parameter (fraction of inter-community edges)
average_degree: 5
min_community: 10
```

### Ring of Cliques

Graph constructed by connecting multiple cliques in a ring topology.

**Config:** `exp_configs/data/roc_single_graph.yaml`

```yaml
graph_type: ring_of_cliques
num_cliques: 4
clique_size: 5
```

### Synthetic Graphs

Various random graph models.

```yaml
dataset_name: synthetic
dataset_config:
  graph_type: "erdos_renyi"  # or "regular", "tree", "watts_strogatz", etc.
  n: 20
  num_graphs: 100
```

## Synthetic Graph Generators

Available in `src/tmgg/data/synthetic_graphs.py`:

### Regular Graphs

D-regular graphs where every node has exactly d neighbors.

```python
from tmgg.data import generate_regular_graphs

graphs = generate_regular_graphs(n=20, d=3, num_graphs=100)
```

### Erdős-Rényi Graphs

Random graphs with independent edge probability p.

```python
from tmgg.data import generate_erdos_renyi_graphs

graphs = generate_erdos_renyi_graphs(n=20, p=0.3, num_graphs=100)
```

### Tree Graphs

Random trees with n-1 edges.

```python
from tmgg.data import generate_tree_graphs

graphs = generate_tree_graphs(n=20, num_graphs=100)
```

### Watts-Strogatz Graphs

Small-world graphs with tunable clustering and path length.

```python
from tmgg.data import generate_watts_strogatz_graphs

graphs = generate_watts_strogatz_graphs(
    n=20,
    num_graphs=100,
    k=4,    # Each node connected to k nearest neighbors
    p=0.3   # Rewiring probability
)
```

### Random Geometric Graphs

Nodes placed uniformly in a unit square, edges between nearby nodes.

```python
from tmgg.data import generate_random_geometric_graphs

graphs = generate_random_geometric_graphs(
    n=20,
    num_graphs=100,
    radius=0.3  # Edge threshold distance
)
```

### Configuration Model Graphs

Random graphs with a specified degree sequence.

```python
from tmgg.data import generate_configuration_model_graphs

graphs = generate_configuration_model_graphs(
    n=20,
    num_graphs=100,
    degree_sequence=[3] * 20  # All nodes degree 3
)
```

### SyntheticGraphDataset Class

Unified interface for all synthetic graphs:

```python
from tmgg.data import SyntheticGraphDataset

# Aliases available: er, ws, rg, cm
dataset = SyntheticGraphDataset("ws", num_nodes=20, num_graphs=100, k=4, p=0.2)
train, val, test = dataset.train_val_test_split()
tensor = dataset.to_torch()  # Returns torch.Tensor
```

## Noise Types

Five noise models are available for training and evaluation.

### Gaussian Noise

Additive Gaussian noise to the adjacency matrix.

```python
from tmgg.data import add_gaussian_noise

noisy = add_gaussian_noise(adjacency, eps=0.1)
```

The noise level `eps` controls the standard deviation.

### Rotation Noise

Rotates the adjacency matrix in eigenspace using a skew-symmetric matrix. Preserves spectral properties while perturbing structure.

```python
from tmgg.data import add_rotation_noise, random_skew_symmetric_matrix

skew = random_skew_symmetric_matrix(k=20)  # k = num eigenvectors
noisy = add_rotation_noise(adjacency, eps=0.1, skew=skew)
```

### DiGress Noise (Categorical Transition)

Implements the forward diffusion process from Vignac et al. (2023). For binary adjacency matrices, this applies a categorical transition matrix that interpolates between the identity (no change) and a uniform distribution over edge states. At noise level `eps`, the transition probability is `Q_t = (1 - eps) * I + eps * uniform`, so each edge independently stays the same with probability `(1 - eps/2)` or flips with probability `eps/2`. This produces half the flip rate of simple edge flipping at the same noise level.

```python
from tmgg.data import add_digress_noise

noisy = add_digress_noise(adjacency, eps=0.1)
```

### Edge Flip Noise

Simple Bernoulli edge flipping: each edge is independently flipped (0 to 1 or 1 to 0) with probability `eps`. This is the straightforward binary noise model, distinct from the DiGress categorical transition above.

```python
from tmgg.data import add_edge_flip_noise

noisy = add_edge_flip_noise(adjacency, eps=0.1)  # 10% flip probability
```

### Logit Noise

Applies Gaussian noise in logit (log-odds) space, producing soft adjacency values in (0, 1). The input is clamped to avoid numerical issues, transformed via `logit(A) = log(A / (1 - A))`, perturbed with symmetric Gaussian noise of standard deviation `sigma`, then mapped back through sigmoid. Unlike flip-based noise types, the output is continuous rather than binary. The `sigma` parameter controls perturbation severity in log-odds space: values around 0.5 cause mild perturbation where most edges retain their original polarity, while values around 5.0 cause aggressive flipping.

```python
from tmgg.data import add_logit_noise

noisy = add_logit_noise(adjacency, sigma=1.0)
```

## Configuring Noise

In YAML configs:

```yaml
noise_type: "digress"  # gaussian, rotation, digress, edge_flip, or logit
noise_levels: [0.005, 0.02, 0.05, 0.1, 0.25, 0.4, 0.5]
```

During training, a noise level is sampled uniformly from `noise_levels` for each batch.

## Data Flow

1. **Setup**: `GraphDataModule.setup()` loads or generates base graphs
2. **Permutation**: Each base graph is permuted `num_samples_per_graph` times
3. **Splitting**: Data split into train/val/test sets
4. **Batching**: DataLoader creates batches
5. **Noise**: Noise applied to each batch during training (clean data preserved for loss computation)
