# Models

This document describes the model architectures available in the framework.

## Model Hierarchy

```
BaseModel (src/tmgg/models/base.py)
└── DenoisingModel
    ├── MultiLayerAttention (attention/attention.py)
    ├── GNN (gnn/gnn.py)
    ├── GNNSymmetric (gnn/gnn_sym.py)
    ├── NodeVarGNN (gnn/nvgnn.py)
    ├── SequentialDenoisingModel (hybrid/hybrid.py)
    ├── Spectral denoisers (spectral_denoisers/)
    │   ├── LinearPE
    │   ├── GraphFilterBank
    │   ├── SelfAttentionDenoiser
    │   ├── BilinearDenoiser
    │   ├── BilinearDenoiserWithMLP
    │   └── MultiLayerBilinearDenoiser
    └── Shrinkage wrappers (spectral_denoisers/shrinkage_wrapper.py)
        ├── StrictShrinkageWrapper
        └── RelaxedShrinkageWrapper
```

All models inherit from `DenoisingModel`, which provides configuration utilities and prediction methods.

## Attention Models

### MultiLayerAttention

Multi-layer transformer attention for denoising adjacency matrices. Processes the full adjacency matrix through stacked self-attention layers.

**Location:** `src/tmgg/models/attention/attention.py`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | required | Model dimension (typically num_nodes) |
| `num_heads` | int | required | Number of attention heads |
| `num_layers` | int | required | Number of transformer layers |
| `d_k` | int | None | Key dimension (defaults to d_model // num_heads) |
| `d_v` | int | None | Value dimension (defaults to d_model // num_heads) |
| `dropout` | float | 0.0 | Dropout rate |
| `bias` | bool | True | Use bias in linear layers |

**Config:** `exp_configs/models/attention/multi_layer_attention.yaml`

**Usage:**

```python
from tmgg.models.attention import MultiLayerAttention

model = MultiLayerAttention(
    d_model=20,
    num_heads=4,
    num_layers=4,
)
output = model(adjacency_matrix)  # (batch, n, n) -> (batch, n, n)
```

## GNN Models

All GNN models use spectral embeddings via eigendecomposition of the adjacency matrix.

### GNN

Standard graph neural network with polynomial graph convolution filters.

**Location:** `src/tmgg/models/gnn/gnn.py`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_layers` | int | required | Number of GCN layers |
| `num_terms` | int | 3 | Polynomial filter terms |
| `feature_dim_in` | int | 10 | Input feature dimension |
| `feature_dim_out` | int | 10 | Output feature dimension |
| `eigenvalue_reg` | float | 0.0 | Eigenvalue regularization |

**Config:** `exp_configs/models/gnn/standard_gnn.yaml`

**Returns:** Tuple of (X, Y) embeddings for reconstruction via X @ Y.T

### GNNSymmetric

Symmetric GNN with shared X and Y embeddings. Uses a single embedding pathway, making the reconstruction inherently symmetric.

**Location:** `src/tmgg/models/gnn/gnn_sym.py`

**Config:** `exp_configs/models/gnn/symmetric_gnn.yaml`

**Returns:** Tuple of (reconstructed_adjacency, X_embeddings)

### NodeVarGNN

Node-variant GNN with per-node filter coefficients. Allows heterogeneous processing across nodes.

**Location:** `src/tmgg/models/gnn/nvgnn.py`

**Config:** `exp_configs/models/gnn/nodevar_gnn.yaml`

## Hybrid Models

### SequentialDenoisingModel

Combines a GNN embedding model with a transformer denoising model. The GNN extracts node embeddings, which are then processed by attention layers.

**Location:** `src/tmgg/models/hybrid/hybrid.py`

**Usage:**

```python
from tmgg.models.hybrid import create_sequential_model

model = create_sequential_model(
    gnn_config={"num_layers": 2, "feature_dim_out": 10},
    transformer_config={"num_heads": 4, "num_layers": 2}
)
```

**Config:** `exp_configs/models/hybrid/hybrid_with_transformer.yaml`

## Spectral Denoisers

Models operating in the spectral domain, using the eigendecomposition of the noisy adjacency matrix. All output raw logits; use `model.predict(logits)` for [0,1] probabilities.

**Location:** `src/tmgg/models/spectral_denoisers/`

### LinearPE

Linear transformation in eigenspace with optional bias correction.

**Formula:**
```
Â = V W V^T + 1 b^T + b 1^T
```

where V ∈ R^{n×k} are top-k eigenvectors, W ∈ R^{k×k} is learnable, and b ∈ R^{max_n} is a bias vector capturing node-specific degree corrections.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of eigenvectors |
| `max_nodes` | int | 200 | Maximum graph size (for bias) |
| `use_bias` | bool | True | Enable node-specific bias term |

**Config:** `exp_configs/models/spectral/linear_pe.yaml`

### GraphFilterBank

Spectral polynomial filter with learnable coefficient matrices.

**Formula:**
```
W = Σ_{ℓ=0}^{K-1} Λ^ℓ ⊙ H^{(ℓ)}
Â = V W V^T
```

where H^{(ℓ)} ∈ R^{k×k} are learnable coefficient matrices. The polynomial filter allows learning frequency-dependent transformations: different eigenvalue magnitudes can be amplified or attenuated differently.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of eigenvectors |
| `polynomial_degree` | int | 5 | Polynomial degree K |

**Config:** `exp_configs/models/spectral/filter_bank.yaml`

### SelfAttentionDenoiser

Scaled dot-product attention on eigenvector embeddings.

**Formula:**
```
Q = V W_Q,  K = V W_K,  Val = V W_V
attn = softmax(Q K^T / √d_k)
H = attn · Val
A_hat = (H W_out_Q) (H W_out_K)^T / √d_out
```

where V ∈ R^{n×k} are eigenvectors, W_Q, W_K, W_V ∈ R^{k×d_k} are learnable
projections, and W_out_Q, W_out_K ∈ R^{d_k×d_out} reconstruct the adjacency
from the attended representations.

**Registry alias:** `self_attention`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of eigenvectors (input dim) |
| `d_k` | int | 64 | Query/key/value dimension |
| `d_out` | int | None | Readout projection dimension (defaults to `d_k`) |

**Config:** `exp_configs/models/spectral/self_attention.yaml`

### BilinearDenoiser

Scaled bilinear form on eigenvector embeddings. Despite naming inherited from transformer literature, there is no softmax normalization and no value projection -- the core operation is a pure bilinear form.

**Formula:**
```
Q = V W_Q,  K = V W_K
Â = Q K^T / √d_k
```

where V ∈ R^{n×k} are top-k eigenvectors and W_Q, W_K ∈ R^{k×d_k} are learnable projection matrices. The output is raw logits (unbounded); use `model.predict(logits)` for [0,1] probabilities via sigmoid.

**Location:** `src/tmgg/models/spectral_denoisers/bilinear.py`

**Registry alias:** `bilinear`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of eigenvectors (input dim) |
| `d_k` | int | 64 | Query/key dimension |

**Usage:**

```python
from tmgg.models.factory import create_model

model = create_model("bilinear", {"k": 8, "d_k": 64})
```

### BilinearDenoiserWithMLP

Extends BilinearDenoiser by passing the bilinear output through an element-wise MLP before producing adjacency logits. The MLP treats each edge independently, preserving permutation equivariance.

**Formula:**
```
Q = V W_Q,  K = V W_K
H = Q K^T / √d_k
Â = MLP(H)
```

The MLP reshapes H to scalar-per-edge, applies hidden layers with ReLU activations, and outputs one logit per edge.

**Location:** `src/tmgg/models/spectral_denoisers/bilinear.py`

**Registry aliases:** `self_attention_mlp`, `bilinear_mlp`

> **Note:** The `self_attention_mlp` alias refers to `BilinearDenoiserWithMLP`, not to `SelfAttentionDenoiser` combined with an MLP. This is a common source of confusion.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of eigenvectors (input dim) |
| `d_k` | int | 64 | Query/key dimension |
| `mlp_hidden_dim` | int | 128 | Hidden dimension of MLP layers |
| `mlp_num_layers` | int | 2 | Number of MLP layers (including output) |

**Config:** `exp_configs/models/spectral/self_attention_mlp.yaml`

**Usage:**

```python
from tmgg.models.factory import create_model

model = create_model("self_attention_mlp", {"k": 8, "d_k": 64, "mlp_hidden_dim": 128})
```

### MultiLayerBilinearDenoiser

Multilayer transformer-style denoiser operating on eigenvectors. Stacks transformer blocks (multi-head self-attention + residual connections + optional feed-forward MLP) on eigenvector embeddings, then reconstructs the adjacency via final Q/K projections. This is the deep extension of BilinearDenoiser, enabling more complex spectral-to-adjacency mappings.

**Architecture:**
```
V (n×k)  →  Linear (k → d_model)
         →  L × TransformerBlock(attention + residual + [MLP + residual])
         →  Q = W_Q(h),  K = W_K(h)
         →  Â = Q K^T / √d_model
```

Each TransformerBlock uses post-norm style (LayerNorm after residual addition), multi-head self-attention, and an optional GELU-activated feed-forward MLP.

**Location:** `src/tmgg/models/spectral_denoisers/bilinear.py`

**Registry aliases:** `multilayer_self_attention`, `multilayer_attention`, `multilayer_bilinear`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of eigenvectors (input dim) |
| `d_model` | int | 64 | Hidden dimension for transformer blocks |
| `num_heads` | int | 4 | Number of attention heads (must divide d_model) |
| `num_layers` | int | 2 | Number of stacked transformer blocks |
| `use_mlp` | bool | True | Include feed-forward MLP in each block |
| `mlp_hidden_dim` | int | 4*d_model | MLP hidden dimension (config key: `transformer_mlp_hidden_dim`) |
| `dropout` | float | 0.0 | Dropout probability |

**Config:** `exp_configs/models/spectral/multilayer_self_attention.yaml`

**Usage:**

```python
from tmgg.models.factory import create_model

model = create_model("multilayer_self_attention", {
    "k": 8, "d_model": 64, "num_heads": 4, "num_layers": 3,
})
```

### StrictShrinkageWrapper

Wraps an inner spectral denoiser (by default BilinearDenoiser) to enforce denoising via SVD singular value shrinkage. The inner model extracts learned features that are aggregated to graph level and fed to an MLP predicting per-singular-value shrinkage coefficients. Sigmoid gating constrains coefficients to [0, 1], guaranteeing that singular values can only decrease -- corresponding to a denoising interpretation where noise adds energy that should be removed.

**Formula:**
```
U, S, V^T = SVD(A_noisy)
F = inner_model.get_features(A_noisy)
h = aggregate(F)                          # mean or attention pooling
alpha = sigmoid(MLP(h))                   # alpha in [0, 1] per singular value
A_hat = U diag(alpha * S) V^T
A_hat = (A_hat + A_hat^T) / 2            # symmetrize
```

**Location:** `src/tmgg/models/spectral_denoisers/shrinkage_wrapper.py`

**Registry alias:** `self_attention_strict_shrinkage`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | required | Number of eigenvectors for inner BilinearDenoiser |
| `d_k` | int | 64 | Key dimension for inner BilinearDenoiser |
| `shrinkage_max_rank` | int | 50 | Maximum number of singular values to retain |
| `shrinkage_aggregation` | str | "mean" | Feature aggregation: "mean" or "attention" |
| `shrinkage_hidden_dim` | int | 128 | Hidden dimension of the shrinkage MLP |
| `shrinkage_mlp_layers` | int | 2 | Number of layers in the shrinkage MLP |

**Config:** `exp_configs/models/spectral/self_attention_strict_shrinkage.yaml`

> **Note:** This module is experimental and not yet used in standard experiment sweeps.

### RelaxedShrinkageWrapper

Like StrictShrinkageWrapper but uses FiLM-style affine modulation (`scale * S + shift`) instead of sigmoid gating. This allows both shrinkage and expansion of singular values, making it more expressive at the cost of losing the monotone-decrease guarantee. The scale passes through softplus to ensure positivity.

**Formula:**
```
(scale_raw, shift) = split(MLP(h))
scale = softplus(scale_raw)
S_mod = scale * S + shift
A_hat = U diag(S_mod) V^T
```

**Location:** `src/tmgg/models/spectral_denoisers/shrinkage_wrapper.py`

**Registry alias:** `self_attention_relaxed_shrinkage`

**Parameters:** Same as StrictShrinkageWrapper (uses identical config keys).

**Config:** `exp_configs/models/spectral/self_attention_relaxed_shrinkage.yaml`

> **Note:** This module is experimental and not yet used in standard experiment sweeps.

## DiGress Models

Diffusion-based transformer for graph denoising. Used as baseline for comparing spectral architectures.

For a detailed comparison of DiGress attention vs. standard scaled dot-product attention, see [Attention Mechanisms: DiGress vs Standard Graph Transformer](attention_comparison.md).

**Location:** `src/tmgg/models/digress/`

### DiGress Transformer

Score-based model operating on graph structure. Two variants are used in experiments:

**Official settings** (`digress_sbm_small.yaml`):
- LR=0.0002, AdamW + amsgrad, weight_decay=1e-12
- No LR scheduling

**High LR variant** (`digress_sbm_small_highlr.yaml`):
- LR=1e-2 (matching spectral models)
- For fair comparison with spectral architectures

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_layers` | int | 4 | Number of transformer layers |
| `k` | int | 50 | Number of eigenvectors (input feature dimension) |
| `use_eigenvectors` | bool | True | Use eigenvector features |
| `hidden_dims.dx` | int | 128 | Node hidden dimension |
| `hidden_dims.de` | int | 32 | Edge hidden dimension |
| `hidden_dims.dy` | int | 128 | Global (graph-level) hidden dimension |
| `hidden_dims.n_head` | int | 4 | Number of attention heads |

**Config:** `exp_configs/models/digress/digress_sbm_small.yaml`

## Baseline Models

Simple baselines for comparison.

**Location:** `src/tmgg/experiments/lin_mlp_baseline_denoising/`

### Linear Baseline

Linear transformation initialized at identity.

**Formula:**
```
A_pred = W @ A @ W^T + b
```

**Config:** `exp_configs/models/baselines/linear.yaml`

### MLP Baseline

Multi-layer perceptron applied to flattened adjacency.

**Config:** `exp_configs/models/baselines/mlp.yaml`

## Layers

Shared layers used by the models:

### EigenEmbedding

Computes eigenvector embeddings from adjacency matrices.

**Location:** `src/tmgg/models/layers/eigen_embedding.py`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eigenvalue_reg` | float | 0.0 | Diagonal regularization for gradient stability |

Eigendecomposition gradients can be unstable when eigenvalues are close together. Setting `eigenvalue_reg` to a small value (e.g., 1e-3) adds a diagonal perturbation that spreads eigenvalues apart.

### GraphConvolutionLayer

Polynomial graph convolution layer.

**Location:** `src/tmgg/models/layers/gcn.py`

### MultiHeadAttention

Standard multi-head attention layer.

**Location:** `src/tmgg/models/layers/mha_layer.py`

## Graph Embeddings

The embeddings module provides tools for analyzing minimal embedding dimensions for exact graph reconstruction. Given a graph, it finds the smallest dimension d such that the graph can be perfectly reconstructed from node embeddings.

**Location:** `src/tmgg/models/embeddings/`

### Embedding Types

| Type | Symmetric | Asymmetric | Reconstruction |
|------|-----------|------------|----------------|
| LPCA | `LPCASymmetric` | `LPCAAsymmetric` | A ≈ σ(X·Xᵀ) |
| Dot Product | `DotProductSymmetric` | `DotProductAsymmetric` | A ≈ X·Xᵀ |
| Dot Threshold | `DotThresholdSymmetric` | `DotThresholdAsymmetric` | A_ij = 1 iff xᵢ·xⱼ > τ |
| Distance Threshold | `DistanceThresholdSymmetric` | — | A_ij = 1 iff ‖xᵢ-xⱼ‖ < τ |
| Orthogonal | `OrthogonalRepSymmetric` | — | A_ij = 1 iff |xᵢ·xⱼ| > ε |

### Fitters

Two fitting strategies are available:

- **Gradient**: Adam optimizer with BCE/MSE loss, temperature annealing for threshold models
- **Spectral**: SVD-based closed-form solution for dot product, initialization for others

### Dimension Search

`DimensionSearcher` finds the minimal embedding dimension via binary search:

```python
from tmgg.models.embeddings.dimension_search import DimensionSearcher, EmbeddingType

searcher = DimensionSearcher(
    tol_fnorm=0.01,      # Frobenius norm tolerance
    tol_accuracy=0.99,   # Edge accuracy threshold
    fitter="both",       # "gradient", "spectral", or "both"
)

result = searcher.find_min_dimension(
    adjacency,
    EmbeddingType.DOT_PRODUCT_SYMMETRIC,
)
print(f"Min dimension: {result.min_dimension}")
print(f"Final accuracy: {result.final_accuracy}")
```

The search starts at ceil(√n) and first searches downward if successful, then upward if needed.

## Eigenvalue Regularization

GNN models can experience gradient instability due to eigendecomposition. When eigenvalues are close together, gradients involve terms like 1/(λ_i - λ_j) that can explode.

To mitigate this, set `eigenvalue_reg` to a small positive value:

```python
model = GNN(num_layers=2, eigenvalue_reg=1e-3)
```

This adds `eigenvalue_reg * I` to the adjacency matrix before eigendecomposition, spreading eigenvalues apart. Use values between 1e-4 and 1e-2 if you observe NaN gradients or unstable training.
