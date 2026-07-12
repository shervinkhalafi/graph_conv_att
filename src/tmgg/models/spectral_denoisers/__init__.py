"""Spectral denoising models for graph diffusion tasks.

This package provides denoising architectures that operate on the eigenspace
of noisy adjacency matrices. All models follow a common pattern: extract
top-k eigenvectors, apply a learnable transformation, reconstruct adjacency.

All models output raw logits. Use model.predict(logits) to get [0,1] probabilities.

Models
------
LinearPE
    Simple linear transformation: Â = V W V^T + bias
GraphFilterBank
    Spectral polynomial filter: W = Σ Λ^ℓ H^{(ℓ)}
SelfAttentionDenoiser
    Vaswani et al. (2017) self-attention with softmax + value projection,
    followed by bilinear readout: Â = (attn @ Val) readout
BilinearDenoiser
    Scaled bilinear form: Â = Q K^T / √d_k (no softmax, no values)
BilinearDenoiserWithMLP
    Bilinear with MLP post-processing: Â = MLP(Q K^T / √d_k)
MultiLayerBilinearDenoiser
    Stacked transformer blocks on eigenvectors with bilinear readout

Layers
------
TopKEigenLayer
    Extract top-k eigenvectors with sign normalization

Base Classes
------------
SpectralDenoiser
    Abstract base for all spectral denoising models
"""

from tmgg.models.layers.topk_eigen import (
    EigenDecompositionError,
    TopKEigenLayer,
)
from tmgg.models.spectral_denoisers.base_spectral import SpectralDenoiser
from tmgg.models.spectral_denoisers.bilinear import (
    BilinearDenoiser,
    BilinearDenoiserWithMLP,
    MultiLayerBilinearDenoiser,
)
from tmgg.models.spectral_denoisers.filter_bank import GraphFilterBank
from tmgg.models.spectral_denoisers.linear_pe import LinearPE
from tmgg.models.spectral_denoisers.self_attention import SelfAttentionDenoiser

__all__ = [
    # Base classes
    "SpectralDenoiser",
    # Layers
    "EigenDecompositionError",
    "TopKEigenLayer",
    # Models
    "LinearPE",
    "GraphFilterBank",
    "BilinearDenoiser",
    "BilinearDenoiserWithMLP",
    "MultiLayerBilinearDenoiser",
    "SelfAttentionDenoiser",
]
