"""Experiment modules for graph denoising and generation.

Each subdirectory is a self-contained experiment with its own runner,
Lightning module, and Hydra configuration.  Import from the specific
experiment module rather than from this package directly.

Experiments
-----------
digress_denoising
    DiGress transformer-based graph denoising (single-step).
gnn_denoising
    GNN-based graph denoising baseline.
gnn_transformer_denoising
    Hybrid GNN + Transformer denoising architecture.
spectral_arch_denoising
    Spectral graph denoisers (LinearPE, GraphFilterBank, BilinearDenoiser).
lin_mlp_baseline_denoising
    Linear and MLP baselines for pipeline sanity checks.
discrete_diffusion_generative
    Discrete categorical diffusion for graph generation (DiGress-style).
gaussian_diffusion_generative
    Gaussian diffusion for graph generation with MMD evaluation.
eigenstructure_study
    Spectral analysis: eigendecomposition collection, band/spectral gap
    analysis, noise perturbation studies.
embedding_study
    Embedding dimension analysis across datasets.

Shared infrastructure
---------------------
``tmgg.training``
    Lightning modules, evaluation metrics, orchestration, and spectral
    utilities shared across experiments.  See ``tmgg.training`` for the
    public API.

``exp_configs/``
    Hydra configuration files.  Base configs per experiment type, model
    presets, data configs, and multi-stage definitions.

CLI entry points
----------------
``tmgg-digress``
    Run DiGress denoising experiments.
``tmgg-gnn``
    Run GNN denoising experiments.
``tmgg-experiment``
    Unified multi-stage experiment runner.
"""
