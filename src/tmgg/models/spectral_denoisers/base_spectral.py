"""Base class for spectral graph denoising models.

All spectral denoisers share a common pattern: extract embeddings (either
eigenvectors or learned positional encodings like PEARL) from the noisy
adjacency matrix, process them through an architecture-specific transformation,
and reconstruct the denoised adjacency matrix.
"""

# pyright: reportAttributeAccessIssue=false
# torch.nn.functional.pad is not fully typed in the PyTorch stubs; the pad
# function exists at runtime but pyright cannot resolve it via the stub.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import torch

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphDistribution,
)
from tmgg.models.base import (
    EdgeSource,
    GraphModel,
    _coerce_input_to,
    _coerce_output_to,
    read_edge_scalar,
)
from tmgg.models.layers.pearl_embedding import PEARLEmbedding
from tmgg.models.layers.topk_eigen import TopKEigenLayer

EmbeddingSource = Literal["eigenvector", "pearl_random", "pearl_basis"]


class SpectralDenoiser(GraphModel, ABC):
    """Abstract base class for spectral graph denoising models.

    Spectral denoisers operate on node embeddings extracted from the adjacency
    matrix. They support multiple embedding sources:

    - "eigenvector": Top-k eigenvectors via eigendecomposition (default)
    - "pearl_random": R-PEARL positional encodings (random init + GNN)
    - "pearl_basis": B-PEARL positional encodings (basis vectors + GNN)

    Parameters
    ----------
    k : int
        Embedding dimension. For eigenvectors, this is the number of top-k
        eigenvectors. For PEARL, this is the output dimension.
    embedding_source : {"eigenvector", "pearl_random", "pearl_basis"}
        Source of node embeddings. PEARL variants use learned GNN message
        passing instead of eigendecomposition. Default is "eigenvector".
    pearl_num_layers : int
        Number of GNN layers for PEARL embeddings. Ignored if embedding_source
        is "eigenvector". Default is 3.
    pearl_hidden_dim : int
        Hidden dimension for PEARL GNN layers. Default is 64.
    pearl_input_samples : int
        Number of random input samples for R-PEARL. Default is 32.
    pearl_max_nodes : int
        Maximum graph size for B-PEARL basis vectors. Default is 200.

    Attributes
    ----------
    embedding_layer : TopKEigenLayer | PEARLEmbedding
        Layer for extracting node embeddings.
    embedding_source : str
        The embedding source being used.

    Notes
    -----
    PEARL reference: E. Hejin et al., "PEARL: A Scalable and Effective Random
    Positional Encoding." ICLR 2025. https://github.com/ehejin/Pearl-PE
    """

    _internal_in: ClassVar[type] = DenseGraphDistribution
    _internal_out: ClassVar[type] = DenseGraphDistribution

    def __init__(
        self,
        k: int,
        embedding_source: EmbeddingSource = "eigenvector",
        pearl_num_layers: int = 3,
        pearl_hidden_dim: int = 64,
        pearl_input_samples: int = 32,
        pearl_max_nodes: int = 200,
        edge_source: EdgeSource = "feat",
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ):
        super().__init__()
        self.k = k
        self.embedding_source = embedding_source
        self.edge_source: EdgeSource = edge_source
        self.output_dims_x_class = output_dims_x_class
        self.output_dims_x_feat = output_dims_x_feat
        self.output_dims_e_class = output_dims_e_class
        self.output_dims_e_feat = output_dims_e_feat
        self._output_target: EdgeSource = (
            "class" if output_dims_e_class is not None else "feat"
        )

        if embedding_source == "eigenvector":
            self.embedding_layer: TopKEigenLayer | PEARLEmbedding = TopKEigenLayer(k=k)
            # Backward compat: subclasses may reference eigen_layer directly
            self.eigen_layer: TopKEigenLayer | None = self.embedding_layer
        elif embedding_source in ("pearl_random", "pearl_basis"):
            mode = "random" if embedding_source == "pearl_random" else "basis"
            self.embedding_layer = PEARLEmbedding(
                output_dim=k,
                num_layers=pearl_num_layers,
                mode=mode,
                hidden_dim=pearl_hidden_dim,
                input_samples=pearl_input_samples,
                max_nodes=pearl_max_nodes,
            )
            # PEARL has no eigen_layer - subclasses using eigen_layer will fail
            # with a clear error
            self.eigen_layer = None
        else:
            raise ValueError(
                f"embedding_source must be 'eigenvector', 'pearl_random', or "
                f"'pearl_basis', got {embedding_source!r}"
            )

    def forward(
        self,
        data: GraphData,
        t: torch.Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> GraphDistribution | DenseGraphDistribution:
        """Denoise graph via spectral transformation.

        Coerces the input to a :class:`DenseGraphState` (lossless lifts are
        applied automatically; lossy distribution → state collapse must be
        caller-explicit), reads the dense scalar adjacency through
        :func:`read_edge_scalar` (which respects ``self.edge_source``), and
        writes the prediction to the configured split edge field. Optionally
        appends the timestep to ``y`` so downstream subclasses can opt into
        ``t`` conditioning. The state-typed output is then converted to a
        distribution and emitted in the requested layout via
        :func:`_coerce_output_to`.

        Parameters
        ----------
        data
            Graph features in any of the four GraphData variants.
        t
            Diffusion timestep tensor, or None for unconditional denoising.
        output_dense
            If True, return a :class:`DenseGraphDistribution`; otherwise
            return the sparse :class:`GraphDistribution` (default).

        Returns
        -------
        GraphDistribution | DenseGraphDistribution
            Denoised graph distribution in the requested layout, with the
            prediction in the configured edge field.
        """
        d = _coerce_input_to(data, target=DenseGraphDistribution)
        assert isinstance(d, DenseGraphDistribution)
        A = read_edge_scalar(d, self.edge_source)

        if self.embedding_source == "eigenvector":
            assert isinstance(self.embedding_layer, TopKEigenLayer)
            V, Lambda = self.embedding_layer(A)

            actual_k = V.shape[-1]
            if actual_k < self.k:
                pad_size = self.k - actual_k
                V = torch.nn.functional.pad(V, (0, pad_size))
                Lambda = torch.nn.functional.pad(Lambda, (0, pad_size))
        else:
            assert isinstance(self.embedding_layer, PEARLEmbedding)
            V = self.embedding_layer(A)
            unbatched = V.ndim == 2
            if unbatched:
                Lambda = torch.zeros(self.k, device=V.device, dtype=V.dtype)
            else:
                Lambda = torch.zeros(V.shape[0], self.k, device=V.device, dtype=V.dtype)

        result_adj = self._spectral_forward(V, Lambda, A)
        if self._output_target == "feat":
            out_dense = DenseGraphState.from_structure_only(d.node_mask, result_adj)
        else:  # "class"
            out_dense = DenseGraphState.from_edge_scalar(
                result_adj, node_mask=d.node_mask, target="E_class"
            )
        out_dense = out_dense.replace(y=d.y)
        if t is not None:
            new_y = torch.cat([out_dense.y, t.unsqueeze(-1)], dim=-1)
            out_dense = out_dense.replace(y=new_y)
        out_dist = out_dense.to_distribution()
        target = DenseGraphDistribution if output_dense else GraphDistribution
        return _coerce_output_to(out_dist, target=target)  # type: ignore[return-value]

    @abstractmethod
    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Architecture-specific spectral processing.

        Parameters
        ----------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k).
        Lambda : torch.Tensor
            Corresponding eigenvalues of shape (batch, k) or (k,).
        A : torch.Tensor
            The (possibly transformed) input adjacency matrix, provided for
            architectures that need access to the full matrix.

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency matrix (raw, before output transform).
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """Get model configuration for logging/saving.

        Returns
        -------
        dict
            Dictionary containing model hyperparameters.
        """
        return {
            "k": self.k,
            "embedding_source": self.embedding_source,
            "edge_source": self.edge_source,
            "output_dims_x_class": self.output_dims_x_class,
            "output_dims_x_feat": self.output_dims_x_feat,
            "output_dims_e_class": self.output_dims_e_class,
            "output_dims_e_feat": self.output_dims_e_feat,
        }

    @property
    def feature_dim(self) -> int:
        """Output dimension of ``get_features()``.

        Base implementation returns ``self.k`` (eigenvector dimension).
        Subclasses that project to a different space override this.
        """
        return self.k

    def get_features(self, data: GraphData) -> torch.Tensor:
        """Extract learned features for each node.

        Wrapper architectures like ShrinkageWrapper use this to aggregate
        features for graph-level predictions.

        Parameters
        ----------
        data
            Graph features in any of the four GraphData variants. The
            dense scalar adjacency is read via the ``edge_source`` selector
            (matching the selector used by ``forward``).

        Returns
        -------
        torch.Tensor
            Node features of shape ``(batch, n, feature_dim)``.
        """
        d = _coerce_input_to(data, target=DenseGraphDistribution)
        assert isinstance(d, DenseGraphDistribution)
        A = read_edge_scalar(d, self.edge_source)

        if self.embedding_source == "eigenvector":
            assert isinstance(self.embedding_layer, TopKEigenLayer)
            V, _Lambda = self.embedding_layer(A)

            actual_k = V.shape[-1]
            if actual_k < self.k:
                pad_size = self.k - actual_k
                V = torch.nn.functional.pad(V, (0, pad_size))  # PyTorch stub gap
        else:
            assert isinstance(self.embedding_layer, PEARLEmbedding)
            V = self.embedding_layer(A)

        return V
