"""Gradient-based fitting for graph embeddings.

Uses Adam optimizer with BCE or MSE loss to fit embedding parameters
to reconstruct a target adjacency matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.optim as optim

from ..base import EmbeddingResult, GraphEmbedding


@dataclass
class FitConfig:
    """Configuration for gradient-based fitting.

    Attributes
    ----------
    lr
        Learning rate for Adam optimizer.
    max_steps
        Maximum optimization steps.
    tol_fnorm
        Frobenius norm tolerance for early stopping.
    tol_accuracy
        Edge accuracy tolerance for early stopping.
    patience
        Steps without improvement before early stopping.
    loss_type
        Loss function: "bce" or "mse".
    log_interval
        Steps between progress logging (0 to disable).
    """

    lr: float = 0.01
    max_steps: int = 10000
    tol_fnorm: float = 0.01
    tol_accuracy: float = 0.99
    patience: int = 500
    loss_type: Literal["bce", "mse"] = "bce"
    log_interval: int = 0


class GradientFitter:
    """Fits graph embeddings via gradient descent.

    Uses Adam optimizer with optional early stopping when reconstruction
    quality reaches target thresholds.
    """

    def __init__(self, config: FitConfig | None = None) -> None:
        """Initialize gradient fitter.

        Parameters
        ----------
        config
            Fitting configuration. Uses defaults if None.
        """
        self.config = config or FitConfig()

    def fit(
        self,
        embedding: GraphEmbedding,
        target: torch.Tensor,
    ) -> EmbeddingResult:
        """Fit embedding to reconstruct target adjacency.

        Parameters
        ----------
        embedding
            The embedding model to optimize.
        target
            Target adjacency matrix of shape (n, n).

        Returns
        -------
        EmbeddingResult
            Result containing fitted embeddings and metrics.
        """
        cfg = self.config
        optimizer = optim.Adam(embedding.parameters(), lr=cfg.lr)

        best_fnorm = float("inf")
        steps_without_improvement = 0
        converged = False

        for step in range(cfg.max_steps):
            optimizer.zero_grad()
            loss = embedding.compute_loss(target, loss_type=cfg.loss_type)
            loss.backward()
            optimizer.step()

            # Evaluate periodically
            if step % 100 == 0 or step == cfg.max_steps - 1:
                fnorm, accuracy = embedding.evaluate(target)

                if cfg.log_interval > 0 and step % cfg.log_interval == 0:
                    print(
                        f"Step {step}: loss={loss.item():.6f}, "
                        f"fnorm={fnorm:.4f}, accuracy={accuracy:.4f}"
                    )

                # Check early stopping criteria
                if fnorm < cfg.tol_fnorm and accuracy >= cfg.tol_accuracy:
                    converged = True
                    break

                # Track improvement for patience
                if fnorm < best_fnorm - 1e-6:
                    best_fnorm = fnorm
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 100

                if steps_without_improvement >= cfg.patience:
                    break

        result = embedding.to_result(target)
        result.converged = converged
        return result

    def fit_with_temperature_annealing(
        self,
        embedding: GraphEmbedding,
        target: torch.Tensor,
        init_temperature: float = 1.0,
        final_temperature: float = 50.0,
        anneal_steps: int = 5000,
    ) -> EmbeddingResult:
        """Fit embedding with temperature annealing for threshold models.

        Gradually increases temperature to sharpen the sigmoid relaxation.
        Only applicable to threshold-based embeddings.

        Parameters
        ----------
        embedding
            The embedding model (must have a 'temperature' attribute).
        target
            Target adjacency matrix.
        init_temperature
            Starting temperature (soft threshold).
        final_temperature
            Ending temperature (sharp threshold).
        anneal_steps
            Steps over which to anneal temperature.

        Returns
        -------
        EmbeddingResult
            Result containing fitted embeddings and metrics.
        """
        if not hasattr(embedding, "temperature"):
            raise ValueError(
                "Temperature annealing requires embedding with 'temperature' attribute"
            )

        cfg = self.config
        optimizer = optim.Adam(embedding.parameters(), lr=cfg.lr)

        best_fnorm = float("inf")
        steps_without_improvement = 0
        converged = False

        for step in range(cfg.max_steps):
            # Anneal temperature
            if step < anneal_steps:
                progress = step / anneal_steps
                new_temp = (
                    init_temperature + (final_temperature - init_temperature) * progress
                )
            else:
                new_temp = final_temperature
            # Use object.__setattr__ to bypass nn.Module's type checking
            object.__setattr__(embedding, "temperature", new_temp)

            optimizer.zero_grad()
            loss = embedding.compute_loss(target, loss_type=cfg.loss_type)
            loss.backward()
            optimizer.step()

            if step % 100 == 0 or step == cfg.max_steps - 1:
                fnorm, accuracy = embedding.evaluate(target)

                if cfg.log_interval > 0 and step % cfg.log_interval == 0:
                    current_temp = getattr(embedding, "temperature", 0.0)
                    print(
                        f"Step {step}: loss={loss.item():.6f}, "
                        f"fnorm={fnorm:.4f}, accuracy={accuracy:.4f}, "
                        f"temp={current_temp:.1f}"
                    )

                if fnorm < cfg.tol_fnorm and accuracy >= cfg.tol_accuracy:
                    converged = True
                    break

                if fnorm < best_fnorm - 1e-6:
                    best_fnorm = fnorm
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 100

                if steps_without_improvement >= cfg.patience:
                    break

        result = embedding.to_result(target)
        result.converged = converged
        return result
