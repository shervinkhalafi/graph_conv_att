"""Embedding dimension study utilities.

This module provides tools for collecting embedding dimension statistics
across datasets and summarizing the results.
"""

from tmgg.experiments.embedding_study.analysis import analyze_results
from tmgg.experiments.embedding_study.collector import EmbeddingCollector

__all__ = ["EmbeddingCollector", "analyze_results"]
