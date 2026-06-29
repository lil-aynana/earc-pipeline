"""
Pipeline Statistics Collector

Stores statistics from each stage of the EARC pipeline.
This module is independent of all pipeline layers and is
used later by rag_pipeline.py and the Streamlit UI.
"""

from __future__ import annotations

from typing import Any


class PipelineStats:
    """
    Collects statistics produced by different pipeline layers.
    """

    def __init__(self) -> None:
        """Initialize an empty statistics dictionary."""
        self._stats: dict[str, dict[str, Any]] = {}

    def add(self, layer: str, **kwargs: Any) -> None:
        """
        Add or update statistics for a pipeline layer.

        Parameters
        ----------
        layer : str
            Name of the pipeline layer.
        **kwargs
            Key-value statistics to store.
        """
        if layer not in self._stats:
            self._stats[layer] = {}

        self._stats[layer].update(kwargs)

    def get(self) -> dict[str, dict[str, Any]]:
        """
        Return all collected statistics.
        """
        return self._stats

    def clear(self) -> None:
        """
        Remove all stored statistics.
        """
        self._stats.clear()