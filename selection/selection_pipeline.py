"""Module 3 orchestration: Layers 7-9."""

from __future__ import annotations

from typing import Any

from selection import adaptive_budget, evidence_diversity_guard
from selection.reasoning_chain_graph import ReasoningChainGraph


class SelectionPipeline:
    """Runs reasoning graph, budget selection, and diversity guard."""

    def __init__(self):
        self.reasoning_graph = ReasoningChainGraph()

    def run(
        self,
        query_info: dict[str, Any],
        scored_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return selected/candidate evidence after layers 7-9."""
        if not scored_records:
            return {
                "selected_sentences": [],
                "candidate_sentences": [],
                "stats": {
                    "reasoning": {
                        "total_sentences": 0,
                        "bridge_nodes": 0,
                        "non_bridge_nodes": 0,
                    },
                    "budget": {
                        "query_type": query_info.get("query_type", "descriptive"),
                        "budget": 0,
                        "tokens_used": 0,
                        "tokens_remaining": 0,
                        "total_input_sentences": 0,
                        "total_selected_sentences": 0,
                        "bridge_selected": 0,
                        "non_bridge_selected": 0,
                    },
                    "diversity": {
                        "query_entities": 0,
                        "covered_entities": 0,
                        "coverage_ratio": 0.0,
                        "missing_entities": [],
                        "query_keywords": 0,
                        "covered_keywords": 0,
                        "keyword_coverage_ratio": 0.0,
                        "missing_keywords": [],
                        "sentence_swaps": 0,
                        "coverage_improved": False,
                        "coverage_before": 0,
                        "coverage_after": 0,
                        "coverage_delta": 0,
                    },
                },
            }

        layer7_output = self.reasoning_graph.run(query_info, scored_records)
        layer8_output = adaptive_budget.run(query_info, layer7_output)
        layer8_output = self._normalize_layer8(layer8_output)
        layer9_output = evidence_diversity_guard.run(query_info, layer8_output)
        return layer9_output

    @staticmethod
    def _normalize_layer8(layer8_output: dict[str, Any]) -> dict[str, Any]:
        """Bridge Layer 8's two output shapes into the schema Layer 9 expects.

        When the whole candidate set fits the token budget, ``adaptive_budget.run``
        short-circuits and returns a ``sentences`` key instead of the
        ``selected_sentences`` / ``candidate_sentences`` pair. Layer 9 requires
        the latter, so we adapt without modifying the upstream selection layer.
        """
        if "selected_sentences" in layer8_output:
            return layer8_output

        sentences = layer8_output.get("sentences", [])
        return {
            "selected_sentences": sentences,
            "candidate_sentences": [],
            "stats": layer8_output.get("stats", {}),
        }
