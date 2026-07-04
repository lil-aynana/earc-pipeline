"""
selection/selection_pipeline.py
================================

Orchestrator for the EARC (Evidence Acquisition, Ranking and Curation)
Selection Module.

The Selection Module is composed of four independently implemented,
already-complete layers:

    Layer 7  : reasoning_chain_graph.py    (Reasoning Chain Graph)
    Layer 8  : adaptive_budget.py          (Adaptive Token Budget)
    Layer 9  : evidence_diversity_guard.py (Evidence Diversity Guard)
    Layer 10 : evidence_sufficiency.py     (Evidence Sufficiency Verification)

This module contains NONE of their algorithms. Its sole responsibility
is to:

    1. Execute Layers 7 -> 8 -> 9 -> 10 in that exact order.
    2. Pass the output of each layer, unmodified, as the input to the
       next.
    3. Return Layer 10's final output in a clean, generic shape.

Everything happens in memory. No intermediate result is ever
serialized, cached, or written to disk, Google Drive, or any other
persistent store. If any layer raises an exception, this orchestrator
does not suppress it or continue -- it fails immediately, attaching
context about which layer failed while preserving the original
exception via chaining.

The Selection module is intentionally unaware of who consumes its
output. Deciding where ``selected_sentences``, ``stats``, and
``candidate_sentences`` go next (e.g. Generation, Utils/UI, debugging)
is the responsibility of the master EARC pipeline, not this module.
"""

from typing import Any, Dict, List

from selection import reasoning_chain_graph
from selection import adaptive_budget
from selection import evidence_diversity_guard
from selection import evidence_sufficiency


# ==========================================================================
# Pipeline-level error type
# ==========================================================================

class SelectionPipelineError(RuntimeError):
    """Raised when a layer within the Selection Pipeline fails.

    This wraps whatever exception a Selection layer (7-10) raised,
    adding context about which layer failed. The original exception is
    preserved and remains inspectable via ``__cause__`` (standard
    Python exception chaining, ``raise ... from err``) -- it is never
    swallowed or replaced.
    """


# ==========================================================================
# Layer execution helper
# ==========================================================================

def _run_layer(
    layer_name: str,
    layer_callable: Any,
    *args: Any,
) -> Any:
    """Execute a single Selection layer and re-raise failures with context.

    This helper performs no algorithmic work of its own. It only calls
    ``layer_callable(*args)`` and, if that call raises, wraps the
    failure in a ``SelectionPipelineError`` that identifies which layer
    failed, while chaining the original exception so nothing is
    suppressed.

    Args:
        layer_name: Human-readable identifier for the layer being run,
            e.g. ``"Layer 7 (reasoning_chain_graph)"``. Used only for
            error messages.
        layer_callable: The layer's public ``run`` entry point.
        *args: Positional arguments to forward to ``layer_callable``.

    Returns:
        Whatever ``layer_callable(*args)`` returns, unmodified.

    Raises:
        SelectionPipelineError: If ``layer_callable`` raises any
            exception. The original exception is available as
            ``err.__cause__``.
    """
    try:
        return layer_callable(*args)
    except Exception as err:
        raise SelectionPipelineError(
            f"{layer_name} failed during Selection Pipeline execution: {err}"
        ) from err


# ==========================================================================
# Public entry point
# ==========================================================================

def run(
    query_analysis: Dict[str, Any],
    scoring_output: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full Selection Module (Layers 7 through 10) in sequence.

    Executes, in strict order and entirely in memory:

        Scoring Module output
            -> reasoning_chain_graph.run()      (Layer 7)
            -> adaptive_budget.run()            (Layer 8)
            -> evidence_diversity_guard.run()   (Layer 9)
            -> evidence_sufficiency.run()       (Layer 10)

    Each layer receives exactly the output produced by the previous
    layer, without modification, caching, or side effects. No
    intermediate result is written to disk, pickled, serialized to
    JSON/CSV, or persisted anywhere outside this function's local
    variables.

    Layer 10's output is returned unaltered, in a clean, generic shape.
    This function does not know about, and does not route to, any
    downstream consumer (Generation, Utils/UI, or otherwise) -- that
    decision belongs to the master EARC pipeline that calls this
    function.

    Args:
        query_analysis: Layer 1 output describing the query (``query``,
            ``query_type``, ``entities``, ``keywords``, and any other
            fields Layers 7-10 require).
        scoring_output: Scoring Module (Layer 6) output -- a list of
            sentence dictionaries (each carrying fields such as
            ``text``, ``doc_id``, ``sent_idx``, ``position``,
            ``retrieval_rank``, ``parent_bm25``, ``parent_dense``,
            ``parent_rrf``, ``temporal_year``, ``score``, ``embedding``,
            and ``is_bridge``).

    Returns:
        Dict with exactly three top-level keys::

            {
                "selected_sentences": [...],
                "candidate_sentences": [...],
                "stats": {...}
            }

        ``selected_sentences`` and ``candidate_sentences`` are Layer
        10's final evidence and leftover candidates, respectively;
        ``stats`` is the full accumulated statistics dict (reasoning +
        budget + diversity + sufficiency) from Layers 7-10. What each
        of these is used for downstream is outside this module's
        concern.

    Raises:
        SelectionPipelineError: If any of Layers 7-10 raises an
            exception. The pipeline fails immediately; no partial or
            fallback result is returned. The original exception from
            the failing layer is chained and inspectable via
            ``err.__cause__``.
    """
    layer7_output = _run_layer(
        "Layer 7 (reasoning_chain_graph)",
        reasoning_chain_graph.run,
        query_analysis,
        scoring_output,
    )

    layer8_output = _run_layer(
        "Layer 8 (adaptive_budget)",
        adaptive_budget.run,
        query_analysis,
        layer7_output,
    )

    print("Layer 8 keys:", layer8_output.keys())

    # Call validator directly
    evidence_diversity_guard._validate_layer8_output(layer8_output)

    print("Layer 9 validation PASSED")

    layer9_output = _run_layer(
        "Layer 9 (evidence_diversity_guard)",
        evidence_diversity_guard.run,
        query_analysis,
        layer8_output,
    )

    layer10_output = _run_layer(
        "Layer 10 (evidence_sufficiency)",
        evidence_sufficiency.run,
        query_analysis,
        layer9_output,
    )

    return {
        "selected_sentences": layer10_output["selected_sentences"],
        "candidate_sentences": layer10_output["candidate_sentences"],
        "stats": layer10_output["stats"],
    }