"""Adaptive Token Budget Allocation module for the RAG pipeline (Layer 8).

Receives the sentence list from Layer 7 (Reasoning Chain Graph) and returns
a filtered subset that fits within the token budget configured for the
detected query type.  Bridge sentences are prioritised over non-bridge
sentences.  The module operates entirely in RAM and produces no side effects.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer

from config import CONFIG

# ---------------------------------------------------------------------------
# Module-level tokenizer (single load; treated as read-only global state).
# Requires CONFIG["tokenizer_model"] to be set in config.py.
# ---------------------------------------------------------------------------
_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_model"])

# Fields that every input sentence must contain.
_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"text", "score", "is_bridge", "doc_id", "sent_idx", "position"}
)


def _validate(sentences: list[dict[str, Any]]) -> None:
    """Raise ValueError if any sentence is missing a required field.

    Args:
        sentences: Sentence dictionaries to validate.

    Raises:
        ValueError: If a required field is absent from any sentence.
    """
    for i, sent in enumerate(sentences):
        missing = _REQUIRED_FIELDS - sent.keys()
        if missing:
            raise ValueError(
                f"Sentence at index {i} is missing required fields: {missing}"
            )


def _token_count(text: str) -> int:
    """Return the number of tokens in *text* using the module-level tokenizer.

    Args:
        text: Raw sentence string.

    Returns:
        Integer token count, excluding special tokens.
    """
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _greedy_select(
    sentences: list[dict[str, Any]],
    token_counts: dict[tuple[int, int], int],
    remaining: int,
) -> tuple[list[dict[str, Any]], int]:
    """Greedily add sentences from *sentences* while staying within *remaining* tokens.

    Sentences are assumed to be pre-sorted by score descending.  Each
    sentence is identified by its stable ``(doc_id, sent_idx)`` key in
    *token_counts*.

    Args:
        sentences: Candidate sentence dictionaries, sorted by score descending.
        token_counts: Maps ``(doc_id, sent_idx)`` to the pre-computed token count.
        remaining: Token budget still available.

    Returns:
        A 2-tuple of (selected sentence dicts, updated remaining budget).
    """
    selected: list[dict[str, Any]] = []
    for sent in sentences:
        count = token_counts[(sent["doc_id"], sent["sent_idx"])]
        if count <= remaining:
            selected.append(sent)
            remaining -= count
    return selected, remaining


def run(
    query_analysis: dict[str, Any],
    layer7_output: dict[str, Any],
) -> dict[str, Any]:
    """Select sentences that fit within the query-type token budget.

    Algorithm
    ---------
    1. Extract ``sentences`` and upstream ``stats`` from *layer7_output*.
    2. Validate that ``"query_type"`` is present in *query_analysis*.
    3. Determine the token budget from ``CONFIG["token_budget"]`` keyed by
       ``query_analysis["query_type"]``.
    4. Compute token counts for every sentence.
    5. If the total fits within the budget, return the full list immediately.
    6. Otherwise split into bridge and non-bridge groups, sort each by score
       descending, and greedily fill the budget with bridge sentences first,
       then non-bridge sentences.
    7. Restore original retrieval order (doc_id, position) before returning.
    8. Merge Layer 7 statistics with Layer 8 budget statistics and return.

    Args:
        query_analysis: Query-analysis output from Layer 1.  Must contain
            ``"query_type"`` matching a key in ``CONFIG["token_budget"]``.
        layer7_output: Output dict from Layer 7.  Must contain
            ``"sentences"`` (list of sentence dicts).  May contain
            ``"stats"`` (dict) which is preserved and extended.

    Returns:
        A dict with keys:

        * ``"sentences"`` – filtered list of original sentence dicts in
          natural document order, with bridge sentences prioritised during
          selection.
        * ``"stats"`` – merged dict of Layer 7 stats and a ``"budget"``
          namespace owned by Layer 8.

        Layer 8 stat keys (nested under ``"budget"``): ``query_type``,
        ``budget``, ``tokens_used``, ``tokens_remaining``,
        ``total_input_sentences``, ``total_selected_sentences``,
        ``bridge_selected``, ``non_bridge_selected``.

    Raises:
        ValueError: If ``"query_type"`` is absent from *query_analysis*, if
            the query type is not found in the token budget configuration, or
            if any sentence is missing a required field.
    """
    sentences: list[dict[str, Any]] = layer7_output["sentences"]
    layer7_stats: dict[str, Any] = layer7_output.get("stats", {})

    if not sentences:
        return {"sentences": [], "stats": layer7_stats}

    _validate(sentences)

    # --- Step 1: resolve budget ------------------------------------------
    if "query_type" not in query_analysis:
        raise ValueError(
            "'query_type' is missing from query_analysis. "
            "Layer 1 must populate this field before calling Layer 8."
        )
    query_type: str = query_analysis["query_type"]
    budget_map: dict[str, int] = CONFIG["token_budget"]
    if query_type not in budget_map:
        raise ValueError(
            f"Query type '{query_type}' not found in CONFIG['token_budget']. "
            f"Available types: {list(budget_map.keys())}"
        )
    budget: int = budget_map[query_type]

    # --- Step 2: compute token counts once, keyed by (doc_id, sent_idx) --
    token_counts: dict[tuple[int, int], int] = {
        (sent["doc_id"], sent["sent_idx"]): _token_count(sent["text"])
        for sent in sentences
    }

    # --- Step 3: short-circuit if already within budget ------------------
    total: int = sum(token_counts.values())
    if total <= budget:
        budget_stats = {
            "query_type": query_type,
            "budget": budget,
            "tokens_used": total,
            "tokens_remaining": budget - total,
            "total_input_sentences": len(sentences),
            "total_selected_sentences": len(sentences),
            "bridge_selected": sum(1 for s in sentences if s["is_bridge"]),
            "non_bridge_selected": sum(1 for s in sentences if not s["is_bridge"]),
        }
        return {"sentences": sentences, "stats": {**layer7_stats, "budget": budget_stats}}

    # --- Steps 4–5: split and sort both groups by score descending -------
    bridge: list[dict[str, Any]] = sorted(
        (s for s in sentences if s["is_bridge"]),
        key=lambda s: s["score"],
        reverse=True,
    )
    non_bridge: list[dict[str, Any]] = sorted(
        (s for s in sentences if not s["is_bridge"]),
        key=lambda s: s["score"],
        reverse=True,
    )

    # --- Steps 6–7: greedy fill, bridge first ----------------------------
    selected_bridge, remaining = _greedy_select(bridge, token_counts, budget)
    selected_non_bridge, remaining = _greedy_select(non_bridge, token_counts, remaining)

    # --- Step 8: restore natural document order --------------------------
    selected: list[dict[str, Any]] = selected_bridge + selected_non_bridge
    selected.sort(key=lambda s: (s["doc_id"], s.get("position", s["sent_idx"])))

    selected_keys = {
    (s["doc_id"], s["sent_idx"])
    for s in selected
    }

    candidate_sentences = [
        s for s in sentences
        if (s["doc_id"], s["sent_idx"]) not in selected_keys
    ]

    tokens_used = budget - remaining
    budget_stats = {
        "query_type": query_type,
        "budget": budget,
        "tokens_used": tokens_used,
        "tokens_remaining": remaining,
        "total_input_sentences": len(sentences),
        "total_selected_sentences": len(selected),
        "bridge_selected": len(selected_bridge),
        "non_bridge_selected": len(selected_non_bridge),
    }
    return {"selected_sentences": selected, "candidate_sentences": candidate_sentences, "stats": {**layer7_stats, "budget": budget_stats}}