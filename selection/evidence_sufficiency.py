"""
selection/evidence_sufficiency.py
==================================

Layer 10 of the EARC (Evidence Acquisition, Ranking and Curation) pipeline:
Evidence Sufficiency Verification + Controlled Iterative Expansion.

This is the FINAL layer of the Selection module, acting as the last
quality gate before the Generation module performs prompt construction
and LLM inference.

Responsibility (and ONLY responsibility)
-----------------------------------------
1. Verify whether the evidence selected by Layer 9 is sufficient for
   answer generation, using deterministic, rule-based checks.
2. If insufficient, expand the evidence by appending the minimum number
   of additional candidate sentences necessary, one at a time, until
   sufficiency is reached or an expansion limit / candidate exhaustion
   is hit.

Explicitly OUT of scope for this layer (owned by other layers/modules):
    - document retrieval, BM25/FAISS search, embedding computation
    - rescoring evidence, computing bridge nodes
    - replacing or reordering evidence
    - prompt construction/ordering or LLM inference
    - any I/O (file writes, JSON serialization, pickle, network/Drive access)

This module performs no LLM calls. All sufficiency determination and
expansion logic is deterministic and based solely on data already
present in the Layer 9 output and query analysis.
"""

from typing import Any, Dict, List, Optional, Tuple

from selection import config


# ==========================================================================
# Input validation
# ==========================================================================

def _validate_inputs(
    query_analysis: Dict[str, Any],
    layer9_output: Dict[str, Any],
) -> None:
    """Validate the structural integrity of inputs to Layer 10.

    Args:
        query_analysis: Output of Layer 1, expected to contain
            ``query``, ``query_type``, ``entities``, and ``keywords``.
        layer9_output: Output of Layer 9, expected to contain
            ``selected_sentences``, ``candidate_sentences``, and a
            ``stats`` dict with ``reasoning``, ``budget``, and
            ``diversity`` sub-dicts.

    Raises:
        TypeError: If a required field has the wrong type.
        ValueError: If a required field is missing or structurally
            invalid.
    """
    if not isinstance(query_analysis, dict):
        raise TypeError("query_analysis must be a dict.")

    for field in ("query", "query_type", "entities", "keywords"):
        if field not in query_analysis:
            raise ValueError(f"query_analysis is missing required field: '{field}'")

    if not isinstance(query_analysis["query"], str):
        raise TypeError("query_analysis['query'] must be a str.")
    if not isinstance(query_analysis["query_type"], str):
        raise TypeError("query_analysis['query_type'] must be a str.")
    if not isinstance(query_analysis["entities"], list):
        raise TypeError("query_analysis['entities'] must be a list.")
    if not isinstance(query_analysis["keywords"], list):
        raise TypeError("query_analysis['keywords'] must be a list.")

    if not isinstance(layer9_output, dict):
        raise TypeError("layer9_output must be a dict.")

    for field in ("selected_sentences", "candidate_sentences", "stats"):
        if field not in layer9_output:
            raise ValueError(f"layer9_output is missing required field: '{field}'")

    if not isinstance(layer9_output["selected_sentences"], list):
        raise TypeError("layer9_output['selected_sentences'] must be a list.")
    if not isinstance(layer9_output["candidate_sentences"], list):
        raise TypeError("layer9_output['candidate_sentences'] must be a list.")

    stats = layer9_output["stats"]
    if not isinstance(stats, dict):
        raise TypeError("layer9_output['stats'] must be a dict.")

    for field in ("reasoning", "budget", "diversity"):
        if field not in stats:
            raise ValueError(f"layer9_output['stats'] is missing required field: '{field}'")

    diversity = stats["diversity"]
    if not isinstance(diversity, dict):
        raise TypeError("layer9_output['stats']['diversity'] must be a dict.")
    for field in ("missing_entities", "missing_keywords"):
        if field not in diversity:
            raise ValueError(
                f"layer9_output['stats']['diversity'] is missing required field: '{field}'"
            )


# ==========================================================================
# Query complexity estimation
# ==========================================================================

def _compute_query_complexity(query_analysis: Dict[str, Any]) -> str:
    """Estimate query complexity deterministically, without an LLM.

    Uses simple surface features -- query word count, entity count, and
    keyword count -- compared against explicit thresholds defined in
    ``selection.config.COMPLEXITY_THRESHOLDS``.

    A query is classified as the lowest tier for which it satisfies
    every feature's threshold; if it exceeds the "medium" thresholds on
    any feature, it is classified as "high".

    Args:
        query_analysis: Dict containing ``query``, ``entities``, and
            ``keywords``.

    Returns:
        One of ``"low"``, ``"medium"``, or ``"high"``.
    """
    query_words = len(query_analysis["query"].split())
    entity_count = len(query_analysis["entities"])
    keyword_count = len(query_analysis["keywords"])

    low = config.COMPLEXITY_THRESHOLDS["low"]
    medium = config.COMPLEXITY_THRESHOLDS["medium"]

    if (
        query_words <= low["max_query_words"]
        and entity_count <= low["max_entities"]
        and keyword_count <= low["max_keywords"]
    ):
        return "low"

    if (
        query_words <= medium["max_query_words"]
        and entity_count <= medium["max_entities"]
        and keyword_count <= medium["max_keywords"]
    ):
        return "medium"

    return "high"


# ==========================================================================
# Required evidence count
# ==========================================================================

def _required_evidence_count(query_type: str, complexity: str) -> int:
    """Compute the minimum required evidence count for this query.

    Combines the base requirement for the query type with the
    complexity-driven bump, both defined explicitly in
    ``selection.config``.

    Args:
        query_type: One of ``"factoid"``, ``"descriptive"``,
            ``"multi-hop"`` (case-insensitive).
        complexity: One of ``"low"``, ``"medium"``, ``"high"``.

    Returns:
        The minimum number of evidence sentences required.
    """
    normalized_type = query_type.strip().lower()

    base = config.BASE_MINIMUM_EVIDENCE.get(
        normalized_type, config.DEFAULT_BASE_MINIMUM_EVIDENCE
    )
    bump = config.COMPLEXITY_EVIDENCE_BUMP.get(complexity, 0)

    return base + bump


# ==========================================================================
# Rule 1: Coverage completeness
# ==========================================================================

def _is_coverage_complete(diversity_stats: Dict[str, Any]) -> bool:
    """Check whether Layer 9's diversity stats report complete coverage.

    Coverage is complete only if there are no missing entities and no
    missing keywords reported by Layer 9.

    Args:
        diversity_stats: ``layer9_output["stats"]["diversity"]`` dict,
            expected to contain ``missing_entities`` and
            ``missing_keywords`` lists.

    Returns:
        True if both lists are empty, False otherwise.
    """
    missing_entities = diversity_stats.get("missing_entities", [])
    missing_keywords = diversity_stats.get("missing_keywords", [])
    return len(missing_entities) == 0 and len(missing_keywords) == 0


# ==========================================================================
# Rule 3: Bridge / reasoning requirement
# ==========================================================================

def _bridge_requirement_met(
    query_type: str, reasoning_stats: Dict[str, Any]
) -> bool:
    """Check whether the multi-hop bridge requirement is satisfied.

    Only relevant for multi-hop queries. For all other query types this
    requirement is trivially satisfied (not applicable).

    Args:
        query_type: The query type string (case-insensitive).
        reasoning_stats: ``layer9_output["stats"]["reasoning"]`` dict,
            expected to contain a boolean-ish ``has_bridge_sentence`` or
            equivalent indicator.

    Returns:
        True if the bridge requirement is satisfied or not applicable,
        False if a bridge is required but absent.
    """
    if query_type.strip().lower() != "multi-hop":
        return True

    # Layer 7 (Reasoning Chain Graph) is the source of truth for bridge
    # sentence presence. Support a couple of reasonable key names
    # defensively, since the exact key emitted by Layer 7/9 reasoning
    # stats is not respecified here, while staying read-only.
    if "has_bridge_sentence" in reasoning_stats:
        return bool(reasoning_stats["has_bridge_sentence"])

    bridge_sentences = reasoning_stats.get("bridge_sentences")
    if bridge_sentences is not None:
        return len(bridge_sentences) > 0

    # No recognizable bridge indicator present: treat conservatively as
    # not met, since multi-hop sufficiency explicitly depends on it.
    return False


# ==========================================================================
# Combined sufficiency check
# ==========================================================================

def _is_sufficient(
    query_type: str,
    selected_count: int,
    required_count: int,
    coverage_complete: bool,
    bridge_met: bool,
) -> bool:
    """Combine all three rules into the final sufficiency decision.

    Args:
        query_type: The query type string.
        selected_count: Current number of selected evidence sentences.
        required_count: Minimum required evidence count.
        coverage_complete: Result of ``_is_coverage_complete``.
        bridge_met: Result of ``_bridge_requirement_met``.

    Returns:
        True only if coverage is complete, the minimum evidence count
        is met, and (for multi-hop) the bridge requirement is met.
    """
    minimum_evidence_met = selected_count >= required_count
    return coverage_complete and minimum_evidence_met and bridge_met


# ==========================================================================
# Candidate ranking
# ==========================================================================

def _sentence_key(sentence: Dict[str, Any]) -> Tuple[Any, Any]:
    """Build the deterministic tie-break key ``(doc_id, sent_idx)``.

    Args:
        sentence: A candidate sentence dict.

    Returns:
        Tuple of ``(doc_id, sent_idx)``, defaulting missing fields to
        sentinel values that sort consistently.
    """
    return (sentence.get("doc_id", ""), sentence.get("sent_idx", 0))


def _covers_any(sentence: Dict[str, Any], targets: List[Any]) -> bool:
    """Check whether a sentence's entities/keywords intersect ``targets``.

    Args:
        sentence: A candidate sentence dict, expected to optionally
            carry ``entities`` and/or ``keywords`` lists describing
            what it covers.
        targets: List of missing entities or keywords to check against.

    Returns:
        True if any of the sentence's covered entities/keywords appear
        in ``targets``.
    """
    if not targets:
        return False

    target_set = set(targets)
    covered = set(sentence.get("entities", [])) | set(sentence.get("keywords", []))
    return len(covered & target_set) > 0


def _is_bridge_sentence(sentence: Dict[str, Any]) -> bool:
    """Check whether a candidate sentence is flagged as a bridge sentence.

    Args:
        sentence: A candidate sentence dict, optionally carrying an
            ``is_bridge`` boolean flag.

    Returns:
        True if the sentence is marked as a bridge sentence.
    """
    return bool(sentence.get("is_bridge", False))


def _rank_candidates(
    candidates: List[Dict[str, Any]],
    missing_entities: List[Any],
    missing_keywords: List[Any],
    bridge_required: bool,
) -> List[Dict[str, Any]]:
    """Rank candidate sentences by the deterministic priority order.

    Priority (highest first):
        1. Covers a missing entity
        2. Covers a missing keyword
        3. Is a bridge sentence (only relevant if ``bridge_required``)
        4. Higher score
        5. Shorter sentence (by text length)
        6. (doc_id, sent_idx) ascending, for full determinism

    Args:
        candidates: List of candidate sentence dicts not yet selected.
        missing_entities: Entities still missing per diversity stats.
        missing_keywords: Keywords still missing per diversity stats.
        bridge_required: Whether a bridge sentence is currently needed.

    Returns:
        A new list of candidates sorted by descending priority. The
        input list is not mutated.
    """

    def sort_key(sentence: Dict[str, Any]) -> Tuple[int, int, int, float, int, Tuple[Any, Any]]:
        covers_entity = 1 if _covers_any(sentence, missing_entities) else 0
        covers_keyword = 1 if _covers_any(sentence, missing_keywords) else 0
        is_bridge = 1 if (bridge_required and _is_bridge_sentence(sentence)) else 0
        score = float(sentence.get("score", 0.0))
        text_length = len(sentence.get("text", ""))
        tie_break = _sentence_key(sentence)

        # Negate fields that should sort descending (covers_entity,
        # covers_keyword, is_bridge, score), keep ascending for
        # text_length and tie_break, all within a single ascending sort.
        return (
            -covers_entity,
            -covers_keyword,
            -is_bridge,
            -score,
            text_length,
            tie_break,
        )

    return sorted(candidates, key=sort_key)


def _select_best_candidate(
    ranked_candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Select the top-ranked candidate, if any remain.

    Args:
        ranked_candidates: Output of ``_rank_candidates``.

    Returns:
        The highest-priority candidate sentence, or None if the list is
        empty.
    """
    if not ranked_candidates:
        return None
    return ranked_candidates[0]


# ==========================================================================
# Controlled iterative expansion
# ==========================================================================

def _expand_once(
    selected: List[Dict[str, Any]],
    remaining_candidates: List[Dict[str, Any]],
    missing_entities: List[Any],
    missing_keywords: List[Any],
    bridge_required: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Add exactly one candidate sentence to the selected evidence.

    Does not mutate the input lists; returns new lists reflecting the
    single addition.

    Args:
        selected: Current selected evidence sentences.
        remaining_candidates: Candidate sentences not yet selected.
        missing_entities: Entities still missing per diversity stats.
        missing_keywords: Keywords still missing per diversity stats.
        bridge_required: Whether a bridge sentence is currently needed.

    Returns:
        A tuple of ``(new_selected, new_remaining_candidates, added)``
        where ``added`` is the candidate that was appended, or None if
        no candidates were available to add.
    """
    if not remaining_candidates:
        return list(selected), list(remaining_candidates), None

    ranked = _rank_candidates(
        remaining_candidates, missing_entities, missing_keywords, bridge_required
    )
    best = _select_best_candidate(ranked)

    if best is None:
        return list(selected), list(remaining_candidates), None

    new_selected = selected + [best]
    new_remaining = [c for c in remaining_candidates if c is not best]

    return new_selected, new_remaining, best


def _iterative_expansion(
    query_type: str,
    selected: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    diversity_stats: Dict[str, Any],
    reasoning_stats: Dict[str, Any],
    required_count: int,
    expansion_limit: int,
) -> Tuple[List[Dict[str, Any]], int, str, bool, bool, bool]:
    """Repeatedly add single candidates until sufficient or exhausted.

    After every addition, sufficiency is fully recomputed (coverage,
    minimum evidence, and bridge requirement) using the same rules as
    the initial check. Coverage tracking against missing
    entities/keywords uses Layer 9's reported missing lists as the
    target set throughout expansion; Layer 10 does not recompute
    diversity scoring itself (that remains Layer 9's responsibility),
    it only checks whether newly added candidates happen to cover
    those already-identified gaps.

    Args:
        query_type: The query type string.
        selected: Initial selected evidence sentences (not mutated).
        candidates: Initial candidate sentences (not mutated).
        diversity_stats: Layer 9's diversity stats dict.
        reasoning_stats: Layer 9's reasoning stats dict.
        required_count: Minimum required evidence count.
        expansion_limit: Maximum number of sentences that may be added.

    Returns:
        Tuple of:
            final_selected: Selected sentences after expansion.
            expansions: Number of sentences actually added.
            stopped_reason: One of "sufficient", "coverage_complete",
                "expansion_limit", "no_useful_candidates".
            coverage_complete: Final coverage completeness flag.
            minimum_evidence_met: Final minimum evidence flag.
            bridge_met: Final bridge requirement flag.
    """
    current_selected = list(selected)
    current_candidates = list(candidates)

    missing_entities = list(diversity_stats.get("missing_entities", []))
    missing_keywords = list(diversity_stats.get("missing_keywords", []))
    bridge_required = query_type.strip().lower() == "multi-hop"

    coverage_complete = _is_coverage_complete(diversity_stats)
    bridge_met = _bridge_requirement_met(query_type, reasoning_stats)
    minimum_evidence_met = len(current_selected) >= required_count

    expansions = 0
    stopped_reason = "sufficient"

    while True:
        sufficient = _is_sufficient(
            query_type,
            len(current_selected),
            required_count,
            coverage_complete,
            bridge_met,
        )
        if sufficient:
            stopped_reason = "sufficient"
            break

        if expansions >= expansion_limit:
            stopped_reason = "expansion_limit"
            break

        current_selected, current_candidates, added = _expand_once(
            current_selected,
            current_candidates,
            missing_entities,
            missing_keywords,
            bridge_required,
        )

        if added is None:
            stopped_reason = "no_useful_candidates"
            break

        expansions += 1

        # Diversity coverage in this layer is bound by what Layer 9
        # already identified as missing; if the newly added sentence
        # covers any of those gaps, treat that specific gap as closed
        # for purposes of *this layer's* sufficiency tracking. This
        # does not recompute or replace Layer 9's diversity scoring --
        # it only updates Layer 10's local view of remaining gaps so
        # iterative expansion can converge.
        missing_entities = [
            e for e in missing_entities if e not in added.get("entities", [])
        ]
        missing_keywords = [
            k for k in missing_keywords if k not in added.get("keywords", [])
        ]
        coverage_complete = len(missing_entities) == 0 and len(missing_keywords) == 0

        minimum_evidence_met = len(current_selected) >= required_count

        if bridge_required and _is_bridge_sentence(added):
            bridge_met = True

    return (
        current_selected,
        expansions,
        stopped_reason,
        coverage_complete,
        minimum_evidence_met,
        bridge_met,
    )


# ==========================================================================
# Stats construction
# ==========================================================================

def _build_sufficiency_stats(
    is_sufficient: bool,
    coverage_complete: bool,
    minimum_evidence_met: bool,
    bridge_requirement_met: bool,
    query_complexity: str,
    required_evidence: int,
    final_evidence_count: int,
    expansions: int,
    expansion_limit: int,
    stopped_reason: str,
) -> Dict[str, Any]:
    """Assemble the ``stats["sufficiency"]`` dict in the documented shape.

    Args:
        is_sufficient: Final overall sufficiency decision.
        coverage_complete: Whether coverage was complete in the end.
        minimum_evidence_met: Whether the minimum evidence count was met.
        bridge_requirement_met: Whether the bridge requirement was met.
        query_complexity: Estimated complexity tier.
        required_evidence: Minimum required evidence count.
        final_evidence_count: Evidence count after any expansion.
        expansions: Number of sentences added during expansion.
        expansion_limit: Configured maximum expansions for this query type.
        stopped_reason: Why the expansion loop terminated.

    Returns:
        Dict matching the specification's sufficiency stats shape.
    """
    return {
        "is_sufficient": is_sufficient,
        "coverage_complete": coverage_complete,
        "minimum_evidence_met": minimum_evidence_met,
        "bridge_requirement_met": bridge_requirement_met,
        "query_complexity": query_complexity,
        "required_evidence": required_evidence,
        "final_evidence_count": final_evidence_count,
        "expansions": expansions,
        "expansion_limit": expansion_limit,
        "stopped_reason": stopped_reason,
    }


# ==========================================================================
# Public entry point
# ==========================================================================

def run(
    query_analysis: Dict[str, Any],
    layer9_output: Dict[str, Any],
) -> Dict[str, Any]:
    """Run Layer 10: Evidence Sufficiency Verification + Controlled Expansion.

    This is the sole public entry point for Layer 10. It verifies
    whether ``layer9_output["selected_sentences"]`` is sufficient for
    answer generation using deterministic, rule-based checks, and if
    not, performs controlled iterative expansion by appending candidate
    sentences one at a time (never replacing existing evidence).

    No LLM is called. No retrieval, rescoring, reordering, or I/O is
    performed. Input objects are never mutated.

    Args:
        query_analysis: Layer 1 output containing ``query``,
            ``query_type``, ``entities``, and ``keywords``.
        layer9_output: Layer 9 output containing ``selected_sentences``,
            ``candidate_sentences``, and ``stats`` (with ``reasoning``,
            ``budget``, ``diversity`` sub-dicts).

    Returns:
        Dict with the same shape as ``layer9_output``, except:
            - ``selected_sentences`` may have additional sentences
              appended (never replaced or reordered relative to their
              original relative order; new sentences are appended at
              the end).
            - ``candidate_sentences`` has any newly-selected sentences
              removed.
            - ``stats`` is a copy of the original stats with a new
              ``stats["sufficiency"]`` key appended; ``reasoning``,
              ``budget``, and ``diversity`` are preserved unmodified.

    Raises:
        TypeError: If inputs have incorrect types.
        ValueError: If inputs are missing required fields.
    """
    _validate_inputs(query_analysis, layer9_output)

    query_type = query_analysis["query_type"]
    stats = layer9_output["stats"]
    diversity_stats = stats["diversity"]
    reasoning_stats = stats["reasoning"]

    selected_sentences = list(layer9_output["selected_sentences"])
    candidate_sentences = list(layer9_output["candidate_sentences"])

    complexity = _compute_query_complexity(query_analysis)
    required_count = _required_evidence_count(query_type, complexity)

    normalized_type = query_type.strip().lower()
    expansion_limit = config.MAX_EXPANSION_BY_QUERY_TYPE.get(
        normalized_type, config.DEFAULT_MAX_EXPANSION
    )

    initial_coverage_complete = _is_coverage_complete(diversity_stats)
    initial_bridge_met = _bridge_requirement_met(query_type, reasoning_stats)
    initial_sufficient = _is_sufficient(
        query_type,
        len(selected_sentences),
        required_count,
        initial_coverage_complete,
        initial_bridge_met,
    )

    if initial_sufficient:
        final_selected = selected_sentences
        final_candidates = candidate_sentences
        expansions = 0
        stopped_reason = "sufficient"
        coverage_complete = initial_coverage_complete
        minimum_evidence_met = len(selected_sentences) >= required_count
        bridge_met = initial_bridge_met
        is_sufficient = True
    else:
        (
            final_selected,
            expansions,
            stopped_reason,
            coverage_complete,
            minimum_evidence_met,
            bridge_met,
        ) = _iterative_expansion(
            query_type,
            selected_sentences,
            candidate_sentences,
            diversity_stats,
            reasoning_stats,
            required_count,
            expansion_limit,
        )

        selected_ids = {id(s) for s in final_selected}
        final_candidates = [
            c for c in candidate_sentences if id(c) not in selected_ids
        ]

        is_sufficient = _is_sufficient(
            query_type,
            len(final_selected),
            required_count,
            coverage_complete,
            bridge_met,
        )

    sufficiency_stats = _build_sufficiency_stats(
        is_sufficient=is_sufficient,
        coverage_complete=coverage_complete,
        minimum_evidence_met=minimum_evidence_met,
        bridge_requirement_met=bridge_met,
        query_complexity=complexity,
        required_evidence=required_count,
        final_evidence_count=len(final_selected),
        expansions=expansions,
        expansion_limit=expansion_limit,
        stopped_reason=stopped_reason,
    )

    new_stats = dict(stats)
    new_stats["sufficiency"] = sufficiency_stats

    return {
        "selected_sentences": final_selected,
        "candidate_sentences": final_candidates,
        "stats": new_stats,
    }