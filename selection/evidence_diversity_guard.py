"""
Layer 9: Evidence Diversity Guard
==================================

This module implements Layer 9 of the EARC (Evidence Acquisition,
Ranking, and Curation) pipeline's Selection module.

Layer 9 is the final evidence refinement stage before Sufficiency
Verification. It does not retrieve, score, embed, or expand evidence.
Instead, it inspects the evidence already selected by Layer 8 and
performs *like-for-like replacements* (never additions) of weaker
selected sentences with stronger candidate sentences, whenever doing
so improves coverage of the query's entities and keywords without
reducing overall coverage.

Coverage matching is token-based rather than raw substring matching:
sentence and term text are tokenized and lightly normalized (case
folding plus a minimal plural-stripping heuristic) so that, e.g., the
query term "satellite" is recognized inside the sentence "satellites
orbit earth", and "car" is recognized inside "cars". This remains a
lightweight, dependency-free heuristic -- it intentionally stops short
of a full NLP pipeline (no stemmer/lemmatizer dependency, no POS
tagging) so Layer 9 stays cheap and deterministic.

The "diversity" terminology used throughout this module (and in the
output stats) refers specifically to *query coverage diversity* --
i.e. how many distinct query entities/keywords the selected evidence
set represents -- and NOT document-source diversity, deduplication,
or redundancy removal. Those concerns belong to other layers.

The entire pipeline executes in RAM. This module performs no I/O of
any kind: no disk writes, no JSON/pickle serialization, no network
or Google Drive calls.

Design goals:
    - Deterministic: identical inputs always produce identical outputs.
    - Pure: no mutation of input sentence dictionaries.
    - Modular: each responsibility is isolated in its own helper.
    - Defensive: thorough validation with descriptive exceptions.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

__all__ = ["run"]


# ---------------------------------------------------------------------------
# Required field contracts
# ---------------------------------------------------------------------------

_REQUIRED_SENTENCE_FIELDS: Tuple[str, ...] = (
    "text",
    "score",
    "embedding",
    "is_bridge",
    "doc_id",
    "sent_idx",
    "position",
)

_REQUIRED_LAYER8_FIELDS: Tuple[str, ...] = (
    "selected_sentences",
    "candidate_sentences",
    "stats",
)

_REQUIRED_STATS_FIELDS: Tuple[str, ...] = ("reasoning", "budget")

_REQUIRED_QUERY_FIELDS: Tuple[str, ...] = (
    "query",
    "query_type",
    "entities",
    "keywords",
)


# ---------------------------------------------------------------------------
# Step 1: Input validation
# ---------------------------------------------------------------------------

def _validate_query_analysis(query_analysis: Any) -> None:
    """
    Validate the structure of ``query_analysis``.

    Raises:
        ValueError: if ``query_analysis`` is not a dict or is missing
            any required field.
    """
    if not isinstance(query_analysis, dict):
        raise ValueError(
            f"query_analysis must be a dict, got {type(query_analysis).__name__}"
        )

    for field in _REQUIRED_QUERY_FIELDS:
        if field not in query_analysis:
            raise ValueError(
                f"query_analysis is missing required field: '{field}'"
            )

    if not isinstance(query_analysis["entities"], list):
        raise ValueError("query_analysis['entities'] must be a list")

    if not isinstance(query_analysis["keywords"], list):
        raise ValueError("query_analysis['keywords'] must be a list")


def _validate_sentence(sentence: Any, container_name: str, index: int) -> None:
    """
    Validate a single sentence dictionary.

    Args:
        sentence: The sentence object to validate.
        container_name: Name of the list it came from (for error messages).
        index: Position of the sentence within that list.

    Raises:
        ValueError: if the sentence is not a dict or is missing any
            required field.
    """
    if not isinstance(sentence, dict):
        raise ValueError(
            f"{container_name}[{index}] must be a dict, "
            f"got {type(sentence).__name__}"
        )

    missing = [f for f in _REQUIRED_SENTENCE_FIELDS if f not in sentence]
    if missing:
        raise ValueError(
            f"{container_name}[{index}] is missing required field(s): "
            f"{missing}. Sentence keys present: {sorted(sentence.keys())}"
        )

    if not isinstance(sentence["text"], str):
        raise ValueError(
            f"{container_name}[{index}]['text'] must be a string, "
            f"got {type(sentence['text']).__name__}"
        )

    if not isinstance(sentence["is_bridge"], bool):
        raise ValueError(
            f"{container_name}[{index}]['is_bridge'] must be a bool, "
            f"got {type(sentence['is_bridge']).__name__}"
        )

    if not isinstance(sentence["score"], (int, float)):
        raise ValueError(
            f"{container_name}[{index}]['score'] must be numeric, "
            f"got {type(sentence['score']).__name__}"
        )


def _validate_sentence_list(sentences: Any, container_name: str) -> None:
    """
    Validate that ``sentences`` is a list of well-formed sentence dicts.

    Raises:
        ValueError: if ``sentences`` is not a list, or any element
            fails sentence validation.
    """
    if not isinstance(sentences, list):
        raise ValueError(
            f"{container_name} must be a list, got {type(sentences).__name__}"
        )

    for idx, sentence in enumerate(sentences):
        _validate_sentence(sentence, container_name, idx)


def _validate_layer8_output(layer8_output: Any) -> None:
    """
    Validate the top-level structure of ``layer8_output``.

    Raises:
        ValueError: if ``layer8_output`` is malformed in any way.
    """
    if not isinstance(layer8_output, dict):
        raise ValueError(
            f"layer8_output must be a dict, got {type(layer8_output).__name__}"
        )

    for field in _REQUIRED_LAYER8_FIELDS:
        if field not in layer8_output:
            raise ValueError(
                f"layer8_output is missing required field: '{field}'"
            )

    _validate_sentence_list(layer8_output["selected_sentences"], "selected_sentences")
    _validate_sentence_list(layer8_output["candidate_sentences"], "candidate_sentences")

    stats = layer8_output["stats"]
    if not isinstance(stats, dict):
        raise ValueError(f"layer8_output['stats'] must be a dict, got {type(stats).__name__}")

    missing_stats = [f for f in _REQUIRED_STATS_FIELDS if f not in stats]
    if missing_stats:
        raise ValueError(
            f"layer8_output['stats'] is missing required field(s): {missing_stats}"
        )


# ---------------------------------------------------------------------------
# Step 2: Lightweight token-based term matching
# ---------------------------------------------------------------------------
#
# Improvement #1: matching is now token-based instead of raw substring
# matching. A query term such as "satellite" must now match the
# sentence "satellites orbit earth", and "car" must match "cars". This
# is achieved with case folding plus a minimal, dependency-free plural
# -stripping heuristic -- not a full lemmatizer/stemmer -- to keep
# Layer 9 lightweight per the layer's design constraints.
#
# Improvement #5: tokenization results are memoized (via lru_cache, on
# the immutable sentence/term text) so repeated coverage computations
# over the same small sentence pool don't re-tokenize the same string
# multiple times.

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _normalize_token(token: str) -> str:
    """
    Lightly normalize a single lowercase word token.

    Applies a minimal heuristic plural-stripping rule (not a full
    lemmatizer) so that simple plural/singular variants are treated
    as the same concept.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"

    # Remove "es" only for common English plural endings
    if (
        len(token) > 4
        and token.endswith("es")
        and (
            token[:-2].endswith(("s", "x", "z"))
            or token[:-2].endswith(("ch", "sh"))
        )
    ):
        return token[:-2]

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token

@lru_cache(maxsize=None)
def _tokenize(text: str) -> FrozenSet[str]:
    """Tokenize and normalize free text into a set of comparable tokens."""
    return frozenset(_normalize_token(w) for w in _WORD_RE.findall(text.lower()))


def _term_in_sentence(term: str, sentence_text: str) -> bool:
    """
    Check whether ``term`` is represented within ``sentence_text``.

    A term is considered present when every normalized token making up
    the term also appears among the sentence's normalized tokens. This
    allows multi-word terms (e.g. "United Nations") to match
    regardless of surrounding punctuation/case, while still being far
    cheaper than full NLP entity matching.
    """
    term_tokens = _tokenize(term)
    if not term_tokens:
        return False
    sentence_tokens = _tokenize(sentence_text)
    return term_tokens.issubset(sentence_tokens)


def _compute_covered_terms(
    sentences: List[Dict[str, Any]], terms: List[str]
) -> Set[str]:
    """
    Determine which of ``terms`` are covered by the given ``sentences``.

    Args:
        sentences: Sentences whose text will be scanned.
        terms: Candidate terms (entities or keywords) to check for.

    Returns:
        The subset of ``terms`` that appear in at least one sentence's text.
    """
    covered: Set[str] = set()
    for term in terms:
        for sentence in sentences:
            if _term_in_sentence(term, sentence["text"]):
                covered.add(term)
                break
    return covered


def _coverage_score(
    sentences: List[Dict[str, Any]], entities: List[str], keywords: List[str]
) -> int:
    """
    Compute a simple integer coverage score: the total number of unique
    distinct query entities and keywords represented across ``sentences``.

    This is the sole metric Layer 9 uses to compare two evidence sets
    (e.g. before/after a candidate replacement) -- it measures *query
    coverage diversity*, not document-source diversity.
    """
    covered_entities = _compute_covered_terms(sentences, entities)
    covered_keywords = _compute_covered_terms(sentences, keywords)
    return len(covered_entities) + len(covered_keywords)


# ---------------------------------------------------------------------------
# Step 3: Removable sentence pool
# ---------------------------------------------------------------------------

def _removable_sentences(
    selected_sentences: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Determine which currently-selected sentences are eligible for removal.

    Bridge sentences are excluded from consideration entirely unless
    every selected sentence is a bridge sentence (i.e. no non-bridge
    alternative exists), matching the rule that bridge sentences should
    never be removed if a non-bridge alternative is available.
    """
    non_bridge = [s for s in selected_sentences if not s["is_bridge"]]
    return non_bridge if non_bridge else list(selected_sentences)


# ---------------------------------------------------------------------------
# Step 4: Replacement evaluation
# ---------------------------------------------------------------------------

def _missing_terms(
    selected_sentences: List[Dict[str, Any]], entities: List[str], keywords: List[str]
) -> Set[str]:
    """Return the query entities/keywords not yet covered by ``selected_sentences``."""
    covered_entities = _compute_covered_terms(selected_sentences, entities)
    covered_keywords = _compute_covered_terms(selected_sentences, keywords)
    missing_entities = set(entities) - covered_entities
    missing_keywords = set(keywords) - covered_keywords
    return missing_entities | missing_keywords


def _candidate_missing_coverage(
    candidate: Dict[str, Any], missing_terms: Set[str]
) -> int:
    """Count how many currently-missing query terms ``candidate`` covers."""
    return sum(1 for term in missing_terms if _term_in_sentence(term, candidate["text"]))


def _replacement_pair_key(
    improvement: int,
    missing_count: int,
    candidate: Dict[str, Any],
    removed: Dict[str, Any],
) -> Tuple[Any, ...]:
    """
    Deterministic ranking key for a candidate/removal replacement pair.

    Lower keys are preferred. Ordering (per the candidate-ranking and
    pairwise tie-break rules):

        1. Greatest overall coverage improvement (primary criterion).
        2. Candidate covers the most currently-missing query concepts.
        3. Candidate is a bridge sentence.
        4. Candidate has a higher score.
        5. Candidate sentence is shorter.
        6. Deterministic tie break on (candidate doc_id, sent_idx),
           then (removed doc_id, sent_idx).
    """
    return (
        -improvement,
        -missing_count,
        0 if candidate["is_bridge"] else 1,
        -float(candidate["score"]),
        len(candidate["text"]),
        candidate["doc_id"],
        candidate["sent_idx"],
        removed["doc_id"],
        removed["sent_idx"],
    )


def _attempt_replacement(
    selected_sentences: List[Dict[str, Any]],
    candidate_sentences: List[Dict[str, Any]],
    entities: List[str],
    keywords: List[str],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    
    missing = _missing_terms(selected_sentences, entities, keywords)
    if not missing:
        return None

    eligible_candidates = [
        c for c in candidate_sentences if _candidate_missing_coverage(c, missing) > 0
    ]
    if not eligible_candidates:
        return None

    removal_pool = _removable_sentences(selected_sentences)
    if not removal_pool:
        return None

    baseline_score = _coverage_score(selected_sentences, entities, keywords)

    # Made explicit per-sentence for readability: how much coverage each
    # removable sentence currently contributes to the selected set (its
    # marginal loss if removed alone, in isolation from any addition).
    # This is coverage-loss-based, not a count of query terms inside the
    # sentence's own text -- see _compute_marginal_loss docstring.

    best_pair: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
    best_key: Optional[Tuple[Any, ...]] = None

    for candidate in eligible_candidates:
        missing_count = _candidate_missing_coverage(candidate, missing)
        for removed in removal_pool:
            hypothetical_selected = [
                s for s in selected_sentences if s is not removed
            ] + [candidate]
            hypothetical_score = _coverage_score(hypothetical_selected, entities, keywords)
            improvement = hypothetical_score - baseline_score

            if improvement <= 0:
                continue

            key = _replacement_pair_key(improvement, missing_count, candidate, removed)
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (removed, candidate)

    return best_pair


def _run_replacement_loop(
    selected_sentences: List[Dict[str, Any]],
    candidate_sentences: List[Dict[str, Any]],
    entities: List[str],
    keywords: List[str],
    max_iterations: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Repeatedly perform beneficial replacements until query coverage is
    complete or no beneficial replacement remains.

    Args:
        selected_sentences: Working copy of the selected sentence list.
        candidate_sentences: Working copy of the candidate sentence list.
        entities: Query entities to cover.
        keywords: Query keywords to cover.
        max_iterations: Safety bound on loop iterations, equal to the
            number of candidate sentences (the maximum number of swaps
            that could ever be beneficial), guaranteeing termination.

    Returns:
        Tuple of (final_selected, final_candidates, swap_count).
    """
    selected = list(selected_sentences)
    candidates = list(candidate_sentences)
    swap_count = 0

    for _ in range(max_iterations):
        result = _attempt_replacement(selected, candidates, entities, keywords)
        if result is None:
            break

        removed_sentence, added_sentence = result

        selected = [s for s in selected if s is not removed_sentence]
        selected.append(added_sentence)

        candidates = [c for c in candidates if c is not added_sentence]
        candidates.append(removed_sentence)

        swap_count += 1

    return selected, candidates, swap_count


# ---------------------------------------------------------------------------
# Statistics assembly
# ---------------------------------------------------------------------------

def _build_diversity_stats(
    original_selected: List[Dict[str, Any]],
    final_selected: List[Dict[str, Any]],
    entities: List[str],
    keywords: List[str],
    swap_count: int,
) -> Dict[str, Any]:
    """
    Build the ``stats["diversity"]`` block.

    Note on naming: "diversity" here refers to *query coverage
    diversity* -- how many distinct query entities/keywords the final
    selected evidence set represents -- and not document-source
    diversity, deduplication, or redundancy removal. This block
    describes coverage before/after refinement and the number of
    swaps performed to improve it.

    In addition to the per-entity/per-keyword ratios, this block
    reports the raw aggregate coverage score (entities + keywords
    covered, see ``_coverage_score``) before and after refinement via
    ``coverage_before`` / ``coverage_after``, and their difference via
    ``coverage_delta`` (always >= 0, since replacements are only ever
    accepted when they strictly improve coverage).
    """
    covered_entities = _compute_covered_terms(final_selected, entities)
    covered_keywords = _compute_covered_terms(final_selected, keywords)

    missing_entities = sorted(set(entities) - covered_entities)
    missing_keywords = sorted(set(keywords) - covered_keywords)

    entity_count = len(entities)
    keyword_count = len(keywords)

    coverage_ratio = (
        len(covered_entities) / entity_count if entity_count > 0 else 1.0
    )
    keyword_coverage_ratio = (
        len(covered_keywords) / keyword_count if keyword_count > 0 else 1.0
    )

    coverage_before = _coverage_score(original_selected, entities, keywords)
    coverage_after = _coverage_score(final_selected, entities, keywords)

    return {
        "query_entities": entity_count,
        "covered_entities": len(covered_entities),
        "coverage_ratio": coverage_ratio,
        "missing_entities": missing_entities,
        "query_keywords": keyword_count,
        "covered_keywords": len(covered_keywords),
        "keyword_coverage_ratio": keyword_coverage_ratio,
        "missing_keywords": missing_keywords,
        "sentence_swaps": swap_count,
        "coverage_improved": coverage_after > coverage_before,
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "coverage_delta": coverage_after - coverage_before,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(query_analysis: Dict[str, Any], layer8_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Layer 9 (Evidence Diversity Guard) of the EARC Selection module.

    Layer 9 refines the evidence selected by Layer 8 by swapping weaker
    selected sentences for stronger candidate sentences when doing so
    improves coverage of the query's entities and keywords. It never
    changes the number of selected sentences, never recomputes scores,
    embeddings, or bridge nodes, and never performs any disk or network
    I/O -- it operates entirely on in-memory data structures.

    Args:
        query_analysis: Dict containing ``query``, ``query_type``,
            ``entities`` (list[str]), and ``keywords`` (list[str]).
        layer8_output: Dict containing ``selected_sentences``,
            ``candidate_sentences``, and ``stats`` (with ``reasoning``
            and ``budget`` sub-dicts), as produced by Layer 8.

    Returns:
        Dict with the same shape as ``layer8_output``, except:
            - ``selected_sentences`` / ``candidate_sentences`` reflect
              any replacements performed.
            - ``stats["diversity"]`` is added, describing the query
              coverage analysis and swaps performed.
            - ``stats["reasoning"]`` and ``stats["budget"]`` are passed
              through unchanged.

    Raises:
        ValueError: if ``query_analysis`` or ``layer8_output`` is
            malformed, or if any sentence is missing required fields.
    """
    _validate_query_analysis(query_analysis)
    _validate_layer8_output(layer8_output)

    entities: List[str] = list(query_analysis["entities"])
    keywords: List[str] = list(query_analysis["keywords"])

    original_selected: List[Dict[str, Any]] = layer8_output["selected_sentences"]
    original_candidates: List[Dict[str, Any]] = layer8_output["candidate_sentences"]
    original_stats: Dict[str, Any] = layer8_output["stats"]

    # Loop iterations are bounded by the candidate pool size: each swap
    # consumes exactly one candidate, so this bound guarantees termination
    # even if no early-exit condition is hit.
    max_iterations = len(original_candidates)

    final_selected, final_candidates, swap_count = _run_replacement_loop(
        original_selected, original_candidates, entities, keywords, max_iterations
    )

    diversity_stats = _build_diversity_stats(
        original_selected, final_selected, entities, keywords, swap_count
    )

    return {
        "selected_sentences": final_selected,
        "candidate_sentences": final_candidates,
        "stats": {
            "reasoning": original_stats["reasoning"],
            "budget": original_stats["budget"],
            "diversity": diversity_stats,
        },
    }