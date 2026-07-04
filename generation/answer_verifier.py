"""
generation/answer_verifier.py
=============================

Layer 13 of the EARC pipeline: Answer Verification & Citation Grounding.

The final quality gate. It inspects the answer produced by Layer 12 against
the evidence assembled by Layer 11 and reports how well the answer is
*grounded* in that evidence. It does NOT call an LLM, retrieve, rescore, or
perform any I/O.

For each answer sentence it measures the token overlap with the evidence
context. A sentence whose content-word overlap meets a configurable
threshold is considered "grounded". It also resolves the inline ``[n]``
citation markers back to their source documents and flags any markers that
point outside the available evidence range.

Answers that are deliberate refusals (e.g. "I don't have enough information
to answer") are detected up front and scored separately: declining to answer
is treated as correct, safe behavior, not as an ungrounded claim.

The output is advisory: it annotates the result with a faithfulness score
and a list of unsupported sentences but never rewrites the answer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from config import CONFIG

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CITATION_RE = re.compile(r"\[\d+\]")

# Lightweight stop-word list so common function words don't inflate overlap.
_STOPWORDS = frozenset(
    """
    a an the of to in on at for and or but if then else with without within
    is are was were be been being am do does did has have had this that these
    those it its as by from into over under between about above below up down
    out off again further once here there all any both each few more most other
    some such no nor not only own same so than too very can will just
    """.split()
)

# Prefixes that identify a deliberate "no answer" response from
# AnswerGenerator's own fallback paths (empty evidence, or a negated query the
# evidence can't support). These are NOT hallucinations and must not be
# scored as ungrounded.
_NO_ANSWER_PREFIXES = (
    "i don't have enough information to answer",
    "the retrieved evidence describes the included/affirmative set",
)


def _is_no_answer_response(answer: str) -> bool:
    """True if ``answer`` is a deliberate refusal-to-answer, not a real claim."""
    normalized = (answer or "").strip().lower()
    return any(normalized.startswith(prefix) for prefix in _NO_ANSWER_PREFIXES)


def _content_tokens(text: str) -> List[str]:
    """Lower-case alphanumeric tokens with stop-words and citation markers removed."""
    text = _CITATION_RE.sub(" ", text or "")
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS
    ]


def _split_sentences(text: str) -> List[str]:
    """
    Split the answer into logical sentences while keeping citation markers
    attached to the sentence they belong to.

    Example
    -------
    Input:
        Python was created by Guido van Rossum. [1]
        Python was first released in 1991. [2]

    Output:
        [
            "Python was created by Guido van Rossum. [1]",
            "Python was first released in 1991. [2]"
        ]
    """
    pieces = re.split(r"(?<=[.!?])\s+", (text or "").strip())

    merged: List[str] = []

    for piece in pieces:
        piece = piece.strip()

        if not piece:
            continue

        # Piece starts with a citation marker.
        # Example:
        # "[1] Python was first released in 1991."
        m = re.match(r"^(\[\d+\])\s*(.*)$", piece)

        if m and merged:
            citation = m.group(1)
            remainder = m.group(2)

            # Attach citation to previous sentence.
            merged[-1] += f" {citation}"

            # Remaining text becomes the next sentence.
            if remainder:
                merged.append(remainder)

            continue

        # Standalone citation.
        # Example:
        # "[2]"
        if re.fullmatch(r"\[\d+\]", piece):
            if merged:
                merged[-1] += f" {piece}"
            continue

        merged.append(piece)

    return merged


def _cited_markers(text: str) -> List[int]:
    """Extract all [n] citation markers from a piece of text."""
    return [int(m) for m in re.findall(r"\[(\d+)\]", text or "")]


def verify(
    answer: str,
    citations: List[Dict[str, Any]],
    evidence_context: str,
) -> Dict[str, Any]:
    """Assess grounding of ``answer`` against the evidence."""

    # Refusal responses are correct-by-design, not ungrounded claims — score
    # them separately and skip the overlap machinery entirely.
    if _is_no_answer_response(answer):
        return {
            "grounded": True,
            "faithfulness": None,
            "mean_overlap": None,
            "supported_sentences": 0,
            "scored_sentences": 0,
            "unsupported_sentences": [],
            "citation_count": 0,
            "distinct_citations": [],
            "invalid_citations": [],
            "has_citations": False,
            "is_refusal": True,
        }

    threshold = float(
        CONFIG.get("generation", {}).get(
            "grounding_overlap_threshold",
            0.5,
        )
    )

    evidence_tokens = set(_content_tokens(evidence_context))
    valid_markers = {c["marker"] for c in citations}

    sentences = _split_sentences(answer)

    supported = 0
    unsupported: List[str] = []
    overlaps: List[float] = []

    for sent in sentences:
        tokens = _content_tokens(sent)

        if not tokens:
            continue

        overlap = sum(
            1 for t in tokens if t in evidence_tokens
        ) / len(tokens)

        overlaps.append(overlap)

        if overlap >= threshold:
            supported += 1
        else:
            unsupported.append(sent)

    scored_sentences = len(overlaps)

    faithfulness = (
        supported / scored_sentences
        if scored_sentences
        else 0.0
    )

    mean_overlap = (
        sum(overlaps) / scored_sentences
        if scored_sentences
        else 0.0
    )

    used_markers = _cited_markers(answer)

    invalid_markers = sorted(
        {
            m
            for m in used_markers
            if m not in valid_markers
        }
    )

    return {
        "grounded": bool(scored_sentences) and not unsupported,
        "faithfulness": round(faithfulness, 4),
        "mean_overlap": round(mean_overlap, 4),
        "supported_sentences": supported,
        "scored_sentences": scored_sentences,
        "unsupported_sentences": unsupported,
        "citation_count": len(used_markers),
        "distinct_citations": sorted(set(used_markers)),
        "invalid_citations": invalid_markers,
        "has_citations": len(used_markers) > 0,
        "is_refusal": False,
    }
