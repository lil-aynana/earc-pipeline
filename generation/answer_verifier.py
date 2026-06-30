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

The output is advisory: it annotates the result with a faithfulness score
and a list of unsupported sentences but never rewrites the answer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from config import CONFIG

_TOKEN_RE = re.compile(r"[a-z0-9]+")

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


def _content_tokens(text: str) -> List[str]:
    """Lower-case alphanumeric tokens with stop-words removed."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def _split_sentences(text: str) -> List[str]:
    """Naive, dependency-free sentence splitter."""
    # strip citation markers before splitting so "[1]." doesn't fragment.
    pieces = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in pieces if p.strip()]


def _cited_markers(text: str) -> List[int]:
    """Extract all [n] citation markers from a piece of text."""
    return [int(m) for m in re.findall(r"\[(\d+)\]", text or "")]


def verify(
    answer: str,
    citations: List[Dict[str, Any]],
    evidence_context: str,
) -> Dict[str, Any]:
    """Assess grounding of ``answer`` against the evidence.

    Args:
        answer: The Layer 12 answer text (may contain ``[n]`` markers).
        citations: Layer 11 citation metadata (one entry per marker).
        evidence_context: The concatenated evidence text used in the prompt.

    Returns:
        A dict describing grounding, faithfulness, citation validity and any
        unsupported answer sentences.
    """
    threshold = float(
        CONFIG.get("generation", {}).get("grounding_overlap_threshold", 0.5)
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
        overlap = sum(1 for t in tokens if t in evidence_tokens) / len(tokens)
        overlaps.append(overlap)
        if overlap >= threshold:
            supported += 1
        else:
            unsupported.append(sent)

    scored_sentences = len(overlaps)
    faithfulness = (supported / scored_sentences) if scored_sentences else 0.0
    mean_overlap = (sum(overlaps) / scored_sentences) if scored_sentences else 0.0

    used_markers = _cited_markers(answer)
    invalid_markers = sorted({m for m in used_markers if m not in valid_markers})

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
    }
