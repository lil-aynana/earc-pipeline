"""
generation/prompt_builder.py
============================

Layer 11 of the EARC pipeline: Prompt Construction.

Takes the evidence selected by the Selection module (Layers 7-10) and the
query analysis from Module 1, and deterministically assembles:

    * an ordered, numbered, citation-tagged evidence context block, and
    * a query-type-aware instruction prompt that asks an LLM to answer the
      question using ONLY that evidence and to cite sources with [n] markers.

This layer performs no LLM calls, no scoring, no retrieval, and no I/O. It is
pure and deterministic: identical inputs always produce identical prompts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from config import CONFIG

# Characters that legitimately close a complete sentence.
_TERMINAL_PUNCT = '.!?…"\')]'


def _sanitize_evidence_text(text: str) -> str:
    """Safety net for chunk-boundary truncation that slips past segmentation.

    The corpus was built with a character-based chunker, so some sentences can
    be severed mid-word (e.g. ``"...managed as a waste pr"``). Module 1's
    segmenter now drops such trailing fragments, but this acts as a second
    line of defence so the answer text never exposes a half-word.

    Behaviour (deliberately conservative):
        * If the text already ends in terminal punctuation, it is returned
          unchanged.
        * Otherwise the final whitespace-delimited token is treated as a
          possible truncated word and removed, and a period is appended so the
          sentence reads cleanly.
        * If trimming the trailing partial word would remove more than half the
          sentence, the truncation is too severe to safely repair — an empty
          string is returned so the caller drops the sentence rather than risk
          asserting a grammatically clean but factually incomplete claim.
    """
    text = (text or "").strip()
    if not text or text[-1] in _TERMINAL_PUNCT:
        return text
    # Trim a single trailing partial word, then re-terminate.
    trimmed = re.sub(r"\s+\S+$", "", text).rstrip()
    if not trimmed or len(trimmed) < 0.5 * len(text):
        return ""
    if trimmed[-1] not in _TERMINAL_PUNCT:
        trimmed += "."
    return trimmed


def _max_context_sentences(query_type: str) -> int:
    """Return the cap on evidence sentences for a given query type."""
    gen_cfg = CONFIG.get("generation", {})
    per_type = gen_cfg.get("max_context_sentences", {})
    return int(
        per_type.get(
            query_type,
            gen_cfg.get("default_max_context_sentences", 8),
        )
    )


def _dedupe_exact(
    selected_sentences: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop exact-duplicate evidence sentences, keeping first occurrence.

    Two sentences are considered duplicates when their whitespace-normalised,
    lower-cased text is identical. This is a cheap guard against the same
    sentence being surfaced twice (e.g. via different retrieval paths) and
    wasting a citation slot; semantic/paraphrase dedup is intentionally out of
    scope here.
    """
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for s in selected_sentences:
        key = re.sub(r"\s+", " ", str(s.get("text", "")).strip().lower())
        if key and key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped


def _ordered_evidence(
    selected_sentences: List[Dict[str, Any]],
    query_type: str,
    max_context: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Order evidence for the prompt: anchor first, then bridges, then support.

    The anchor is the highest-scoring non-bridge sentence — typically the
    sentence that introduces the main entity/topic the question is about.
    Bridge sentences (multi-hop connectors, flagged upstream by the reasoning
    chain graph) come next, so the model sees the reasoning glue immediately
    after being told what the anchor topic is. Remaining sentences follow by
    descending score, with a stable tie-break on ``(doc_id, position)``.

    If every selected sentence is flagged as a bridge (no non-bridge anchor
    candidate exists), ordering falls back to plain score-descending order.

    ``max_context`` overrides the per-query-type sentence cap when provided
    (used by the standard-RAG baseline, which must keep the full retrieved
    context uncompressed). When ``None`` the config cap for ``query_type``
    applies.
    """
    limit = (
        _max_context_sentences(query_type)
        if max_context is None
        else int(max_context)
    )

    selected_sentences = _dedupe_exact(selected_sentences)

    non_bridges = [s for s in selected_sentences if not s.get("is_bridge", False)]
    anchor = (
        max(non_bridges, key=lambda s: float(s.get("score", 0.0) or 0.0))
        if non_bridges
        else None
    )

    def _stable_key(s: Dict[str, Any]) -> tuple:
        """Identity key that survives copying/serialisation of the dict."""
        return (
            str(s.get("doc_id", "")),
            s.get("position", s.get("sent_idx", 0)) or 0,
            str(s.get("text", "")),
        )

    # Compare by a stable key rather than object identity so anchor detection
    # keeps working even if the sentence list is copied upstream.
    anchor_key = _stable_key(anchor) if anchor is not None else None

    def sort_key(item):
        sent = item
        is_anchor = anchor_key is not None and _stable_key(sent) == anchor_key
        is_bridge = bool(sent.get("is_bridge", False))
        score = float(sent.get("score", 0.0) or 0.0)
        doc_id = str(sent.get("doc_id", ""))
        position = sent.get("position", sent.get("sent_idx", 0)) or 0
        # rank: anchor (0) -> bridges (1) -> everything else (2), then by score, then stable
        rank = 0 if is_anchor else (1 if is_bridge else 2)
        return (rank, -score, doc_id, position)

    ordered = sorted(selected_sentences, key=sort_key)
    return ordered[:limit]

def build_context(
    selected_sentences: List[Dict[str, Any]],
    query_type: str = "descriptive",
    max_context: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the numbered, citation-tagged evidence context block.

    Returns a dict with:
        ``context``   : str  — the formatted "[1] ... [2] ..." evidence text
        ``citations`` : list — per-marker citation metadata
        ``evidence``  : list — the ordered evidence dicts actually used

    ``max_context`` overrides the per-query-type sentence cap when provided.
    """
    ordered = _ordered_evidence(selected_sentences, query_type, max_context)

    lines: List[str] = []
    citations: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    marker = 1
    for sent in ordered:
        text = _sanitize_evidence_text(str(sent.get("text", "")))
        if not text:
            # Empty or unrecoverably truncated — drop it so it never gets a
            # citation marker or an empty "[n] " line in the context block.
            continue
        lines.append(f"[{marker}] {text}")
        citations.append(
            {
                "marker": marker,
                "doc_id": sent.get("doc_id"),
                "dataset": sent.get("dataset"),
                "title": sent.get("title"),
                "is_bridge": bool(sent.get("is_bridge", False)),
                "score": float(sent.get("score", 0.0) or 0.0),
                "text": text,
            }
        )
        # Shallow-copy with sanitized text so downstream layers (e.g. the
        # extractive backend) reuse clean text without mutating the upstream
        # selection objects. ``marker`` is preserved so citations stay correct
        # even when a non-contiguous subset is later stitched (negation path).
        ev = dict(sent)
        ev["text"] = text
        ev["marker"] = marker
        evidence.append(ev)
        marker += 1

    return {
        "context": "\n".join(lines),
        "citations": citations,
        "evidence": evidence,
    }


def _instruction(query_type: str, has_negation: bool = False) -> str:
    """Return a query-type-specific instruction line.

    When the query contains a negation/exclusion (``has_negation``), an
    explicit directive is prepended so the model answers the *excluded* set
    rather than the affirmative one, and is told to say so when the evidence
    does not support enumerating the exclusion.
    """
    negation_directive = (
        "IMPORTANT: This question is negated/exclusionary (e.g. 'not', "
        "'except', 'without'). Answer the EXCLUSION, not the affirmative. "
        "If the evidence only lists the included/affirmative items and does "
        "not support identifying what is excluded, say so explicitly instead "
        "of listing the included items. "
    )
    prefix = negation_directive if has_negation else ""

    qt = (query_type or "").strip().lower()
    if qt == "factoid":
        body = (
            "Answer the question with a single, precise fact in one short "
            "sentence. Use only the evidence above and cite the sentence "
            "number(s) you used with [n]."
        )
    elif qt == "multi_hop":
        body = (
            "Answer the question by connecting facts across the evidence "
            "above. Explain the link between the relevant pieces in 2-4 "
            "sentences and cite every supporting sentence with [n]."
        )
    else:
        # descriptive / default
        body = (
            "Answer the question thoroughly using only the evidence above. "
            "Write 2-4 sentences and cite each supporting sentence with [n]."
        )
    return prefix + body


def build_prompt(
    query: str,
    selected_sentences: List[Dict[str, Any]],
    query_type: str = "descriptive",
    has_negation: bool = False,
    max_context: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble the full LLM prompt for Layer 12.

    Returns a dict with ``prompt`` (the full instruction text the LLM
    receives), plus ``context``, ``citations`` and ``evidence`` carried
    through from :func:`build_context` so downstream layers can reuse them.

    ``max_context`` overrides the per-query-type sentence cap when provided
    (used by the standard-RAG baseline to keep the full retrieved context).
    """
    ctx = build_context(selected_sentences, query_type, max_context)

    if not ctx["evidence"]:
        prompt = (
            "You are a careful question-answering assistant.\n\n"
            "No evidence was provided.\n\n"
            f"Question: {query}\n\n"
            "Reply exactly: \"I don't have enough information to answer.\""
        )
    else:
        prompt = (
            "You are a careful question-answering assistant. Some of the "
            "numbered evidence below may be irrelevant to the question — "
            "ignore it. Answer strictly from the relevant evidence and never "
            "invent facts or use outside knowledge. If the evidence does not "
            "contain the answer, reply exactly: \"I don't have enough "
            "information to answer.\" Do not guess.\n\n"
            "Evidence:\n"
            f"{ctx['context']}\n\n"
            f"Question: {query}\n\n"
            f"{_instruction(query_type, has_negation)}\n\n"
            "Answer:"
        )

    return {
        "prompt": prompt,
        "context": ctx["context"],
        "citations": ctx["citations"],
        "evidence": ctx["evidence"],
    }
