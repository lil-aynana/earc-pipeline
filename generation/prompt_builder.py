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
from typing import Any, Dict, List

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
          sentence reads cleanly. If that would empty the string, the original
          is returned unchanged.
    """
    text = (text or "").strip()
    if not text or text[-1] in _TERMINAL_PUNCT:
        return text
    # Trim a single trailing partial word, then re-terminate.
    trimmed = re.sub(r"\s+\S+$", "", text).rstrip()
    if not trimmed:
        return text
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


def _ordered_evidence(
    selected_sentences: List[Dict[str, Any]],
    query_type: str,
) -> List[Dict[str, Any]]:
    """Order evidence for the prompt: bridges first, then by score desc.

    Ordering is deterministic. Bridge sentences (multi-hop connectors) are
    surfaced first so the model sees the reasoning glue early, then the
    remaining sentences by descending score, with a stable tie-break on
    ``(doc_id, position)``.
    """
    limit = _max_context_sentences(query_type)

    def sort_key(item):
        sent = item
        is_bridge = bool(sent.get("is_bridge", False))
        score = float(sent.get("score", 0.0) or 0.0)
        doc_id = str(sent.get("doc_id", ""))
        position = sent.get("position", sent.get("sent_idx", 0)) or 0
        # bridge first (0 before 1), then high score first (negate), then stable
        return (0 if is_bridge else 1, -score, doc_id, position)

    ordered = sorted(selected_sentences, key=sort_key)
    return ordered[:limit]


def build_context(
    selected_sentences: List[Dict[str, Any]],
    query_type: str = "descriptive",
) -> Dict[str, Any]:
    """Build the numbered, citation-tagged evidence context block.

    Returns a dict with:
        ``context``   : str  — the formatted "[1] ... [2] ..." evidence text
        ``citations`` : list — per-marker citation metadata
        ``evidence``  : list — the ordered evidence dicts actually used
    """
    ordered = _ordered_evidence(selected_sentences, query_type)

    lines: List[str] = []
    citations: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    for i, sent in enumerate(ordered, 1):
        text = _sanitize_evidence_text(str(sent.get("text", "")))
        lines.append(f"[{i}] {text}")
        citations.append(
            {
                "marker": i,
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
        # selection objects.
        ev = dict(sent)
        ev["text"] = text
        evidence.append(ev)

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
) -> Dict[str, Any]:
    """Assemble the full LLM prompt for Layer 12.

    Returns a dict with ``prompt`` (the full instruction text the LLM
    receives), plus ``context``, ``citations`` and ``evidence`` carried
    through from :func:`build_context` so downstream layers can reuse them.
    """
    ctx = build_context(selected_sentences, query_type)

    if not ctx["evidence"]:
        prompt = (
            "You are a careful question-answering assistant.\n\n"
            "No evidence was provided.\n\n"
            f"Question: {query}\n\n"
            "Reply exactly: \"I don't have enough information to answer.\""
        )
    else:
        prompt = (
            "You are a careful question-answering assistant. Answer strictly "
            "from the numbered evidence and never invent facts.\n\n"
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
