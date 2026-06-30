"""
evaluation/metrics.py
======================

Deterministic, dependency-free evaluation metrics for the EARC pipeline.

Implements SQuAD-style answer metrics (Exact Match and token-level F1) plus
the EARC-specific efficiency metric (context compression ratio). No external
NLP/eval libraries are required, so this runs anywhere the pipeline runs.

All functions are pure: identical inputs always produce identical outputs.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[\d+\]")


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/space.

    Citation markers like ``[1]`` are stripped first so an answer's inline
    citations never affect the match.
    """
    text = _CITATION_RE.sub(" ", text or "")
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def exact_match(prediction: str, gold_answers: List[str]) -> float:
    """Return 1.0 if the prediction exactly matches any gold answer, else 0.0."""
    pred = normalize_answer(prediction)
    return 1.0 if any(pred == normalize_answer(g) for g in gold_answers) else 0.0


def _f1(prediction: str, gold: str) -> float:
    """Token-level F1 between a prediction and a single gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, gold_answers: List[str]) -> float:
    """Return the best token-level F1 over all gold answers."""
    if not gold_answers:
        return 0.0
    return max(_f1(prediction, g) for g in gold_answers)


def answer_contains_gold(prediction: str, gold_answers: List[str]) -> float:
    """Lenient match: 1.0 if any normalized gold answer is a substring.

    Useful for generative answers that embed the gold span inside a longer
    sentence (e.g. "The telephone was invented by Alexander Graham Bell").
    """
    pred = normalize_answer(prediction)
    for g in gold_answers:
        gold = normalize_answer(g)
        if gold and gold in pred:
            return 1.0
    return 0.0


def compression_ratio(retrieved_tokens: int, selected_tokens: int) -> float:
    """Fraction of retrieved context kept after selection (lower = more compression).

    Returns ``selected / retrieved``. A value of 0.25 means the selected
    evidence is a quarter of the originally retrieved context.
    """
    if retrieved_tokens <= 0:
        return 0.0
    return selected_tokens / retrieved_tokens


def token_count(text: str) -> int:
    """Whitespace token count, matching ``SentenceObject.token_count``."""
    return len((text or "").split())
