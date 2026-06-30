"""
retrieval/
──────────
Module 1: Retrieval Layer — Stages 1, 2, 3 of the EARC pipeline.

Public API (import these in pipeline.py and other modules):

    from retrieval import RetrievalLayer, SentenceObject
    from retrieval.loader import load_corpus_artifacts
"""

from retrieval.retrieval_pipeline import RetrievalLayer
from retrieval.sentence_object import SentenceObject

__all__ = ['RetrievalLayer', 'SentenceObject']
