"""
retrieval/retrieval_pipeline.py
────────────────────────────────
RetrievalLayer — top-level interface for Module 1 (Stages 1, 2, 3).

Production usage (in-RAM handoff to Module 2):
    layer = RetrievalLayer(faiss_index, bm25_index, all_chunks, all_metadata, model)
    m2_input, query_info = layer.retrieve(query)

    m2_input   → List[dict]  — exact format Module 2 expects, in RAM, no pickle
    query_info → dict {query, query_type, keywords, entities, has_negation}

If you need the internal SentenceObject list (for tests / Streamlit UI):
    sentences, query_info = layer.retrieve_as_objects(query)
"""

import logging
import time
from typing import Dict, List, Tuple

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_analyser import QueryAnalyzer
from retrieval.segmenter import segment_to_sentences
from retrieval.sentence_object import SentenceObject

log = logging.getLogger('EARC-M1')


class RetrievalLayer:
    """
    Module 1 top-level interface — Stages 1, 2, 3.

    Parameters
    ----------
    faiss_index  : pre-loaded FAISS index
    bm25_index   : pre-loaded BM25Okapi object
    all_chunks   : List[str] — raw chunk texts
    all_metadata : List[dict] — metadata parallel to all_chunks
    embed_model  : pre-loaded SentenceTransformer
    """

    def __init__(
        self,
        faiss_index  : faiss.Index,
        bm25_index   : BM25Okapi,
        all_chunks   : List[str],
        all_metadata : List[dict],
        embed_model  : SentenceTransformer,
    ):
        self.analyzer  = QueryAnalyzer()
        self.retriever = HybridRetriever(
            faiss_index, bm25_index, all_chunks, all_metadata, embed_model
        )

    def _run_stages(self, query: str) -> Tuple[List[SentenceObject], Dict]:
        """Internal: run Stage 1 → 2 → 3, return (sentences, query_info)."""
        log.info('=' * 60)
        log.info('Query: %r', query)
        t0 = time.time()

        query_info = self.analyzer.analyze(query)
        chunks     = self.retriever.fused_retrieve(query, query_info)
        sentences  = segment_to_sentences(
            chunks,
            query_info['entities'],
            query_info['keywords'],
        )

        log.info(
            'Module 1 total: %.2fs → %d sentences | type=%s | negation=%s',
            time.time() - t0,
            len(sentences),
            query_info['query_type'],
            query_info['has_negation'],
        )
        log.info('=' * 60)
        return sentences, query_info

    def retrieve(self, query: str) -> Tuple[List[dict], Dict]:
        """
        Run Stage 1 → 2 → 3 and return Module 2-compatible output.

        This is the production handoff method.
        Everything is in RAM — no pickle, no disk I/O.

        Returns
        -------
        (m2_input, query_info)

        m2_input   : List[dict] — each dict matches Module 2's input format exactly
        query_info : dict — {query, query_type, keywords, entities, has_negation}
                     propagated unchanged through all downstream modules
        """
        sentences, query_info = self._run_stages(query)
        m2_input = [s.to_m2_dict() for s in sentences]
        return m2_input, query_info

    def retrieve_as_objects(self, query: str) -> Tuple[List[SentenceObject], Dict]:
        """
        Same as retrieve() but returns internal SentenceObject list.

        Use this in:
        - Unit tests (test_query_analyser.py, test_segmenter.py)
        - Streamlit UI (ui/app.py)
        - Notebooks for inspection / debugging

        NOT the production path to Module 2 — use retrieve() for that.
        """
        return self._run_stages(query)
