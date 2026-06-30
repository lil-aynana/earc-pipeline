"""
retrieval/retrieval_pipeline.py
────────────────────────────────
RetrievalLayer — top-level interface for Module 1 (Stages 1, 2, 3).

Production usage (in-RAM handoff to Module 2):
    layer = RetrievalLayer(faiss_index, bm25_index, all_chunks, all_metadata, model)
    sentences, query_info = layer.retrieve(query)
    # pass directly to Module 2 — no pickle, no disk I/O

sentences  → List[SentenceObject]  (Module 2 fills embeddings + scores)
query_info → dict {query, query_type, keywords, entities, has_negation}
             propagated unchanged through all downstream modules
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

    def retrieve(self, query: str) -> Tuple[List[SentenceObject], Dict]:
        """
        Run Stage 1 → Stage 2 → Stage 3 and return Module 1 output.

        Parameters
        ----------
        query : raw user query string

        Returns
        -------
        (sentences, query_info)
        sentences  : List[SentenceObject], embedding=None on all objects
        query_info : dict with keys query, query_type, keywords, entities, has_negation
        """
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
