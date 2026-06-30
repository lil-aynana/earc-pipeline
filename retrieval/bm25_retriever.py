"""
retrieval/bm25_retriever.py
────────────────────────────
BM25Retriever — sparse keyword retrieval using rank_bm25.

Takes the pre-loaded BM25Okapi index and the chunk list.
Called by HybridRetriever in hybrid_retriever.py.
"""

import logging
import time
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

log = logging.getLogger('EARC-M1')


class BM25Retriever:
    """
    Wraps a pre-loaded BM25Okapi index for keyword retrieval.

    Parameters
    ----------
    bm25_index : pre-loaded BM25Okapi object
    all_chunks : List[str] — parallel to the FAISS index

    Usage
    -----
    retriever = BM25Retriever(bm25_index, all_chunks)
    results, elapsed = retriever.retrieve(keywords, top_k=15)
    # results: List[Tuple[chunk_idx: int, score: float]]
    """

    def __init__(self, bm25_index: BM25Okapi, all_chunks: List[str]):
        self._bm25   = bm25_index
        self._chunks = all_chunks

    def retrieve(
        self,
        keywords: List[str],
        top_k   : int,
    ) -> Tuple[List[Tuple[int, float]], float]:
        """
        Query BM25 with content keywords.
        Multi-word entity keywords are split into tokens (BM25 is token-level).

        Parameters
        ----------
        keywords : lemmatised content words from QueryAnalyzer
        top_k    : maximum number of results to return

        Returns
        -------
        (results, elapsed_seconds)
        results : List[(chunk_idx, bm25_score)] — only chunks with score > 0
        """
        t0 = time.time()

        if not keywords:
            log.debug('BM25: empty keyword list — returning no results')
            return [], time.time() - t0

        # BM25 is token-level; split any multi-word keywords (e.g. entity names)
        bm25_tokens: List[str] = []
        for kw in keywords:
            bm25_tokens.extend(kw.split())

        scores      = self._bm25.get_scores(bm25_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results     = [
            (int(i), float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

        return results, time.time() - t0
