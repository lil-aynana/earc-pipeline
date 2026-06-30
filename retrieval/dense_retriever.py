"""
retrieval/dense_retriever.py
─────────────────────────────
DenseRetriever — semantic retrieval using FAISS IndexFlatIP.

Embeds the full query with SentenceTransformer and performs
inner-product search (cosine similarity since embeddings are normalised).
Called by HybridRetriever in hybrid_retriever.py.
"""

import logging
import time
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger('EARC-M1')


class DenseRetriever:
    """
    Wraps a pre-loaded FAISS index for dense semantic retrieval.

    Parameters
    ----------
    faiss_index : pre-loaded FAISS IndexFlatIP (or similar)
    embed_model : pre-loaded SentenceTransformer

    Usage
    -----
    retriever = DenseRetriever(faiss_index, embed_model)
    results, elapsed = retriever.retrieve(query, top_k=15)
    # results: List[Tuple[chunk_idx: int, cosine_score: float]]
    """

    def __init__(self, faiss_index: faiss.Index, embed_model: SentenceTransformer):
        self._faiss = faiss_index
        self._model = embed_model

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> Tuple[List[Tuple[int, float]], float]:
        """
        Embed the full query and search FAISS IndexFlatIP.

        normalize_embeddings=True → inner product == cosine similarity.

        Parameters
        ----------
        query : raw query string (full, not keyword-filtered)
        top_k : number of nearest neighbours to return

        Returns
        -------
        (results, elapsed_seconds)
        results : List[(chunk_idx, cosine_score)] — only valid (idx >= 0) results
        """
        t0 = time.time()

        q_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype('float32')

        scores, indices = self._faiss.search(q_vec, top_k)

        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0  # FAISS returns -1 for padding when k > ntotal
        ]

        return results, time.time() - t0
