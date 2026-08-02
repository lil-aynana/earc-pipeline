"""
retrieval/hybrid_retriever.py
──────────────────────────────
HybridRetriever — Stage 2: Adaptive Hybrid Retrieval.

BM25 (sparse, keyword) + FAISS (dense, semantic) fused by Reciprocal Rank Fusion.

    RRF score: rrf(c) = Σ_r [ 1 / (k + rank_r(c)) ]

Individual bm25_score and faiss_score are stored per chunk alongside rrf_score
so that Module 2 can use raw retrieval strength as an additional scoring signal
without re-running retrieval.

Granular timing is logged separately for BM25, FAISS, and RRF merge steps.
"""

import logging
import time
from typing import Dict, List, Tuple

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.retrieval_config import K_BY_TYPE, RRF_K

log = logging.getLogger('EARC-M1')


class HybridRetriever:
    """
    Stage 2 — Adaptive Hybrid Retrieval.

    Parameters
    ----------
    faiss_index  : pre-loaded FAISS index
    bm25_index   : pre-loaded BM25Okapi object
    all_chunks   : List[str] — raw chunk texts parallel to FAISS vectors
    all_metadata : List[dict] — metadata dicts parallel to all_chunks
    embed_model  : pre-loaded SentenceTransformer

    Usage
    -----
    retriever = HybridRetriever(faiss_index, bm25_index, all_chunks, all_metadata, model)
    chunks = retriever.fused_retrieve(query, query_info)
    """

    def __init__(
        self,
        faiss_index  : faiss.Index,
        bm25_index   : BM25Okapi,
        all_chunks   : List[str],
        all_metadata : List[dict],
        embed_model  : SentenceTransformer,
    ):
        self._chunks   = all_chunks
        self._metadata = all_metadata
        self._bm25_ret = BM25Retriever(bm25_index, all_chunks)
        self._dense_ret = DenseRetriever(faiss_index, embed_model)

    # ── RRF merge ─────────────────────────────────────────────────────────────────

    def _rrf_merge(
        self,
        bm25_results  : List[Tuple[int, float]],
        faiss_results : List[Tuple[int, float]],
        top_k_final   : int,
    ) -> Tuple[List[Tuple[int, float]], float]:
        """
        Reciprocal Rank Fusion.
        Chunks appearing in both lists accumulate higher RRF scores.

        Returns
        -------
        (merged_list, elapsed_seconds)
        merged_list : List[(chunk_idx, rrf_score)] sorted descending, length ≤ top_k_final
        """
        t0     = time.time()
        scores : Dict[int, float] = {}
        for rank0, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank0 + 1)
        for rank0, (idx, _) in enumerate(faiss_results):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank0 + 1)
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k_final]
        return merged, time.time() - t0

    # ── Main entry point ──────────────────────────────────────────────────────────

    def fused_retrieve(self, query: str, query_info: Dict) -> List[Dict]:
        """
        Run BM25 + FAISS + RRF. Returns a list of chunk dicts containing:
            chunk_idx, chunk_text, rrf_rank, rrf_score,
            bm25_score, faiss_score,
            doc_id, dataset, title, year

        Retrieval depths are determined by query_type via K_BY_TYPE config.

        Parameters
        ----------
        query      : raw query string
        query_info : dict from QueryAnalyzer.analyze()
        """
        qt    = query_info['query_type']
        kw    = query_info['keywords']
        k_cfg = K_BY_TYPE.get(qt)
        if k_cfg is None:
            log.warning('Unknown query type %r — falling back to factoid depths', qt)
            k_cfg = K_BY_TYPE['factoid']

        bm25_r,  t_bm25  = self._bm25_ret.retrieve(kw, k_cfg['bm25'])
        faiss_r, t_faiss = self._dense_ret.retrieve(query, k_cfg['faiss'])
        merged,  t_rrf   = self._rrf_merge(bm25_r, faiss_r, k_cfg['final'])

        log.info(
            'Retrieval [%s]: BM25=%d(%.3fs) FAISS=%d(%.3fs) RRF=%d(%.3fs)',
            qt,
            len(bm25_r),  t_bm25,
            len(faiss_r), t_faiss,
            len(merged),  t_rrf,
        )

        # Build lookup dicts for individual scores
        bm25_score_map  = {idx: score for idx, score in bm25_r}
        faiss_score_map = {idx: score for idx, score in faiss_r}

        results = []
        for rank0, (chunk_idx, rrf_score) in enumerate(merged):
            meta = self._metadata[chunk_idx]
            results.append({
                'chunk_idx'  : chunk_idx,
                'chunk_text' : self._chunks[chunk_idx],
                'rrf_score'  : rrf_score,
                'rrf_rank'   : rank0 + 1,
                'bm25_score' : bm25_score_map.get(chunk_idx, 0.0),
                'faiss_score': faiss_score_map.get(chunk_idx, 0.0),
                'doc_id'     : meta.get('doc_id', ''),
                'dataset'    : meta.get('dataset', ''),
                'title'      : meta.get('title', ''),
                'year'       : meta.get('year', None),
                'char_start' : meta.get('char_start', None),
            })
        return results
