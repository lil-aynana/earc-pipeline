"""
retrieval/loader.py
────────────────────
load_corpus_artifacts() — loads FAISS index, BM25 index, chunk shards,
metadata shards, and the embedding model from disk.

Called once at startup in pipeline.py (or the Colab notebook).
Returns (faiss_index, bm25_index, all_chunks, all_metadata, embed_model).
"""

import logging
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from retrieval.retrieval_config import EMBED_DIM

log = logging.getLogger('EARC-M1')


def load_corpus_artifacts(
    faiss_path      : Path,
    bm25_path       : Path,
    chunks_dir      : Path,
    metadata_dir    : Path,
    embed_model_name: str,
) -> Tuple:
    """
    Load all corpus artifacts from disk.

    Parameters
    ----------
    faiss_path       : path to the FAISS flat-IP index file
    bm25_path        : path to the pickled BM25Okapi object
    chunks_dir       : directory containing chunks_*.pkl shards
    metadata_dir     : directory containing metadata_*.pkl shards
    embed_model_name : HuggingFace model name for SentenceTransformer

    Returns
    -------
    (faiss_index, bm25_index, all_chunks, all_metadata, embed_model)

    Raises
    ------
    AssertionError if chunk/FAISS vector counts or FAISS dimension don't match config.
    """
    t0 = time.time()

    log.info('Loading FAISS index ...')
    faiss_index = faiss.read_index(str(faiss_path))
    log.info('  FAISS: %d vectors, dim=%d', faiss_index.ntotal, faiss_index.d)
    assert faiss_index.d == EMBED_DIM, (
        f'FAISS dim mismatch: index has {faiss_index.d}, config expects {EMBED_DIM}'
    )

    log.info('Loading BM25 index ...')
    with open(bm25_path, 'rb') as f:
        bm25_index = pickle.load(f)
    log.info('  BM25: %d documents', len(bm25_index.idf))

    log.info('Loading chunk shards ...')
    all_chunks: List[str] = []
    for shard_path in sorted(chunks_dir.glob('chunks_*.pkl')):
        with open(shard_path, 'rb') as f:
            all_chunks.extend(pickle.load(f))
        log.info('  %s: running total %d', shard_path.name, len(all_chunks))
    assert len(all_chunks) == faiss_index.ntotal, (
        f'Chunk/FAISS mismatch: {len(all_chunks)} chunks vs {faiss_index.ntotal} vectors'
    )

    log.info('Loading metadata shards ...')
    all_metadata: List[dict] = []
    for meta_path in sorted(metadata_dir.glob('metadata_*.pkl')):
        with open(meta_path, 'rb') as f:
            all_metadata.extend(pickle.load(f))
        log.info('  %s: running total %d', meta_path.name, len(all_metadata))
    assert len(all_metadata) == len(all_chunks), (
        f'Metadata/chunk mismatch: {len(all_metadata)} vs {len(all_chunks)}'
    )

    log.info('Loading embedding model: %s ...', embed_model_name)
    embed_model = SentenceTransformer(embed_model_name)
    log.info('  dim=%d', embed_model.get_sentence_embedding_dimension())

    log.info('All artifacts loaded in %.1fs', time.time() - t0)
    return faiss_index, bm25_index, all_chunks, all_metadata, embed_model
