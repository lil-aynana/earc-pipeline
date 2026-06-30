"""
retrieval/sentence_object.py
─────────────────────────────
SentenceObject — the shared data model that flows through the entire EARC pipeline.

    Module 1 populates : all structural / identity fields + contains_query_entity
    Module 2 fills in  : embedding (real ndarray), all score fields
    Module 3 sets      : force_include

sentence_id is a stable string identifier:
    '{dataset}:{doc_id}:{chunk_id}:{position}'
Used for deduplication, debugging, and evaluation logging.

embedding is None until Module 2 fills it in.
Storing None instead of np.zeros(384) saves ~1.5 KB per sentence
across 100+ candidate sentences per query.

retrieval_score is the normalised RRF score from Module 1.
Reserved for Module 2 to use as an additional ranking signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SentenceObject:
    # ── Identity / structural (Module 1) ─────────────────────────────────────────
    sentence_id           : str            # stable: dataset:doc_id:chunk_id:position
    text                  : str
    doc_id                : str
    dataset               : str
    title                 : str
    position              : int            # sentence index within its chunk
    retrieval_rank        : int            # 1-based RRF rank of parent chunk
    chunk_id              : int            # index into all_chunks / FAISS
    year                  : Optional[int]  # document year for temporal scoring
    bm25_score            : float          # raw BM25 score of parent chunk
    faiss_score           : float          # raw FAISS cosine score of parent chunk
    retrieval_score       : float          # normalised RRF score (reserved for Module 2)
    contains_query_entity : bool           # entity OR keyword overlap with query
    token_count           : int            # whitespace-split word count

    # ── Embedding (Module 2 fills this) ──────────────────────────────────────────
    embedding             : Optional[np.ndarray] = None  # 384-dim; None until Module 2

    # ── Scores (Module 2 fills these) ────────────────────────────────────────────
    semantic_score        : float = 0.0
    evidence_score        : float = 0.0
    evidentiality_score   : float = 0.0
    claim_density_score   : float = 0.0
    temporal_score        : float = 0.0
    final_score           : float = 0.0

    # ── Selection flag (Module 3 sets this) ──────────────────────────────────────
    force_include         : bool = False

    def __repr__(self) -> str:
        return (
            f'SentenceObject(id={self.sentence_id!r}, '
            f'rank={self.retrieval_rank}, score={self.final_score:.3f}, '
            f'entity={self.contains_query_entity}, '
            f'text={self.text[:60]!r})'
        )
