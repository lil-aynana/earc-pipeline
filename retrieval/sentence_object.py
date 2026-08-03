"""
retrieval/sentence_object.py
─────────────────────────────
SentenceObject — Module 1's internal data model.

    Module 1 populates : all structural / identity fields + contains_query_entity
    Module 2 fills in  : embedding (real ndarray), all score fields
    Module 3 sets      : force_include

sentence_id (internal): '{dataset}:{doc_id}:{chunk_id}:{sent_idx}'
Used for deduplication, debugging, and evaluation logging.

to_m2_dict() converts to the exact dict format Module 2 expects — in RAM,
no pickle, no disk I/O. This is the canonical handoff contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SentenceObject:
    # ── Identity / structural (Module 1) ─────────────────────────────────────────
    sentence_id           : str            # internal: dataset:doc_id:chunk_id:sent_idx
    text                  : str
    doc_id                : str
    dataset               : str
    title                 : str
    position              : int            # sentence index within its chunk (== sent_idx)
    retrieval_rank        : int            # 1-based RRF rank of parent chunk
    chunk_id              : int            # index into all_chunks / FAISS
    year                  : Optional[int]  # document year for temporal scoring
    bm25_score            : float          # raw BM25 score of parent chunk
    faiss_score           : float          # raw FAISS cosine score of parent chunk
    retrieval_score       : float          # normalised RRF score
    contains_query_entity : bool           # entity OR keyword overlap with query
    token_count           : int            # whitespace-split word count

    # ── Embedding (Module 2 fills this) ──────────────────────────────────────────
    embedding             : Optional[np.ndarray] = None  # None until Module 2

    # ── Scores (Module 2 fills these) ────────────────────────────────────────────
    semantic_score        : float = 0.0
    evidence_score        : float = 0.0
    evidentiality_score   : float = 0.0
    claim_density_score   : float = 0.0
    temporal_score        : float = 0.0
    final_score           : float = 0.0

    # ── Selection flags ───────────────────────────────────────────────────────────
    is_bridge             : bool = False   # Module 3 sets this
    force_include         : bool = False   # Module 3 sets this

    # ── Per-sentence query coverage (Layer 10 uses these) ────────────────────────
    # Lists of the query's entities/keywords that this sentence actually covers.
    # Populated by the segmenter; empty lists mean "no coverage" (safe default).
    sentence_entities     : list = None    # set in segmenter
    sentence_keywords     : list = None    # set in segmenter

    def __post_init__(self):
        if self.sentence_entities is None:
            self.sentence_entities = []
        if self.sentence_keywords is None:
            self.sentence_keywords = []

    # ── Module 2 handoff ─────────────────────────────────────────────────────────

    def to_m2_dict(self) -> dict:
        """
        Convert to the exact dict format Module 2 expects.

        Called in RetrievalLayer.retrieve() — returns a plain dict in RAM.
        No pickle, no disk I/O.

        Module 2 input format:
            sentence_id          : "d{doc_id}:c{chunk_id}:{sent_idx}"
            text                 : str
            doc_id               : str
            dataset              : str
            title                : str
            position             : int
            sent_idx             : int
            retrieval_rank       : int
            chunk_id             : int
            temporal_year        : int | None
            parent_bm25          : float
            parent_dense         : float
            parent_rrf           : float
            contains_query_entity: bool
            token_count          : int
            embedding            : None      ← Module 2 fills this
            score                : 0.0       ← Module 2 fills this
            is_bridge            : False     ← Module 3 sets this
            force_include        : bool
        """
        return {
            'sentence_id'          : f"d{self.doc_id}:c{self.chunk_id}:{self.position}",
            'text'                 : self.text,
            'doc_id'               : self.doc_id,
            'dataset'              : self.dataset,
            'title'                : self.title,
            'position'             : self.position,
            'sent_idx'             : self.position,   # same value, Module 2's name
            'retrieval_rank'       : self.retrieval_rank,
            'chunk_id'             : self.chunk_id,
            'temporal_year'        : self.year,
            'parent_bm25'          : self.bm25_score,
            'parent_dense'         : self.faiss_score,
            'parent_rrf'           : self.retrieval_score,
            'contains_query_entity': self.contains_query_entity,
            'token_count'          : self.token_count,
            'embedding'            : None,    # Module 2 fills this
            'score'                : 0.0,     # Module 2 fills this
            'is_bridge'            : False,   # Module 3 sets this
            'force_include'        : self.force_include,
            # Layer 10 (evidence_sufficiency) requires these fields on every sentence.
            # They contain the subset of query entities/keywords this sentence covers.
            'entities'             : list(self.sentence_entities),
            'keywords'             : list(self.sentence_keywords),
        }

    def __repr__(self) -> str:
        return (
            f'SentenceObject(id={self.sentence_id!r}, '
            f'rank={self.retrieval_rank}, score={self.final_score:.3f}, '
            f'entity={self.contains_query_entity}, '
            f'text={self.text[:60]!r})'
        )
