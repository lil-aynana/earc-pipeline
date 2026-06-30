"""
pipeline.py
────────────
EARC Pipeline — top-level entry point.

Initialises all module artifacts once and exposes a single run(query) function
that chains Module 1 → Module 2 → Module 3 → Module 4 in RAM.

Module 2 (scoring), Module 3 (selection), and Module 4 (generation) must each
expose a compatible pipeline class; stubs are provided below until those modules
are implemented by teammates.

Usage
-----
    from pipeline import EARCPipeline
    pipe = EARCPipeline()
    result = pipe.run("Who invented the telephone?")
"""

import logging
from pathlib import Path

from retrieval.loader import load_corpus_artifacts
from retrieval.retrieval_pipeline import RetrievalLayer
from retrieval.retrieval_config import (
    BM25_PATH,
    CHUNKS_DIR,
    EMBED_MODEL,
    FAISS_PATH,
    METADATA_DIR,
)
from scoring.scoring_pipeline import ScoringPipeline
from selection.selection_pipeline import SelectionPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('EARC')


class EARCPipeline:
    """
    End-to-end EARC pipeline.

    Loads all corpus artifacts once at __init__ time.
    Downstream modules (2, 3, 4) are plugged in as they are implemented.

    Parameters
    ----------
    faiss_path       : override default FAISS index path
    bm25_path        : override default BM25 index path
    chunks_dir       : override default chunks shard directory
    metadata_dir     : override default metadata shard directory
    embed_model_name : override default embedding model name
    """

    def __init__(
        self,
        faiss_path       : Path = FAISS_PATH,
        bm25_path        : Path = BM25_PATH,
        chunks_dir       : Path = CHUNKS_DIR,
        metadata_dir     : Path = METADATA_DIR,
        embed_model_name : str  = EMBED_MODEL,
    ):
        faiss_index, bm25_index, all_chunks, all_metadata, embed_model = \
            load_corpus_artifacts(
                faiss_path, bm25_path, chunks_dir, metadata_dir, embed_model_name
            )

        # Module 1 — Retrieval Layer (your module)
        self.retrieval_layer = RetrievalLayer(
            faiss_index, bm25_index, all_chunks, all_metadata, embed_model
        )

        # Module 2 — Scoring (Stages 4-6)
        self.scoring_pipeline = ScoringPipeline()

        # Module 3 — Selection (Stages 7-9)
        self.selection_pipeline = SelectionPipeline()

        # Module 4 — Generation (plug in when ready)
        # from generation.generation_pipeline import GenerationPipeline
        # self.generation_pipeline = GenerationPipeline()

        log.info('EARCPipeline ready.')

    def run(self, query: str) -> dict:
        """
        Run the full pipeline for a single query.

        Returns
        -------
        dict with at minimum:
            query      : str
            query_info : dict (query_type, keywords, entities, has_negation)
            sentences  : List[SentenceObject]   ← Module 2 output (scored + deduped)
            selected_sentences : list[dict]     ← Module 3 output
            candidate_sentences: list[dict]     ← Module 3 output
            selection_stats    : dict           ← Module 3 stats
            # answer   : str                    ← added when Module 4 is wired in
        """
        # Stage 1–3: Retrieval
        sentences, query_info = self.retrieval_layer.retrieve(query)

        # Stage 4–6: Scoring
        sentences = self.scoring_pipeline.run(query_info, sentences)
        scored_records = self.scoring_pipeline.to_selection_records(sentences)

        # Stage 7–9: Selection
        selection_output = self.selection_pipeline.run(query_info, scored_records)

        # Stage 10–12: Generation  (stub — wire in Module 4 here)
        # answer = self.generation_pipeline.generate(selected, query_info)

        return {
            'query'     : query,
            'query_info': query_info,
            'sentences' : sentences,
            'selected_sentences': selection_output['selected_sentences'],
            'candidate_sentences': selection_output['candidate_sentences'],
            'selection_stats': selection_output['stats'],
        }


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    pipe = EARCPipeline()

    test_queries = [
        'Who invented the telephone?',
        'What did Marie Curie and Albert Einstein both contribute to physics?',
        'What countries are not members of NATO?',
    ]

    for q in test_queries:
        result = pipe.run(q)
        print(f"\nQuery      : {result['query']}")
        print(f"Type       : {result['query_info']['query_type']}")
        print(f"Sentences  : {len(result['sentences'])}")
        entity_count = sum(1 for s in result['sentences'] if s.contains_query_entity)
        print(f"With entity: {entity_count}")
