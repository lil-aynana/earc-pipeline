"""Module 2 orchestration: Layers 4-6."""

from __future__ import annotations

from typing import Any

from retrieval.sentence_object import SentenceObject
from scoring.multi_signal_scorer import MultiSignalScorer
from scoring.query_first_embedder import QueryFirstEmbedder
from scoring.redundancy_remover import RedundancyRemover


class ScoringPipeline:
    """Runs embedding, multi-signal scoring, and redundancy removal."""

    def __init__(self):
        self.embedder = QueryFirstEmbedder()
        self.scorer = MultiSignalScorer()
        self.remover = RedundancyRemover()

    def run(self, query_info: dict[str, Any], sentences: list[SentenceObject]) -> list[SentenceObject]:
        """Score Module 1 `SentenceObject` output and return deduplicated results."""
        if not sentences:
            return sentences

        query = query_info["query"]
        query_type = query_info["query_type"]

        sentences = self.embedder.embed_sentences(query, sentences)
        query_embedding = self.embedder.get_query_embedding(query)
        sentences = self.scorer.score_sentences(query, query_type, sentences, query_embedding)
        sentences = self.remover.remove_redundancy(sentences)
        return sentences

    def to_selection_records(self, sentences: list[SentenceObject]) -> list[dict[str, Any]]:
        """Convert scored `SentenceObject`s to Module 3's dict schema."""
        records: list[dict[str, Any]] = []
        for sent in sentences:
            records.append(
                {
                    "sentence_id": sent.sentence_id,
                    "text": sent.text,
                    "doc_id": sent.doc_id,
                    "dataset": sent.dataset,
                    "title": sent.title,
                    "position": sent.position,
                    "sent_idx": sent.position,
                    "retrieval_rank": sent.retrieval_rank,
                    "chunk_id": sent.chunk_id,
                    "year": sent.year,
                    "bm25_score": sent.bm25_score,
                    "faiss_score": sent.faiss_score,
                    "retrieval_score": sent.retrieval_score,
                    "embedding": sent.embedding,
                    "contains_query_entity": sent.contains_query_entity,
                    "token_count": sent.token_count,
                    "score": sent.final_score,
                    "semantic_score": sent.semantic_score,
                    "evidence_score": sent.evidence_score,
                    "evidentiality_score": sent.evidentiality_score,
                    "claim_density_score": sent.claim_density_score,
                    "temporal_score": sent.temporal_score,
                    "is_bridge": bool(sent.force_include),
                    "force_include": bool(sent.force_include),
                }
            )
        return records
