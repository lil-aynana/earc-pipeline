
"""Layer 6: Redundancy removal for scored sentence candidates."""

from __future__ import annotations

import numpy as np

from config import CONFIG
from retrieval.sentence_object import SentenceObject


class RedundancyRemover:
    """Greedy duplicate removal using embedding similarity thresholds."""

    def __init__(self):
        self.exact_threshold = float(CONFIG["redundancy_exact_threshold"])
        self.soft_threshold = float(CONFIG["redundancy_soft_threshold"])

    def remove_redundancy(self, sentences: list[SentenceObject]) -> list[SentenceObject]:
        if not sentences:
            return sentences

        sorted_sentences = sorted(
            sentences,
            key=lambda s: s.final_score,
            reverse=True,
        )

        kept: list[SentenceObject] = []
        for candidate in sorted_sentences:
            if candidate.embedding is None:
                kept.append(candidate)
                continue

            should_keep = True
            for existing in kept:
                if existing.embedding is None:
                    continue
                similarity = self._cosine_sim(candidate.embedding, existing.embedding)

                if similarity >= self.exact_threshold:
                    should_keep = False
                    break

                if self.soft_threshold <= similarity < self.exact_threshold:
                    if candidate.doc_id == existing.doc_id:
                        should_keep = False
                        break

            if should_keep:
                kept.append(candidate)

        # Restore stable order for downstream modules.
        kept.sort(key=lambda s: (s.retrieval_rank, s.doc_id, s.position))
        return kept

    @staticmethod
    def _cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
