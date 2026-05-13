
# scoring/redundancy_remover.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import CONFIG


class RedundancyRemover:
    """
    Remove duplicate and near-duplicate sentences using
    embedding similarity and document-aware rules.
    """

    def __init__(self):
        self.exact_threshold = CONFIG["redundancy_exact_threshold"]  # 0.92
        self.soft_threshold = CONFIG["redundancy_soft_threshold"]    # 0.80
        print(f"RedundancyRemover initialized")
        print(f"  Exact duplicate threshold: {self.exact_threshold}")
        print(f"  Soft duplicate threshold:  {self.soft_threshold}")

    def remove_redundancy(self, sentences: list[dict]) -> list[dict]:
        """
        Greedy deduplication using cosine similarity thresholds.

        Args:
            sentences: List of scored sentence dicts with embeddings

        Returns:
            Deduplicated sentence list
        """
        print(f"\nRemoving redundancy from {len(sentences)} sentences...")

        # Sort by score descending (keep highest-scoring duplicates)
        sorted_sentences = sorted(
            sentences,
            key=lambda x: x['score'],
            reverse=True
        )

        kept_sentences = []
        removed_count = 0

        for candidate in sorted_sentences:
            should_keep = True

            # Compare against all kept sentences
            for kept in kept_sentences:
                similarity = self._cosine_sim(
                    candidate['embedding'],
                    kept['embedding']
                )

                # Rule 1: Exact duplicate (≥0.92 similarity)
                if similarity >= self.exact_threshold:
                    should_keep = False
                    removed_count += 1
                    break

                # Rule 2: Soft duplicate (0.80-0.92) from same document
                if (self.soft_threshold <= similarity < self.exact_threshold):
                    if candidate['doc_id'] == kept['doc_id']:
                        # Same document: remove lower-scoring one (candidate)
                        should_keep = False
                        removed_count += 1
                        break
                    # Different documents: keep both (complementary framing)

            if should_keep:
                kept_sentences.append(candidate)

        print(f"✓ Removed {removed_count} redundant sentences")
        print(f"✓ Kept {len(kept_sentences)} unique sentences")

        return kept_sentences

    def _cosine_sim(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        sim = cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0][0]
        return float(sim)
