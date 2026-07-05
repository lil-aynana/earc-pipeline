# scoring/redundancy_remover.py
 
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import CONFIG
 
_NEG = re.compile(
    r"\b(not|no|never|without|cannot|didn\'t|doesn\'t|wasn\'t|isn\'t|aren\'t)\b",
    re.I
)
 
 
def _contradicts(a: str, b: str) -> bool:
    """Cheap heuristic: differing negation-word counts flags a likely
    contradiction between two otherwise-similar sentences."""
    return abs(len(_NEG.findall(a)) - len(_NEG.findall(b))) >= 1
 
 
class RedundancyRemover:
    """
    Remove duplicate and near-duplicate sentences using
    embedding similarity and document-aware rules.
    """
 
    def __init__(self):
        self.exact_threshold = CONFIG["redundancy_exact_threshold"]  # e.g. 0.92
        self.soft_threshold = CONFIG["redundancy_soft_threshold"]    # e.g. 0.80
        print("RedundancyRemover initialized")
        print(f"  Exact duplicate threshold: {self.exact_threshold}")
        print(f"  Soft duplicate threshold:  {self.soft_threshold}")
 
    def remove_redundancy(self, sentences: list[dict]):
        """
        Greedy deduplication using cosine similarity thresholds.
 
        Returns:
            (kept_sentences, stats)
        """
        print(f"\nRemoving redundancy from {len(sentences)} sentences...")
 
        sorted_sentences = sorted(sentences, key=lambda x: x['score'], reverse=True)
 
        kept_sentences = []
        removed_count = 0
 
        for candidate in sorted_sentences:
            should_keep = True
 
            for kept in kept_sentences:
                similarity = self._cosine_sim(candidate['embedding'], kept['embedding'])
 
                # Rule 1: exact duplicate
                if similarity >= self.exact_threshold:
                    should_keep = False
                    removed_count += 1
                    break
 
                # Rule 2: soft duplicate (0.80-0.92)
                if self.soft_threshold <= similarity < self.exact_threshold:
                    if candidate['doc_id'] == kept['doc_id']:
                        # same document -> definite redundancy, drop candidate
                        should_keep = False
                        removed_count += 1
                        break
                    elif _contradicts(candidate['text'], kept['text']):
                        # cross-doc contradiction: kept sentence already scored
                        # higher, so drop the lower-scoring contradicting candidate
                        # NOTE: this intentionally overrides "preserve cross-doc
                        # diversity" when the two claims conflict - flag to your
                        # team if you want plain diverse-but-similar sentences from
                        # different docs to always survive regardless of contradiction.
                        should_keep = False
                        removed_count += 1
                        break
                    # else: different doc, no contradiction -> keep both (diversity)
 
            if should_keep:
                kept_sentences.append(candidate)
 
        stats = {
            "step": 6,
            "input_sentences": len(sorted_sentences),
            "output_sentences": len(kept_sentences),
            "removed": removed_count,
            "compression_ratio": round(len(kept_sentences) / max(len(sorted_sentences), 1), 3),
            "sample_output": {k: v for k, v in kept_sentences[0].items() if k != "embedding"} if kept_sentences else {},
        }
        print("\n=== STEP 6 OUTPUT ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return kept_sentences, stats
 
    def _cosine_sim(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        return float(sim)
