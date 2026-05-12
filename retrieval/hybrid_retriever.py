# retrieval/hybrid_retriever.py

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from config import CONFIG


class HybridRetriever:

    def __init__(self):
        self.bm25  = BM25Retriever()
        self.dense = DenseRetriever()
        self.documents = []
        self.is_built = False

    def build_index(self, documents: list[str]) -> None:
        """
        Build both BM25 and FAISS indices from document list.
        Call once at startup.
        """
        self.documents = documents

        print("\nBuilding BM25 index...")
        self.bm25.build_index(documents)

        print("\nBuilding FAISS dense index...")
        self.dense.build_index(documents)

        self.is_built = True
        print("\nHybrid retriever ready")

    def _reciprocal_rank_fusion(
        self,
        bm25_results:  list[tuple],
        dense_results: list[tuple],
        k_rrf: int = 60
    ) -> list[tuple[int, float]]:
        """
        Merge BM25 and dense ranked lists using RRF.

        RRF score = sum of 1 / (k_rrf + rank) across both lists.

        Returns:
            list of (doc_index, rrf_score) sorted by rrf_score descending
        """
        rrf_scores = {}

        for rank, (doc, score, idx) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0)
            rrf_scores[idx] += 1.0 / (k_rrf + rank + 1)

        for rank, (doc, score, idx) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0)
            rrf_scores[idx] += 1.0 / (k_rrf + rank + 1)

        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results

    def retrieve(
        self,
        query:      str,
        query_type: str
    ) -> list[dict]:
        """
        Retrieve top-k documents using hybrid BM25 + dense + RRF.

        Returns:
            list of dicts with keys:
                text        — document text
                doc_index   — original corpus index
                bm25_score  — raw BM25 score (0 if not in BM25 top-k)
                dense_score — raw dense similarity score (0 if not in dense top-k)
                rrf_score   — combined RRF score
                rank        — final rank after RRF (1 = best)
        """
        if not self.is_built:
            raise RuntimeError("Call build_index() before retrieve()")

        k = CONFIG["retrieval_k"][query_type]

        # Run both retrievers with larger k for better fusion
        fetch_k = k * 2

        bm25_results  = self.bm25.retrieve(query, query_type)
        dense_results = self.dense.retrieve(query, query_type)

        # Build score lookup dicts
        bm25_score_map  = {idx: score for _, score, idx in bm25_results}
        dense_score_map = {idx: score for _, score, idx in dense_results}

        # Apply RRF
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results)

        # Build result objects
        results = []
        for rank, (idx, rrf_score) in enumerate(fused[:k]):
            results.append({
                "text":        self.documents[idx],
                "doc_index":   idx,
                "bm25_score":  round(bm25_score_map.get(idx, 0.0), 4),
                "dense_score": round(dense_score_map.get(idx, 0.0), 4),
                "rrf_score":   round(rrf_score, 6),
                "rank":        rank + 1
            })

        return results
