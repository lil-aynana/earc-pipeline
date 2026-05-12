# retrieval/bm25_retriever.py

from rank_bm25 import BM25Okapi
from config import CONFIG


class BM25Retriever:

    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.is_built = False

    def build_index(self, documents: list[str]) -> None:
        """
        Build BM25 index from list of document strings.
        Call once at startup — not per query.

        Args:
            documents: list of document text strings
        """
        self.documents = documents
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.is_built = True
        print(f"BM25 index built — {len(documents)} documents")

    def retrieve(
        self,
        query: str,
        query_type: str
    ) -> list[tuple[str, float, int]]:
        """
        Retrieve top-k documents for query using BM25.

        Args:
            query:      natural language query string
            query_type: one of factoid, descriptive, multi_hop

        Returns:
            list of (document_text, bm25_score, doc_index) tuples
        """
        if not self.is_built:
            raise RuntimeError("Call build_index() before retrieve()")

        k = CONFIG["retrieval_k"][query_type]
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:k]

        results = [
            (self.documents[i], float(scores[i]), int(i))
            for i in top_indices
        ]

        return results
