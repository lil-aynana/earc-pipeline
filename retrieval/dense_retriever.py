# retrieval/dense_retriever.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import CONFIG


class DenseRetriever:

    def __init__(self):
        print(f"Loading embedding model: {CONFIG['embedding_model']}...")
        self.model = SentenceTransformer(CONFIG["embedding_model"])
        self.index = None
        self.documents = []
        self.embeddings = None
        self.is_built = False
        print("Dense retriever loaded")

    def build_index(self, documents: list[str]) -> None:
        """
        Encode all documents and build FAISS index.
        Call once at startup — not per query.
        """
        self.documents = documents
        print(f"Encoding {len(documents)} documents for FAISS index...")

        embeddings = self.model.encode(
            documents,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.embeddings = embeddings.astype("float32")
        faiss.normalize_L2(self.embeddings)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

        self.is_built = True
        print(f"FAISS index built — {self.index.ntotal} vectors · {dimension} dimensions")

    def retrieve(
        self,
        query: str,
        query_type: str
    ) -> list[tuple[str, float, int]]:
        """
        Retrieve top-k documents using dense similarity search.

        Returns:
            list of (document_text, similarity_score, doc_index)
        """
        if not self.is_built:
            raise RuntimeError("Call build_index() before retrieve()")

        k = CONFIG["retrieval_k"][query_type]

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = [
            (self.documents[idx], float(scores[0][i]), int(idx))
            for i, idx in enumerate(indices[0])
            if idx != -1
        ]

        return results

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Return standalone query embedding for use in scoring step.
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(embedding)
        return embedding[0]
