
"""Layer 4: Query-first sentence embedding for Module 2."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CONFIG
from retrieval.sentence_object import SentenceObject


class QueryFirstEmbedder:
    """Embeds candidate sentences conditioned on the input query."""

    def __init__(self, model: Optional[SentenceTransformer] = None):
        self.model = model or SentenceTransformer(CONFIG["embedding_model"])

    def embed_sentences(
        self,
        query: str,
        sentences: list[SentenceObject],
        batch_size: int = 64,
    ) -> list[SentenceObject]:
        """Fill `embedding` for each sentence with query-conditioned vectors."""
        if not sentences:
            return sentences

        joint_inputs = [f"{query} [SEP] {s.text}" for s in sentences]
        embeddings = self.model.encode(
            joint_inputs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        for i, sent in enumerate(sentences):
            sent.embedding = embeddings[i].astype(np.float32)
        return sentences

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Return a normalized embedding for the raw query."""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0].astype(np.float32)
