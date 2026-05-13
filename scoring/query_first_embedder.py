
# scoring/query_first_embedder.py

import numpy as np
from sentence_transformers import SentenceTransformer
from config import CONFIG


class QueryFirstEmbedder:
    """
    Novel technique: embed sentences jointly with the query
    to capture query-specific relevance in the embedding space.
    """

    def __init__(self):
        print(f"Loading embedding model: {CONFIG['embedding_model']}...")
        self.model = SentenceTransformer(CONFIG['embedding_model'])
        self.cache = {}  # Store embeddings to avoid recomputation
        print("QueryFirstEmbedder ready")

    def embed_sentences(
        self,
        query: str,
        sentences: list[dict],
        batch_size: int = 64
    ) -> list[dict]:
        """
        Embed all sentences jointly with the query.

        Args:
            query: The user's question
            sentences: List of sentence dicts from retrieval pipeline
            batch_size: Number of sentences to encode at once

        Returns:
            Same sentence list with 'embedding' field filled
        """
        print(f"\nEmbedding {len(sentences)} sentences with query-first technique...")

        # Build joint input strings: "[query] [SEP] [sentence]"
        joint_inputs = []
        for sent in sentences:
            joint_text = f"{query} [SEP] {sent['text']}"
            joint_inputs.append(joint_text)

        # Encode all at once in batches
        embeddings = self.model.encode(
            joint_inputs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )

        # Attach embeddings to sentence objects
        for i, sent in enumerate(sentences):
            sent['embedding'] = embeddings[i].astype('float32')
            # Store in cache using sentence text as key
            cache_key = f"{query}||{sent['text']}"
            self.cache[cache_key] = embeddings[i]

        print(f"✓ Embedded {len(sentences)} sentences")
        return sentences

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get standalone query embedding for scoring comparisons.
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0].astype('float32')
