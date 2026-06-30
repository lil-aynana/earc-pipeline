
"""Layer 5: Multi-signal scorer for Module 2."""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import spacy

from config import CONFIG
from retrieval.sentence_object import SentenceObject

nlp = spacy.load(CONFIG["spacy_model"])

EVIDENTIALITY_PATTERNS = {
    "definitions": [
        r"\bis\s+a\b", r"\bwas\s+defined\s+as\b", r"\brefers\s+to\b", r"\bmeans\b", r"\brepresents\b",
    ],
    "biographies": [
        r"\bborn\s+in\b", r"\bdied\s+in\b", r"\bwon\s+the\b", r"\bfounded\b", r"\bcreated\b", r"\binvented\b",
    ],
    "causality": [
        r"\bbecause\b", r"\btherefore\b", r"\bas\s+a\s+result\b", r"\bwhich\s+led\s+to\b", r"\bcaused\b", r"\bresulted\s+in\b",
    ],
    "superlatives": [
        r"\bthe\s+first\b", r"\bthe\s+only\b", r"\bthe\s+largest\b", r"\bthe\s+most\b", r"\bthe\s+best\b",
    ],
    "temporal": [
        r"\bin\s+\d{4}\b", r"\bduring\s+\d{4}\b", r"\bfrom\s+\d{4}\s+to\s+\d{4}\b",
    ],
}

COMPILED_PATTERNS = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in EVIDENTIALITY_PATTERNS.items()
}


class MultiSignalScorer:
    """Compute semantic and factual quality signals, then aggregate them."""

    def score_sentences(
        self,
        query: str,
        query_type: str,
        sentences: list[SentenceObject],
        query_embedding: np.ndarray,
    ) -> list[SentenceObject]:
        if not sentences:
            return sentences

        weights = CONFIG["scoring_weights"].get(
            query_type,
            CONFIG["scoring_weights"]["descriptive"],
        )
        has_temporal_context = any(
            kw in query.lower() for kw in ("recent", "latest", "current", "now", "today", "when")
        )

        for sent in sentences:
            sim_score = self._semantic_similarity(sent.embedding, query_embedding)
            evidence_score = self._evidence_score(sent.text)
            evidentiality_score = self._evidentiality_score(sent.text)
            claim_density = self._claim_density(sent.text)

            temporal_score = 0.0
            if has_temporal_context and sent.year is not None:
                temporal_score = self._temporal_recency(sent.year)

            composite = (
                weights["sim"] * sim_score
                + weights["evidence"] * evidence_score
                + weights["evidentiality"] * evidentiality_score
                + weights["density"] * claim_density
                + weights["temporal"] * temporal_score
            )

            sent.semantic_score = float(sim_score)
            sent.evidence_score = float(evidence_score)
            sent.evidentiality_score = float(evidentiality_score)
            sent.claim_density_score = float(claim_density)
            sent.temporal_score = float(temporal_score)
            sent.final_score = float(composite)

        return sentences

    @staticmethod
    def _semantic_similarity(sentence_embedding: np.ndarray | None, query_embedding: np.ndarray) -> float:
        if sentence_embedding is None:
            return 0.0
        # Embeddings are pre-normalized by encoder, so dot product is cosine similarity.
        return float(np.dot(sentence_embedding, query_embedding))

    @staticmethod
    def _evidence_score(text: str) -> float:
        doc = nlp(text)
        if len(doc) == 0:
            return 0.0
        entity_count = len(doc.ents)
        number_count = sum(1 for token in doc if token.like_num)
        factual_indicators = sum(1 for token in doc if token.pos_ in {"PROPN", "NUM"} or token.ent_type_)
        normalized = (entity_count + number_count + factual_indicators) / len(doc)
        return min(normalized * 2.0, 1.0)

    @staticmethod
    def _evidentiality_score(text: str) -> float:
        matches = 0
        for patterns in COMPILED_PATTERNS.values():
            for pattern in patterns:
                if pattern.search(text):
                    matches += 1
                    break
        return matches / max(len(COMPILED_PATTERNS), 1)

    @staticmethod
    def _claim_density(text: str) -> float:
        doc = nlp(text)
        token_count = len([t for t in doc if not t.is_space and not t.is_punct])
        if token_count == 0:
            return 0.0
        entity_count = len(doc.ents)
        number_count = sum(1 for token in doc if token.like_num)
        verb_count = sum(1 for token in doc if token.pos_ == "VERB")
        return min((entity_count + number_count + verb_count) / token_count, 1.0)

    @staticmethod
    def _temporal_recency(year: int) -> float:
        current_year = datetime.utcnow().year
        if year > current_year:
            return 0.0
        years_ago = current_year - year
        if years_ago <= 2:
            return 1.0
        if years_ago <= 5:
            return 0.9
        if years_ago <= 10:
            return 0.7
        if years_ago <= 20:
            return 0.5
        return 0.3
