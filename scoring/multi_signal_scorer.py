
# scoring/multi_signal_scorer.py

import re
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import CONFIG

# Load spaCy once at module level
nlp = spacy.load(CONFIG["spacy_model"])

# Evidentiality patterns
EVIDENTIALITY_PATTERNS = {
    "definitions": [
        r'\bis\s+a\b', r'\bwas\s+defined\s+as\b', r'\brefers\s+to\b',
        r'\bmeans\b', r'\brepresents\b'
    ],
    "biographies": [
        r'\bborn\s+in\b', r'\bdied\s+in\b', r'\bwon\s+the\b',
        r'\bfounded\b', r'\bcreated\b', r'\binvented\b'
    ],
    "causality": [
        r'\bbecause\b', r'\btherefore\b', r'\bas\s+a\s+result\b',
        r'\bwhich\s+led\s+to\b', r'\bcaused\b', r'\bresulted\s+in\b'
    ],
    "superlatives": [
        r'\bthe\s+first\b', r'\bthe\s+only\b', r'\bthe\s+largest\b',
        r'\bthe\s+most\b', r'\bthe\s+best\b'
    ],
    "temporal": [
        r'\bin\s+\d{4}\b', r'\bduring\s+\d{4}\b', r'\bfrom\s+\d{4}\s+to\s+\d{4}\b'
    ]
}

# Compile all patterns
COMPILED_PATTERNS = {}
for category, patterns in EVIDENTIALITY_PATTERNS.items():
    COMPILED_PATTERNS[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

# Year extraction pattern
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')


class MultiSignalScorer:
    """
    Novel multi-signal scoring combining semantic similarity,
    evidence density, evidentiality, claim density, and temporal signals.
    """

    def __init__(self):
        self.query_embedding = None
        print("MultiSignalScorer initialized")

    def score_sentences(
        self,
        query: str,
        query_type: str,
        sentences: list[dict],
        query_embedding: np.ndarray
    ) -> list[dict]:
        """
        Compute composite relevance score for each sentence.

        Args:
            query: The user's question
            query_type: One of 'factoid', 'descriptive', 'multi_hop'
            sentences: List of sentence dicts with embeddings filled
            query_embedding: Standalone query embedding vector

        Returns:
            Same sentence list with 'score' field filled
        """
        print(f"\nScoring {len(sentences)} sentences with multi-signal approach...")
        print(f"Query type: {query_type}")

        self.query_embedding = query_embedding
        weights = CONFIG["scoring_weights"][query_type]

        # Check if query has temporal keywords
        temporal_keywords = ["recent", "latest", "current", "now", "today", "when"]
        has_temporal_context = any(kw in query.lower() for kw in temporal_keywords)

        for sent in sentences:
            # Signal 1: Semantic similarity
            sim_score = self._semantic_similarity(sent['embedding'])

            # Signal 2: Evidence score
            evidence_score = self._evidence_score(sent['text'])

            # Signal 3: Evidentiality score
            evidentiality_score = self._evidentiality_score(sent['text'])

            # Signal 4: Claim density
            claim_density = self._claim_density(sent['text'])

            # Signal 5: Temporal recency
            temporal_score = 0.0
            if has_temporal_context and sent['temporal_year']:
                temporal_score = self._temporal_recency(sent['temporal_year'])

            # Weighted combination
            composite_score = (
                weights['sim'] * sim_score +
                weights['evidence'] * evidence_score +
                weights['evidentiality'] * evidentiality_score +
                weights['density'] * claim_density +
                weights['temporal'] * temporal_score
            )

            sent['score'] = round(composite_score, 4)

        print(f"✓ Scored {len(sentences)} sentences")
        return sentences

    def _semantic_similarity(self, sentence_embedding: np.ndarray) -> float:
        """Cosine similarity between sentence and query embeddings."""
        sim = cosine_similarity(
            sentence_embedding.reshape(1, -1),
            self.query_embedding.reshape(1, -1)
        )[0][0]
        return float(sim)

    def _evidence_score(self, text: str) -> float:
        """Count named entities, numbers, and factual indicators."""
        doc = nlp(text)

        # Count entities
        entity_count = len(doc.ents)

        # Count numbers
        number_count = sum(1 for token in doc if token.like_num)

        # Count factual indicators (specific nouns, proper nouns)
        factual_indicators = sum(
            1 for token in doc
            if token.pos_ in ['PROPN', 'NUM'] or token.ent_type_
        )

        # Normalize by sentence length
        total = entity_count + number_count + factual_indicators
        normalized = total / len(doc) if len(doc) > 0 else 0

        # Scale to 0-1 range (cap at reasonable maximum)
        return min(normalized * 2, 1.0)

    def _evidentiality_score(self, text: str) -> float:
        """Detect factual sentence structures using regex patterns."""
        matches = 0

        for category, patterns in COMPILED_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(text):
                    matches += 1
                    break  # Count each category once max

        # Normalize: max 5 categories, so divide by 5
        return matches / 5.0

    def _claim_density(self, text: str) -> float:
        """Information per token: (entities + numbers + verbs) / token_count."""
        doc = nlp(text)

        entity_count = len(doc.ents)
        number_count = sum(1 for token in doc if token.like_num)
        verb_count = sum(1 for token in doc if token.pos_ == 'VERB')

        token_count = len([t for t in doc if not t.is_space and not t.is_punct])

        if token_count == 0:
            return 0.0

        density = (entity_count + number_count + verb_count) / token_count
        return min(density, 1.0)  # Cap at 1.0

    def _temporal_recency(self, year: int) -> float:
        """Score based on year recency with exponential decay."""
        current_year = 2026  # As per your system prompt date

        if year > current_year:
            return 0.0  # Future years get no bonus

        years_ago = current_year - year

        # Decay function: recent years score higher
        if years_ago <= 2:
            return 1.0
        elif years_ago <= 5:
            return 0.9
        elif years_ago <= 10:
            return 0.7
        elif years_ago <= 20:
            return 0.5
        else:
            return 0.3
