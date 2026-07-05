# scoring/multi_signal_scorer.py
 
import re
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import CONFIG
 
nlp = spacy.load(CONFIG["spacy_model"])
 
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
 
COMPILED_PATTERNS = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in EVIDENTIALITY_PATTERNS.items()
}
 
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
 
 
class MultiSignalScorer:
    """
    Novel multi-signal scoring combining semantic similarity,
    evidence density, evidentiality, claim density, and temporal signals.
    """
 
    def __init__(self):
        self.query_embedding = None
        print("MultiSignalScorer initialized")
 
    def score_sentences(self, query: str, query_type: str, sentences: list[dict], query_embedding: np.ndarray):
        """
        Compute composite relevance score for each sentence.
 
        Returns:
            (sentences, stats) - same list with 'score' + individual signal
            fields filled ('semantic_score', 'evidence_score',
            'evidentiality_score', 'claim_density_score', 'temporal_score'),
            plus a stats dict.
        """
        print(f"\nScoring {len(sentences)} sentences with multi-signal approach...")
        print(f"Query type: {query_type}")
 
        self.query_embedding = query_embedding
        weights = CONFIG["scoring_weights"][query_type]
 
        temporal_keywords = ["recent", "latest", "current", "now", "today", "when"]
        has_temporal_context = any(kw in query.lower() for kw in temporal_keywords)
 
        for sent in sentences:
            sim_score = self._semantic_similarity(sent['embedding'])
            evidence_score = self._evidence_score(sent['text'])
            evidentiality_score = self._evidentiality_score(sent['text'])
            claim_density = self._claim_density(sent['text'])
 
            temporal_score = 0.0
            # FIX: field name aligned to schema - was sent['year'] mismatch
            # upstream; scorer and schema both use 'temporal_year'.
            if has_temporal_context and sent.get('temporal_year'):
                temporal_score = self._temporal_recency(sent['temporal_year'])
 
            composite_score = (
                weights['sim'] * sim_score +
                weights['evidence'] * evidence_score +
                weights['evidentiality'] * evidentiality_score +
                weights['density'] * claim_density +
                weights['temporal'] * temporal_score
            )
 
            # Keep the individual signals on the sentence dict too (not just
            # the composite) - matches the "multi-signal" framing in the
            # architecture and gives Module 3 visibility into *why* something
            # scored the way it did.
            sent['semantic_score'] = round(sim_score, 4)
            sent['evidence_score'] = round(evidence_score, 4)
            sent['evidentiality_score'] = round(evidentiality_score, 4)
            sent['claim_density_score'] = round(claim_density, 4)
            sent['temporal_score'] = round(temporal_score, 4)
            sent['score'] = round(composite_score, 4)
 
        print(f"Scored {len(sentences)} sentences")
        scores = [s['score'] for s in sentences]
        stats = {
            "step": 5,
            "query_type": query_type,
            "weights": weights,
            "total_scored": len(sentences),
            "max_score": round(max(scores), 4),
            "min_score": round(min(scores), 4),
            "mean_score": round(float(np.mean(scores)), 4),
            "top3": [s['text'][:60] for s in sorted(sentences, key=lambda x: x['score'], reverse=True)[:3]],
        }
        print("\n=== STEP 5 OUTPUT ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return sentences, stats
 
    def _semantic_similarity(self, sentence_embedding: np.ndarray) -> float:
        sim = cosine_similarity(
            sentence_embedding.reshape(1, -1),
            self.query_embedding.reshape(1, -1)
        )[0][0]
        return float(sim)
 
    def _evidence_score(self, text: str) -> float:
        doc = nlp(text)
        entity_count = len(doc.ents)
        number_count = sum(1 for token in doc if token.like_num)
        factual_indicators = sum(
            1 for token in doc
            if token.pos_ in ['PROPN', 'NUM'] or token.ent_type_
        )
        total = entity_count + number_count + factual_indicators
        normalized = total / len(doc) if len(doc) > 0 else 0
        return min(normalized * 2, 1.0)
 
    def _evidentiality_score(self, text: str) -> float:
        matches = 0
        for category, patterns in COMPILED_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(text):
                    matches += 1
                    break
        return matches / 5.0
 
    def _claim_density(self, text: str) -> float:
        doc = nlp(text)
        entity_count = len(doc.ents)
        number_count = sum(1 for token in doc if token.like_num)
        verb_count = sum(1 for token in doc if token.pos_ == 'VERB')
        token_count = len([t for t in doc if not t.is_space and not t.is_punct])
        if token_count == 0:
            return 0.0
        density = (entity_count + number_count + verb_count) / token_count
        return min(density, 1.0)
 
    def _temporal_recency(self, year: int) -> float:
        current_year = 2026
 
        if year > current_year:
            return 0.0
 
        years_ago = current_year - year
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
