"""
retrieval/query_analyser.py
────────────────────────────
QueryAnalyzer — Stage 1: Query Analysis & Classification.

Classification priority order:
  1. multi_hop  — 2+ substantive entities, OR 2+ finite verbs with a WH-word,
                  OR multiple WH-words, OR explicit comparison/contrast tokens,
                  OR compound conjunction between two question clauses
  2. factoid    — single factoid WH-word, single finite verb, no multi_hop signals
  3. descriptive — how/why, no WH-word, or long query

Negation detection is separate from classification:
  has_negation is always set in query_info regardless of query type,
  so Module 2/3 can use it without re-inspecting the query.

Returns
-------
dict with keys:
    query        : str        — original query string (propagated to all modules)
    query_type   : str        — 'factoid' | 'multi_hop' | 'descriptive'
    keywords     : List[str]  — lemmatised content words for BM25
    entities     : List[str]  — named entity strings for Stage 9 entity guard
    has_negation : bool       — True if query contains a negation signal
"""

import logging
from typing import Dict, List

import spacy

from retrieval.retrieval_config import (
    ALL_WH,
    AUX_DEPS,
    CLAUSE_DEPS,
    COMPARISON_TOKENS,
    CONTENT_POS,
    DESCRIPTIVE_WH,
    DETERMINERS,
    FACTOID_WH,
    NEGATION_TOKENS,
    SUBSTANTIVE_ENT_TYPES,
)

log = logging.getLogger('EARC-M1')

# Shared spaCy model — loaded once at module import time.
# pipeline.py / the notebook must call spacy.load before importing this module,
# OR this module can own the load (fine for scripts; Colab loads it in Cell 2).
_nlp: spacy.Language = None


def get_nlp() -> spacy.Language:
    """Return the shared spaCy model, loading it on first call."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_sm')
        log.info('spaCy loaded: %s', _nlp.meta['name'])
    return _nlp


class QueryAnalyzer:
    """Stage 1 — Query Analysis & Classification."""

    def __init__(self):
        self._nlp = get_nlp()

    # ── Public entry point ────────────────────────────────────────────────────────

    def analyze(self, query: str) -> Dict:
        doc          = self._nlp(query)
        has_negation = self._detect_negation(doc)
        query_type   = self._classify(doc)
        keywords     = self._extract_keywords(doc)
        entities     = self._extract_entities(doc)

        log.info(
            'QueryAnalyzer: type=%-12s negation=%-5s | keywords=%s | entities=%s',
            query_type, has_negation, keywords, entities,
        )
        return {
            'query'       : query,
            'query_type'  : query_type,
            'keywords'    : keywords,
            'entities'    : entities,
            'has_negation': has_negation,
        }

    # ── Negation detection ────────────────────────────────────────────────────────

    def _detect_negation(self, doc) -> bool:
        """
        Return True if the query contains a negation or exclusion signal.

        Checks both:
        - spaCy dep_ == 'neg' (syntactic negation: 'not', 'never', 'no')
        - Lexical set NEGATION_TOKENS (catches 'except', 'without',
          'excluding', 'non-member', 'outside' which spaCy may not
          parse as negation)

        has_negation does NOT affect BM25 keywords (BM25 cannot handle negation).
        has_negation does NOT affect the query string sent to FAISS (full query used).
        """
        for token in doc:
            if token.dep_ == 'neg':
                return True
            if token.text.lower() in NEGATION_TOKENS:
                return True
        return False

    # ── Classification ────────────────────────────────────────────────────────────

    def _classify(self, doc) -> str:
        """
        Classify query type using spaCy parse output.

        multi_hop signals (any one is sufficient):
          A. 2+ substantive named entities (cross-doc evidence needed)
          B. 2+ finite verbs AND a factoid WH-word (compound question)
          C. 2+ WH-words of any kind ('When and where was Tesla born?')
          D. Explicit comparison/contrast token present
          E. Coordinating conjunction between two named entities OR
             two question clauses (heuristic: CC dep + 2 content branches)

        factoid    : single factoid WH-word, no multi_hop signals.
        descriptive: how/why, no WH-word, or neither of the above.
        """
        token_texts_lower = [t.text.lower() for t in doc]
        token_set         = set(token_texts_lower)

        # Substantive named entities only
        subst_entities = [
            ent for ent in doc.ents
            if ent.label_ in SUBSTANTIVE_ENT_TYPES
        ]

        # Finite clause verbs (ROOT / relcl / ccomp / advcl / acl)
        finite_verbs = [
            t for t in doc
            if t.pos_ == 'VERB' and t.dep_ in CLAUSE_DEPS
        ]

        # Count WH-words
        wh_words_found = [w for w in token_texts_lower if w in ALL_WH]

        has_factoid_wh     = bool(token_set & FACTOID_WH)
        has_descriptive_wh = bool(token_set & DESCRIPTIVE_WH)
        has_comparison     = bool(token_set & COMPARISON_TOKENS)

        # ── multi_hop signals ─────────────────────────────────────────────────────

        # A. Two or more substantive named entities
        if len(subst_entities) >= 2:
            return 'multi_hop'

        # B. Compound question: 2+ finite verbs with a factoid WH-word
        if len(finite_verbs) >= 2 and has_factoid_wh:
            return 'multi_hop'

        # C. Multiple WH-words
        if len(wh_words_found) >= 2:
            return 'multi_hop'

        # D. Comparison / contrast token
        if has_comparison:
            return 'multi_hop'

        # E. Coordinating conjunction with 2+ content branches
        if self._has_compound_conjunction(doc):
            return 'multi_hop'

        # ── factoid ───────────────────────────────────────────────────────────────
        if has_factoid_wh and not has_descriptive_wh:
            return 'factoid'

        # ── descriptive ───────────────────────────────────────────────────────────
        return 'descriptive'

    def _has_compound_conjunction(self, doc) -> bool:
        """
        Heuristic: return True if there is a coordinating conjunction (CC dep)
        whose head has 2+ content-word children on each side.
        Catches 'Newton and Einstein', 'TCP and UDP' without relying on NER.
        """
        for token in doc:
            if token.dep_ == 'cc':
                head = token.head
                # Count conjuncts attached to the same head
                conjuncts = [
                    c for c in head.children
                    if c.dep_ == 'conj' and c.pos_ in {'NOUN', 'PROPN', 'VERB'}
                ]
                if conjuncts and head.pos_ in {'NOUN', 'PROPN', 'VERB'}:
                    return True
        return False

    # ── Keyword extraction ────────────────────────────────────────────────────────

    def _extract_keywords(self, doc) -> List[str]:
        """
        Extract lemmatised content words for BM25 query.

        Includes : NOUN, PROPN, VERB (non-auxiliary), ADJ, NUM
        Excludes : stop words, punctuation, spaces, auxiliary verbs,
                   negation tokens (BM25 cannot invert meaning)
        """
        keywords = []
        seen     = set()
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            if token.pos_ not in CONTENT_POS:
                continue
            if token.dep_ in AUX_DEPS:
                continue
            lemma = token.lemma_.lower()
            if lemma in seen:
                continue
            seen.add(lemma)
            keywords.append(lemma)
        return keywords

    # ── Entity extraction ─────────────────────────────────────────────────────────

    def _extract_entities(self, doc) -> List[str]:
        """
        Return unique named entity strings from the query.

        Leading determiners ('the', 'a', 'an') are stripped.
        Only substantive entity types are returned.
        """
        result = []
        seen   = set()
        for ent in doc.ents:
            if ent.label_ not in SUBSTANTIVE_ENT_TYPES:
                continue
            words = ent.text.strip().split()
            if words and words[0].lower() in DETERMINERS:
                words = words[1:]
            text = ' '.join(words).strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                result.append(text)
        return result
