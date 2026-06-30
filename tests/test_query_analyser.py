"""
tests/test_query_analyser.py
─────────────────────────────
Unit tests for QueryAnalyzer — classification and negation detection.

Run: pytest tests/test_query_analyser.py -v
"""

import pytest
from retrieval.query_analyser import QueryAnalyzer


@pytest.fixture(scope='module')
def qa():
    return QueryAnalyzer()


# ── Classification ─────────────────────────────────────────────────────────────

CLASSIFICATION_CASES = [
    # Core cases
    ('Who invented the telephone?',                                              'factoid'),
    ('What is the capital of France?',                                           'factoid'),
    ('How does photosynthesis work?',                                            'descriptive'),
    ('What did Marie Curie and Albert Einstein both contribute to physics?',     'multi_hop'),
    ('Which film won the Academy Award for Best Picture in 2020 and who directed it?', 'multi_hop'),
    # v4 edge cases
    ('In which year did the Berlin Wall fall?',                                  'factoid'),
    ('What did Einstein publish in Germany in 1905?',                            'factoid'),
    ('Why did the Roman Empire collapse?',                                       'descriptive'),
    ('When and where was Nikola Tesla born?',                                    'multi_hop'),
    ('Who invented C++ and when did it become popular?',                         'multi_hop'),
    ('Compare Newton and Einstein on gravity',                                   'multi_hop'),
    ('What is the difference between TCP and UDP?',                              'multi_hop'),
]


@pytest.mark.parametrize('query,expected', CLASSIFICATION_CASES)
def test_classification(qa, query, expected):
    result = qa.analyze(query)
    assert result['query_type'] == expected, (
        f'Query: {query!r}\n'
        f'  Expected: {expected}\n'
        f'  Got     : {result["query_type"]}'
    )


# ── Negation detection ─────────────────────────────────────────────────────────

NEGATION_CASES = [
    ('What countries are not members of NATO?', True),
    ('Who invented the telephone?',             False),
    ('Countries excluding NATO membership',     True),
    ('Neither A nor B is correct',              True),
    ('What is the capital of France?',          False),
]


@pytest.mark.parametrize('query,expected_negation', NEGATION_CASES)
def test_negation_detection(qa, query, expected_negation):
    result = qa.analyze(query)
    assert result['has_negation'] == expected_negation, (
        f'Query: {query!r}\n'
        f'  Expected has_negation={expected_negation}\n'
        f'  Got     has_negation={result["has_negation"]}'
    )


def test_negation_not_in_keywords(qa):
    """'not' must not appear in BM25 keywords (BM25 cannot handle negation)."""
    result = qa.analyze('What countries are not members of NATO?')
    assert 'not' not in result['keywords'], (
        f"'not' should be excluded from keywords; got: {result['keywords']}"
    )


# ── Keyword extraction ─────────────────────────────────────────────────────────

def test_keywords_are_lemmatised(qa):
    """Keywords must be lemmatised (e.g. 'telephones' → 'telephone')."""
    result = qa.analyze('Who invented telephones?')
    assert 'telephone' in result['keywords'] or 'invent' in result['keywords'], (
        f'Expected lemmatised keywords; got: {result["keywords"]}'
    )


def test_keywords_exclude_stopwords(qa):
    result = qa.analyze('What is the capital of France?')
    assert 'the' not in result['keywords']
    assert 'is' not in result['keywords']
    assert 'of' not in result['keywords']


# ── query_info structure ───────────────────────────────────────────────────────

def test_query_info_keys(qa):
    result = qa.analyze('Who invented the telephone?')
    for key in ('query', 'query_type', 'keywords', 'entities', 'has_negation'):
        assert key in result, f'Missing key: {key}'


def test_query_propagated_unchanged(qa):
    q = 'Who invented the telephone?'
    result = qa.analyze(q)
    assert result['query'] == q
