"""
tests/test_segmenter.py
────────────────────────
Unit tests for segment_to_sentences() — Stage 3.

Run: pytest tests/test_segmenter.py -v
"""

import pytest
from retrieval.segmenter import segment_to_sentences, _is_fragment, _approx_tokens
from retrieval.sentence_object import SentenceObject


# ── Fragment filter tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize('text,expected', [
    ('This is a proper sentence.', False),
    ('lowercase start — mid-chunk cut', True),
    ('• bullet point', True),
    ('– dash artefact', True),
    ('Another clean sentence.', False),
])
def test_is_fragment(text, expected):
    assert _is_fragment(text) == expected


# ── Token count ────────────────────────────────────────────────────────────────

def test_approx_tokens():
    assert _approx_tokens('hello world') == 2
    assert _approx_tokens('one') == 1
    assert _approx_tokens('') == 0


# ── Integration: segment_to_sentences ─────────────────────────────────────────

def _make_chunk(text, idx=0, rrf_rank=1):
    return {
        'chunk_idx'  : idx,
        'chunk_text' : text,
        'rrf_rank'   : rrf_rank,
        'rrf_score'  : 0.05,
        'bm25_score' : 1.0,
        'faiss_score': 0.8,
        'doc_id'     : 'doc_001',
        'dataset'    : 'wiki',
        'title'      : 'Test Title',
        'year'       : 2020,
    }


def test_returns_sentence_objects():
    chunks = [_make_chunk('Alexander Graham Bell invented the telephone in 1876.')]
    results = segment_to_sentences(chunks, [], [])
    assert all(isinstance(s, SentenceObject) for s in results)


def test_empty_chunks():
    results = segment_to_sentences([], [], [])
    assert results == []


def test_entity_flag_set():
    chunks = [_make_chunk('Alexander Graham Bell invented the telephone in 1876.')]
    results = segment_to_sentences(chunks, ['Alexander Graham Bell'], ['invent'])
    assert any(s.contains_query_entity for s in results)


def test_entity_flag_not_set_for_unrelated():
    chunks = [_make_chunk('The cat sat on the mat.')]
    results = segment_to_sentences(chunks, ['Einstein'], ['physics'])
    assert not any(s.contains_query_entity for s in results)


def test_sentence_id_format():
    chunks = [_make_chunk('Alexander Graham Bell invented the telephone.')]
    results = segment_to_sentences(chunks, [], [])
    for s in results:
        parts = s.sentence_id.split(':')
        assert len(parts) == 4, f'Unexpected sentence_id format: {s.sentence_id}'


def test_embedding_is_none():
    chunks = [_make_chunk('Alexander Graham Bell invented the telephone.')]
    results = segment_to_sentences(chunks, [], [])
    assert all(s.embedding is None for s in results)


def test_metadata_propagated():
    chunks = [_make_chunk('Bell invented the telephone.', idx=42, rrf_rank=3)]
    results = segment_to_sentences(chunks, [], [])
    for s in results:
        assert s.chunk_id == 42
        assert s.retrieval_rank == 3
        assert s.doc_id == 'doc_001'
        assert s.dataset == 'wiki'
        assert s.year == 2020


def test_too_short_filtered():
    chunks = [_make_chunk('Ok.')]  # 1 token — below MIN_SENT_TOKENS
    results = segment_to_sentences(chunks, [], [], min_tokens=5)
    assert results == []


def test_keyword_lemma_matching():
    """'invented' in chunk should match keyword 'invent'."""
    chunks = [_make_chunk('Bell invented the telephone in 1876.')]
    results = segment_to_sentences(chunks, [], ['invent'])
    assert any(s.contains_query_entity for s in results)
