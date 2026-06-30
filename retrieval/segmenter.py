"""
retrieval/segmenter.py
───────────────────────
segment_to_sentences() — Stage 3: Sentence Segmentation & Metadata Attachment.

Uses nlp.pipe() to batch all retrieved chunk texts through spaCy in one pass,
then filters and converts each sentence into a SentenceObject.

Key design decisions:
- nlp.pipe() batch processing — single spaCy call for all chunks (much faster than
  calling nlp() per chunk on large retrieved sets)
- NER disabled during segmentation (not needed; NER was already run on queries)
- Fragment filter — drops lowercase-start sentences (mid-chunk cuts) and bullet artefacts
- Length filter — MIN_SENT_TOKENS / MAX_SENT_TOKENS from config
- Entity/keyword overlap flag — uses lemma matching for morphological variants
  ('telephones' → 'telephone', 'invented' → 'invent')
- embedding=None — Module 2 fills this in; None saves ~1.5 KB per sentence vs np.zeros(384)
- Stable sentence_id — '{dataset}:{doc_id}:{chunk_idx}:{sent_idx}'
"""

import logging
import re
import time
from typing import Dict, List, Optional

import spacy

from retrieval.query_analyser import get_nlp
from retrieval.retrieval_config import (
    FRAGMENT_START_CHARS,
    MAX_SENT_TOKENS,
    MIN_SENT_TOKENS,
    SPACY_BATCH_SIZE,
)
from retrieval.sentence_object import SentenceObject

log = logging.getLogger('EARC-M1')

_WHITESPACE_RE = re.compile(r'\s+')


# ── Text utilities ─────────────────────────────────────────────────────────────


def _clean(text: str) -> str:
    """Collapse internal whitespace and strip leading/trailing whitespace."""
    return _WHITESPACE_RE.sub(' ', text).strip()


def _is_fragment(text: str) -> bool:
    """
    True if text looks like a chunk boundary fragment or list artefact.
    Catches: lowercase starts (mid-sentence cuts) and bullet/dash artefacts.
    """
    if not text:
        return True
    return text[0] in FRAGMENT_START_CHARS


def _approx_tokens(text: str) -> int:
    """Whitespace-split word count as a fast token approximation."""
    return len(text.split())


def _has_entity_or_keyword(
    sentence : str,
    entities : List[str],
    keywords : List[str],
    sent_doc,             # pre-computed spaCy span doc for this sentence
) -> bool:
    """
    True if sentence contains any query entity (substring) or keyword (lemma).

    Level 1: entity substring match (case-insensitive).
    Level 2: keyword lemma match against sentence token lemmas.
             Handles morphological variants via spaCy lemmatizer:
             'telephones' → 'telephone', 'invented' → 'invent', 'members' → 'member'.

    sent_doc is passed in (pre-computed by nlp.pipe batch) to avoid redundant spaCy calls.
    """
    s_lower = sentence.lower()

    # Entity substring match
    if any(ent.lower() in s_lower for ent in entities if ent):
        return True

    # Keyword lemma match
    if keywords and sent_doc is not None:
        sent_lemmas = {
            t.lemma_.lower() for t in sent_doc
            if not t.is_punct and not t.is_space
        }
        if any(kw in sent_lemmas for kw in keywords):
            return True

    return False


# ── Main segmentation function ─────────────────────────────────────────────────


def segment_to_sentences(
    retrieved_chunks : List[Dict],
    query_entities   : List[str],
    query_keywords   : List[str],
    min_tokens       : int = MIN_SENT_TOKENS,
    max_tokens       : int = MAX_SENT_TOKENS,
) -> List[SentenceObject]:
    """
    Stage 3 — Sentence Segmentation & Metadata Attachment.

    Uses nlp.pipe() to process all chunk texts in a single batched spaCy call
    (senter + tagger + lemmatizer only; NER disabled).

    For each sentence:
      1. Fragment filter  — drop lowercase-start / bullet artefacts
      2. Length filter    — drop sentences outside [min_tokens, max_tokens]
      3. Entity/keyword overlap flag — using pre-computed sentence lemmas
      4. SentenceObject creation with all Module 1 fields populated

    Returns
    -------
    List[SentenceObject] with embedding=None (Module 2 fills this in).
    """
    if not retrieved_chunks:
        log.warning('segment_to_sentences: no chunks received.')
        return []

    t0 = time.time()
    nlp = get_nlp()

    # Batch all chunk texts through spaCy in one pass.
    # Keep parser (en_core_web_sm uses it for .sents, not a standalone senter).
    # Disable NER — not needed at segmentation stage.
    chunk_texts = [chunk['chunk_text'] for chunk in retrieved_chunks]
    spacy_docs  = list(nlp.pipe(
        chunk_texts,
        batch_size=SPACY_BATCH_SIZE,
        disable=['ner'],
    ))

    results     : List[SentenceObject] = []
    n_fragments = 0
    n_too_short = 0
    n_too_long  = 0

    for chunk, spacy_doc in zip(retrieved_chunks, spacy_docs):
        for sent_idx, sent in enumerate(spacy_doc.sents):
            text = _clean(sent.text)
            if not text:
                continue

            if _is_fragment(text):
                n_fragments += 1
                continue

            tok_count = _approx_tokens(text)
            if tok_count < min_tokens:
                n_too_short += 1
                continue
            if tok_count > max_tokens:
                n_too_long += 1
                continue

            # Extract the sentence span as its own Doc for lemma-based keyword matching.
            # Using the already-computed spacy_doc avoids redundant full-pipeline calls.
            sent_span_doc = sent.as_doc()

            has_entity = _has_entity_or_keyword(
                text, query_entities, query_keywords, sent_span_doc
            )

            # Stable sentence ID: dataset:doc_id:chunk_idx:sent_idx
            sentence_id = (
                f"{chunk['dataset']}:{chunk['doc_id']}"
                f":{chunk['chunk_idx']}:{sent_idx}"
            )

            results.append(SentenceObject(
                sentence_id           = sentence_id,
                text                  = text,
                doc_id                = chunk['doc_id'],
                dataset               = chunk['dataset'],
                title                 = chunk['title'],
                position              = sent_idx,
                retrieval_rank        = chunk['rrf_rank'],
                chunk_id              = chunk['chunk_idx'],
                year                  = chunk.get('year', None),
                bm25_score            = chunk['bm25_score'],
                faiss_score           = chunk['faiss_score'],
                retrieval_score       = chunk['rrf_score'],
                embedding             = None,        # Module 2 fills this
                contains_query_entity = has_entity,
                token_count           = tok_count,
            ))

    t_seg = time.time() - t0
    log.info(
        'Segmentation: %d sentences from %d chunks in %.3fs '
        '(dropped: %d fragments, %d too short, %d too long)',
        len(results), len(retrieved_chunks), t_seg,
        n_fragments, n_too_short, n_too_long,
    )
    return results
