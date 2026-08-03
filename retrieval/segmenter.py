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
from bisect import bisect_right
from collections import defaultdict
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

# A sentence that does not end in one of these is very likely a chunk-boundary
# cut ("...the first generation iPhone was") rather than a real sentence end.
_TERMINAL_PUNCT = ('.', '!', '?', '…', '"', "'", ')', ']')

# Minimum suffix/prefix character overlap for two consecutive chunks of the same
# document to be considered adjacent (they share ~chunk_overlap chars by design).
# Kept comfortably below the builder's chunk_overlap (100) to tolerate the
# whitespace stripping applied when chunks were created.
_MIN_STITCH_OVERLAP = 20

# Cap the suffix/prefix scan window (chars) so stitching stays O(n) per pair.
_MAX_STITCH_SCAN = 400


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


def _is_truncated_tail(text: str) -> bool:
    """
    True if a sentence looks cut off at the end (no terminal punctuation).

    After chunk stitching this should only ever fire on the final sentence of a
    reconstructed segment — i.e. a genuine edge of the retrieved context where
    the continuation was never retrieved and therefore cannot be recovered.
    """
    if not text:
        return True
    return not text.endswith(_TERMINAL_PUNCT)


def _suffix_prefix_overlap(a: str, b: str) -> int:
    """
    Longest overlap length k such that a[-k:] == b[:k].

    Used to splice two consecutive chunks of the same document back together
    without duplicating their shared (chunk_overlap) region. Returns 0 if no
    overlap of at least ``_MIN_STITCH_OVERLAP`` chars is found.
    """
    max_k = min(len(a), len(b), _MAX_STITCH_SCAN)
    for k in range(max_k, _MIN_STITCH_OVERLAP - 1, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def _stitch_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Reconstruct contiguous document text from overlapping retrieved chunks.

    Chunks were produced by a character-based splitter (chunk_size=800,
    chunk_overlap=100), which cuts sentences at chunk boundaries. Here we group
    chunks by document, order them by original position, and splice adjacent
    chunks together (removing their shared overlap) so that sentences spanning a
    boundary are reassembled whole before segmentation.

    Returns a list of "segments", each a dict:
        {
          "text":       reconstructed text,
          "boundaries": [(char_offset_in_text, source_chunk), ...]  # ascending
        }
    ``boundaries`` lets each resulting sentence be attributed back to the chunk
    it originated from (for rank/score metadata).
    """
    grouped: Dict = defaultdict(list)
    for ch in chunks:
        grouped[(ch.get('dataset', ''), ch.get('doc_id', ''))].append(ch)

    segments: List[Dict] = []
    for group in grouped.values():
        # Order by original document position. char_start is exact; fall back to
        # the globally-sequential chunk_idx when it is unavailable.
        group_sorted = sorted(
            group,
            key=lambda c: (
                c.get('char_start')
                if c.get('char_start') is not None
                else c.get('chunk_idx', 0)
            ),
        )

        for ch in group_sorted:
            text = ch.get('chunk_text') or ''
            if not text:
                continue

            if not segments or segments[-1].get('_doc') != (
                ch.get('dataset', ''), ch.get('doc_id', '')
            ):
                segments.append({
                    'text': text,
                    'boundaries': [(0, ch)],
                    '_doc': (ch.get('dataset', ''), ch.get('doc_id', '')),
                })
                continue

            seg = segments[-1]
            k = _suffix_prefix_overlap(seg['text'], text)
            if k >= _MIN_STITCH_OVERLAP:
                piece = text[k:]
                if piece:  # skip fully-duplicated chunks
                    seg['boundaries'].append((len(seg['text']), ch))
                    seg['text'] += piece
            else:
                # No detectable overlap => a gap between retrieved chunks; start
                # a fresh segment so unrelated text is not concatenated.
                segments.append({
                    'text': text,
                    'boundaries': [(0, ch)],
                    '_doc': (ch.get('dataset', ''), ch.get('doc_id', '')),
                })

    return segments



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

    # Stitch overlapping chunks back into contiguous document segments so that
    # sentences cut across chunk boundaries are reassembled before segmentation.
    segments = _stitch_chunks(retrieved_chunks)

    # Batch all reconstructed segment texts through spaCy in one pass.
    # Keep parser (en_core_web_sm uses it for .sents, not a standalone senter).
    # Disable NER — not needed at segmentation stage.
    segment_texts = [seg['text'] for seg in segments]
    spacy_docs    = list(nlp.pipe(
        segment_texts,
        batch_size=SPACY_BATCH_SIZE,
        disable=['ner'],
    ))

    results     : List[SentenceObject] = []
    n_fragments = 0
    n_truncated = 0
    n_too_short = 0
    n_too_long  = 0

    for seg, spacy_doc in zip(segments, spacy_docs):
        # Ascending char offsets marking where each source chunk's content begins
        # within the reconstructed segment (for per-sentence metadata attribution).
        boundaries    = seg['boundaries']
        boundary_offs = [off for off, _ in boundaries]

        for sent_idx, sent in enumerate(spacy_doc.sents):
            text = _clean(sent.text)
            if not text:
                continue

            if _is_fragment(text):
                n_fragments += 1
                continue

            # After stitching, a truncated tail means the continuation was never
            # retrieved (true context edge) — only then do we drop it.
            if _is_truncated_tail(text):
                n_truncated += 1
                continue

            tok_count = _approx_tokens(text)
            if tok_count < min_tokens:
                n_too_short += 1
                continue
            if tok_count > max_tokens:
                n_too_long += 1
                continue

            # Attribute this sentence to the source chunk whose content region
            # contains the sentence's start offset.
            b_idx = bisect_right(boundary_offs, sent.start_char) - 1
            if b_idx < 0:
                b_idx = 0
            chunk = boundaries[b_idx][1]

            # Extract the sentence span as its own Doc for lemma-based keyword matching.
            # Using the already-computed spacy_doc avoids redundant full-pipeline calls.
            sent_span_doc = sent.as_doc()

            has_entity = _has_entity_or_keyword(
                text, query_entities, query_keywords, sent_span_doc
            )

            # Compute per-sentence entity/keyword coverage for Layer 10.
            s_lower = text.lower()
            matched_entities = [e for e in query_entities if e and e.lower() in s_lower]
            if query_keywords and sent_span_doc is not None:
                sent_lemmas = {
                    t.lemma_.lower() for t in sent_span_doc
                    if not t.is_punct and not t.is_space
                }
                matched_keywords = [kw for kw in query_keywords if kw in sent_lemmas]
            else:
                matched_keywords = []

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
                sentence_entities     = matched_entities,
                sentence_keywords     = matched_keywords,
            ))

    t_seg = time.time() - t0
    log.info(
        'Segmentation: %d sentences from %d chunks (%d stitched segments) in %.3fs '
        '(dropped: %d fragments, %d truncated, %d too short, %d too long)',
        len(results), len(retrieved_chunks), len(segments), t_seg,
        n_fragments, n_truncated, n_too_short, n_too_long,
    )
    return results
