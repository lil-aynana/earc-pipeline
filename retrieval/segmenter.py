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
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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

# Overlap-stitching bounds. The corpus chunker uses chunk_overlap=100 chars, so
# adjacent chunks of a document share ~100 chars. We search a slightly wider
# window (to tolerate whitespace normalisation) and require a minimum match to
# avoid treating a coincidental short repeat as a real overlap.
_MIN_OVERLAP_CHARS = 10
_MAX_OVERLAP_CHARS = 260



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


# Characters that legitimately end a complete sentence.
_TERMINAL_PUNCT = '.!?…"\')]'


def _is_truncated_tail(text: str) -> bool:
    """
    True if `text` looks like the trailing half of a sentence that was cut
    mid-word by the fixed-width (character-based) corpus chunker (e.g.
    "...Warsaw Pact co", "...as a waste pr"). Such tails never end in terminal
    punctuation.

    NOTE: This is used only for logging/diagnostics. Truncated tails are NOT
    discarded — see `_reconstruct_runs`, which stitches adjacent retrieved
    chunks back together so cut sentences become whole. A tail only stays
    truncated when the continuation chunk was not retrieved, in which case the
    partial sentence is still kept (it may carry useful evidence; the
    generation layer sanitises any dangling partial word for display).
    """
    if not text:
        return True
    return text[-1] not in _TERMINAL_PUNCT


def _dedupe_overlap(
    prev_text: str,
    next_text: str,
    min_overlap: int = _MIN_OVERLAP_CHARS,
    max_overlap: int = _MAX_OVERLAP_CHARS,
) -> Tuple[str, int]:
    """
    Remove from `next_text` the leading region it shares with the suffix of
    `prev_text`, returning ``(remainder, overlap_len)``.

    The corpus chunker overlaps adjacent chunks by ~100 chars, so when two
    chunks are truly adjacent the suffix of one equals the prefix of the next.
    We return the largest such overlap so the shared text appears only once in
    the stitched output. ``overlap_len == 0`` means no reliable overlap was
    found (the chunks are not adjacent / there is a retrieval gap).
    """
    if not prev_text or not next_text:
        return next_text, 0
    upper = min(len(prev_text), len(next_text), max_overlap)
    for k in range(upper, min_overlap - 1, -1):
        if prev_text[-k:] == next_text[:k]:
            return next_text[k:], k
    return next_text, 0


def _reconstruct_runs(retrieved_chunks: List[Dict]) -> List[Dict]:
    """
    Stitch adjacent retrieved chunks of the same document back into continuous
    text, so sentences severed at chunk boundaries become whole again.

    Returns a list of "runs". Each run is::

        {'text': <continuous cleaned text>,
         'spans': [(start_offset_in_text, chunk_dict), ...]}

    Chunks are grouped by (dataset, doc_id) and ordered by reading position
    (char_start, then chunk_idx). Two consecutive chunks are merged only when
    BOTH (a) their chunk_idx is consecutive and (b) a genuine character overlap
    confirms adjacency. A retrieval gap (non-adjacent chunks) starts a new run,
    so unrelated passages are never concatenated into a false sentence. The
    ``spans`` map lets each reconstructed sentence be attributed back to the
    chunk it originated from (for retrieval scores / ids).
    """
    groups: "defaultdict[Tuple[str, str], List[Dict]]" = defaultdict(list)
    order: List[Tuple[str, str]] = []
    for ch in retrieved_chunks:
        key = (ch.get('dataset', ''), ch.get('doc_id', ''))
        if key not in groups:
            order.append(key)
        groups[key].append(ch)

    runs: List[Dict] = []
    for key in order:
        chunks = groups[key]
        # Reading order: char_start if present, else chunk_idx.
        chunks.sort(key=lambda c: (
            c.get('char_start') if c.get('char_start') is not None else c.get('chunk_idx', 0),
            c.get('chunk_idx', 0),
        ))

        cur_text: Optional[str] = None
        cur_spans: List[Tuple[int, Dict]] = []
        prev_idx: Optional[int] = None

        for ch in chunks:
            ctext = _clean(ch['chunk_text'])
            if not ctext:
                continue
            cidx = ch.get('chunk_idx')

            if cur_text is None:
                cur_text, cur_spans, prev_idx = ctext, [(0, ch)], cidx
                continue

            remainder, overlap = _dedupe_overlap(cur_text, ctext)
            idx_adjacent = (
                cidx is not None and prev_idx is not None and cidx == prev_idx + 1
            )
            # Merge only when adjacency is corroborated by overlap. If chunk_idx
            # is unavailable, fall back to overlap alone.
            can_merge = overlap > 0 and (idx_adjacent or cidx is None or prev_idx is None)

            if can_merge:
                cur_spans.append((len(cur_text), ch))
                cur_text += remainder
            else:
                runs.append({'text': cur_text, 'spans': cur_spans})
                cur_text, cur_spans = ctext, [(0, ch)]
            prev_idx = cidx

        if cur_text is not None:
            runs.append({'text': cur_text, 'spans': cur_spans})

    return runs


def _chunk_for_pos(spans: List[Tuple[int, Dict]], pos: int) -> Dict:
    """Return the source chunk whose contributed span contains char `pos`."""
    chosen = spans[0][1]
    for offset, ch in spans:
        if offset <= pos:
            chosen = ch
        else:
            break
    return chosen



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

    Adjacent retrieved chunks of the same document are first stitched back into
    continuous text (`_reconstruct_runs`) so sentences severed at chunk
    boundaries are recovered whole, then each run is processed with nlp.pipe()
    in a single batched spaCy call (senter + tagger + lemmatizer; NER disabled).

    For each sentence:
      1. Fragment filter  — drop lowercase-start / bullet artefacts (leading
         fragments at the start of a run whose previous chunk wasn't retrieved)
      2. Length filter    — drop sentences outside [min_tokens, max_tokens]
      3. Entity/keyword overlap flag — using pre-computed sentence lemmas
      4. SentenceObject creation with all Module 1 fields populated, attributed
         back to its source chunk via the run span map

    Truncated tails are NOT discarded: stitching heals interior cuts, and any
    tail that stays truncated (continuation chunk not retrieved) is still kept
    as partial evidence.

    Returns
    -------
    List[SentenceObject] with embedding=None (Module 2 fills this in).
    """
    if not retrieved_chunks:
        log.warning('segment_to_sentences: no chunks received.')
        return []

    t0 = time.time()
    nlp = get_nlp()

    # Stitch adjacent chunks of each document into continuous runs so sentences
    # cut at chunk boundaries are made whole before segmentation.
    runs = _reconstruct_runs(retrieved_chunks)

    # Batch all run texts through spaCy in one pass.
    # Keep parser (en_core_web_sm uses it for .sents, not a standalone senter).
    # Disable NER — not needed at segmentation stage.
    run_texts  = [run['text'] for run in runs]
    spacy_docs = list(nlp.pipe(
        run_texts,
        batch_size=SPACY_BATCH_SIZE,
        disable=['ner'],
    ))

    results       : List[SentenceObject] = []
    n_fragments   = 0
    n_too_short   = 0
    n_too_long    = 0
    n_truncated   = 0
    per_chunk_seq : Dict[int, int] = {}  # chunk_idx -> running sentence count

    for run, spacy_doc in zip(runs, spacy_docs):
        sents    = list(spacy_doc.sents)
        last_idx = len(sents) - 1
        for sent_idx, sent in enumerate(sents):
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

            # A tail can only remain truncated at the very end of a run, when the
            # continuation chunk wasn't retrieved. Keep it (partial evidence);
            # just count it for diagnostics.
            if sent_idx == last_idx and _is_truncated_tail(text):
                n_truncated += 1

            # Attribute this sentence back to its originating chunk.
            chunk    = _chunk_for_pos(run['spans'], sent.start_char)
            chunk_id = chunk['chunk_idx']
            local_id = per_chunk_seq.get(chunk_id, 0)
            per_chunk_seq[chunk_id] = local_id + 1

            # Extract the sentence span as its own Doc for lemma-based keyword matching.
            # Using the already-computed spacy_doc avoids redundant full-pipeline calls.
            sent_span_doc = sent.as_doc()

            has_entity = _has_entity_or_keyword(
                text, query_entities, query_keywords, sent_span_doc
            )

            # Stable sentence ID: dataset:doc_id:chunk_idx:local_sent_idx
            sentence_id = (
                f"{chunk['dataset']}:{chunk['doc_id']}"
                f":{chunk_id}:{local_id}"
            )

            results.append(SentenceObject(
                sentence_id           = sentence_id,
                text                  = text,
                doc_id                = chunk['doc_id'],
                dataset               = chunk['dataset'],
                title                 = chunk['title'],
                position              = local_id,
                retrieval_rank        = chunk['rrf_rank'],
                chunk_id              = chunk_id,
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
        'Segmentation: %d sentences from %d chunks (%d stitched runs) in %.3fs '
        '(dropped: %d fragments, %d too short, %d too long; '
        '%d tails kept truncated)',
        len(results), len(retrieved_chunks), len(runs), t_seg,
        n_fragments, n_too_short, n_too_long, n_truncated,
    )
    return results
