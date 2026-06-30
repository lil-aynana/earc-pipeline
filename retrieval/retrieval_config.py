"""
retrieval/retrieval_config.py
─────────────────────────────
All tunable parameters and closed linguistic sets for Module 1.
Every other retrieval file imports from here — never hardcode values elsewhere.
"""

from pathlib import Path

# ── Corpus artifact paths (override in pipeline.py for non-Colab environments) ──

DRIVE_BASE   = Path('/content/drive/MyDrive/RAG_Project')
FAISS_PATH   = DRIVE_BASE / 'faiss.index'
BM25_PATH    = DRIVE_BASE / 'bm25.pkl'
CHUNKS_DIR   = DRIVE_BASE / 'chunks'
METADATA_DIR = DRIVE_BASE / 'metadata'

# ── Embedding model ──────────────────────────────────────────────────────────────

EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBED_DIM   = 384

# ── RRF ──────────────────────────────────────────────────────────────────────────

RRF_K = 60  # standard RRF smoothing constant

# ── Retrieval depth per query type ───────────────────────────────────────────────
# Controls BM25/FAISS candidate pool sizes and final RRF top-N.
# Token budget for the LLM prompt is handled downstream in Module 3.

K_BY_TYPE = {
    'factoid'    : {'bm25': 15, 'faiss': 15, 'final': 8},
    'descriptive': {'bm25': 20, 'faiss': 20, 'final': 12},
    'multi_hop'  : {'bm25': 30, 'faiss': 30, 'final': 20},
}

# ── Sentence length bounds for Stage 3 segmentation ─────────────────────────────

MIN_SENT_TOKENS  = 5    # drops fragment labels and bullet artefacts
MAX_SENT_TOKENS  = 150  # drops garbled OCR rows and HTML table cells
SPACY_BATCH_SIZE = 32   # spaCy batch size for nlp.pipe() in Stage 3

# ── Closed linguistic sets — do not change unless English grammar changes ─────────

# Named entity types that count as substantive query subjects.
# Incidental GPE/LOC/DATE/TIME/CARDINAL alone do NOT trigger multi_hop.
SUBSTANTIVE_ENT_TYPES = {
    'PERSON', 'ORG', 'WORK_OF_ART', 'EVENT', 'PRODUCT', 'LAW', 'NORP', 'FAC'
}

# POS tags for BM25 content keywords.
CONTENT_POS = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'}

# Wh-words by query category.
FACTOID_WH     = {'who', 'what', 'when', 'where', 'which', 'whom'}
DESCRIPTIVE_WH = {'how', 'why'}
ALL_WH         = FACTOID_WH | DESCRIPTIVE_WH

# Dependency labels marking auxiliary verbs — excluded from keywords.
AUX_DEPS = {'aux', 'auxpass'}

# Clause-level dependency labels for finite verb detection.
CLAUSE_DEPS = {'ROOT', 'relcl', 'ccomp', 'advcl', 'acl'}

# Comparison/contrast lexical signals — presence suggests multi-entity reasoning.
# These are surface forms, checked after lowercasing.
COMPARISON_TOKENS = {
    'compare', 'comparison', 'versus', 'vs', 'difference',
    'differences', 'contrast', 'similar', 'both', 'neither',
    'between', 'respectively',
}

# Negation tokens — used to set has_negation flag (NOT stripped from query).
NEGATION_TOKENS = {
    'not', 'never', 'no', 'except', 'without', 'excluding',
    'neither', 'nor', 'non', 'non-member', 'outside',
}

# Characters that mark chunk boundary fragments or list artefacts.
FRAGMENT_START_CHARS = set('abcdefghijklmnopqrstuvwxyz*•–—|·')

# Leading determiners to strip from entity spans.
DETERMINERS = {'the', 'a', 'an'}
