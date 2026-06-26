"""
build_bm25.py
─────────────
Builds a BM25Okapi index from all existing chunk shards and saves it as
a single bm25.pkl file. Run this once after corpus generation is complete.

Usage:
    python build_bm25.py

Output:
    /content/drive/MyDrive/RAG_Project/bm25.pkl
    /content/drive/MyDrive/RAG_Project/tokenized_chunks.pkl
"""

import gc
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ─── LOGGING ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_bm25")

# ─── CONFIGURATION ──────────────────────────────────────────
DRIVE_BASE      = Path("/content/drive/MyDrive/RAG_Project")
CHUNKS_DIR      = DRIVE_BASE / "chunks"
OUTPUT_PATH     = DRIVE_BASE / "bm25.pkl"
TOKENIZED_PATH  = DRIVE_BASE / "tokenized_chunks.pkl"

# Pre-compiled token pattern: contiguous lowercase alphanumeric runs.
# Applied after .lower(), so the regex only needs to match [a-z0-9].
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Lightweight stopword list. High-frequency function words that add noise
# to BM25 IDF scores without contributing discriminative signal.
# Extend this set here if needed — tokenize() picks up changes automatically.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "in", "on", "at", "of", "to", "for", "and",
    "or", "but", "with", "by", "from", "as",
    "that", "this", "these", "those", "be",
    "been", "being", "it", "its",
}


# ─── TOKENIZATION ───────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """
    Lowercase → regex-extract alphanumeric tokens → remove stopwords
    and empty tokens.

    "Who invented the telephone?"
    → ["who", "invented", "telephone"]          (stopword "the" removed)

    "Alexander Graham Bell invented the telephone."
    → ["alexander", "graham", "bell", "invented", "telephone"]

    Returns an empty list for blank or non-string input rather than
    raising, so corrupted chunks can be handled gracefully by callers.
    """
    if not isinstance(text, str):
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


# ─── SHARD DISCOVERY ────────────────────────────────────────

def discover_shards(chunks_dir: Path) -> List[Path]:
    """
    Return all chunks_NNNNN.pkl files in chunks_dir, sorted numerically.
    Raises RuntimeError if the directory is missing or contains no shards.
    """
    if not chunks_dir.exists():
        raise RuntimeError(
            f"Chunks directory not found: {chunks_dir}\n"
            f"Make sure the corpus pipeline has been run and the path is correct."
        )
    if not chunks_dir.is_dir():
        raise RuntimeError(f"{chunks_dir} exists but is not a directory.")

    shards = sorted(chunks_dir.glob("chunks_*.pkl"))
    if not shards:
        raise RuntimeError(
            f"No shard files matching 'chunks_*.pkl' found in {chunks_dir}.\n"
            f"Directory contents: {list(chunks_dir.iterdir())[:10]}"
        )
    return shards


# ─── SHARD LOADING ──────────────────────────────────────────

def load_all_chunks(shards: List[Path]) -> List[str]:
    """
    Load every shard in order, collecting all chunk strings into a single
    list. Corrupted shards are skipped with a warning. Non-string entries
    within a shard are skipped individually with a warning.

    Memory note: all chunks are held in RAM simultaneously before BM25Okapi
    is built (BM25Okapi itself requires the full corpus in memory). For
    very large corpora this is the expected memory cost; reduce CHUNKS_DIR
    scope if RAM is tight.
    """
    all_chunks: List[str] = []
    corrupted_shards  = 0
    corrupted_entries = 0

    for shard_path in tqdm(shards, desc="Loading shards", unit="shard"):
        # ── existence check (race condition guard) ──────────────────
        if not shard_path.exists():
            log.warning("Shard disappeared before loading: %s — skipping.",
                        shard_path.name)
            corrupted_shards += 1
            continue

        # ── load with corruption guard ──────────────────────────────
        try:
            with open(shard_path, "rb") as f:
                shard_data = pickle.load(f)
        except Exception as exc:
            log.warning("Cannot load shard %s (%s) — skipping.",
                        shard_path.name, exc)
            corrupted_shards += 1
            continue

        # ── type check: shard must be a list ────────────────────────
        if not isinstance(shard_data, list):
            log.warning("Shard %s contains %s, expected list — skipping.",
                        shard_path.name, type(shard_data).__name__)
            corrupted_shards += 1
            continue

        # ── entry-level validation ───────────────────────────────────
        shard_good: List[str] = []
        for i, entry in enumerate(shard_data):
            if isinstance(entry, str):
                shard_good.append(entry)
            else:
                log.warning(
                    "Non-string entry at index %d in %s (type=%s) — skipping entry.",
                    i, shard_path.name, type(entry).__name__,
                )
                corrupted_entries += 1

        all_chunks.extend(shard_good)
        log.info("  Loaded %-30s  %6d chunks  (running total: %d)",
                 shard_path.name, len(shard_good), len(all_chunks))

    if corrupted_shards:
        log.warning("%d shard(s) were skipped due to errors.", corrupted_shards)
    if corrupted_entries:
        log.warning("%d individual entries were skipped due to type errors.",
                    corrupted_entries)

    return all_chunks


# ─── TOKENIZATION PASS ──────────────────────────────────────

def tokenize_corpus(chunks: List[str]) -> List[List[str]]:
    """
    Tokenize every chunk. Shows a progress bar; reports any empty results.
    Stopword removal is handled inside tokenize().
    """
    log.info("Tokenizing %d chunks …", len(chunks))
    tokenized: List[List[str]] = []
    empty_count = 0

    for chunk in tqdm(chunks, desc="Tokenizing", unit="chunk"):
        tokens = tokenize(chunk)
        tokenized.append(tokens)
        if not tokens:
            empty_count += 1

    if empty_count:
        log.warning("%d chunks produced zero tokens (blank or non-text content).",
                    empty_count)

    return tokenized


# ─── SAVE TOKENIZED CORPUS ──────────────────────────────────

def save_tokenized(tokenized: List[List[str]], path: Path) -> float:
    """
    Atomically save the tokenized corpus via temp file + rename.
    Returns the file size in MB.

    Saving the tokenized corpus separately lets future experiments rebuild
    BM25 (or try alternative retrieval approaches) without re-processing
    the raw chunk text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")

    log.info("Saving tokenized corpus to %s …", path)
    with open(tmp_path, "wb") as f:
        pickle.dump(tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.rename(path)

    size_mb = path.stat().st_size / 1e6
    log.info("Saved %s  (%.1f MB)", path.name, size_mb)
    return size_mb


# ─── BM25 BUILD ─────────────────────────────────────────────

def build_bm25(tokenized: List[List[str]]) -> BM25Okapi:
    """
    Construct a BM25Okapi index from the pre-tokenized corpus.
    BM25Okapi stores one dict per document — this is the expected RAM cost.
    """
    log.info("Building BM25Okapi index over %d documents …", len(tokenized))
    t0 = time.time()

    # BM25Okapi does not expose a progress bar; wrap with a note.
    print("  [BM25] Fitting … (this may take a few minutes for large corpora)")
    index = BM25Okapi(tokenized)

    elapsed = time.time() - t0
    log.info("BM25Okapi built in %.1f seconds.", elapsed)
    return index


# ─── SAVE BM25 ──────────────────────────────────────────────

def save_bm25(index: BM25Okapi, output_path: Path) -> float:
    """
    Atomically save the BM25 index via a temp file + rename so a crash
    during writing never leaves a half-written bm25.pkl on disk.
    Returns the file size in MB.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")

    log.info("Saving BM25 index to %s …", output_path)
    with open(tmp_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.rename(output_path)

    size_mb = output_path.stat().st_size / 1e6
    log.info("Saved %s  (%.1f MB)", output_path.name, size_mb)
    return size_mb


# ─── MAIN ───────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    log.info("=" * 60)
    log.info("build_bm25.py  —  BM25 index builder")
    log.info("  Chunks dir      : %s", CHUNKS_DIR)
    log.info("  BM25 output     : %s", OUTPUT_PATH)
    log.info("  Tokenized output: %s", TOKENIZED_PATH)
    log.info("=" * 60)

    # ── 1. Discover shards ────────────────────────────────────────────
    shards = discover_shards(CHUNKS_DIR)
    log.info("Found %d shard file(s).", len(shards))
    for s in shards:
        log.info("  %s  (%.1f MB)", s.name, s.stat().st_size / 1e6)

    # ── 2. Load all chunks ────────────────────────────────────────────
    chunks = load_all_chunks(shards)
    if not chunks:
        raise RuntimeError(
            "No valid chunks were loaded from any shard. "
            "Cannot build a BM25 index over an empty corpus."
        )
    n_chunks = len(chunks)
    log.info("Total chunks loaded: %d", n_chunks)

    # ── 3. Tokenize ───────────────────────────────────────────────────
    tokenized = tokenize_corpus(chunks)
    # Raw chunk strings no longer needed; free RAM before saves.
    del chunks
    gc.collect()

    # ── 4. Save tokenized corpus ──────────────────────────────────────
    tokenized_size_mb = save_tokenized(tokenized, TOKENIZED_PATH)

    # ── 5. Build BM25 ─────────────────────────────────────────────────
    index = build_bm25(tokenized)

    # ── 6. Save BM25 ─────────────────────────────────────────────────
    bm25_size_mb = save_bm25(index, OUTPUT_PATH)

    # ── 7. Statistics ─────────────────────────────────────────────────
    elapsed     = time.time() - t0
    vocab_size  = len(index.idf)
    doc_lengths = [len(t) for t in tokenized]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

    print("\n" + "=" * 60)
    print("  BM25 BUILD STATISTICS")
    print("=" * 60)
    print(f"  Shard files loaded          : {len(shards):>10,}")
    print(f"  Total chunks loaded         : {n_chunks:>10,}")
    print(f"  Vocabulary size             : {vocab_size:>10,}")
    print(f"  Avg document length (tokens): {avg_doc_len:>10.1f}")
    print(f"  bm25.pkl size on disk       : {bm25_size_mb:>10.1f}  MB")
    print(f"  tokenized_chunks.pkl size   : {tokenized_size_mb:>10.1f}  MB")
    print(f"  Total runtime               : {elapsed:>10.1f}  seconds")
    print("=" * 60 + "\n")

    log.info("Done. BM25 index saved to %s", OUTPUT_PATH)
    log.info("Done. Tokenized corpus saved to %s", TOKENIZED_PATH)


# ─── ENTRY POINT ────────────────────────────────────────────
if __name__ == "__main__":
    main()
