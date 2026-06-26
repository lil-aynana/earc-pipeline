import gc
import hashlib
import json
import logging
import os
import pickle
import time
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── LOGGING ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RAG")


# ─── CUSTOM EXCEPTION ───────────────────────────────────────
class SchemaError(RuntimeError):
    """Raised when a required field cannot be found in a dataset."""


# ─── CONFIGURATION ──────────────────────────────────────────
CONFIG = dict(
    drive_base       = "/content/drive/MyDrive/RAG_Project",
    chunk_size       = 800,
    chunk_overlap    = 100,
    embed_model      = "sentence-transformers/all-MiniLM-L6-v2",
    embed_batch_size = 256,
    checkpoint_every = 5_000,
    batch_size       = 10_000,
    max_rows         = dict(
        nq_passages = None,
        nq_answers  = None,
        hotpot      = None,
        trivia      = None,
    ),
)


# ─── GOOGLE DRIVE MOUNT ─────────────────────────────────────

def mount_drive(base_path: str) -> Path:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        log.info("Google Drive mounted.")
    except ImportError:
        log.warning("Not in Colab — skipping Drive mount.")
    except Exception as exc:
        log.error("Drive mount failed: %s", exc)

    p = Path(base_path)
    for sub in ("ckpt", "chunks", "metadata", "qa_pairs"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    log.info("Project directory: %s", p)
    return p


# ─── PROGRESS TRACKING ──────────────────────────────────────

PROGRESS_FILE = "progress.json"

_DEFAULT_PROGRESS = {
    "dataset_offsets": {
        "nq_passages": 0,
        "nq_answers":  0,
        "hotpot":      0,
        "trivia":      0,
    },
    "next_chunk_id":      0,
    "completed_runs":     0,
    "completed_batches":  0,
    "faiss_total_vectors": 0,
    "last_run_finished_at": None,
}


def load_progress(save_dir: Path) -> Dict[str, Any]:
    path = save_dir / PROGRESS_FILE
    if not path.exists():
        log.info("No progress.json found — starting from scratch.")
        return json.loads(json.dumps(_DEFAULT_PROGRESS))

    with open(path) as f:
        progress = json.load(f)

    for k, v in _DEFAULT_PROGRESS.items():
        if k not in progress:
            progress[k] = v
    for k, v in _DEFAULT_PROGRESS["dataset_offsets"].items():
        progress["dataset_offsets"].setdefault(k, v)

    log.info("Loaded progress.json  offsets=%s  batches=%d  faiss_vectors=%d",
             progress["dataset_offsets"],
             progress["completed_batches"],
             progress["faiss_total_vectors"])
    return progress


def save_progress(save_dir: Path, progress: Dict[str, Any]) -> None:
    path = save_dir / PROGRESS_FILE
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    tmp.rename(path)
    log.info("progress.json updated  offsets=%s", progress["dataset_offsets"])


# ─── OPTION-A AUTO-ROLLBACK ─────────────────────────────────
#
# Detects the "progress.json advanced but 0 chunks were written" state.
# The symptom is: completed_batches == N but the newest shard in
# manifest.json is empty (faiss_start == faiss_end). Two distinct causes
# produce this same signature:
#   (a) the old dedup bug, where cross-run dedup incorrectly skipped all
#       documents in a batch; or
#   (b) a mid-FAISS-encoding crash, where the run died after committing
#       an empty shard placeholder but before finalize_run() completed.
# Both are repaired identically, so one helper covers both. This MUST run
# before check_corpus_integrity()/validate_manifest(), which would
# otherwise hard-raise on this exact (recoverable) state.
#
# When detected, the function:
#   1. Rolls progress.json offsets back to the start of the bad batch.
#   2. Deletes doc_hashes.json so it is rebuilt with the corrected
#      _hash_text function (which now includes dataset + row_idx).
#   3. Removes the empty shard entry from manifest.json.
#   4. Deletes the empty shard pkl files (chunks / metadata / qa_pairs)
#      that were written for that batch.
#
# After rollback the normal startup continues and the repaired batch is
# processed correctly on this same run.

def _rollback_empty_batch_if_needed(
    save_dir : Path,
    progress : Dict[str, Any],
    manifest : Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Check whether the most-recently committed batch produced 0 FAISS vectors
    and, if so, roll it back so the batch is retried cleanly.

    Returns (possibly-updated progress, possibly-updated manifest).
    """
    shards = manifest.get("shards", [])
    if not shards:
        return progress, manifest

    # Sort by batch_idx so we always inspect the last committed shard.
    last_shard = max(shards, key=lambda s: s["batch_idx"])
    if last_shard["faiss_end"] > last_shard["faiss_start"]:
        # Last shard has vectors — nothing to repair.
        return progress, manifest

    bad_batch_idx = last_shard["batch_idx"]
    log.warning(
        "AUTO-ROLLBACK: batch %d produced 0 FAISS vectors "
        "(faiss_start == faiss_end == %d). "
        "This was caused by the old _hash_text bug. Rolling back …",
        bad_batch_idx, last_shard["faiss_start"],
    )

    # ── 1. Figure out what the offsets were BEFORE the bad batch ──────────
    # Each dataset is processed batch_size rows per batch.  The bad batch
    # advanced offsets by batch_size; subtract to get the prior offsets.
    batch_size = CONFIG["batch_size"]
    old_offsets = {
        name: max(0, offset - batch_size)
        for name, offset in progress["dataset_offsets"].items()
    }
    progress["dataset_offsets"]   = old_offsets
    progress["completed_batches"] = bad_batch_idx          # back to N-1 done
    progress["faiss_total_vectors"] = last_shard["faiss_start"]
    # next_chunk_id doesn't change because 0 chunks were written.

    # ── 2. Remove the empty shard from manifest ────────────────────────────
    manifest["shards"] = [s for s in shards if s["batch_idx"] != bad_batch_idx]

    # ── 3. Delete the stale shard pkl files for the bad batch ─────────────
    for subdir, kind in [("chunks", "chunks"), ("metadata", "metadata"),
                         ("qa_pairs", "qa")]:
        p = _shard_path(save_dir, subdir, kind, bad_batch_idx)
        if p.exists():
            p.unlink()
            log.info("AUTO-ROLLBACK: deleted stale shard %s", p.name)

    # ── 4. Delete doc_hashes.json so it is rebuilt with the fixed hash ─────
    hashes_path = save_dir / _DOC_HASHES_FILE
    if hashes_path.exists():
        hashes_path.unlink()
        log.info(
            "AUTO-ROLLBACK: deleted doc_hashes.json — it will be rebuilt "
            "with the corrected _hash_text function this run."
        )

    # ── 5. Persist the repaired state ─────────────────────────────────────
    save_progress(save_dir, progress)
    save_manifest(save_dir, manifest)

    log.info(
        "AUTO-ROLLBACK complete: rolled back to offsets=%s, "
        "completed_batches=%d. Re-running batch %d now.",
        old_offsets, bad_batch_idx, bad_batch_idx,
    )
    return progress, manifest


# ─── MANIFEST ───────────────────────────────────────────────

MANIFEST_FILE = "manifest.json"


def load_manifest(save_dir: Path) -> Dict[str, Any]:
    path = save_dir / MANIFEST_FILE
    if not path.exists():
        return {"shards": []}
    with open(path) as f:
        return json.load(f)


def save_manifest(save_dir: Path, manifest: Dict[str, Any]) -> None:
    path = save_dir / MANIFEST_FILE
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.rename(path)
    log.info("manifest.json updated (%d shards)", len(manifest["shards"]))


def shard_for_vector(manifest: Dict[str, Any], vector_idx: int) -> Optional[Dict]:
    """Binary-search the manifest to find which shard owns FAISS vector i."""
    shards = manifest["shards"]
    lo, hi = 0, len(shards) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s = shards[mid]
        if vector_idx < s["faiss_start"]:
            hi = mid - 1
        elif vector_idx >= s["faiss_end"]:
            lo = mid + 1
        else:
            return s
    return None


def validate_manifest(manifest: Dict[str, Any], faiss_index: faiss.IndexFlatIP,
                       progress: Dict[str, Any]) -> None:
    """
    Verifies:
      - manifest shard count matches progress["completed_batches"]
      - shard batch_idx values are exactly 0..N-1, no gaps/duplicates
      - FAISS ranges are contiguous (no gaps, no overlaps)
      - final manifest range equals faiss_index.ntotal
    """
    shards = manifest["shards"]
    expected_batches = progress["completed_batches"]

    if len(shards) != expected_batches:
        raise RuntimeError(
            f"manifest.json has {len(shards)} shards but progress.json says "
            f"{expected_batches} completed batches. Restore from backup."
        )

    sorted_shards = sorted(shards, key=lambda s: s["batch_idx"])
    expected_idx, cursor = 0, 0
    for s in sorted_shards:
        if s["batch_idx"] != expected_idx:
            raise RuntimeError(
                f"manifest.json missing or duplicated batch_idx: expected "
                f"{expected_idx}, found {s['batch_idx']}."
            )
        if s["faiss_start"] != cursor:
            raise RuntimeError(
                f"manifest.json gap or overlap at batch {s['batch_idx']}: "
                f"expected faiss_start={cursor}, found {s['faiss_start']}."
            )
        if s["faiss_end"] <= s["faiss_start"]:
            raise RuntimeError(
                f"manifest.json batch {s['batch_idx']} has an empty or "
                f"negative range [{s['faiss_start']}, {s['faiss_end']})."
            )
        cursor = s["faiss_end"]
        expected_idx += 1

    if shards and cursor != faiss_index.ntotal:
        raise RuntimeError(
            f"manifest.json final faiss_end={cursor} does not match "
            f"faiss.index.ntotal={faiss_index.ntotal}."
        )

    log.info("manifest.json validated: %d shards, contiguous [0, %d), "
              "matches faiss.index.ntotal.", len(shards), cursor)


def check_corpus_integrity(
    save_dir    : Path,
    manifest    : Dict[str, Any],
    faiss_index : faiss.IndexFlatIP,
    progress    : Dict[str, Any],
) -> None:
    """
    Full startup integrity validation:
      - manifest/FAISS consistency (via validate_manifest)
      - every metadata shard's length must match its manifest range size
      - every chunks shard's length must match its metadata shard's length

    Each shard pair is loaded one at a time and discarded immediately, so
    memory stays O(1) regardless of corpus size.
    """
    validate_manifest(manifest, faiss_index, progress)

    for shard in manifest["shards"]:
        bidx = shard["batch_idx"]
        expected_n = shard["faiss_end"] - shard["faiss_start"]

        meta_path = _shard_path(save_dir, "metadata", "metadata", bidx)
        if not meta_path.exists():
            raise RuntimeError(
                f"INTEGRITY CHECK FAILED: metadata shard {meta_path.name} "
                f"is missing."
            )
        with open(meta_path, "rb") as f:
            meta_len = len(pickle.load(f))
        if meta_len != expected_n:
            raise RuntimeError(
                f"INTEGRITY CHECK FAILED: metadata shard {meta_path.name} "
                f"has {meta_len} entries but manifest expects {expected_n} "
                f"(batch {bidx})."
            )

        chunks_path = _shard_path(save_dir, "chunks", "chunks", bidx)
        if not chunks_path.exists():
            raise RuntimeError(
                f"INTEGRITY CHECK FAILED: chunks shard {chunks_path.name} "
                f"is missing."
            )
        with open(chunks_path, "rb") as f:
            chunks_len = len(pickle.load(f))
        if chunks_len != meta_len:
            raise RuntimeError(
                f"INTEGRITY CHECK FAILED: chunks shard {chunks_path.name} "
                f"has {chunks_len} entries but metadata shard "
                f"{meta_path.name} has {meta_len} (batch {bidx})."
            )
        del meta_len

        # QA shard: must exist and be loadable. Size is not checked against
        # FAISS/metadata because QA pairs come from separate datasets and
        # can legitimately be 0 when all rows were filtered out.
        qa_path = _shard_path(save_dir, "qa_pairs", "qa", bidx)
        if not qa_path.exists():
            raise RuntimeError(
                f"INTEGRITY CHECK FAILED: QA shard {qa_path.name} is missing."
            )
        try:
            with open(qa_path, "rb") as f:
                pickle.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"INTEGRITY CHECK FAILED: QA shard {qa_path.name} "
                f"cannot be loaded: {exc}"
            ) from exc

    log.info("Corpus integrity check passed: %d shards, all metadata/chunk/QA "
              "shards consistent with manifest, FAISS ntotal=%d.",
              len(manifest["shards"]), faiss_index.ntotal)


# ─── BATCH-LEVEL SHARD FILE NAMING ──────────────────────────

def _shard_name(kind: str, batch_idx: int) -> str:
    return f"{kind}_{batch_idx:05d}.pkl"


def _shard_path(save_dir: Path, subdir: str, kind: str, batch_idx: int) -> Path:
    return save_dir / subdir / _shard_name(kind, batch_idx)


# ─── BATCH RANGES ───────────────────────────────────────────

def compute_batch_ranges(
    progress        : Dict[str, Any],
    batch_size      : int,
    dataset_lengths : Dict[str, int],
    max_rows        : Dict[str, Optional[int]],
) -> Dict[str, Tuple[int, int]]:
    ranges: Dict[str, Tuple[int, int]] = {}
    for name, offset in progress["dataset_offsets"].items():
        available = dataset_lengths.get(name, 0)
        cap = max_rows.get(name)
        if cap is not None:
            available = min(available, cap)

        start = min(offset, available)
        end   = min(start + batch_size, available)
        ranges[name] = (start, end)

        if end > start:
            log.info("  %-12s batch rows [%d, %d)  (of %d available)",
                     name, start, end, available)
        else:
            log.info("  %-12s already complete (%d/%d rows) — skipping",
                     name, start, available)
    return ranges


def all_datasets_exhausted(ranges: Dict[str, Tuple[int, int]]) -> bool:
    return all(end <= start for start, end in ranges.values())


# ─── DATASET LOADING ────────────────────────────────────────

def load_datasets(ranges: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
    log.info("Loading datasets for this batch …")
    raw: Dict[str, Any] = {}

    def _load(name, start, end, *args, **kwargs):
        ds = load_dataset(*args, **kwargs)
        if end <= start:
            return ds.select(range(0))
        end = min(end, len(ds))
        ds = ds.select(range(start, end))
        log.info("  %-15s  rows [%d, %d)  (%d rows)", name, start, end, len(ds))
        return ds

    s, e = ranges["nq_passages"]
    raw["nq_passages"] = _load("nq_passages", s, e,
        "sentence-transformers/natural-questions", split="train")

    s, e = ranges["nq_answers"]
    raw["nq_answers"] = _load("nq_answers", s, e,
        "google-research-datasets/nq_open", split="train")

    s, e = ranges["hotpot"]
    raw["hotpot"] = _load("hotpot", s, e,
        "hotpotqa/hotpot_qa", "distractor", split="train")

    s, e = ranges["trivia"]
    raw["trivia"] = _load("trivia", s, e,
        "mandarjoshi/trivia_qa", "rc.wikipedia", split="train",
        trust_remote_code=True)

    log.info("Batch datasets loaded.")
    return raw


def get_dataset_lengths(
    save_dir      : Path,
    max_rows_hint : Dict[str, Optional[int]],
) -> Dict[str, int]:
    """
    Returns the number of rows in each dataset. On the first call the
    lengths are computed by loading each dataset once, then written to
    dataset_lengths.json. On every subsequent call the cached file is
    returned immediately — no datasets are loaded.

    The cache is invalidated if max_rows_hint changes, but since those
    are corpus-level constants in CONFIG that never change between runs,
    this is intentionally not checked here; edit or delete
    dataset_lengths.json to force a re-probe.
    """
    cache_path = save_dir / "dataset_lengths.json"

    if cache_path.exists():
        with open(cache_path) as f:
            lengths = json.load(f)
        log.info("Loaded cached dataset lengths from dataset_lengths.json: %s", lengths)
        return lengths

    log.info("dataset_lengths.json not found — probing dataset sizes (once only) …")
    lengths: Dict[str, int] = {}

    ds = load_dataset("sentence-transformers/natural-questions", split="train")
    lengths["nq_passages"] = len(ds); del ds

    ds = load_dataset("google-research-datasets/nq_open", split="train")
    lengths["nq_answers"] = len(ds); del ds

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    lengths["hotpot"] = len(ds); del ds

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="train",
                      trust_remote_code=True)
    lengths["trivia"] = len(ds); del ds

    gc.collect()

    tmp = cache_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(lengths, f, indent=2)
    tmp.rename(cache_path)
    log.info("Dataset lengths cached to dataset_lengths.json: %s", lengths)
    return lengths


# ─── PERSISTENT DOCUMENT HASH SET ──────────────────────────
#
# Extension point for future cross-run deduplication.
#
# Each committed document's content fingerprint (first 200 chars, same
# as the existing batch-local dedup key) is stored as a hex-digest in a
# persistent set on disk. This costs ~40 bytes per document on disk and
# ~80 bytes in RAM — negligible for millions of documents.
#
# TODAY: the set is loaded at the start of each batch and used to skip
# documents already seen in prior runs, then updated and persisted at
# the end of `create_documents`. Batch-local dedup (seen_keys) is
# unchanged and still runs first.
#
# FUTURE: swap `_hash_text` for a stronger function, or swap the
# set-on-disk for a bloom filter, without touching any other code.

_DOC_HASHES_FILE = "doc_hashes.json"


def _hash_text(text: str, dataset: str = "", row_idx: int = -1) -> str:
    """
    Fingerprint of a document keyed on dataset + row position + leading 200
    chars.  Including dataset and row_idx prevents false-positive dedup when
    two distinct rows share the same opening text (e.g. Wikipedia passages
    that all start with the same article introduction).
    """
    key = f"{dataset}:{row_idx}:{text[:200]}"
    return hashlib.md5(key.encode("utf-8", errors="replace")).hexdigest()


def load_doc_hashes(save_dir: Path) -> Set[str]:
    path = save_dir / _DOC_HASHES_FILE
    if not path.exists():
        return set()
    with open(path) as f:
        return set(json.load(f))


def save_doc_hashes(save_dir: Path, hashes: Set[str]) -> None:
    path = save_dir / _DOC_HASHES_FILE
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(sorted(hashes), f)
    tmp.rename(path)
    log.info("doc_hashes.json updated (%d total hashes).", len(hashes))

_Q_HINTS: Set[str] = {"question", "query"}
_A_HINTS: Set[str] = {"answer", "answers", "target", "short_answers"}
_C_HINTS: Set[str] = {
    "context", "passage", "passages", "document", "documents",
    "text", "content", "search_results",
}
_T_HINTS: Set[str] = {"title", "titles", "subject"}

_FIELD_RULES: Dict[str, Dict[str, List[str]]] = {
    "nq_passages": {"required": [],                                "optional": ["query", "pos", "neg"]},
    "nq_answers" : {"required": ["question", "answers"],           "optional": []},
    "hotpot"     : {"required": ["question", "answers", "context"],"optional": ["title"]},
    "trivia"     : {"required": ["question", "answers"],           "optional": ["context"]},
}


def _detect_field(cols: List[str], hints: Set[str]) -> Optional[str]:
    for col in cols:
        for h in hints:
            if h in col.lower():
                return col
    return None


def inspect_schema(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    schemas: Dict[str, Dict[str, Any]] = {}
    print("\n" + "=" * 64)
    print("  DATASET SCHEMA INSPECTION")
    print("=" * 64)

    for name, ds in raw.items():
        cols   = list(ds.features.keys())
        schema = {
            "columns" : cols,
            "question": _detect_field(cols, _Q_HINTS),
            "answers" : _detect_field(cols, _A_HINTS),
            "context" : _detect_field(cols, _C_HINTS),
            "title"   : _detect_field(cols, _T_HINTS),
        }
        if name == "hotpot":
            if "context" in cols:
                schema["context"] = "context"
            else:
                raise SchemaError(f"[hotpot] No 'context' column. Available: {cols}")
        if name == "nq_passages":
            schema["question"] = _detect_field(cols, _Q_HINTS)
            schema["context"]  = None

        schemas[name] = schema
        print(f"\n[{name}]")
        print(f"  All columns  : {cols}")
        for k in ("question", "answers", "context", "title"):
            print(f"  → {k:10s}: {schema[k]}")
        if len(ds) > 0:
            print("  Sample row   :")
            for k, v in ds[0].items():
                print(f"    {k:30s}: {str(v)[:120].replace(chr(10), ' ')}")
        else:
            print("  (0 rows in this batch)")

    print("\n" + "=" * 64 + "\n")

    errors: List[str] = []
    for ds_name, rules in _FIELD_RULES.items():
        for field in rules["required"]:
            if schemas[ds_name].get(field) is None:
                errors.append(f"[{ds_name}] Required field '{field}' not found. "
                               f"Columns: {schemas[ds_name]['columns']}")
    if errors:
        raise SchemaError("Schema validation failed:\n" +
                          "\n".join(f"  • {e}" for e in errors))

    log.info("Schema validation passed.")
    return schemas


# ─── HELPERS ────────────────────────────────────────────────

def _make_doc_id(dataset: str, global_row_idx: int, text: str) -> str:
    digest = hashlib.md5(text[:64].encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{dataset}_{global_row_idx:07d}_{digest}"


def _flatten_answers(raw_ans: Any) -> List[str]:
    if raw_ans is None:
        return []
    if isinstance(raw_ans, str):
        s = raw_ans.strip()
        return [s] if s else []
    if isinstance(raw_ans, list):
        out: List[str] = []
        for item in raw_ans:
            out.extend(_flatten_answers(item))
        return out
    if isinstance(raw_ans, dict):
        for key in ("value", "normalized_value", "aliases",
                    "normalized_aliases", "human_answers"):
            if key in raw_ans:
                result = _flatten_answers(raw_ans[key])
                if result:
                    return result
        return [str(raw_ans)]
    return [str(raw_ans)]


def _extract_passages(
    row    : Dict[str, Any],
    schema : Dict[str, Any],
    ds_name: str,
) -> List[Tuple[str, str]]:
    if ds_name == "nq_passages":
        text = str(row.get("answer", "")).strip()
        return [("", text)] if text else []

    if ds_name == "trivia":
        out = []
        sr = row.get("search_results")
        if isinstance(sr, dict):
            descriptions = sr.get("description") or []
            sr_titles    = sr.get("title") or []
            for i, text in enumerate(descriptions):
                text = str(text).strip() if text else ""
                if text:
                    out.append((str(sr_titles[i] if i < len(sr_titles) else ""), text))
        ep = row.get("entity_pages")
        if isinstance(ep, dict):
            wiki_contexts = ep.get("wiki_context") or []
            ep_titles     = ep.get("title") or []
            for i, text in enumerate(wiki_contexts):
                text = str(text).strip() if text else ""
                if text:
                    out.append((str(ep_titles[i] if i < len(ep_titles) else ""), text))
        if not out:
            ctx_col = schema.get("context")
            if ctx_col:
                raw_ctx = row.get(ctx_col)
                if isinstance(raw_ctx, dict):
                    title_key = next((k for k in raw_ctx if "title" in k.lower()), None)
                    titles = raw_ctx.get(title_key, []) if title_key else []
                    for k, v in raw_ctx.items():
                        if k == title_key or not isinstance(v, list):
                            continue
                        for i, item in enumerate(v):
                            text = str(item).strip() if item else ""
                            if text:
                                out.append((str(titles[i] if i < len(titles) else ""), text))
                elif isinstance(raw_ctx, str) and raw_ctx.strip():
                    out.append(("", raw_ctx.strip()))
        return out

    if ds_name == "hotpot":
        ctx_col = schema.get("context")
        raw_ctx = row.get(ctx_col) if ctx_col else None
        if not isinstance(raw_ctx, dict):
            return []
        title_list = raw_ctx.get("title") or []
        sent_lists = raw_ctx.get("sentences") or []
        out = []
        for i, sents in enumerate(sent_lists):
            t    = title_list[i] if i < len(title_list) else ""
            text = " ".join(s for s in sents if isinstance(s, str)).strip() \
                   if isinstance(sents, list) else str(sents).strip()
            if text:
                out.append((str(t), text))
        return out

    ctx_col = schema.get("context")
    if not ctx_col:
        return []
    raw_ctx = row.get(ctx_col)
    if raw_ctx is None:
        return []
    if isinstance(raw_ctx, str):
        return [("", raw_ctx.strip())] if raw_ctx.strip() else []
    if isinstance(raw_ctx, list):
        return [("", str(item).strip()) for item in raw_ctx if str(item).strip()]
    return [("", str(raw_ctx).strip())]


# ─── ATOMIC SAVE ────────────────────────────────────────────

def _save_pkl(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    log.info("  Saved  %s  (%.1f MB)", path.name, path.stat().st_size / 1e6)


# ─── FAISS LOADING ──────────────────────────────────────────

def load_existing_faiss(save_dir: Path, dim: int) -> faiss.IndexFlatIP:
    path = save_dir / "faiss.index"
    if path.exists():
        index = faiss.read_index(str(path))
        if index.d != dim:
            raise RuntimeError(
                f"Existing faiss.index has dim={index.d} but embed_model "
                f"produces dim={dim}. Did you change embed_model?"
            )
        log.info("Loaded existing faiss.index: %d vectors", index.ntotal)
        return index

    log.info("No existing faiss.index — creating IndexFlatIP(dim=%d).", dim)
    return faiss.IndexFlatIP(dim)


def reconcile_faiss_with_manifest(
    faiss_index : faiss.IndexFlatIP,
    manifest    : Dict[str, Any],
    save_dir    : Path,
) -> faiss.IndexFlatIP:
    """
    Crash recovery: if a previous run wrote faiss.index but crashed before
    committing manifest.json/progress.json, faiss.index can have MORE
    vectors than the manifest knows about ("orphaned" vectors from the
    aborted batch). Truncate back to the last committed boundary AND
    immediately persist the repaired index to disk (atomic temp-file +
    rename) so disk state and in-memory state are identical.
    """
    expected_end = manifest["shards"][-1]["faiss_end"] if manifest["shards"] else 0

    if faiss_index.ntotal == expected_end:
        return faiss_index

    if faiss_index.ntotal < expected_end:
        raise RuntimeError(
            f"CRASH RECOVERY FAILED: faiss.index has only "
            f"{faiss_index.ntotal} vectors but manifest.json expects "
            f"{expected_end}. The index and manifest are inconsistent in "
            f"a way that cannot be auto-repaired — restore both from a "
            f"consistent backup before re-running."
        )

    orphaned = faiss_index.ntotal - expected_end
    log.warning(
        "Detected %d orphaned FAISS vector(s) from an interrupted run "
        "(faiss.index.ntotal=%d, manifest expects %d) — truncating back "
        "to the last committed boundary.",
        orphaned, faiss_index.ntotal, expected_end,
    )
    kept = faiss_index.reconstruct_n(0, expected_end)
    rebuilt = faiss.IndexFlatIP(faiss_index.d)
    rebuilt.add(kept)

    faiss_path = save_dir / "faiss.index"
    faiss_tmp  = save_dir / "faiss.index.tmp"
    faiss.write_index(rebuilt, str(faiss_tmp))
    os.replace(faiss_tmp, faiss_path)
    log.info("Persisted repaired faiss.index to disk (%d vectors).",
              rebuilt.ntotal)

    return rebuilt


def verify_alignment(
    manifest    : Dict[str, Any],
    faiss_index : faiss.IndexFlatIP,
    new_metadata: List[Dict],
    faiss_start : int,
) -> None:
    expected_start = (
        manifest["shards"][-1]["faiss_end"] if manifest["shards"] else 0
    )
    if faiss_start != expected_start:
        raise RuntimeError(
            f"ALIGNMENT BROKEN: new shard faiss_start={faiss_start} but "
            f"manifest expects {expected_start}. Refusing to save."
        )
    expected_end = faiss_start + len(new_metadata)
    if faiss_index.ntotal != expected_end:
        raise RuntimeError(
            f"ALIGNMENT BROKEN: faiss_index.ntotal={faiss_index.ntotal} but "
            f"faiss_start({faiss_start}) + len(new_metadata)({len(new_metadata)}) "
            f"= {expected_end}. Refusing to save."
        )


# ─── DOCUMENT CREATION ──────────────────────────────────────

def create_documents(
    raw          : Dict[str, Any],
    schemas      : Dict[str, Dict[str, Any]],
    batch_starts : Dict[str, int],
    save_dir     : Path,
) -> List[Dict[str, Any]]:
    documents  : List[Dict[str, Any]] = []
    # Batch-local dedup: prefix-keyed set, cleared each run.
    seen_keys  : Set[str] = set()
    # Cross-run dedup: persisted hash set, loaded from prior runs.
    seen_hashes: Set[str] = load_doc_hashes(save_dir)
    prior_count = len(seen_hashes)
    cross_run_skipped = 0

    def _add(ds_name, global_row_idx, title, text):
        nonlocal cross_run_skipped
        text = text.strip()
        if not text:
            return
        # Fast batch-local check (prefix key, same as before).
        local_key = text[:200]
        if local_key in seen_keys:
            return
        seen_keys.add(local_key)
        # Cross-run check via persistent hash (keyed on dataset + row + text).
        h = _hash_text(text, ds_name, global_row_idx)
        if h in seen_hashes:
            cross_run_skipped += 1
            return
        seen_hashes.add(h)
        documents.append({
            "doc_id"  : _make_doc_id(ds_name, global_row_idx, text),
            "dataset" : ds_name,
            "title"   : (title or "").strip(),
            "text"    : text,
            "row_idx" : global_row_idx,
        })

    for ds_name in ("nq_passages", "hotpot", "trivia"):
        ds = raw[ds_name]
        if len(ds) == 0:
            continue
        schema       = schemas[ds_name]
        start_offset = batch_starts[ds_name]
        log.info("Extracting documents from %s (global rows %d..%d) …",
                 ds_name, start_offset, start_offset + len(ds) - 1)
        for local_idx, row in enumerate(tqdm(ds, desc=f"{ds_name} docs")):
            global_row_idx = start_offset + local_idx
            for title, text in _extract_passages(row, schema, ds_name):
                _add(ds_name, global_row_idx, title, text)

    # Persist the updated hash set (only if new hashes were added).
    new_hashes = len(seen_hashes) - prior_count
    if new_hashes > 0:
        save_doc_hashes(save_dir, seen_hashes)

    log.info(
        "New documents this batch: %d  (cross-run dedup skipped: %d, "
        "total corpus hashes: %d)",
        len(documents), cross_run_skipped, len(seen_hashes),
    )
    return documents


# ─── QA PAIR CREATION ───────────────────────────────────────

def create_qa_pairs(
    raw          : Dict[str, Any],
    schemas      : Dict[str, Dict[str, Any]],
    documents    : List[Dict[str, Any]],
    batch_starts : Dict[str, int],
) -> List[Dict[str, Any]]:
    qa_pairs  : List[Dict[str, Any]] = []

    # Build a row→doc_id map only for datasets where passages and QA rows
    # share the same row index (hotpot, trivia). nq_answers is a completely
    # separate dataset from nq_passages and must NOT be looked up here.
    row_to_doc: Dict[Tuple[str, int], str] = {}
    for doc in documents:
        if doc["dataset"] in ("hotpot", "trivia"):
            key = (doc["dataset"], doc["row_idx"])
            if key not in row_to_doc:
                row_to_doc[key] = doc["doc_id"]

    def _make_qid(ds_name, global_idx):
        return f"{ds_name}_q{global_idx:07d}"

    for ds_name, hf_name in [
        ("nq_answers", "nq"),
        ("hotpot",     "hotpot"),
        ("trivia",     "trivia"),
    ]:
        ds = raw[ds_name] if ds_name in raw else raw.get(ds_name)
        if ds is None or len(ds) == 0:
            continue
        log.info("Creating QA pairs from %s …", ds_name)
        q_col        = schemas[ds_name]["question"]
        a_col        = schemas[ds_name]["answers"]
        start_offset = batch_starts[ds_name]
        for local_idx, row in enumerate(tqdm(ds, desc=f"{ds_name} QA")):
            global_idx = start_offset + local_idx
            question   = str(row.get(q_col) or "").strip()
            answers    = _flatten_answers(row.get(a_col))
            if not question or not answers:
                continue

            # nq_answers rows cannot be linked to nq_passages doc_ids:
            # they are independent datasets with no shared row key.
            if ds_name == "nq_answers":
                doc_id = None
            else:
                doc_id = row_to_doc.get((ds_name, global_idx), None)

            qa_pairs.append({
                "question_id": _make_qid(hf_name, global_idx),
                "question"   : question,
                "answers"    : answers,
                "dataset"    : hf_name,
                "doc_id"     : doc_id,
            })

    log.info("New QA pairs this batch: %d", len(qa_pairs))
    return qa_pairs


# ─── CHUNKING ───────────────────────────────────────────────

def chunk_documents(
    documents     : List[Dict[str, Any]],
    chunk_size    : int,
    chunk_overlap : int,
    next_chunk_id : int,
) -> Tuple[List[str], List[Dict[str, Any]], int]:
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    chunks  : List[str]  = []
    metadata: List[Dict] = []
    step = chunk_size - chunk_overlap
    cid  = next_chunk_id

    for doc in tqdm(documents, desc="Chunking"):
        text = doc["text"]
        if not text:
            continue
        start = 0
        while start < len(text):
            end   = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({
                    "chunk_id"  : cid,
                    "doc_id"    : doc["doc_id"],
                    "dataset"   : doc["dataset"],
                    "title"     : doc["title"],
                    "char_start": start,
                })
                cid += 1
            start += step

    assert len(chunks) == len(metadata), \
        f"BUG: chunks ({len(chunks)}) ≠ metadata ({len(metadata)})"
    log.info("New chunks this batch: %d  (chunk_id %d → %d)",
             len(chunks), next_chunk_id, cid - 1 if cid > next_chunk_id else next_chunk_id)
    return chunks, metadata, cid


# ─── FAISS EMBEDDING & APPEND ───────────────────────────────

def _estimate_faiss_memory(n_new: int, dim: int) -> None:
    index_mb = (n_new * dim * 4) / 1e6
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e6
        total = psutil.virtual_memory().total / 1e6
        avail_str = f"{avail:,.0f} MB available / {total:,.0f} MB total"
    except ImportError:
        avail_str = "psutil not installed"

    print("\n" + "─" * 56)
    print("  FAISS MEMORY ESTIMATE (this batch's NEW vectors)")
    print("─" * 56)
    print(f"  New chunks to embed : {n_new:>10,}")
    print(f"  Embedding dim        : {dim:>10,}")
    print(f"  New vectors RAM      : {index_mb:>10,.1f} MB")
    print(f"  System RAM           : {avail_str}")
    print("─" * 56 + "\n")


def embed_and_append(
    new_chunks       : List[str],
    faiss_index      : faiss.IndexFlatIP,
    model            : SentenceTransformer,
    batch_size       : int,
    checkpoint_every : int,
    save_dir         : Path,
) -> Tuple[faiss.IndexFlatIP, SentenceTransformer]:
    """
    Takes an already-loaded SentenceTransformer instead of a model name —
    the embedding model is loaded exactly once per run (in main()) and
    reused here, instead of being instantiated a second time.
    """
    ckpt_index_path = save_dir / "ckpt" / "faiss_partial.index"
    ckpt_meta_path  = save_dir / "ckpt" / "faiss_ckpt_meta.json"

    dim = model.get_sentence_embedding_dimension()
    log.info("Embedding model dim=%d (reusing already-loaded model).", dim)

    n_new = len(new_chunks)
    _estimate_faiss_memory(n_new, dim)

    base_ntotal  = faiss_index.ntotal
    start_idx    = 0
    last_ckpt_at = base_ntotal

    if ckpt_index_path.exists() and ckpt_meta_path.exists():
        try:
            with open(ckpt_meta_path) as f:
                ckpt_meta = json.load(f)
            saved_base  = ckpt_meta.get("base_ntotal", -1)
            saved_n_new = ckpt_meta.get("n_new_added", 0)
            if saved_base == base_ntotal and 0 < saved_n_new <= n_new:
                candidate = faiss.read_index(str(ckpt_index_path))
                if candidate.ntotal == base_ntotal + saved_n_new and candidate.d == dim:
                    faiss_index  = candidate
                    start_idx    = saved_n_new
                    last_ckpt_at = candidate.ntotal
                    log.info("Resuming FAISS batch from checkpoint: %d/%d new vectors.",
                             start_idx, n_new)
                else:
                    log.warning("Stale FAISS partial checkpoint — ignoring.")
            else:
                log.warning("FAISS checkpoint base_ntotal mismatch — re-embedding batch.")
        except Exception as exc:
            log.warning("Cannot load FAISS checkpoint (%s) — re-embedding batch.", exc)

    log.info("Encoding new chunks %d → %d …", start_idx, n_new)
    pbar = tqdm(total=n_new - start_idx, desc="FAISS encode (new)")

    for batch_start in range(start_idx, n_new, batch_size):
        batch_end = min(batch_start + batch_size, n_new)
        batch     = new_chunks[batch_start:batch_end]

        embs = model.encode(
            batch,
            batch_size           = batch_size,
            show_progress_bar    = False,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        ).astype(np.float32)

        faiss_index.add(embs)
        pbar.update(len(batch))

        vectors_since_ckpt = faiss_index.ntotal - last_ckpt_at
        is_last = (batch_end == n_new)
        if vectors_since_ckpt >= checkpoint_every or is_last:
            faiss.write_index(faiss_index, str(ckpt_index_path))
            with open(ckpt_meta_path, "w") as f:
                json.dump({"base_ntotal": base_ntotal,
                           "n_new_added": faiss_index.ntotal - base_ntotal}, f)
            last_ckpt_at = faiss_index.ntotal
            log.info("  [faiss ckpt] %d/%d new vectors (total=%d).",
                     faiss_index.ntotal - base_ntotal, n_new, faiss_index.ntotal)

        del embs, batch
        gc.collect()

    pbar.close()
    log.info("FAISS: %d total vectors (was %d, +%d new), dim=%d",
             faiss_index.ntotal, base_ntotal,
             faiss_index.ntotal - base_ntotal, dim)

    for p in (ckpt_index_path, ckpt_meta_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    return faiss_index, model


# ─── FINALIZE RUN ───────────────────────────────────────────

def finalize_run(
    save_dir      : Path,
    batch_idx     : int,
    new_chunks    : List[str],
    new_metadata  : List[Dict],
    new_qa_pairs  : List[Dict],
    faiss_index   : faiss.IndexFlatIP,
    faiss_start   : int,
    manifest      : Dict[str, Any],
    progress      : Dict[str, Any],
    new_offsets   : Dict[str, int],
    next_chunk_id : int,
) -> Dict[str, str]:

    verify_alignment(manifest, faiss_index, new_metadata, faiss_start)

    save_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    for obj, subdir, kind in [
        (new_chunks,   "chunks",   "chunks"),
        (new_metadata, "metadata", "metadata"),
        (new_qa_pairs, "qa_pairs", "qa"),
    ]:
        p = _shard_path(save_dir, subdir, kind, batch_idx)
        _save_pkl(obj, p)
        paths[f"{kind}_{batch_idx:05d}"] = str(p)

    faiss_path = save_dir / "faiss.index"
    faiss_tmp  = save_dir / "faiss.index.tmp"
    faiss.write_index(faiss_index, str(faiss_tmp))
    os.replace(faiss_tmp, faiss_path)
    log.info("  Saved  faiss.index  (%.1f MB)", faiss_path.stat().st_size / 1e6)
    paths["faiss"] = str(faiss_path)

    manifest["shards"].append({
        "batch_idx"  : batch_idx,
        "faiss_start": faiss_start,
        "faiss_end"  : faiss_index.ntotal,
    })
    save_manifest(save_dir, manifest)
    paths["manifest"] = str(save_dir / MANIFEST_FILE)

    progress["dataset_offsets"]      = new_offsets
    progress["next_chunk_id"]        = next_chunk_id
    progress["completed_runs"]       = progress.get("completed_runs", 0) + 1
    progress["completed_batches"]    = batch_idx + 1
    progress["faiss_total_vectors"]  = faiss_index.ntotal
    progress["last_run_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_progress(save_dir, progress)

    return paths


# ─── STATISTICS ─────────────────────────────────────────────

def print_statistics(
    progress    : Dict[str, Any],
    manifest    : Dict[str, Any],
    new_qa_pairs: List[Dict],
    new_chunks  : List[str],
    new_metadata: List[Dict],
    faiss_index : faiss.IndexFlatIP,
    paths       : Dict[str, str],
) -> None:
    total_shards  = len(manifest["shards"])
    total_vectors = faiss_index.ntotal
    avg_len       = (sum(len(c) for c in new_chunks) / len(new_chunks)
                     if new_chunks else 0)

    chunk_by_ds: Dict[str, int] = {}
    for m in new_metadata:
        chunk_by_ds[m["dataset"]] = chunk_by_ds.get(m["dataset"], 0) + 1

    qa_by_ds: Dict[str, int] = {}
    for q in new_qa_pairs:
        qa_by_ds[q["dataset"]] = qa_by_ds.get(q["dataset"], 0) + 1

    print("\n" + "=" * 60)
    print("  CUMULATIVE STATISTICS (after this run)")
    print("=" * 60)
    print(f"  Committed batches   : {total_shards:>10,}")
    print(f"  FAISS total vectors : {total_vectors:>10,}")
    print(f"  QA pairs this batch : {len(new_qa_pairs):>10,}")
    print(f"  Chunks this batch   : {len(new_chunks):>10,}")
    print(f"  Avg chunk length    : {avg_len:>10.1f}  chars")
    print()
    print("  Dataset row offsets (rows processed so far):")
    for ds, off in sorted(progress["dataset_offsets"].items()):
        print(f"    {ds:20s}: {off:>10,}")
    print()
    print("  Chunks this batch by dataset:")
    for ds, cnt in sorted(chunk_by_ds.items()):
        print(f"    {ds:20s}: {cnt:>10,}")
    print()
    print("  QA pairs this batch by dataset:")
    for ds, cnt in sorted(qa_by_ds.items()):
        print(f"    {ds:20s}: {cnt:>10,}")
    print()
    print("  FAISS shard manifest:")
    for s in manifest["shards"]:
        n = s["faiss_end"] - s["faiss_start"]
        print(f"    batch {s['batch_idx']:05d}: vectors [{s['faiss_start']:>8,}, {s['faiss_end']:>8,})  ({n:>7,} vecs)")
    print()
    print("  Saved artifacts (this batch):")
    for k, v in sorted(paths.items()):
        p = Path(v)
        size_mb = p.stat().st_size / 1e6 if p.exists() else 0
        print(f"    {k:30s}: {p.name}  ({size_mb:.1f} MB)")
    print("=" * 60 + "\n")


# ─── SANITY CHECK ───────────────────────────────────────────

def sanity_check(
    new_chunks  : List[str],
    new_metadata: List[Dict],
    faiss_index : faiss.IndexFlatIP,
    faiss_start : int,
    model       : SentenceTransformer,
    top_k       : int = 3,
) -> None:
    if not new_chunks:
        log.info("Sanity check skipped — no new chunks this batch.")
        return

    # ── 1. Assert the index grew by exactly the number of new vectors ──
    expected_total = faiss_start + len(new_chunks)
    if faiss_index.ntotal != expected_total:
        raise RuntimeError(
            f"SANITY CHECK FAILED: faiss_index.ntotal={faiss_index.ntotal} "
            f"but faiss_start({faiss_start}) + len(new_chunks)({len(new_chunks)}) "
            f"= {expected_total}. FAISS did not grow by the expected amount."
        )
    log.info("Sanity check: FAISS vector count correct (%d total, +%d this batch).",
             faiss_index.ntotal, len(new_chunks))

    # ── 2. Pick a representative query from this batch ─────────────────
    # Use the first chunk that is long enough to be meaningful. This
    # avoids any hardcoded domain assumption and exercises actual content.
    probe_chunk = next(
        (c for c in new_chunks if len(c.split()) >= 10),
        new_chunks[0],
    )
    # Truncate to the first sentence / 120 chars for a tighter query.
    first_sentence = probe_chunk.split(".")[0][:120].strip()
    query = first_sentence if len(first_sentence.split()) >= 4 else probe_chunk[:120].strip()

    print("\n" + "─" * 56)
    print(f"  SANITY CHECK")
    print(f"  Probe query (from this batch): '{query[:80]}…'" if len(query) > 80 else
          f"  Probe query (from this batch): '{query}'")
    print("─" * 56)

    # ── 3. FAISS search — top results should include new-batch vectors ──
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores_f, idxs_f = faiss_index.search(q_emb, min(top_k, faiss_index.ntotal))

    from_new_batch = 0
    print(f"\nFAISS top results (global index, {faiss_index.ntotal} vectors):")
    for rank, (score, i) in enumerate(zip(scores_f[0], idxs_f[0]), 1):
        if faiss_start <= i < faiss_start + len(new_chunks):
            local_i = i - faiss_start
            m   = new_metadata[local_i]
            txt = new_chunks[local_i][:200].strip()
            print(f"  {rank}. [{m['dataset']}] {m['title']!r}  (score={score:.4f})")
            print(f"     {txt} …")
            from_new_batch += 1
        else:
            print(f"  {rank}. faiss_idx={i} from prior batch  (score={score:.4f})")

    if from_new_batch == 0:
        log.warning(
            "Sanity check: none of the top-%d FAISS results came from the "
            "current batch — the query may be too generic or the batch is "
            "very small compared to the full corpus.", top_k
        )
    else:
        log.info("Sanity check: %d/%d top results came from the current batch.",
                 from_new_batch, top_k)

    print("─" * 56 + "\n")


# ─── QUERY-TIME RETRIEVAL HELPERS ───────────────────────────

_SHARD_CACHE_SIZE = 8
_metadata_cache: "OrderedDict[Tuple[str, int], List[Dict]]" = OrderedDict()
_chunk_cache: "OrderedDict[Tuple[str, int], List[str]]" = OrderedDict()

# Process-wide cached embedding model, used by retrieve() when no model
# is explicitly injected.
_shared_embed_model: Optional[SentenceTransformer] = None


def get_shared_embed_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Returns a process-wide cached SentenceTransformer instance, loading
    it on first call only.
    """
    global _shared_embed_model
    if _shared_embed_model is None:
        _shared_embed_model = SentenceTransformer(model_name or CONFIG["embed_model"])
    return _shared_embed_model


def _load_shard_cached(
    cache    : "OrderedDict",
    save_dir : Path,
    subdir   : str,
    kind     : str,
    batch_idx: int,
) -> List:
    key = (subdir, batch_idx)
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    path = _shard_path(save_dir, subdir, kind, batch_idx)
    with open(path, "rb") as f:
        data = pickle.load(f)
    cache[key] = data
    cache.move_to_end(key)
    if len(cache) > _SHARD_CACHE_SIZE:
        cache.popitem(last=False)
    return data


def load_metadata_for_vector(
    save_dir  : Path,
    manifest  : Dict[str, Any],
    vector_idx: int,
) -> Optional[Dict]:
    shard = shard_for_vector(manifest, vector_idx)
    if shard is None:
        return None
    metadata_shard = _load_shard_cached(_metadata_cache, save_dir, "metadata",
                                         "metadata", shard["batch_idx"])
    local_idx = vector_idx - shard["faiss_start"]
    return metadata_shard[local_idx]


def load_chunk_for_vector(
    save_dir  : Path,
    manifest  : Dict[str, Any],
    vector_idx: int,
) -> Optional[str]:
    shard = shard_for_vector(manifest, vector_idx)
    if shard is None:
        return None
    chunks_shard = _load_shard_cached(_chunk_cache, save_dir, "chunks",
                                       "chunks", shard["batch_idx"])
    local_idx = vector_idx - shard["faiss_start"]
    return chunks_shard[local_idx]


def retrieve(
    query    : str,
    save_dir : Path,
    top_k    : int = 5,
    model    : Optional[SentenceTransformer] = None,
) -> List[Dict[str, Any]]:
    """
    FAISS dense retrieval. Results are grouped by owning shard before any
    file is opened, so each metadata/chunk shard is read at most once per
    query. A small LRU cache avoids re-reading shards from recent queries.

    `model`: pass an already-loaded SentenceTransformer to reuse it. If
    omitted, falls back to a process-wide cached singleton so the model
    is only ever loaded once per process.
    """
    manifest    = load_manifest(save_dir)
    faiss_index = faiss.read_index(str(save_dir / "faiss.index"))

    if model is None:
        model = get_shared_embed_model(CONFIG["embed_model"])

    q_emb  = model.encode([query], normalize_embeddings=True).astype(np.float32)
    _, idxs = faiss_index.search(q_emb, top_k)
    idxs = [int(i) for i in idxs[0] if i >= 0]

    by_shard: Dict[int, List[int]] = defaultdict(list)
    for vec_idx in idxs:
        shard = shard_for_vector(manifest, vec_idx)
        if shard is None:
            log.warning("FAISS vector %d has no owning shard in manifest.json "
                        "— skipping.", vec_idx)
            continue
        by_shard[shard["batch_idx"]].append(vec_idx)

    results: List[Dict[str, Any]] = []
    for batch_idx, vec_idxs in by_shard.items():
        shard = next(s for s in manifest["shards"] if s["batch_idx"] == batch_idx)
        metadata_shard = _load_shard_cached(_metadata_cache, save_dir,
                                             "metadata", "metadata", batch_idx)
        chunks_shard   = _load_shard_cached(_chunk_cache, save_dir,
                                             "chunks", "chunks", batch_idx)
        for vec_idx in vec_idxs:
            local_idx = vec_idx - shard["faiss_start"]
            if 0 <= local_idx < len(metadata_shard):
                results.append({
                    "metadata" : metadata_shard[local_idx],
                    "chunk"    : chunks_shard[local_idx],
                    "faiss_idx": vec_idx,
                })

    order = {idx: pos for pos, idx in enumerate(idxs)}
    results.sort(key=lambda r: order.get(r["faiss_idx"], len(idxs)))
    return results


# ─── MAIN ORCHESTRATOR ──────────────────────────────────────

def main() -> None:
    t0  = time.time()
    cfg = CONFIG

    save_dir = mount_drive(cfg["drive_base"])

    progress = load_progress(save_dir)
    manifest = load_manifest(save_dir)

    # ── Embedding model: loaded ONCE, reused for dim + encoding below ──
    model = SentenceTransformer(cfg["embed_model"])
    model_dim = model.get_sentence_embedding_dimension()

    # ── FAISS load + crash-recovery reconciliation ────────────────────
    faiss_index = load_existing_faiss(save_dir, dim=model_dim)
    faiss_index = reconcile_faiss_with_manifest(faiss_index, manifest, save_dir)

    # ── Auto-repair: roll back any empty batch left by the old dedup bug,
    #    OR by a mid-FAISS-encoding crash (same on-disk signature: last
    #    shard has faiss_start == faiss_end). Must run BEFORE the hard
    #    integrity check below, which would otherwise raise on exactly
    #    this recoverable state. ──────────────────────────────────────────
    progress, manifest = _rollback_empty_batch_if_needed(save_dir, progress, manifest)

    # ── Full corpus integrity validation ──────────────────────────────
    check_corpus_integrity(save_dir, manifest, faiss_index, progress)

    dataset_lengths = get_dataset_lengths(save_dir, cfg["max_rows"])
    ranges = compute_batch_ranges(
        progress, cfg["batch_size"], dataset_lengths, cfg["max_rows"]
    )

    if all_datasets_exhausted(ranges):
        log.info("All datasets fully processed — nothing to do.")
        return

    batch_idx    = progress["completed_batches"]
    batch_starts = {name: r[0] for name, r in ranges.items()}

    raw     = load_datasets(ranges)
    schemas = inspect_schema(raw)

    new_documents = create_documents(raw, schemas, batch_starts, save_dir)
    new_qa_pairs  = create_qa_pairs(raw, schemas, new_documents, batch_starts)
    del raw
    gc.collect()

    new_chunks, new_metadata, next_chunk_id = chunk_documents(
        new_documents,
        chunk_size    = cfg["chunk_size"],
        chunk_overlap = cfg["chunk_overlap"],
        next_chunk_id = progress["next_chunk_id"],
    )
    del new_documents
    gc.collect()

    faiss_start = faiss_index.ntotal

    faiss_index, model = embed_and_append(
        new_chunks,
        faiss_index,
        model            = model,
        batch_size       = cfg["embed_batch_size"],
        checkpoint_every = cfg["checkpoint_every"],
        save_dir         = save_dir,
    )

    new_offsets = {name: r[1] for name, r in ranges.items()}
    paths = finalize_run(
        save_dir      = save_dir,
        batch_idx     = batch_idx,
        new_chunks    = new_chunks,
        new_metadata  = new_metadata,
        new_qa_pairs  = new_qa_pairs,
        faiss_index   = faiss_index,
        faiss_start   = faiss_start,
        manifest      = manifest,
        progress      = progress,
        new_offsets   = new_offsets,
        next_chunk_id = next_chunk_id,
    )

    print_statistics(
        progress     = progress,
        manifest     = manifest,
        new_qa_pairs = new_qa_pairs,
        new_chunks   = new_chunks,
        new_metadata = new_metadata,
        faiss_index  = faiss_index,
        paths        = paths,
    )

    sanity_check(new_chunks, new_metadata, faiss_index, faiss_start, model)
    del model
    gc.collect()

    elapsed = time.time() - t0
    log.info("Batch %d complete in %.1f minutes. Re-run to process the next batch.",
             batch_idx, elapsed / 60)


# ─── ENTRY POINT ────────────────────────────────────────────
if __name__ == "__main__":
    main()
