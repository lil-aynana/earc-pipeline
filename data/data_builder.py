import gc
import hashlib
import json
import logging
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── 2. LOGGING ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RAG")


# ─── 3. CUSTOM EXCEPTION ────────────────────────────────────
class SchemaError(RuntimeError):
    """Raised when a required field cannot be found in a dataset."""


# ─── 4. CONFIGURATION ───────────────────────────────────────
CONFIG = dict(
    # Google Drive output directory
    drive_base       = "/content/drive/MyDrive/RAG_Project",

    # Chunking (character-level)
    chunk_size       = 800,
    chunk_overlap    = 100,

    # Embedding model
    embed_model      = "sentence-transformers/all-MiniLM-L6-v2",
    embed_batch_size = 256,    # reduce to 64 if you hit OOM

    # Vectors added to FAISS index before writing a checkpoint
    checkpoint_every = 5_000,

    # Set to a small int (e.g. 1000) for a quick smoke-test
    max_rows = dict(
        nq_passages = 5000,
        nq_answers  = 5000,
        hotpot      = 5000,
        trivia      = 5000,
    ),
)


# ─── 5. GOOGLE DRIVE MOUNT ──────────────────────────────────

def mount_drive(base_path: str) -> Path:
    """Mount Google Drive and create the project directory tree."""
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        log.info("Google Drive mounted.")
    except ImportError:
        log.warning("Not in Colab — skipping Drive mount.")
    except Exception as exc:
        log.error("Drive mount failed: %s", exc)

    p = Path(base_path)
    p.mkdir(parents=True, exist_ok=True)
    (p / "ckpt").mkdir(exist_ok=True)
    log.info("Project directory: %s", p)
    return p


# ─── 6. DATASET LOADING ─────────────────────────────────────

def load_datasets(max_rows: Dict[str, Optional[int]]) -> Dict[str, Any]:
    """
    Load all four dataset splits from the HuggingFace Hub.
    Returns a dict: logical_name → Dataset object.
    """
    log.info("Loading datasets …")
    raw: Dict[str, Any] = {}

    def _load(name: str, *args, **kwargs) -> Any:
        limit = max_rows.get(name)
        ds = load_dataset(*args, **kwargs)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        log.info("  %-15s  %d rows", name, len(ds))
        return ds

    raw["nq_passages"] = _load(
        "nq_passages",
        "sentence-transformers/natural-questions",
        split="train",
    )
    raw["nq_answers"] = _load(
        "nq_answers",
        "google-research-datasets/nq_open",
        split="train",
    )
    raw["hotpot"] = _load(
        "hotpot",
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train",
    )
    raw["trivia"] = _load(
        "trivia",
        "mandarjoshi/trivia_qa",
        "rc.wikipedia",
        split="train",
        trust_remote_code=True,
    )

    log.info("All datasets loaded.")
    return raw


# ─── 7. SCHEMA INSPECTION & VALIDATION ──────────────────────

_Q_HINTS: Set[str] = {"question", "query"}
_A_HINTS: Set[str] = {"answer", "answers", "target", "short_answers"}

_C_HINTS: Set[str] = {
    "context", "passage", "passages", "document", "documents",
    "text", "content", "search_results",
}
_T_HINTS: Set[str] = {"title", "titles", "subject"}

_FIELD_RULES: Dict[str, Dict[str, List[str]]] = {
    "nq_passages": {"required": [],                          "optional": ["query", "pos", "neg"]},
    "nq_answers" : {"required": ["question", "answers"],     "optional": []},
    # BUG 1 FIX: hotpot required field is "context", not "supporting_facts".
    "hotpot"     : {"required": ["question", "answers", "context"], "optional": ["title"]},
    "trivia"     : {"required": ["question", "answers"],     "optional": ["context"]},
}


def _detect_field(cols: List[str], hints: Set[str]) -> Optional[str]:
    """Return the first column whose name contains any hint keyword (case-insensitive)."""
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
        cols = list(ds.features.keys())
        schema: Dict[str, Any] = {
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
                raise SchemaError(
                    "[hotpot] Expected a 'context' column but none found. "
                    f"Available columns: {cols}"
                )

        if name == "nq_passages":
            schema["question"] = _detect_field(cols, _Q_HINTS)   # 'query'
            schema["context"]  = None                             # no single ctx col

        schemas[name] = schema

        print(f"\n[{name}]")
        print(f"  All columns  : {cols}")
        print(f"  → question   : {schema['question']}")
        print(f"  → answers    : {schema['answers']}")
        print(f"  → context    : {schema['context']}")
        print(f"  → title      : {schema['title']}")
        print("  Sample row   :")
        sample = ds[0]
        for k, v in sample.items():
            print(f"    {k:30s}: {str(v)[:120].replace(chr(10), ' ')}")

    print("\n" + "=" * 64 + "\n")

    # Validate required fields and collect all errors before raising
    errors: List[str] = []
    for ds_name, rules in _FIELD_RULES.items():
        schema = schemas[ds_name]
        for field in rules["required"]:
            if schema.get(field) is None:
                errors.append(
                    f"[{ds_name}] Required field '{field}' not found. "
                    f"Available columns: {schema['columns']}"
                )
    if errors:
        msg = "Schema validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        raise SchemaError(msg)

    log.info("Schema validation passed for all datasets.")
    return schemas


# ─── 8. HELPERS ─────────────────────────────────────────────

def _make_doc_id(dataset: str, row_idx: int, text: str) -> str:
    """Stable doc_id: dataset + zero-padded row index + 8-char MD5 prefix."""
    digest = hashlib.md5(text[:64].encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{dataset}_{row_idx:07d}_{digest}"


def _flatten_answers(raw_ans: Any) -> List[str]:
    """
    Recursively normalise any answer representation into List[str].

    Handles:
      • str
      • List[str | dict | …]
      • TriviaQA answer dict  {"value", "aliases", "normalized_value", …}
      • HotpotQA single-string answer
      • nq_open list-of-strings answer
    """
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
    row     : Dict[str, Any],
    schema  : Dict[str, Any],
    ds_name : str,
) -> List[Tuple[str, str]]:
   
    # ── BUG 2 FIX: nq_passages ───────────────────────────────
    if ds_name == "nq_passages":
        text = str(row.get("answer", "")).strip()

        if text:
           return [("", text)]

        return []
    # ── BUG 3 FIX: trivia ────────────────────────────────────
    if ds_name == "trivia":
        out = []

        # Sub-source 1: search_results
        sr = row.get("search_results")
        if isinstance(sr, dict):
            descriptions = sr.get("description") or []
            sr_titles    = sr.get("title") or []
            for i, text in enumerate(descriptions):
                text = str(text).strip() if text else ""
                if not text:
                    continue
                title = sr_titles[i] if i < len(sr_titles) else ""
                out.append((str(title), text))

        # Sub-source 2: entity_pages (Wikipedia passages)
        ep = row.get("entity_pages")
        if isinstance(ep, dict):
            wiki_contexts = ep.get("wiki_context") or []
            ep_titles     = ep.get("title") or []
            for i, text in enumerate(wiki_contexts):
                text = str(text).strip() if text else ""
                if not text:
                    continue
                title = ep_titles[i] if i < len(ep_titles) else ""
                out.append((str(title), text))

        # Fallback: use the detected context column if neither above found content
        if not out:
            ctx_col = schema.get("context")
            if ctx_col:
                raw_ctx = row.get(ctx_col)
                if isinstance(raw_ctx, dict):
                    # Generic dict fallback: extract any list-of-str values
                    title_key = next(
                        (k for k in raw_ctx if "title" in k.lower()), None
                    )
                    titles = raw_ctx.get(title_key, []) if title_key else []
                    for k, v in raw_ctx.items():
                        if k == title_key or not isinstance(v, list):
                            continue
                        for i, item in enumerate(v):
                            text = str(item).strip() if item else ""
                            if not text:
                                continue
                            title = titles[i] if i < len(titles) else ""
                            out.append((str(title), text))
                elif isinstance(raw_ctx, str) and raw_ctx.strip():
                    out.append(("", raw_ctx.strip()))

        return out

    if ds_name == "hotpot":
        ctx_col = schema.get("context")   # always "context" after BUG 1 fix
        raw_ctx = row.get(ctx_col) if ctx_col else None
        if not isinstance(raw_ctx, dict):
            return []

        title_list = raw_ctx.get("title") or []
        sent_lists = raw_ctx.get("sentences") or []
        out = []
        for i, sents in enumerate(sent_lists):
            t = title_list[i] if i < len(title_list) else ""
            if isinstance(sents, list):
                
                text = " ".join(s for s in sents if isinstance(s, str)).strip()
            else:
                text = str(sents).strip()
            if text:
                out.append((str(t), text))
        return out

    # ── Generic fallback for any other dataset ───────────────
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


# ─── 9. PERIODIC SAVE HELPERS ───────────────────────────────

def _save_pkl(obj: Any, path: Path) -> None:
    """Atomic pickle write: write to .tmp then rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    log.info("  Saved  %s  (%.1f MB)", path.name, path.stat().st_size / 1e6)


def _checkpoint_list(
    data  : List[Any],
    path  : Path,
    label : str,
    every : int,
    force : bool = False,
) -> None:
    """
    Save `data` to `path` every `every` items (or when force=True).
    Guards against every==0 to prevent ZeroDivisionError.
    Uses atomic write to protect against mid-write corruption.
    """
    if every <= 0:
        return
    if force or (len(data) > 0 and len(data) % every == 0):
        _save_pkl(data, path)
        log.info("  [checkpoint] %s: %d items → %s", label, len(data), path)


# ─── 10. DOCUMENT CREATION ──────────────────────────────────

def create_documents(
    raw       : Dict[str, Any],
    schemas   : Dict[str, Dict[str, Any]],
    save_dir  : Path,
    ckpt_every: int,
) -> List[Dict[str, Any]]:
    
    out_path  = save_dir / "ckpt" / "documents_ckpt.pkl"
    documents : List[Dict[str, Any]] = []
    seen_keys : Set[str] = set()

    def _add(ds_name: str, row_idx: int, title: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        dedup_key = text[:200]
        if dedup_key in seen_keys:
            return
        seen_keys.add(dedup_key)
        documents.append({
            "doc_id"  : _make_doc_id(ds_name, row_idx, text),
            "dataset" : ds_name,
            "title"   : (title or "").strip(),
            "text"    : text,
            "row_idx" : row_idx,
        })

    for ds_name in ("nq_passages", "hotpot", "trivia"):
        log.info("Extracting documents from %s …", ds_name)
        ds     = raw[ds_name]
        schema = schemas[ds_name]
        prev_len = len(documents)

        for row_idx, row in enumerate(tqdm(ds, desc=f"{ds_name} docs")):
            for title, text in _extract_passages(row, schema, ds_name):
                _add(ds_name, row_idx, title, text)

            added = len(documents) - prev_len
            if ckpt_every > 0 and added > 0 and len(documents) % ckpt_every == 0:
                _save_pkl(documents, out_path)
                log.info(
                    "  [checkpoint] documents: %d items → %s",
                    len(documents), out_path,
                )

    _checkpoint_list(documents, out_path, "documents", ckpt_every, force=True)
    log.info("Total documents (deduplicated): %d", len(documents))
    return documents


# ─── 11. QA PAIR CREATION ───────────────────────────────────

def create_qa_pairs(
    raw       : Dict[str, Any],
    schemas   : Dict[str, Dict[str, Any]],
    documents : List[Dict[str, Any]],
    save_dir  : Path,
    ckpt_every: int,
) -> List[Dict[str, Any]]:
   
    out_path = save_dir / "ckpt" / "qa_pairs_ckpt.pkl"
    qa_pairs : List[Dict[str, Any]] = []

    # ── (dataset, row_idx) → first doc_id seen for that row ──
    row_to_doc: Dict[Tuple[str, int], str] = {}
    for doc in documents:
        key = (doc["dataset"], doc["row_idx"])
        if key not in row_to_doc:
            row_to_doc[key] = doc["doc_id"]

    # ── question text → doc_id from nq_passages ──────────────
    # BUG 2 FIX: use schema["question"] which maps to 'query' column.
    nq_q_col = schemas["nq_passages"].get("question")   # 'query'
    nq_question_to_doc: Dict[str, str] = {}
    if nq_q_col:
        log.info("Building nq question→doc_id map …")
        for row_idx, row in enumerate(tqdm(raw["nq_passages"], desc="nq q→doc")):
            q_text = str(row.get(nq_q_col) or "").strip().lower()
            if q_text and q_text not in nq_question_to_doc:
                doc_id = row_to_doc.get(("nq_passages", row_idx), "")
                if doc_id:
                    nq_question_to_doc[q_text] = doc_id

    def _make_qid(ds_name: str, idx: int) -> str:
        return f"{ds_name}_q{idx:07d}"

    # ── NQ answers ────────────────────────────────────────────
    log.info("Creating QA pairs from nq_answers …")
    q_col = schemas["nq_answers"]["question"]
    a_col = schemas["nq_answers"]["answers"]
    for idx, row in enumerate(tqdm(raw["nq_answers"], desc="nq QA")):
        question = str(row.get(q_col) or "").strip()
        answers  = _flatten_answers(row.get(a_col))
        if not question or not answers:
            continue
        doc_id = ""
        qa_pairs.append({
            "question_id": _make_qid("nq", idx),
            "question"   : question,
            "answers"    : answers,
            "dataset"    : "nq",
            "doc_id"     : doc_id,
        })
        _checkpoint_list(qa_pairs, out_path, "qa_pairs", ckpt_every)

    # ── HotpotQA ─────────────────────────────────────────────
    log.info("Creating QA pairs from hotpot …")
    q_col = schemas["hotpot"]["question"]
    a_col = schemas["hotpot"]["answers"]
    for idx, row in enumerate(tqdm(raw["hotpot"], desc="hotpot QA")):
        question = str(row.get(q_col) or "").strip()
        answers  = _flatten_answers(row.get(a_col))
        if not question or not answers:
            continue
        qa_pairs.append({
            "question_id": _make_qid("hotpot", idx),
            "question"   : question,
            "answers"    : answers,
            "dataset"    : "hotpot",
            "doc_id"     : row_to_doc.get(("hotpot", idx), ""),
        })
        _checkpoint_list(qa_pairs, out_path, "qa_pairs", ckpt_every)

    # ── TriviaQA ─────────────────────────────────────────────
    log.info("Creating QA pairs from trivia …")
    q_col = schemas["trivia"]["question"]
    a_col = schemas["trivia"]["answers"]
    for idx, row in enumerate(tqdm(raw["trivia"], desc="trivia QA")):
        question = str(row.get(q_col) or "").strip()
        answers  = _flatten_answers(row.get(a_col))
        if not question or not answers:
            continue
        qa_pairs.append({
            "question_id": _make_qid("trivia", idx),
            "question"   : question,
            "answers"    : answers,
            "dataset"    : "trivia",
            "doc_id"     : row_to_doc.get(("trivia", idx), ""),
        })
        _checkpoint_list(qa_pairs, out_path, "qa_pairs", ckpt_every)

    _checkpoint_list(qa_pairs, out_path, "qa_pairs", ckpt_every, force=True)
    log.info("Total QA pairs: %d", len(qa_pairs))
    return qa_pairs


# ─── 12. CHUNKING ───────────────────────────────────────────

def chunk_documents(
    documents   : List[Dict[str, Any]],
    chunk_size  : int,
    chunk_overlap: int,
    save_dir    : Path,
    ckpt_every  : int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
   
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    c_path = save_dir / "ckpt" / "chunks_ckpt.pkl"
    m_path = save_dir / "ckpt" / "metadata_ckpt.pkl"

    chunks  : List[str]  = []
    metadata: List[Dict] = []
    step = chunk_size - chunk_overlap

    for doc in tqdm(documents, desc="Chunking"):
        text    = doc["text"]
        doc_id  = doc["doc_id"]
        dataset = doc["dataset"]
        title   = doc["title"]

        if not text:
            continue

        start = 0
        while start < len(text):
            end   = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                cid = len(chunks)
                chunks.append(chunk)
                metadata.append({
                    "chunk_id"  : cid,
                    "doc_id"    : doc_id,
                    "dataset"   : dataset,
                    "title"     : title,
                    "char_start": start,
                })
                if ckpt_every > 0 and len(chunks) % ckpt_every == 0:
                    _save_pkl(chunks,   c_path)
                    _save_pkl(metadata, m_path)
                    log.info("  [checkpoint] chunks/metadata: %d", len(chunks))
            start += step

    # Final checkpoint
    _save_pkl(chunks,   c_path)
    _save_pkl(metadata, m_path)

    assert len(chunks) == len(metadata), (
        f"BUG: chunks ({len(chunks)}) ≠ metadata ({len(metadata)})"
    )
    log.info("Total chunks: %d", len(chunks))
    return chunks, metadata


# ─── 13. BM25 ───────────────────────────────────────────────

def build_bm25(chunks: List[str]) -> BM25Okapi:
    """Build a BM25Okapi index from chunk texts (whitespace tokenisation)."""
    log.info("Tokenising %d chunks for BM25 …", len(chunks))
    tokenised = [c.lower().split() for c in tqdm(chunks, desc="BM25 tokenise")]
    log.info("Fitting BM25 …")
    bm25 = BM25Okapi(tokenised)
    log.info("BM25 ready.")
    return bm25


# ─── 14. FAISS ──────────────────────────────────────────────

def _estimate_faiss_memory(n_chunks: int, dim: int) -> None:
    """
    Print dataset size and expected peak RAM before embedding.
    IndexFlatIP stores float32 vectors: 4 bytes × dim × n_vectors.
    """
    index_mb = (n_chunks * dim * 4) / 1e6
    try:
        import psutil
        avail_mb  = psutil.virtual_memory().available / 1e6
        total_mb  = psutil.virtual_memory().total / 1e6
        avail_str = f"{avail_mb:,.0f} MB available / {total_mb:,.0f} MB total"
    except ImportError:
        avail_str = "psutil not installed — install it for RAM availability info"

    print("\n" + "─" * 56)
    print("  FAISS MEMORY ESTIMATE")
    print("─" * 56)
    print(f"  Chunks to embed   : {n_chunks:>10,}")
    print(f"  Embedding dim     : {dim:>10,}")
    print(f"  Index RAM (flat)  : {index_mb:>10,.1f} MB")
    print(f"  System RAM        : {avail_str}")
    print("─" * 56 + "\n")


def build_faiss(
    chunks          : List[str],
    model_name      : str,
    batch_size      : int,
    checkpoint_every: int,
    save_dir        : Path,
) -> Tuple[faiss.IndexFlatIP, SentenceTransformer]:
    
    ckpt_index_path = save_dir / "ckpt" / "faiss_partial.index"
    ckpt_meta_path  = save_dir / "ckpt" / "faiss_ckpt_meta.json"

    model = SentenceTransformer(model_name)
    dim   = model.get_sentence_embedding_dimension()
    log.info("Embedding model: %s  (dim=%d)", model_name, dim)

    n_chunks = len(chunks)
    _estimate_faiss_memory(n_chunks, dim)

    # ── Resume from a valid partial checkpoint ────────────────
    start_idx    = 0
    last_ckpt_at = 0

    if ckpt_index_path.exists() and ckpt_meta_path.exists():
        try:
            with open(ckpt_meta_path) as f:
                ckpt_meta = json.load(f)
            saved_n = ckpt_meta.get("n_added", 0)
            if 0 < saved_n <= n_chunks:
                index = faiss.read_index(str(ckpt_index_path))
                if index.ntotal == saved_n:
                    start_idx    = saved_n
                    last_ckpt_at = saved_n
                    log.info(
                        "Resuming FAISS from checkpoint: %d vectors already added.",
                        start_idx,
                    )
                else:
                    log.warning(
                        "Checkpoint meta says %d but index has %d — "
                        "deleting stale checkpoint and restarting.",
                        saved_n, index.ntotal,
                    )
                    for p in (ckpt_index_path, ckpt_meta_path):
                        try:
                            p.unlink()
                        except FileNotFoundError:
                            pass
                    index = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexFlatIP(dim)
        except Exception as exc:
            log.warning("Cannot load FAISS checkpoint (%s) — starting fresh.", exc)
            for p in (ckpt_index_path, ckpt_meta_path):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatIP(dim)

    # ── Incremental encode + add ──────────────────────────────
    log.info("Encoding chunks %d → %d …", start_idx, n_chunks)
    pbar = tqdm(total=n_chunks - start_idx, desc="FAISS encode")

    for batch_start in range(start_idx, n_chunks, batch_size):
        batch_end = min(batch_start + batch_size, n_chunks)
        batch     = chunks[batch_start:batch_end]

        embs = model.encode(
            batch,
            batch_size           = batch_size,
            show_progress_bar    = False,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        ).astype(np.float32)

        index.add(embs)
        pbar.update(len(batch))

        is_last_batch = (batch_end == n_chunks)
        vectors_since_ckpt = index.ntotal - last_ckpt_at
        if vectors_since_ckpt >= checkpoint_every or is_last_batch:
            faiss.write_index(index, str(ckpt_index_path))
            with open(ckpt_meta_path, "w") as f:
                json.dump({"n_added": index.ntotal}, f)
            last_ckpt_at = index.ntotal
            log.info(
                "  [faiss ckpt] %d / %d vectors saved.", index.ntotal, n_chunks
            )

        del embs, batch
        gc.collect()

    pbar.close()
    log.info("FAISS index complete: %d vectors, dim=%d", index.ntotal, dim)

    for p in (ckpt_index_path, ckpt_meta_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    return index, model


# ─── 15. SAVE & LOAD ARTIFACTS ──────────────────────────────

def save_artifacts(
    save_dir    : Path,
    chunks      : List[str],
    metadata    : List[Dict],
    qa_pairs    : List[Dict],
    bm25        : BM25Okapi,
    faiss_index : faiss.IndexFlatIP,
) -> Dict[str, str]:
    """
    Write final artifacts to save_dir.  Outputs exactly:
        chunks.pkl · metadata.pkl · qa_pairs.pkl · bm25.pkl · faiss.index
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    for obj, fname in [
        (chunks,   "chunks.pkl"),
        (metadata, "metadata.pkl"),
        (qa_pairs, "qa_pairs.pkl"),
        (bm25,     "bm25.pkl"),
    ]:
        p = save_dir / fname
        _save_pkl(obj, p)
        paths[fname.split(".")[0]] = str(p)

    faiss_path = save_dir / "faiss.index"
    faiss.write_index(faiss_index, str(faiss_path))
    log.info("  Saved  faiss.index  (%.1f MB)", faiss_path.stat().st_size / 1e6)
    paths["faiss"] = str(faiss_path)

    return paths


def load_artifacts(save_dir: Path) -> Dict[str, Any]:
    """
    Reload all five artifacts from disk.
    Returns dict with keys: chunks, metadata, qa_pairs, bm25, faiss.
    """
    def _pkl(name: str) -> Any:
        with open(save_dir / name, "rb") as f:
            return pickle.load(f)

    return {
        "chunks"  : _pkl("chunks.pkl"),
        "metadata": _pkl("metadata.pkl"),
        "qa_pairs": _pkl("qa_pairs.pkl"),
        "bm25"    : _pkl("bm25.pkl"),
        "faiss"   : faiss.read_index(str(save_dir / "faiss.index")),
    }


# ─── 16. STATISTICS ─────────────────────────────────────────

def print_statistics(
    n_documents : int,
    qa_pairs    : List[Dict],
    chunks      : List[str],
    metadata    : List[Dict],
    faiss_index : faiss.IndexFlatIP,
    paths       : Dict[str, str],
) -> None:
    avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

    chunk_by_ds: Dict[str, int] = {}
    for m in metadata:
        chunk_by_ds[m["dataset"]] = chunk_by_ds.get(m["dataset"], 0) + 1

    qa_by_ds: Dict[str, int] = {}
    for q in qa_pairs:
        qa_by_ds[q["dataset"]] = qa_by_ds.get(q["dataset"], 0) + 1

    print("\n" + "=" * 60)
    print("  FINAL STATISTICS")
    print("=" * 60)
    print(f"  Documents         : {n_documents:>10,}")
    print(f"  QA pairs          : {len(qa_pairs):>10,}")
    print(f"  Chunks            : {len(chunks):>10,}")
    print(f"  Avg chunk length  : {avg_len:>10.1f}  chars")
    print(f"  FAISS index size  : {faiss_index.ntotal:>10,}  vectors")
    print()
    print("  Chunks by dataset:")
    for ds, cnt in sorted(chunk_by_ds.items()):
        print(f"    {ds:20s}: {cnt:>10,}")
    print()
    print("  QA pairs by dataset:")
    for ds, cnt in sorted(qa_by_ds.items()):
        print(f"    {ds:20s}: {cnt:>10,}")
    print()
    print("  Saved artifacts:")
    for k, v in paths.items():
        size_mb = Path(v).stat().st_size / 1e6 if Path(v).exists() else 0
        print(f"    {k:12s}: {v}  ({size_mb:.1f} MB)")
    print("=" * 60 + "\n")


# ─── 17. SANITY CHECK ───────────────────────────────────────

def sanity_check(
    chunks      : List[str],
    metadata    : List[Dict],
    bm25        : BM25Okapi,
    faiss_index : faiss.IndexFlatIP,
    model       : SentenceTransformer,
    top_k       : int = 3,
) -> None:
    """
    Run a single query through both BM25 and FAISS to verify the pipeline
    is wired correctly end-to-end.

    Accepts the already-loaded SentenceTransformer model rather than
    loading a second copy, preventing an unnecessary spike in RAM usage.
    The caller is responsible for deleting the model after this call.
    """
    query = "Who invented the telephone?"
    print("\n" + "─" * 56)
    print(f"  SANITY CHECK  —  query: '{query}'")
    print("─" * 56)

    # BM25
    tokens   = query.lower().split()
    scores   = bm25.get_scores(tokens)
    top_bm25 = scores.argsort()[-top_k:][::-1]
    print("\nBM25 top results:")
    for rank, i in enumerate(top_bm25, 1):
        m = metadata[i]
        print(f"  {rank}. [{m['dataset']}] {m['title']!r}")
        print(f"     {chunks[i][:200].strip()} …")

    # FAISS
    q_emb = model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)
    scores_f, idxs_f = faiss_index.search(q_emb, top_k)
    print("\nFAISS top results:")
    for rank, (score, i) in enumerate(zip(scores_f[0], idxs_f[0]), 1):
        m = metadata[i]
        print(f"  {rank}. [{m['dataset']}] {m['title']!r}  (score={score:.4f})")
        print(f"     {chunks[i][:200].strip()} …")

    print("─" * 56 + "\n")


# ─── 18. MAIN ORCHESTRATOR ──────────────────────────────────

def main() -> None:
    t0   = time.time()
    cfg  = CONFIG
    ckpt = cfg["checkpoint_every"]

    # ── Mount Drive ───────────────────────────────────────────
    save_dir = mount_drive(cfg["drive_base"])

    # ── Load ──────────────────────────────────────────────────
    raw = load_datasets(cfg["max_rows"])

    # ── Inspect + validate ────────────────────────────────────
    schemas = inspect_schema(raw)

    # ── Documents ─────────────────────────────────────────────
    documents = create_documents(raw, schemas, save_dir, ckpt)

    # ── QA pairs ─────────────────────────────────────────────
    qa_pairs = create_qa_pairs(raw, schemas, documents, save_dir, ckpt)

    # Free raw datasets
    del raw
    gc.collect()

    # ── Chunks + metadata ─────────────────────────────────────
    chunks, metadata = chunk_documents(
        documents,
        chunk_size    = cfg["chunk_size"],
        chunk_overlap = cfg["chunk_overlap"],
        save_dir      = save_dir,
        ckpt_every    = ckpt,
    )

    n_documents = len(documents)
    del documents
    gc.collect()

    # ── BM25 ─────────────────────────────────────────────────
    bm25 = build_bm25(chunks)
    _save_pkl(bm25, save_dir / "ckpt" / "bm25_ckpt.pkl")

    # ── FAISS ─────────────────────────────────────────────────
    faiss_index, embed_model = build_faiss(
        chunks,
        model_name       = cfg["embed_model"],
        batch_size       = cfg["embed_batch_size"],
        checkpoint_every = ckpt,
        save_dir         = save_dir,
    )

    # ── Save final artifacts ──────────────────────────────────
    paths = save_artifacts(
        save_dir, chunks, metadata, qa_pairs, bm25, faiss_index
    )

    # ── Statistics ────────────────────────────────────────────
    print_statistics(
        n_documents = n_documents,
        qa_pairs    = qa_pairs,
        chunks      = chunks,
        metadata    = metadata,
        faiss_index = faiss_index,
        paths       = paths,
    )

    # ── Sanity check ─────────────────────────────────────────
    sanity_check(chunks, metadata, bm25, faiss_index, embed_model)
    del embed_model
    gc.collect()

    elapsed = time.time() - t0
    log.info("Pipeline complete in %.1f minutes.", elapsed / 60)


# ─── ENTRY POINT ────────────────────────────────────────────
if __name__ == "__main__":
    main()