"""
evaluation/evaluator.py
=======================

End-to-end evaluation harness for the EARC pipeline.

Runs the full pipeline (Layers 1-13) over a sample of labelled QA pairs and
aggregates quality + efficiency metrics:

    * Exact Match (EM) and token-level F1 against gold answers
    * Lenient containment match (gold span found inside the answer)
    * Grounding / faithfulness (from Layer 13, averaged)
    * Context compression ratio (retrieved tokens vs. selected tokens)
    * Mean selected evidence count and mean answer latency

QA pairs use the schema produced by ``data/rag_pipeline.create_qa_pairs``:
    {"question_id", "question", "answers": list[str], "dataset", "doc_id"}

This module performs read-only evaluation. It never mutates the pipeline or
the QA data and only writes a results file when explicitly asked to.
"""

from __future__ import annotations

import pickle
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import CONFIG
from evaluation import metrics


def load_qa_pairs(
    qa_dir: Path,
    datasets: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load and concatenate QA pairs from all ``qa_*.pkl`` shards in a folder.

    Args:
        qa_dir: Directory containing the ``qa_pairs`` shard pickles.
        datasets: Optional whitelist of dataset names (e.g. ``["hotpot"]``)
            to keep. ``None`` keeps every dataset.

    Returns:
        A flat list of QA-pair dicts.
    """
    qa_dir = Path(qa_dir)
    shards = sorted(qa_dir.glob("*.pkl"))
    if not shards:
        raise FileNotFoundError(f"No QA pair pickles found in: {qa_dir}")

    pairs: List[Dict[str, Any]] = []
    for shard in shards:
        with open(shard, "rb") as fh:
            data = pickle.load(fh)
        if isinstance(data, dict):
            data = data.get("qa_pairs", [])
        pairs.extend(data)

    if datasets is not None:
        keep = set(datasets)
        pairs = [p for p in pairs if p.get("dataset") in keep]

    return pairs


def sample_qa_pairs(
    pairs: List[Dict[str, Any]],
    n: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Deterministically sample up to ``n`` QA pairs."""
    if n is None or n >= len(pairs):
        return list(pairs)
    rng = random.Random(seed)
    return rng.sample(pairs, n)


def _retrieved_token_count(result: Dict[str, Any]) -> int:
    """Total whitespace tokens across all scored (retrieved) sentences."""
    total = 0
    for s in result.get("sentences", []):
        tc = getattr(s, "token_count", None)
        if tc is None:
            tc = metrics.token_count(getattr(s, "text", ""))
        total += int(tc or 0)
    return total


def _selected_token_count(result: Dict[str, Any]) -> int:
    """Total whitespace tokens across the final selected evidence."""
    total = 0
    for s in result.get("selected_sentences", []):
        tc = s.get("token_count")
        if tc is None:
            tc = metrics.token_count(s.get("text", ""))
        total += int(tc or 0)
    return total


def evaluate_one(pipe: Any, qa: Dict[str, Any]) -> Dict[str, Any]:
    """Run the pipeline on a single QA pair and compute per-query metrics.

    Returns a per-example record. On failure, returns a record with
    ``error`` set so a single bad example never aborts the whole run.
    """
    question = qa["question"]
    gold = qa.get("answers", []) or []

    start = time.perf_counter()
    try:
        result = pipe.run(question)
    except Exception as exc:  # noqa: BLE001 - record and continue
        return {
            "question_id": qa.get("question_id"),
            "dataset": qa.get("dataset"),
            "error": f"{type(exc).__name__}: {exc}",
        }
    latency = time.perf_counter() - start

    answer = result.get("answer", "")
    verification = result.get("generation", {}).get("verification", {})

    retrieved_tokens = _retrieved_token_count(result)
    selected_tokens = _selected_token_count(result)

    return {
        "question_id": qa.get("question_id"),
        "dataset": qa.get("dataset"),
        "query_type": result.get("query_info", {}).get("query_type"),
        "question": question,
        "gold_answers": gold,
        "answer": answer,
        "exact_match": metrics.exact_match(answer, gold),
        "f1": metrics.f1_score(answer, gold),
        "contains_gold": metrics.answer_contains_gold(answer, gold),
        "grounded": bool(verification.get("grounded", False)),
        "faithfulness": float(verification.get("faithfulness", 0.0) or 0.0),
        "retrieved_tokens": retrieved_tokens,
        "selected_tokens": selected_tokens,
        "compression_ratio": metrics.compression_ratio(retrieved_tokens, selected_tokens),
        "selected_count": len(result.get("selected_sentences", [])),
        "latency_sec": latency,
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-example records into overall + per-dataset summaries."""
    ok = [r for r in records if "error" not in r]
    errored = [r for r in records if "error" in r]

    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "count": len(rows),
            "exact_match": _mean([r["exact_match"] for r in rows]),
            "f1": _mean([r["f1"] for r in rows]),
            "contains_gold": _mean([r["contains_gold"] for r in rows]),
            "grounded_rate": _mean([1.0 if r["grounded"] else 0.0 for r in rows]),
            "faithfulness": _mean([r["faithfulness"] for r in rows]),
            "compression_ratio": _mean([r["compression_ratio"] for r in rows]),
            "mean_selected": _mean([r["selected_count"] for r in rows]),
            "mean_retrieved_tokens": _mean([r["retrieved_tokens"] for r in rows]),
            "mean_selected_tokens": _mean([r["selected_tokens"] for r in rows]),
            "mean_latency_sec": _mean([r["latency_sec"] for r in rows]),
        }

    per_dataset: Dict[str, Any] = {}
    for r in ok:
        per_dataset.setdefault(r["dataset"], []).append(r)
    per_dataset = {ds: summarize(rows) for ds, rows in per_dataset.items()}

    return {
        "overall": summarize(ok),
        "per_dataset": per_dataset,
        "n_evaluated": len(ok),
        "n_errors": len(errored),
        "errors": errored[:20],
    }


def run_evaluation(
    pipe: Any,
    qa_dir: Path,
    sample_size: Optional[int] = None,
    datasets: Optional[List[str]] = None,
    seed: int = 42,
    output_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Run the full evaluation and return aggregated metrics.

    Args:
        pipe: An initialised ``EARCPipeline``.
        qa_dir: Directory containing the QA-pair shard pickles.
        sample_size: Number of QA pairs to evaluate. Defaults to
            ``CONFIG["eval_sample_size"]``. Use ``None`` only via explicit
            ``-1`` semantics in the caller if you want everything.
        datasets: Optional dataset whitelist. Defaults to
            ``CONFIG["datasets"]`` mapped to their short names.
        seed: Sampling seed for reproducibility.
        output_path: If given, pickle the full results dict to this path.
        progress_callback: Optional ``fn(done, total)`` called per example.

    Returns:
        A dict with ``summary`` (aggregate metrics) and ``records``
        (per-example results).
    """
    if sample_size is None:
        sample_size = CONFIG.get("eval_sample_size", 500)

    pairs = load_qa_pairs(qa_dir, datasets=datasets)
    sampled = sample_qa_pairs(pairs, sample_size, seed=seed)

    records: List[Dict[str, Any]] = []
    total = len(sampled)
    for i, qa in enumerate(sampled, 1):
        records.append(evaluate_one(pipe, qa))
        if progress_callback is not None:
            progress_callback(i, total)

    summary = aggregate(records)
    results = {
        "summary": summary,
        "records": records,
        "config": {
            "sample_size": sample_size,
            "requested_datasets": datasets,
            "seed": seed,
            "generation_backend": CONFIG.get("generation", {}).get("backend"),
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            pickle.dump(results, fh)

    return results
