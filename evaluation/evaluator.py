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


def _field(s: Any, name: str, default: Any = "") -> Any:
    """Read a field from a sentence that may be a dict or an attribute object.

    Retrieved/scored sentences flow through the pipeline as plain dicts, so a
    bare ``getattr`` silently returns the default and zeroes out every
    retrieved-side metric. This helper handles both shapes.
    """
    if isinstance(s, dict):
        return s.get(name, default)
    return getattr(s, name, default)


def _retrieved_token_count(result: Dict[str, Any]) -> int:
    """Total whitespace tokens across all scored (retrieved) sentences."""
    total = 0
    for s in result.get("retrieved_sentences", result.get("sentences", [])):
        tc = _field(s, "token_count", None)
        if tc is None:
            tc = metrics.token_count(_field(s, "text", ""))
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


def _retrieved_texts(result: Dict[str, Any]) -> str:
    """Concatenate all scored (retrieved) sentence texts into one string."""
    return " ".join(str(_field(s, "text", "")) for s in result.get("sentences", []))


def _selected_texts(result: Dict[str, Any]) -> str:
    """Concatenate all final selected sentence texts into one string."""
    return " ".join(str(s.get("text", "")) for s in result.get("selected_sentences", []))



def evaluate_one(pipe: Any, qa: Dict[str, Any], run_baseline: bool = True) -> Dict[str, Any]:
    """Run the pipeline on a single QA pair and compute per-query metrics.

    When ``run_baseline`` is True, a standard-RAG baseline answer is also
    generated from the full retrieved context (no selection/compression) so the
    EARC method's token reduction and answer-quality trade-off can be measured.
    The baseline requires a second LLM call per query; set it False for a
    faster EARC-only run.
    """
    question = qa.get("question")
    if question is None:
        return {
            "question_id": qa.get("question_id"),
            "dataset": qa.get("dataset"),
            "error": "KeyError: QA pair missing 'question' field",
        }
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
    generation = result.get("generation", {})
    verification = generation.get("verification", {})

    is_refusal = bool(verification.get("is_refusal", False))
    raw_faithfulness = verification.get("faithfulness", 0.0)
    faithfulness = None if raw_faithfulness is None else float(raw_faithfulness)

    retrieved_tokens = _retrieved_token_count(result)
    selected_tokens = _selected_token_count(result)

    answer_in_retrieved = metrics.answer_contains_gold(_retrieved_texts(result), gold)
    answer_in_selected = metrics.answer_contains_gold(_selected_texts(result), gold)

    # Context-token reduction of the EARC method vs. the standard-RAG baseline
    # (which feeds all retrieved tokens to the LLM). Higher pct = more saving.
    token_reduction = retrieved_tokens - selected_tokens
    token_reduction_pct = (
        100.0 * token_reduction / retrieved_tokens if retrieved_tokens > 0 else 0.0
    )

    record = {
        "question_id": qa.get("question_id"),
        "dataset": qa.get("dataset"),
        "query_type": result.get("query_info", {}).get("query_type"),
        "question": question,
        "gold_answers": gold,
        "answer": answer,
        "backend": generation.get("backend"),
        "exact_match": metrics.exact_match(answer, gold),
        "f1": metrics.f1_score(answer, gold),
        "contains_gold": metrics.answer_contains_gold(answer, gold),
        "answer_in_retrieved": answer_in_retrieved,
        "answer_in_selected": answer_in_selected,
        "grounded": bool(verification.get("grounded", False)),
        "faithfulness": faithfulness,
        "is_refusal": is_refusal,
        "retrieved_tokens": retrieved_tokens,
        "selected_tokens": selected_tokens,
        "compression_ratio": metrics.compression_ratio(retrieved_tokens, selected_tokens),
        "token_reduction": token_reduction,
        "token_reduction_pct": token_reduction_pct,
        "selected_count": len(result.get("selected_sentences", [])),
        "latency_sec": latency,
    }

    if run_baseline:
        try:
            b_start = time.perf_counter()
            b_gen = pipe.generation_pipeline.generate_baseline(
                result.get("query_info", {}),
                result.get("retrieved_sentences", result.get("sentences", [])),
            )
            b_latency = time.perf_counter() - b_start
            b_answer = b_gen.get("answer", "")
            record.update(
                {
                    "baseline_answer": b_answer,
                    "baseline_backend": b_gen.get("backend"),
                    "baseline_exact_match": metrics.exact_match(b_answer, gold),
                    "baseline_f1": metrics.f1_score(b_answer, gold),
                    "baseline_contains_gold": metrics.answer_contains_gold(b_answer, gold),
                    # Standard RAG feeds the full retrieved context to the LLM.
                    "baseline_context_tokens": retrieved_tokens,
                    "baseline_latency_sec": b_latency,
                }
            )
        except Exception as exc:  # noqa: BLE001 - baseline is best-effort
            record["baseline_error"] = f"{type(exc).__name__}: {exc}"

    return record


def _mean(values: List[float]) -> float:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-example records into overall + per-dataset summaries."""
    ok = [r for r in records if "error" not in r]
    errored = [r for r in records if "error" in r]

    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        faithfulness_values = [r.get("faithfulness") for r in rows]
        n_faithfulness_scored = sum(1 for v in faithfulness_values if v is not None)
        baseline_rows = [r for r in rows if "baseline_answer" in r]
        return {
            "count": len(rows),
            "exact_match": _mean([r["exact_match"] for r in rows]),
            "f1": _mean([r["f1"] for r in rows]),
            "contains_gold": _mean([r["contains_gold"] for r in rows]),
            "answer_in_retrieved": _mean([1.0 if r["answer_in_retrieved"] else 0.0 for r in rows]),
            "answer_in_selected": _mean([1.0 if r["answer_in_selected"] else 0.0 for r in rows]),
            "grounded_rate": _mean([1.0 if r["grounded"] else 0.0 for r in rows]),
            "faithfulness": _mean(faithfulness_values),
            "faithfulness_n": n_faithfulness_scored,
            "refusal_rate": _mean([1.0 if r.get("is_refusal") else 0.0 for r in rows]),
            "compression_ratio": _mean([r["compression_ratio"] for r in rows]),
            "token_reduction_pct": _mean([r.get("token_reduction_pct") for r in rows]),
            "mean_selected": _mean([r["selected_count"] for r in rows]),
            "mean_retrieved_tokens": _mean([r["retrieved_tokens"] for r in rows]),
            "mean_selected_tokens": _mean([r["selected_tokens"] for r in rows]),
            "mean_latency_sec": _mean([r["latency_sec"] for r in rows]),
            # Standard-RAG baseline (full retrieved context, no selection).
            "baseline_count": len(baseline_rows),
            "baseline_exact_match": _mean([r.get("baseline_exact_match") for r in baseline_rows]),
            "baseline_f1": _mean([r.get("baseline_f1") for r in baseline_rows]),
            "baseline_contains_gold": _mean([r.get("baseline_contains_gold") for r in baseline_rows]),
            "baseline_mean_context_tokens": _mean([r.get("baseline_context_tokens") for r in baseline_rows]),
            "baseline_mean_latency_sec": _mean([r.get("baseline_latency_sec") for r in baseline_rows]),
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
    run_baseline: bool = True,
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
        run_baseline: If True (default), also generate a standard-RAG baseline
            answer per query (full retrieved context, no selection) so token
            reduction and answer-quality trade-offs are reported. Costs one
            extra LLM call per query; set False for a faster EARC-only run.

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
        records.append(evaluate_one(pipe, qa, run_baseline=run_baseline))
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
            "run_baseline": run_baseline,
            "generation_backend": CONFIG.get("generation", {}).get("backend"),
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            pickle.dump(results, fh)

    return results
