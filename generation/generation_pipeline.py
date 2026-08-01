"""Module 4 orchestration: Layers 11-13 (Generation)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from generation import answer_verifier, prompt_builder
from generation.answer_generator import AnswerGenerator


class GenerationPipeline:
    """Runs prompt construction, answer generation, and answer verification.

    Consumes the Selection module's output (``selected_sentences`` as a list
    of dicts) plus Module 1's ``query_info`` and produces a grounded,
    citation-tagged answer.
    """

    def __init__(self, backend: Optional[str] = None):
        self.generator = AnswerGenerator(backend=backend)

    def generate(
        self,
        query_info: Dict[str, Any],
        selected_sentences: List[Dict[str, Any]],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run Layers 11-13 and return the answer plus supporting metadata.

        If ``verbose`` is True, a formatted generation report (top evidence,
        answer, backend, verification) is printed to the terminal.
        """
        query = query_info.get("query", "")
        query_type = query_info.get("query_type", "descriptive")
        has_negation = bool(query_info.get("has_negation", False))

        # Layer 11 — Prompt Construction
        prompt_bundle = prompt_builder.build_prompt(
            query, selected_sentences, query_type, has_negation=has_negation
        )

        # Layer 12 — Answer Generation
        gen = self.generator.generate(
            prompt_bundle, query, query_type, has_negation=has_negation
        )

        # Layer 13 — Answer Verification & Citation Grounding
        verification = answer_verifier.verify(
            gen["answer"], prompt_bundle["citations"], prompt_bundle["context"]
        )

        result = {
            "answer": gen["answer"],
            "backend": gen["backend"],
            "prompt": prompt_bundle["prompt"],
            "context": prompt_bundle["context"],
            "citations": prompt_bundle["citations"],
            "verification": verification,
        }

        if verbose:
            print_report(query, selected_sentences, result)

        return result

    def generate_baseline(
        self,
        query_info: Dict[str, Any],
        retrieved_sentences: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Standard-RAG baseline: answer from ALL retrieved sentences.

        Feeds the full retrieved context to the same generation backend used by
        the EARC path, bypassing the Selection module (Layers 7-10) and the
        per-query-type context cap. This is the conventional retrieve-then-stuff
        pipeline the EARC method is compared against, so the evaluator can
        quantify token reduction and any answer-quality trade-off.

        Returns a dict with ``answer`` and ``backend`` (mirrors
        :meth:`AnswerGenerator.generate`). No verification is run — the baseline
        exists only for answer-quality and token-cost comparison.
        """
        query = query_info.get("query", "")
        query_type = query_info.get("query_type", "descriptive")
        has_negation = bool(query_info.get("has_negation", False))

        prompt_bundle = prompt_builder.build_prompt(
            query,
            retrieved_sentences,
            query_type,
            has_negation=has_negation,
            max_context=len(retrieved_sentences),
        )
        gen = self.generator.generate(
            prompt_bundle, query, query_type, has_negation=has_negation
        )
        return {"answer": gen["answer"], "backend": gen["backend"]}


def print_report(
    query: str,
    selected_sentences: List[Dict[str, Any]],
    generation_result: Dict[str, Any],
    top_k: int = 5,
) -> None:
    """Pretty-print a generation report to the terminal.

    The evidence table is shown in the exact order the Layer 11 prompt was
    built (anchor -> bridges -> supporting evidence), NOT re-sorted by score.
    This means the ``Rank`` column matches the ``[n]`` citation markers used in
    the answer. The order/markers come from ``generation_result["citations"]``
    (populated by ``build_context``); if those are unavailable, it falls back
    to score-descending order over ``selected_sentences``.

    Also shows the generated answer, the backend used, and the Layer 13
    verification summary. This is a display-only helper — it performs no
    scoring, mutation, or I/O beyond printing.
    """
    citations = generation_result.get("citations")
    if citations:
        # Prompt-built order; ``marker`` is the citation number in the answer.
        rows = [
            (
                c.get("marker"),
                float(c.get("score", 0.0) or 0.0),
                bool(c.get("is_bridge", False)),
                str(c.get("text", "")).strip(),
            )
            for c in citations[:top_k]
        ]
    else:
        # Fallback: no citation metadata, rank by score.
        ordered = sorted(
            selected_sentences,
            key=lambda s: float(s.get("score", 0.0) or 0.0),
            reverse=True,
        )[:top_k]
        rows = [
            (
                i,
                float(s.get("score", 0.0) or 0.0),
                bool(s.get("is_bridge", False)),
                str(s.get("text", "")).strip(),
            )
            for i, s in enumerate(ordered, 1)
        ]

    print("generation module")
    print(f"Query: {query}\n")
    print(f"{'Rank':<5} {'Score':<7} {'Bridge':<7} Evidence")
    print("-" * 70)
    for marker, score, is_bridge, text in rows:
        bridge = "Yes" if is_bridge else "No"
        print(f"{marker:<5} {score:<7.4f} {bridge:<7} {text}")

    verification = generation_result.get("verification", {})
    print()
    print(f"Answer  : {generation_result.get('answer', '')}")
    print(f"Backend : {generation_result.get('backend', '')}")
    print()
    print("Verification:")
    print(f"  {'grounded':<20}: {verification.get('grounded')}")
    print(f"  {'faithfulness':<20}: {verification.get('faithfulness')}")
    print(f"  {'mean_overlap':<20}: {verification.get('mean_overlap')}")
    print(f"  {'citation_count':<20}: {verification.get('citation_count')}")
    print(f"  {'invalid_citations':<20}: {verification.get('invalid_citations')}")
