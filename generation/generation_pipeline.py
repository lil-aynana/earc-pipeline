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
    ) -> Dict[str, Any]:
        """Run Layers 11-13 and return the answer plus supporting metadata."""
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

        return {
            "answer": gen["answer"],
            "backend": gen["backend"],
            "prompt": prompt_bundle["prompt"],
            "context": prompt_bundle["context"],
            "citations": prompt_bundle["citations"],
            "verification": verification,
        }


def print_report(
    query: str,
    selected_sentences: List[Dict[str, Any]],
    generation_result: Dict[str, Any],
    top_k: int = 5,
) -> None:
    """Pretty-print a generation report to the terminal.

    Shows the top-``top_k`` selected evidence sentences (ranked by score),
    the generated answer, the backend used, and the Layer 13 verification
    summary. This is a display-only helper — it performs no scoring, mutation,
    or I/O beyond printing.
    """
    top = sorted(
        selected_sentences,
        key=lambda s: float(s.get("score", 0.0) or 0.0),
        reverse=True,
    )[:top_k]

    print("generation module")
    print(f"Query: {query}\n")
    print(f"{'Rank':<5} {'Score':<7} {'Bridge':<7} Evidence")
    print("-" * 70)
    for rank, sent in enumerate(top, 1):
        score = float(sent.get("score", 0.0) or 0.0)
        bridge = "Yes" if sent.get("is_bridge", False) else "No"
        text = str(sent.get("text", "")).strip()
        print(f"{rank:<5} {score:<7.4f} {bridge:<7} {text}")

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
