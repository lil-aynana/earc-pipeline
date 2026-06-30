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

        # Layer 11 — Prompt Construction
        prompt_bundle = prompt_builder.build_prompt(
            query, selected_sentences, query_type
        )

        # Layer 12 — Answer Generation
        gen = self.generator.generate(prompt_bundle, query, query_type)

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
