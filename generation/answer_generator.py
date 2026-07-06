"""
generation/answer_generator.py
==============================

Layer 12 of the EARC pipeline: Answer Generation.

Turns the prompt built by Layer 11 into a natural-language answer. Several
backends are supported so the pipeline runs anywhere:

    "extractive"   - deterministic, dependency-free. Synthesises an answer
                     directly from the top evidence sentences with inline
                     [n] citations. ALWAYS available (default), works on
                     Colab and fully offline. No model download.
    "transformers" - local HuggingFace seq2seq model (default flan-t5-base).
    "openai"       - OpenAI Chat Completions API (needs OPENAI_API_KEY).
    "ollama"       - local Ollama server (uses CONFIG ollama_url / llm_model).

If a non-extractive backend fails to load or call (missing dependency, no
API key, server down), the generator transparently falls back to the
extractive backend so the pipeline never hard-crashes mid-query.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from config import CONFIG

# Surface cues that indicate a sentence itself expresses a negation/exclusion.
# Used by the extractive backend to answer negated questions with evidence
# that actually addresses the exclusion rather than the affirmative set.
_NEGATION_CUES = (
    " not ", " no ", " non-", " never ", " except", " excluding", " without ",
    " neither ", " nor ", " outside ", "isn't", "aren't", "wasn't", "weren't",
    "non-member", "not a member", "not members",
)


class AnswerGenerator:
    """Layer 12 — pluggable answer generation."""

    def __init__(self, backend: Optional[str] = None):
        gen_cfg = CONFIG.get("generation", {})
        self.backend = (backend or gen_cfg.get("backend", "extractive")).lower()
        self._hf_pipeline = None  # lazily initialised transformers pipeline

    # ── public API ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt_bundle: Dict[str, Any],
        query: str,
        query_type: str,
        has_negation: bool = False,
    ) -> Dict[str, Any]:
        """Generate an answer from a Layer 11 prompt bundle.

        Returns a dict with ``answer`` (str) and ``backend`` (the backend
        actually used, which may differ from the requested one if a
        fallback occurred).
        """
        evidence = prompt_bundle.get("evidence", [])
        if not evidence:
            return {"answer": "I don't have enough information to answer.", "backend": "none"}

        backend = self.backend
        try:
            if backend == "extractive":
                answer = self._extractive(evidence, query, query_type, has_negation)
            elif backend == "transformers":
                answer = self._transformers(prompt_bundle["prompt"])
            elif backend == "openai":
                answer = self._openai(prompt_bundle["prompt"])
            elif backend == "ollama":
                answer = self._ollama(prompt_bundle["prompt"])
            else:
                answer = self._extractive(evidence, query, query_type, has_negation)
                backend = "extractive"
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
            answer = self._extractive(evidence, query, query_type, has_negation)
            backend = f"extractive (fallback from {self.backend}: {type(exc).__name__})"

        return {"answer": answer.strip(), "backend": backend}

    # ── backends ──────────────────────────────────────────────────────────

    @staticmethod
    def _extractive(
        evidence: List[Dict[str, Any]],
        query: str,
        query_type: str,
        has_negation: bool = False,
    ) -> str:
        """Deterministic, LLM-free answer built from the evidence.

        Picks the most relevant evidence sentences (already ordered by
        Layer 11) and stitches them into a concise, cited answer. The number
        of sentences used scales with query type.

        For negated/exclusionary questions, the extractive backend cannot
        compute a set difference, so it (1) prefers evidence sentences that
        themselves express a negation/exclusion, and (2) if none do — i.e. the
        evidence only describes the affirmative set — returns an explicit
        caveat instead of confidently asserting the opposite.
        """
        qt = (query_type or "").strip().lower()

        if has_negation:
            return AnswerGenerator._extractive_negated(evidence)

        if qt == "factoid":
            n = 1
        elif qt == "multi_hop":
            n = 3
        else:
            n = 2

        return AnswerGenerator._stitch(evidence[:n])

    @staticmethod
    def _stitch(chosen: List[Dict[str, Any]]) -> str:
        """Join evidence sentences into a cited answer string."""
        parts: List[str] = []
        for i, sent in enumerate(chosen, 1):
            text = str(sent.get("text", "")).strip()
            if not text:
                continue
            if text[-1] not in ".!?":
                text += "."
            parts.append(f"{text} [{i}]")

        if not parts:
            return "I don't have enough information to answer."
        return " ".join(parts)

    @staticmethod
    def _extractive_negated(evidence: List[Dict[str, Any]]) -> str:
        """Answer a negated/exclusionary question from extractive evidence.

        Prefers sentences that explicitly express exclusion. If the evidence
        contains no such sentence, the exclusion cannot be derived from it, so
        a clear caveat is returned rather than a misleading affirmative.
        """
        negation_hits = [
            sent for sent in evidence
            if any(cue in f" {str(sent.get('text', '')).lower()} " for cue in _NEGATION_CUES)
        ]
        if negation_hits:
            return AnswerGenerator._stitch(negation_hits[:2])

        # Evidence only describes the affirmative/included set.
        lead = str(evidence[0].get("text", "")).strip() if evidence else ""
        if lead and lead[-1] not in ".!?":
            lead += "."
        return (
            "The retrieved evidence describes the included/affirmative set and "
            "does not directly state what is excluded, so the exclusion cannot "
            "be reliably enumerated from it. Most relevant evidence: "
            f"{lead} [1]"
        )


    def _transformers(self, prompt: str) -> str:
        """Local HuggingFace seq2seq generation (e.g. flan-t5)."""
        if self._hf_pipeline is None:
            from transformers import pipeline  # imported lazily

            model = CONFIG.get("generation", {}).get("hf_model", "google/flan-t5-base")
            self._hf_pipeline = pipeline("text2text-generation", model=model)

        max_new = int(CONFIG.get("generation", {}).get("hf_max_new_tokens", 256))
        out = self._hf_pipeline(prompt, max_new_tokens=max_new, do_sample=False)
        return out[0]["generated_text"]

    def _openai(self, prompt: str) -> str:
        """OpenAI Chat Completions backend."""
        from openai import OpenAI  # imported lazily

        client = OpenAI()
        model = CONFIG.get("generation", {}).get("openai_model", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            temperature=CONFIG.get("temperature", 0),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def _ollama(self, prompt: str) -> str:
        """Local Ollama server backend."""
        import requests  # imported lazily

        url = CONFIG.get("ollama_url", "http://localhost:11434/api/generate")
        payload = {
            "model": CONFIG.get("llm_model", "llama3"),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": CONFIG.get("temperature", 0)},
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
