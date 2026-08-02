"""
pipeline.py
────────────
EARC Pipeline — top-level entry point.
"""

import logging
import time
from pathlib import Path

from retrieval.loader import load_corpus_artifacts
from retrieval.retrieval_pipeline import RetrievalLayer
from retrieval.retrieval_config import (
    BM25_PATH,
    CHUNKS_DIR,
    EMBED_MODEL,
    FAISS_PATH,
    METADATA_DIR,
)

# Module 2
from scoring.scoring_pipeline import run as scoring_run

# Module 3
from selection.selection_pipeline import run as selection_run

# Module 4
from generation.generation_pipeline import GenerationPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EARC")


class EARCPipeline:
    """
    End-to-end EARC pipeline.
    """

    def __init__(
        self,
        faiss_path: Path = FAISS_PATH,
        bm25_path: Path = BM25_PATH,
        chunks_dir: Path = CHUNKS_DIR,
        metadata_dir: Path = METADATA_DIR,
        embed_model_name: str = EMBED_MODEL,
        backend: str | None = None,
        llm_model: str | None = None,
    ):

        (
            faiss_index,
            bm25_index,
            all_chunks,
            all_metadata,
            embed_model,
        ) = load_corpus_artifacts(
            faiss_path,
            bm25_path,
            chunks_dir,
            metadata_dir,
            embed_model_name,
        )

        # Module 1
        self.retrieval_layer = RetrievalLayer(
            faiss_index,
            bm25_index,
            all_chunks,
            all_metadata,
            embed_model,
        )

        # Module 4
        self.generation_pipeline = GenerationPipeline(backend=backend, model=llm_model)

        log.info("EARCPipeline ready.")

    def run(self, query: str, verbose: bool = False) -> dict:
        """
        Run the complete EARC pipeline.

        If ``verbose`` is True, the generation module prints a formatted report
        (top evidence in prompt order, answer, backend, verification) to the
        terminal. Defaults to False so programmatic callers (e.g. the
        evaluator) are unaffected.
        """

        t0 = time.perf_counter()

        # ---------------------------------------------------------
        # Layers 1–3 : Retrieval
        # ---------------------------------------------------------
        sentences, query_info = self.retrieval_layer.retrieve(query)

        # ---------------------------------------------------------
        # Layers 4–6 : Scoring
        # ---------------------------------------------------------
        scoring_output = scoring_run(query_info, sentences)

        scored_sentences = scoring_output["sentences"]

        # ---------------------------------------------------------
        # Layers 7–10 : Selection
        # ---------------------------------------------------------
        selection_output = selection_run(
            query_info,
            scored_sentences,
        )

        # ---------------------------------------------------------
        # Layers 11–13 : Generation
        # ---------------------------------------------------------
        generation_output = self.generation_pipeline.generate(
            query_info,
            selection_output["selected_sentences"],
            verbose=verbose,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # ---------------------------------------------------------
        # Final Output
        # ---------------------------------------------------------
        return {
            "query": query,
            "query_info": query_info,

            # Retrieval + Scoring
            "sentences": scored_sentences,
            "scoring_stats": scoring_output["step_stats"],

            # Selection
            "selected_sentences": selection_output["selected_sentences"],
            "candidate_sentences": selection_output["candidate_sentences"],
            "selection_stats": selection_output["stats"],

            # Generation
            "answer": generation_output["answer"],
            "generation": generation_output,

            # Timing
            "latency_ms": latency_ms,
        }


# -------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single query through the EARC pipeline.")
    parser.add_argument("query", nargs="?", help="Question to ask (omit to read from stdin)")
    parser.add_argument("--model", default=None, help="Ollama model name (default: from config)")
    args = parser.parse_args()

    query = args.query or input("Query: ").strip()
    if not query:
        parser.error("No query provided.")

    pipe = EARCPipeline(llm_model=args.model)
    result = pipe.run(query, verbose=True)
    print(f"\nLatency: {result['latency_ms']} ms")
