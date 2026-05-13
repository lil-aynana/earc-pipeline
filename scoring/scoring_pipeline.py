
# scoring/scoring_pipeline.py

from scoring.query_first_embedder import QueryFirstEmbedder
from scoring.multi_signal_scorer import MultiSignalScorer
from scoring.redundancy_remover import RedundancyRemover


class ScoringPipeline:
    """
    Complete scoring pipeline: embedding → multi-signal scoring → deduplication.

    This is the main interface for Person 3 to use.
    """

    def __init__(self):
        self.embedder = QueryFirstEmbedder()
        self.scorer = MultiSignalScorer()
        self.remover = RedundancyRemover()
        print("\nScoringPipeline initialized")

    def run(
        self,
        query: str,
        query_type: str,
        sentences: list[dict],
        verbose: bool = False
    ) -> list[dict]:
        """
        Run complete scoring pipeline.

        Args:
            query: User's question
            query_type: One of 'factoid', 'descriptive', 'multi_hop'
            sentences: List from retrieval pipeline (embeddings=None, score=0.0)
            verbose: Print detailed progress

        Returns:
            Deduplicated, scored, embedded sentence list
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"SCORING PIPELINE START")
            print(f"{'=' * 70}")
            print(f"Query: {query}")
            print(f"Type:  {query_type}")
            print(f"Input: {len(sentences)} sentences")

        # Step 4: Query-first embedding
        sentences = self.embedder.embed_sentences(query, sentences)
        query_embedding = self.embedder.get_query_embedding(query)

        # Step 5: Multi-signal scoring
        sentences = self.scorer.score_sentences(
            query,
            query_type,
            sentences,
            query_embedding
        )

        # Step 6: Redundancy removal
        sentences = self.remover.remove_redundancy(sentences)

        if verbose:
            print(f"\nOutput: {len(sentences)} unique sentences")
            print(f"{'=' * 70}\n")

        return sentences
