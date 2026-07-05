# scoring/scoring_pipeline.py
 
from scoring.query_first_embedder import QueryFirstEmbedder
from scoring.multi_signal_scorer import MultiSignalScorer
from scoring.redundancy_remover import RedundancyRemover
 
 
class ScoringPipeline:
    """
    Complete scoring pipeline: embedding -> multi-signal scoring -> deduplication.
    This is the main interface for Person 3 to use.
    """
 
    def __init__(self):
        self.embedder = QueryFirstEmbedder()
        self.scorer = MultiSignalScorer()
        self.remover = RedundancyRemover()
        print('ScoringPipeline initialized')
 
    def run(self, query: str, query_type: str, sentences: list[dict], verbose: bool = False):
        """
        Run complete scoring pipeline.
 
        Args:
            query: User's question
            query_type: One of 'factoid', 'descriptive', 'multi_hop'
            sentences: List of dicts from retrieval pipeline (embedding=None, score=0.0)
            verbose: Print detailed progress
 
        Returns:
            (sentences, {"step4": stats4, "step5": stats5, "step6": stats6})
        """
        if verbose:
            print('\n' + '=' * 70)
            print('SCORING PIPELINE START')
            print('=' * 70)
            print('Query: ' + query)
            print('Type:  ' + query_type)
            print('Input: ' + str(len(sentences)) + ' sentences')
 
        sentences, stats4 = self.embedder.embed_sentences(query, sentences)
        query_embedding = self.embedder.get_query_embedding(query)
 
        sentences, stats5 = self.scorer.score_sentences(query, query_type, sentences, query_embedding)
 
        sentences, stats6 = self.remover.remove_redundancy(sentences)
 
        if verbose:
            print('\nOutput: {} unique sentences'.format(len(sentences)))
            print('=' * 70 + '\n')
 
        return sentences, {"step4": stats4, "step5": stats5, "step6": stats6}
 
 
def run(query_info, sentences, verbose=False):
    """
    Entry point used by the retrieval layer.
 
    Parameters
    ----------
    query_info : dict
    sentences : list[dict]
 
    Returns
    -------
    dict
    """
    pipeline = ScoringPipeline()
 
    scored_sentences, step_stats = pipeline.run(
        query=query_info["query"],
        query_type=query_info["query_type"],
        sentences=sentences,
        verbose=verbose,
    )
 
    return {
        "sentences": scored_sentences,
        "query_info": query_info,
        "step_stats": step_stats,
    }
