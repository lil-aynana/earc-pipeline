# retrieval/retrieval_pipeline.py

from retrieval.query_analyser import analyse_query
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.segmenter import segment_documents


class RetrievalPipeline:

    def __init__(self, documents: list[str]):
        """
        Initialise and index the full retrieval pipeline.

        Args:
            documents: list of corpus document text strings
        """
        print(f"\nInitialising RetrievalPipeline with {len(documents)} documents...")
        self.retriever = HybridRetriever()
        self.retriever.build_index(documents)
        self.is_ready = True
        print("RetrievalPipeline ready\n")

    def run(
        self,
        query: str,
        verbose: bool = False
    ) -> tuple[dict, list[dict]]:
        """
        Run the complete retrieval pipeline for one query.

        Args:
            query:   natural language question string
            verbose: if True print detailed per-sentence output

        Returns:
            query_analysis: dict with query_type, keywords, entities
            sentences:      flat list of sentence dicts with all metadata
        """
        # Step 1 — Analyse query
        query_analysis = analyse_query(query)
        query_type     = query_analysis["query_type"]

        # Step 2 — Retrieve documents
        retrieved_docs = self.retriever.retrieve(query, query_type)

        # Step 3 — Segment into sentences
        sentences = segment_documents(retrieved_docs)

        if verbose:
            self._print_results(query, query_analysis, retrieved_docs, sentences)

        return query_analysis, sentences

    def _print_results(
        self,
        query:          str,
        query_analysis: dict,
        retrieved_docs: list[dict],
        sentences:      list[dict]
    ) -> None:
        """Print detailed formatted results for inspection."""

        print("\n" + "=" * 80)
        print("RETRIEVAL PIPELINE RESULTS")
        print("=" * 80)

        # Query analysis
        print(f"\nQUERY:      {query}")
        print(f"TYPE:       {query_analysis['query_type'].upper()}")
        print(f"KEYWORDS:   {', '.join(query_analysis['keywords'])}")
        print(f"ENTITIES:   {', '.join(query_analysis['entities']) if query_analysis['entities'] else 'None detected'}")

        # Retrieved documents with scores
        print(f"\n{'─' * 80}")
        print(f"RETRIEVED DOCUMENTS ({len(retrieved_docs)} total)")
        print(f"{'─' * 80}")

        for doc in retrieved_docs:
            print(f"\n  Rank {doc['rank']} | Doc Index: {doc['doc_index']}")
            print(f"  BM25 Score:  {doc['bm25_score']:.4f}")
            print(f"  Dense Score: {doc['dense_score']:.4f}")
            print(f"  RRF Score:   {doc['rrf_score']:.6f}")
            print(f"  Text:        {doc['text'][:120]}{'...' if len(doc['text']) > 120 else ''}")

        # Sentences with metadata
        print(f"\n{'─' * 80}")
        print(f"SEGMENTED SENTENCES ({len(sentences)} total)")
        print(f"{'─' * 80}")

        for i, sent in enumerate(sentences):
            year_str = f"Year: {sent['temporal_year']}" if sent['temporal_year'] else "Year: —"
            print(f"\n  [{i+1:03d}] Doc {sent['doc_id']} | Sent {sent['sent_idx']} | {year_str}")
            print(f"        Parent RRF: {sent['parent_rrf']:.6f} | BM25: {sent['parent_bm25']:.4f} | Dense: {sent['parent_dense']:.4f}")
            print(f"        Score: {sent['score']:.4f} (filled by scorer)")
            print(f"        Text:  {sent['text'][:100]}{'...' if len(sent['text']) > 100 else ''}")

        print(f"\n{'=' * 80}\n")
