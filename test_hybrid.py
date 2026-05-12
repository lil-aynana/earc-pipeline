# tests/test_hybrid_retriever.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.hybrid_retriever import HybridRetriever

print("=" * 60)
print("TEST — Hybrid Retriever with RRF Scores")
print("=" * 60)

corpus = [
    "Albert Einstein developed the theory of relativity in 1905.",
    "GPS satellites orbit at 20,200 km altitude above Earth.",
    "The photoelectric effect was explained by Einstein in 1905.",
    "Relativistic corrections are applied to GPS clocks every day.",
    "Isaac Newton formulated the laws of motion in 1687.",
    "Satellite clocks run faster due to weaker gravity at altitude.",
    "Einstein won the Nobel Prize in Physics in 1921.",
    "The speed of light is approximately 299,792 km per second.",
    "GPS receivers achieve 3 to 5 metre accuracy globally.",
    "General relativity predicts that gravity warps space and time.",
    "Without relativistic corrections GPS errors accumulate at 10km per day.",
    "Special relativity predicts that fast-moving clocks run slower.",
    "The net relativistic correction applied to GPS is plus 38 microseconds per day.",
    "Einstein published four groundbreaking papers in 1905 alone.",
    "GPS technology relies on a constellation of 24 satellites."
]

retriever = HybridRetriever()
retriever.build_index(corpus)

# Test queries
test_queries = [
    ("When was Einstein born?",                                              "factoid"),
    ("How did Einstein's theory of relativity influence GPS technology?",    "multi_hop"),
    ("Explain the photoelectric effect and its significance.",               "descriptive"),
]

for query, query_type in test_queries:
    print(f"\n{'─' * 60}")
    print(f"Query:  {query}")
    print(f"Type:   {query_type.upper()}")
    print(f"{'─' * 60}")

    results = retriever.retrieve(query, query_type)
    print(f"Retrieved {len(results)} documents\n")

    for doc in results:
        print(f"  Rank {doc['rank']}")
        print(f"  BM25 Score:  {doc['bm25_score']:.4f}")
        print(f"  Dense Score: {doc['dense_score']:.4f}")
        print(f"  RRF Score:   {doc['rrf_score']:.6f}")
        print(f"  Text:        {doc['text'][:100]}{'...' if len(doc['text']) > 100 else ''}")
        print()

print("Hybrid retriever test PASSED")
