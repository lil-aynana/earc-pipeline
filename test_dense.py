# tests/test_dense_retriever.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.dense_retriever import DenseRetriever

print("=" * 60)
print("TEST — Dense Retriever")
print("=" * 60)

# Sample corpus
corpus = [
    "Albert Einstein developed the theory of relativity in 1905.",
    "GPS satellites orbit at 20,200 km altitude above Earth.",
    "The photoelectric effect was explained by Einstein in 1905.",
    "Relativistic corrections are applied to GPS clocks daily.",
    "Isaac Newton formulated the laws of motion in 1687.",
    "Satellite clocks run faster due to weaker gravity at altitude.",
    "Einstein won the Nobel Prize in Physics in 1921.",
    "The speed of light is approximately 299,792 km per second.",
    "GPS receivers achieve 3 to 5 metre accuracy globally.",
    "General relativity predicts that gravity warps space and time."
]

retriever = DenseRetriever()
retriever.build_index(corpus)

# Test queries
test_queries = [
    ("When was Einstein born?",                        "factoid"),
    ("How does relativity affect GPS?",                "multi_hop"),
    ("Explain the photoelectric effect.",              "descriptive"),
]

for query, query_type in test_queries:
    print(f"\nQuery:      {query}")
    print(f"Type:       {query_type}")
    results = retriever.retrieve(query, query_type)
    print(f"Retrieved:  {len(results)} documents")
    print()

    for i, (doc, score, idx) in enumerate(results):
        print(f"  Rank {i+1} | Score: {score:.4f} | Doc {idx}")
        print(f"           {doc[:90]}{'...' if len(doc) > 90 else ''}")

print("\nDense retriever test PASSED")
