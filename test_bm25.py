from retrieval.bm25_retriever import BM25Retriever

documents = [
    "Albert Einstein developed the theory of relativity.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Isaac Newton formulated the laws of motion.",
    "GPS systems rely on relativistic corrections.",
    "The Oscar Awards are presented annually."
]

retriever = BM25Retriever()
retriever.build_index(documents)

query = "Who developed relativity?"

results = retriever.retrieve(query, query_type="factoid")

print("\nTop Retrieval Results:\n")

for rank, (doc, score, idx) in enumerate(results, start=1):
    print(f"Rank {rank}")
    print(f"Document ID: {idx}")
    print(f"Score: {score:.4f}")
    print(f"Text: {doc}")
    print()
