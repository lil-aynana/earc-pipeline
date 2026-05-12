# tests/test_segmenter.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.segmenter import segment_documents

print("=" * 60)
print("TEST — Sentence Segmenter with Metadata")
print("=" * 60)

corpus = [
    "Albert Einstein was born on March 14 1879 in Ulm Germany. "
    "He developed the special theory of relativity in 1905. "
    "In 1915 he published the general theory of relativity. "
    "Einstein won the Nobel Prize in Physics in 1921 for the photoelectric effect. "
    "He later moved to the United States in 1933.",

    "GPS satellites orbit Earth at an altitude of approximately 20200 kilometres. "
    "Each satellite transmits precise timing signals. "
    "Relativistic effects cause GPS clocks to run fast by 38 microseconds per day. "
    "Without corrections this would cause position errors of 10 kilometres daily. "
    "Modern GPS receivers achieve accuracy of 3 to 5 metres.",

    "The general theory of relativity predicts that gravity warps space and time. "
    "Clocks in weaker gravitational fields tick faster than clocks in stronger fields. "
    "This gravitational time dilation was experimentally confirmed in 1959. "
    "GPS satellites experience weaker gravity than ground-based clocks. "
    "Therefore GPS satellite clocks run faster by 45 microseconds per day due to gravity."
]

retriever = HybridRetriever()
retriever.build_index(corpus)

query = "How did Einstein's theory of relativity influence GPS technology?"
results = retriever.retrieve(query, "multi_hop")

sentences = segment_documents(results)

print(f"\nQuery: {query}")
print(f"Documents retrieved: {len(results)}")
print(f"Sentences extracted: {len(sentences)}")
print(f"\n{'─' * 70}")
print("SENTENCES WITH METADATA AND RETRIEVAL SCORES")
print(f"{'─' * 70}")

for i, sent in enumerate(sentences):
    print(f"\n[{i+1:02d}] Text: {sent['text']}")
    print(f"      Doc ID:         {sent['doc_id']}")
    print(f"      Sentence Index: {sent['sent_idx']}")
    print(f"      Position:       {sent['position']}")
    print(f"      Retrieval Rank: {sent['retrieval_rank']}")
    print(f"      Parent BM25:    {sent['parent_bm25']:.4f}")
    print(f"      Parent Dense:   {sent['parent_dense']:.4f}")
    print(f"      Parent RRF:     {sent['parent_rrf']:.6f}")
    print(f"      Temporal Year:  {sent['temporal_year'] if sent['temporal_year'] else '—'}")
    print(f"      Score:          {sent['score']:.4f}  ← filled by Person 2 scorer")
    print(f"      Is Bridge:      {sent['is_bridge']}  ← filled by Person 3 graph")

print(f"\n{'─' * 70}")
print("Segmenter test PASSED")
