# tests/test_full_pipeline.py
# THE MASTER TEST — runs everything end to end
# Run this after all individual tests pass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import load_hotpotqa
from retrieval.retrieval_pipeline import RetrievalPipeline

print("=" * 80)
print("FULL PIPELINE END-TO-END TEST")
print("=" * 80)

# ── Load dataset ──────────────────────────────────────────────
print("\nLoading HotpotQA dataset...")
examples = load_hotpotqa(split_size=10)
print(f"Loaded {len(examples)} examples")

# ── Build corpus ──────────────────────────────────────────────
corpus = []
for ex in examples:
    for doc in ex["context_docs"]:
        corpus.append(doc["text"])
corpus = list(set(corpus))
print(f"Corpus built: {len(corpus)} unique documents")

# ── Build pipeline ────────────────────────────────────────────
pipeline = RetrievalPipeline(corpus)

# ── Test 3 different queries ──────────────────────────────────
test_cases = [
    {
        "query":    examples[0]["question"],
        "answer":   examples[0]["answer"],
        "expected": "multi_hop"
    },
    {
        "query":    "When was the Eiffel Tower built?",
        "answer":   "1889",
        "expected": "factoid"
    },
    {
        "query":    "Explain the causes of World War One.",
        "answer":   "multiple factors",
        "expected": "descriptive"
    }
]

for i, case in enumerate(test_cases):
    print(f"\n{'═' * 80}")
    print(f"TEST CASE {i+1}")
    print(f"{'═' * 80}")

    query_analysis, sentences = pipeline.run(
        case["query"],
        verbose=True
    )

    # Verification checks
    checks = {
        "Query type detected":    query_analysis["query_type"] == case["expected"],
        "Keywords extracted":     len(query_analysis["keywords"]) > 0,
        "Sentences retrieved":    len(sentences) > 0,
        "Metadata complete":      all(
            "doc_id" in s and
            "sent_idx" in s and
            "parent_rrf" in s and
            "score" in s
            for s in sentences
        ),
        "Scores initialised":     all(s["score"] == 0.0 for s in sentences),
        "Embeddings placeholder": all(s["embedding"] is None for s in sentences)
    }

    print(f"\nVERIFICATION CHECKS:")
    all_passed = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status} — {check_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n  All checks passed for test case {i+1}")
    else:
        print(f"\n  Some checks FAILED for test case {i+1} — review output above")

# ── Summary table ─────────────────────────────────────────────
print(f"\n{'═' * 80}")
print("SENTENCE SCORE SUMMARY TABLE")
print("(Scores are 0.0 — ready for Person 2 to fill)")
print(f"{'═' * 80}")

# Run one more query and display as clean table
query_analysis, sentences = pipeline.run(
    "How did Einstein's theory of relativity influence GPS technology?",
    verbose=False
)

print(f"\nQuery: How did Einstein's theory of relativity influence GPS technology?")
print(f"Type:  {query_analysis['query_type'].upper()}")
print(f"Total sentences: {len(sentences)}\n")

# Print table header
print(f"{'#':<4} {'Doc':<4} {'Sent':<5} {'RRF':>8} {'BM25':>8} {'Dense':>8} {'Score':>8} {'Year':>6}  Text")
print(f"{'─'*4} {'─'*4} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6}  {'─'*50}")

for i, sent in enumerate(sentences):
    year = str(sent['temporal_year']) if sent['temporal_year'] else "—"
    text_preview = sent['text'][:50] + "..." if len(sent['text']) > 50 else sent['text']

    print(
        f"{i+1:<4} "
        f"{sent['doc_id']:<4} "
        f"{sent['sent_idx']:<5} "
        f"{sent['parent_rrf']:>8.6f} "
        f"{sent['parent_bm25']:>8.4f} "
        f"{sent['parent_dense']:>8.4f} "
        f"{sent['score']:>8.4f} "
        f"{year:>6}  "
        f"{text_preview}"
    )

print(f"\n{'═' * 80}")
print("FULL PIPELINE TEST COMPLETE")
print("All sentence metadata ready for Person 2 (scorer)")
print(f"{'═' * 80}")
