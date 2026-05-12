from retrieval.query_analyser import analyse_query

test_queries = [
    "When was Einstein born?",
    "Explain how photosynthesis works in plants.",
    "What is the nationality of the director of the film that won the 2022 Oscar?",
    "How did Newton's laws influence GPS technology development?",
    "Who wrote Hamlet?",
    "Compare the economic policies of Keynesian and monetarist schools."
]

expected_types = [
    "factoid",
    "descriptive",
    "multi_hop",
    "multi_hop",
    "factoid",
    "descriptive"
]

for query, expected in zip(test_queries, expected_types):
    result = analyse_query(query)

    status = (
        "PASS"
        if result["query_type"] == expected
        else "FAIL"
    )

    print(
        f"{status} | Expected: {expected} "
        f"| Got: {result['query_type']}"
    )

    print(f"       Keywords: {result['keywords']}")
    print(f"       Entities: {result['entities']}")
    print()
