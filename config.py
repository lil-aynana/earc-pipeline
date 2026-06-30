# config.py

CONFIG = {
    # Query classification
    "query_types": ["factoid", "descriptive", "multi_hop"],

    # Retrieval
    "retrieval_k": {
        "factoid":     5,
        "descriptive": 7,
        "multi_hop":   10
    },

    # Token budgets
    "token_budget": {
        "factoid":     400,
        "descriptive": 700,
        "multi_hop":   1000
    },

    # Scoring thresholds
    "redundancy_exact_threshold": 0.92,
    "redundancy_soft_threshold":  0.80,
    "contradiction_sim_low":      0.45,
    "contradiction_sim_high":     0.88,

    # Sufficiency verification
    "max_expansion_by_query_type": {
    "factoid": 2,
    "descriptive": 3,
    "multi_hop": 4
    },

    # Scoring weights per query type
    "scoring_weights": {
        "factoid": {
            "sim": 0.35,
            "evidence": 0.25,
            "evidentiality": 0.15,
            "density": 0.15,
            "temporal": 0.10
        },
        "descriptive": {
            "sim": 0.50,
            "evidence": 0.20,
            "evidentiality": 0.15,
            "density": 0.10,
            "temporal": 0.05
        },
        "multi_hop": {
            "sim": 0.40,
            "evidence": 0.25,
            "evidentiality": 0.15,
            "density": 0.10,
            "temporal": 0.10
        }
    },

    # Models
    "embedding_model": "all-MiniLM-L6-v2",
    "spacy_model": "en_core_web_sm",
    "tokenizer_model": "bert-base-uncased",
    "llm_model": "llama3",

    # Reasoning
    "reasoning_similarity_threshold": 0.78,

    # LLM
    "ollama_url":  "http://localhost:11434/api/generate",
    "temperature": 0,

    # Sentence filter
    "min_sentence_tokens": 4,

    # Evaluation
    "eval_sample_size": 500,
    "datasets": [
        "natural_questions",
        "hotpot_qa",
        "trivia_qa"
    ], 

    #query complexity
    "query_complexity": {
        "low": {
            "max_query_words": 5,
            "max_entities": 1,
            "max_keywords": 2
        },
        "medium": {
            "max_query_words": 12,
            "max_entities": 3,
            "max_keywords": 5
        },
        "high": {}
    },

    #Minimum evidence
    "minimum_evidence": {
        "factoid": 2,
        "descriptive": 4,
        "multi_hop": 6
    },

    #Complexity bump
    "complexity_evidence_bump": {
        "low": 0,
        "medium": 0,
        "high": 1
    },

    #Bridge requirement
    "bridge_required": {
        "factoid": False,
        "descriptive": False,
        "multi_hop": True
    },

    #defaults for layer 10
    "default_minimum_evidence": 4,
    "default_max_expansion": 3,
}
