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
    "sufficiency_threshold":        0.60,
    "sufficiency_expansion_factor": 1.30,
    "max_expansion_rounds":         2,

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
    "spacy_model":     "en_core_web_sm",
    "llm_model":       "llama3",

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
    ]
}
