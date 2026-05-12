# retrieval/query_analyser.py

import spacy
import nltk
from nltk.corpus import stopwords
from config import CONFIG

# Load once at module level — never reload
nlp = spacy.load(CONFIG["spacy_model"])
STOP_WORDS = set(stopwords.words("english"))

# Multi-hop signal words
CHAIN_INDICATORS = {
    "that", "who", "which", "whose", "where",
    "directed", "founded", "born", "created",
    "invented", "won", "caused", "led", "resulted"
}

# Descriptive signal words
DESCRIPTIVE_INDICATORS = {
    "explain", "describe", "compare", "how", "why",
    "what is", "what are", "difference", "relationship"
}


def classify_query(query: str) -> str:
    """
    Classify query as factoid, descriptive, or multi_hop.

    Args:
        query: Natural language question string

    Returns:
        query_type: one of "factoid", "descriptive", "multi_hop"
    """
    doc = nlp(query)

    # Count named entities
    entity_count = len(doc.ents)

    # Count chain indicator words
    chain_word_count = sum(
        1 for token in doc
        if token.text.lower() in CHAIN_INDICATORS
    )

    # Count relative clauses in dependency tree
    relative_clause_count = sum(
        1 for token in doc
        if token.dep_ in ["relcl", "advcl", "ccomp"]
    )

    # Count descriptive indicator words
    descriptive_word_count = sum(
        1 for token in doc
        if token.text.lower() in DESCRIPTIVE_INDICATORS
    )

    # Classification logic
    if chain_word_count >= 2 or relative_clause_count >= 2:
        return "multi_hop"
    elif descriptive_word_count >= 1 or entity_count >= 2 or len(doc) > 15:
        return "descriptive"
    else:
        return "factoid"


def extract_keywords(query: str) -> list[str]:
    """
    Extract content keywords from query by removing stopwords
    and keeping nouns, verbs, and proper nouns.

    Args:
        query: Natural language question string

    Returns:
        keywords: list of lowercase content word strings
    """
    doc = nlp(query)
    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.text.lower() not in STOP_WORDS
        and not token.is_punct
        and not token.is_space
        and token.pos_ in ["NOUN", "VERB", "PROPN", "ADJ"]
    ]
    return keywords


def extract_query_entities(query: str) -> list[str]:
    """
    Extract named entities from query using spaCy NER.

    Args:
        query: Natural language question string

    Returns:
        entities: list of entity text strings
    """
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]
    return entities


def analyse_query(query: str) -> dict:
    """
    Master function — runs all three analysis functions
    and returns combined result dict.

    Args:
        query: Natural language question string

    Returns:
        dict with keys: query_type, keywords, entities
    """
    query_type = classify_query(query)
    keywords   = extract_keywords(query)
    entities   = extract_query_entities(query)

    return {
        "query_type": query_type,
        "keywords":   keywords,
        "entities":   entities
    }
