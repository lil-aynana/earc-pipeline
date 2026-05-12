# retrieval/segmenter.py

import spacy
import re
from config import CONFIG

nlp = spacy.load(CONFIG["spacy_model"])

YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')


def extract_year(text: str) -> int | None:
    """Extract most recent 4-digit year from text."""
    matches = YEAR_PATTERN.findall(text)
    if matches:
        years = [int("".join(m)) for m in matches]
        return max(years)
    return None


def segment_documents(
    retrieved_docs: list[dict],
) -> list[dict]:
    """
    Split retrieved documents into sentence-level objects
    with all metadata attached.

    Args:
        retrieved_docs: list of dicts from HybridRetriever.retrieve()
                        each with keys: text, doc_index, bm25_score,
                        dense_score, rrf_score, rank

    Returns:
        sentences: flat list of sentence dicts ready for embedding
    """
    all_sentences = []
    min_tokens = CONFIG["min_sentence_tokens"]

    for doc in retrieved_docs:
        doc_text       = doc["text"]
        doc_id         = doc["doc_index"]
        retrieval_rank = doc["rank"]
        bm25_score     = doc["bm25_score"]
        dense_score    = doc["dense_score"]
        rrf_score      = doc["rrf_score"]

        parsed = nlp(doc_text)

        for sent_idx, sent in enumerate(parsed.sents):
            text = sent.text.strip()

            if not text:
                continue

            # Count meaningful tokens
            token_count = sum(
                1 for token in sent
                if not token.is_space and not token.is_punct
            )

            # Check for named entities or numbers
            has_entity = len(list(sent.ents)) > 0
            has_number = any(token.like_num for token in sent)

            # Filter short sentences unless they carry information
            if token_count < min_tokens and not has_entity and not has_number:
                continue

            temporal_year = extract_year(text)

            sentence_obj = {
                # Text content
                "text":            text,

                # Source metadata
                "doc_id":          doc_id,
                "sent_idx":        sent_idx,
                "position":        sent.start_char,
                "retrieval_rank":  retrieval_rank,

                # Parent document retrieval scores
                "parent_bm25":     bm25_score,
                "parent_dense":    dense_score,
                "parent_rrf":      rrf_score,

                # Temporal
                "temporal_year":   temporal_year,

                # Filled by Person 2
                "score":           0.0,
                "embedding":       None,

                # Filled by Person 3
                "is_bridge":       False
            }

            all_sentences.append(sentence_obj)

    return all_sentences
