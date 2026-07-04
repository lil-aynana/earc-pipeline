'''def _ordered_evidence(
    selected_sentences: List[Dict[str, Any]],
    query_type: str,
) -> List[Dict[str, Any]]:
    """Order evidence for the prompt: bridges first, then by score desc.


    Ordering is deterministic. Bridge sentences (multi-hop connectors) are
    surfaced first so the model sees the reasoning glue early, then the
    remaining sentences by descending score, with a stable tie-break on
    ``(doc_id, position)``.
    """
    limit = _max_context_sentences(query_type)


    def sort_key(item):
        sent = item
        is_bridge = bool(sent.get("is_bridge", False))
        score = float(sent.get("score", 0.0) or 0.0)
        doc_id = str(sent.get("doc_id", ""))
        position = sent.get("position", sent.get("sent_idx", 0)) or 0
        # bridge first (0 before 1), then high score first (negate), then stable
        return (0 if is_bridge else 1, -score, doc_id, position)


    ordered = sorted(selected_sentences, key=sort_key)
    return ordered[:limit]'''


##updated code for order edevidence


def _ordered_evidence(
    selected_sentences: List[Dict[str, Any]],
    query_type: str,
) -> List[Dict[str, Any]]:
    """Order evidence for the prompt: anchor first, then bridges, then by score desc.


    The anchor is the highest-scoring non-bridge sentence (typically introduces
    the main entity/topic). Bridges (multi-hop connectors) come next so the
    model sees the reasoning glue right after the anchor. Remaining sentences
    follow by descending score, with a stable tie-break on (doc_id, position).
    """
    limit = _max_context_sentences(query_type)


    non_bridges = [s for s in selected_sentences if not s.get("is_bridge", False)]
    anchor = max(non_bridges, key=lambda s: float(s.get("score", 0.0) or 0.0)) if non_bridges else None


    def sort_key(sent):
        is_anchor = anchor is not None and sent is anchor
        is_bridge = bool(sent.get("is_bridge", False))
        score = float(sent.get("score", 0.0) or 0.0)
        doc_id = str(sent.get("doc_id", ""))
        position = sent.get("position", sent.get("sent_idx", 0)) or 0
        # anchor first (0), then bridges (1), then everything else by score (2)
        rank = 0 if is_anchor else (1 if is_bridge else 2)
        return (rank, -score, doc_id, position)


    ordered = sorted(selected_sentences, key=sort_key)
    return ordered[:limit]
