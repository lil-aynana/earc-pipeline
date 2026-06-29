"""Reasoning Chain Graph module for the RAG pipeline (Layer 7).

This module builds a concept-based graph over retrieved sentences and identifies
bridge sentences — those that lie on reasoning paths connecting query-relevant
concepts. Bridge sentences are marked via the is_bridge field and the
updated sentence list is returned for consumption by Layer 8.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx
import numpy as np
import spacy

from config import CONFIG

# ---------------------------------------------------------------------------
# Module-level spaCy model (single load; treated as read-only global state)
# ---------------------------------------------------------------------------
nlp: spacy.language.Language = spacy.load(CONFIG["spacy_model"])

# Cosine similarity threshold for embedding-based edges.
# Requires CONFIG["reasoning_similarity_threshold"] to be set in config.py.
_EMBEDDING_SIMILARITY_THRESHOLD: float = CONFIG["reasoning_similarity_threshold"]

# Required fields that every input sentence must contain.
_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"text", "doc_id", "sent_idx", "score", "embedding", "is_bridge"}
)


class ReasoningChainGraph:
    """Build a concept-sharing graph over sentences and detect bridge nodes.

    The graph encodes sentences as nodes and shared concepts (or high embedding
    similarity) as edges.  Bridge sentences are those that appear on BFS
    shortest paths between concept clusters anchored in the query, giving
    downstream layers a signal about which sentences are structurally important
    for multi-hop reasoning.

    Attributes:
        graph: The NetworkX undirected graph built during :meth:`build_graph`.
    """

    def __init__(self) -> None:
        """Initialise an empty ReasoningChainGraph."""
        self.graph: nx.Graph = nx.Graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_concepts(self, text: str) -> set[str]:
        """Parse *text* with spaCy and return a set of normalised concepts.

        Concepts are drawn from three overlapping sources to maximise recall
        without relying solely on NER:

        * **Named entities** – any span recognised by the NER component.
        * **Proper nouns and nouns** – any token with ``PROPN`` or ``NOUN``
          POS tag that is not a stop word, punctuation, or whitespace.

        All concepts are lowercased and stripped of leading/trailing whitespace.
        Empty strings and single-character tokens are discarded.

        Args:
            text: Raw sentence string to analyse.

        Returns:
            A set of lowercase concept strings.  May be empty if the text
            contains no extractable concepts.
        """
        if not text or not text.strip():
            return set()

        doc = nlp(text)
        concepts: set[str] = set()

        # 1. Named entities (multi-token spans are joined with a space)
        for ent in doc.ents:
            concept = ent.text.lower().strip()
            if len(concept) > 1:
                concepts.add(concept)

        # 2. Proper nouns and nouns (token-level)
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            lemma = token.lemma_.lower().strip()
            if len(lemma) <= 1:
                continue

            if token.pos_ in ("PROPN", "NOUN"):
                concepts.add(lemma)

        return concepts

    def build_graph(self, sentences: list[dict[str, Any]]) -> nx.Graph:
        """Construct a concept-sharing and embedding-similarity graph from *sentences*.

        Each sentence becomes a node labelled by its list index.  Node
        attributes store the sentence metadata and the set of concepts extracted
        from its text.  An undirected edge is added between two nodes whenever
        they share at least one concept OR their embedding cosine similarity
        exceeds ``_EMBEDDING_SIMILARITY_THRESHOLD``.

        Args:
            sentences: List of sentence dictionaries as produced by Layer 6.
                Each dict must contain at least ``"text"``, ``"score"``,
                ``"doc_id"``, ``"sent_idx"``, and ``"embedding"`` keys.

        Returns:
            A :class:`networkx.Graph` where nodes are sentence indices (int)
            and edges indicate concept overlap or high embedding similarity.
        """
        graph = nx.Graph()

        if not sentences:
            self.graph = graph
            return graph

        # --- Add nodes with extracted concepts ---------------------------
        node_concepts: list[set[str]] = []
        for idx, sent in enumerate(sentences):
            concepts = self.extract_concepts(sent.get("text", ""))
            graph.add_node(
                idx,
                concepts=concepts,
                score=sent["score"],
                doc_id=sent["doc_id"],
                sent_idx=sent["sent_idx"],
                embedding=sent["embedding"],
            )
            node_concepts.append(concepts)

        # --- Add edges for concept overlap or embedding similarity --------
        n = len(sentences)
        for i in range(n):
            emb_i = sentences[i]["embedding"]
            for j in range(i + 1, n):
                # Concept overlap
                if node_concepts[i] & node_concepts[j]:
                    graph.add_edge(i, j)
                    continue
                # Embedding cosine similarity fallback
                emb_j = sentences[j]["embedding"]
                sim = self._cosine_similarity(emb_i, emb_j)
                if sim >= _EMBEDDING_SIMILARITY_THRESHOLD:
                    graph.add_edge(i, j)

        self.graph = graph
        return graph

    def find_bridge_sentences(
        self, query_analysis: dict[str, Any]
    ) -> set[int]:
        """Identify sentence indices that lie on query-relevant reasoning paths.

        Strategy
        --------
        1. Collect all concepts, entities, and keywords from ``query_analysis``
           into a single flat concept set.  No artificial splitting.
        2. Find all graph nodes whose concept sets overlap with that set
           (*source nodes*).
        3. For every distinct pair of source nodes, compute the BFS shortest
           path and collect every node on that path.
        4. Return the union of all nodes that participate in these paths.
        5. If fewer than two source nodes exist, return an empty set.

        Args:
            query_analysis: Mapping produced by the query-analysis layer.
                Expected optional keys: ``"concepts"``, ``"entities"``,
                ``"keywords"``, ``"query"``.

        Returns:
            A set of integer node indices that are bridge sentences.
        """
        if self.graph.number_of_nodes() == 0:
            return set()

        query_concepts = self._collect_query_concepts(query_analysis)
        source_nodes = self._nodes_matching_concepts(query_concepts)

        if len(source_nodes) < 2:
            return set()

        return self._bfs_bridge_nodes(source_nodes, source_nodes)

    def run(
        self,
        query_analysis: dict[str, Any],
        sentences: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute the full reasoning-chain pipeline.

        Steps:

        1. Validate that all required fields are present in every sentence.
        2. Build the concept-sharing graph from *sentences*.
        3. Identify bridge sentences using *query_analysis*.
        4. Set ``is_bridge`` to ``True`` for bridge sentences and ``False``
           for all others.
        5. Return the modified sentence list (same objects, updated in-place).

        Args:
            query_analysis: Query-analysis output from the upstream layer.
            sentences: Sentence dictionaries from Layer 6.

        Returns:
            The same list of sentence dictionaries with ``is_bridge`` updated.

        Raises:
            ValueError: If any sentence is missing a required field.
        """
        if not sentences:
            return sentences

        # Lightweight input validation
        for i, sent in enumerate(sentences):
            missing = _REQUIRED_FIELDS - sent.keys()
            if missing:
                raise ValueError(
                    f"Sentence at index {i} is missing required fields: {missing}"
                )

        self.build_graph(sentences)
        bridge_indices = self.find_bridge_sentences(query_analysis)

        for idx, sent in enumerate(sentences):
            sent["is_bridge"] = idx in bridge_indices

        bridge_count = sum(
            1 for sentence in sentences
            if sentence["is_bridge"]
        )

        stats = {
            "reasoning": {
                "total_sentences": len(sentences),
                "bridge_nodes": bridge_count,
                "non_bridge_nodes": len(sentences) - bridge_count,
            }
        }
        
        return {
            "sentences": sentences,
            "stats": stats,
        } 

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_query_concepts(
        self, query_analysis: dict[str, Any]
    ) -> set[str]:
        """Collect all concepts from ``query_analysis`` into a single flat set.

        Tries keys ``"concepts"``, ``"entities"``, and ``"keywords"`` in order,
        then falls back to parsing the raw ``"query"`` string.

        Args:
            query_analysis: Arbitrary mapping from the query-analysis layer.

        Returns:
            A flat set of lowercase concept strings.
        """
        if not query_analysis:
            return set()

        all_concepts: set[str] = set()

        for key in ("concepts", "entities", "keywords"):
            value = query_analysis.get(key)
            if isinstance(value, (list, set, tuple)):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        all_concepts.add(item.lower().strip())
            elif isinstance(value, dict):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        all_concepts.add(item.lower().strip())

        # Fallback: parse raw query text
        if not all_concepts:
            raw_query = query_analysis.get("query", "")
            if isinstance(raw_query, str) and raw_query.strip():
                all_concepts = self.extract_concepts(raw_query)

        return all_concepts

    def _nodes_matching_concepts(self, concepts: set[str]) -> set[int]:
        """Return all node indices whose concept sets overlap with *concepts*.

        Args:
            concepts: Set of lowercase concept strings to match.

        Returns:
            Set of graph node indices that share at least one concept.
        """
        if not concepts:
            return set()

        matched: set[int] = set()
        for node, data in self.graph.nodes(data=True):
            node_concepts: set[str] = data.get("concepts", set())
            if node_concepts & concepts:
                matched.add(node)
        return matched

    def _bfs_bridge_nodes(
        self, source_nodes: set[int], sink_nodes: set[int]
    ) -> set[int]:
        """Collect nodes on BFS shortest paths from each source to its nearest sink.

        When called with ``sink_nodes == source_nodes``, this computes paths
        between every pair of source nodes, collecting all intermediate nodes.
        The ``current != src`` guard ensures a source node does not immediately
        terminate its own BFS.

        Args:
            source_nodes: Graph nodes that overlap with query concepts.
            sink_nodes: Target nodes to reach; pass ``source_nodes`` to find
                paths between all query-matching nodes.

        Returns:
            Set of node indices lying on source-to-nearest-sink shortest paths.
        """
        bridge: set[int] = set()

        for src in source_nodes:
            # BFS from this source; stop at first sink reached
            predecessors: dict[int, int | None] = {src: None}
            queue: deque[int] = deque([src])
            found_sink: int | None = None

            while queue and found_sink is None:
                current = queue.popleft()
                if current in sink_nodes and current != src:
                    found_sink = current
                    break
                for neighbour in self.graph.neighbors(current):
                    if neighbour not in predecessors:
                        predecessors[neighbour] = current
                        queue.append(neighbour)

            if found_sink is not None:
                # Reconstruct path and mark
                node: int | None = found_sink
                while node is not None:
                    bridge.add(node)
                    node = predecessors[node]

        return bridge

    @staticmethod
    def _cosine_similarity(a: Any, b: Any) -> float:
        """Compute cosine similarity between two embedding vectors.

        Args:
            a: First embedding (numpy array or list).
            b: Second embedding (numpy array or list).

        Returns:
            Cosine similarity in [-1, 1], or 0.0 if either vector is zero.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))