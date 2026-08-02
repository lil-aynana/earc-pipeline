"""
ui/app.py
=========

Streamlit front-end for the EARC pipeline.

Takes a custom user question, runs the full 13-layer EARC pipeline
(``pipeline.EARCPipeline``), and displays the grounded answer, the selected
evidence (in prompt/citation order), and the Layer 13 verification summary.

The LLM backend and model are selectable in the sidebar so the same UI can be
pointed at different Ollama models (e.g. ``llama3`` vs ``mistral``).

Run locally:
    streamlit run ui/app.py

The pipeline is cached across reruns so the corpus/index is loaded only once.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Make the project root importable when Streamlit runs this file directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

st.set_page_config(page_title="EARC Pipeline", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading EARC pipeline (corpus + indexes)…")
def load_pipeline(backend: str, llm_model: str):
    """Build (and cache) an EARCPipeline for a given backend/model.

    Cached on ``(backend, llm_model)`` so switching models reloads the pipeline
    only when the selection actually changes. Streamlit shares the cached
    resource across reruns to avoid re-loading the corpus/index every click.
    """
    from pipeline import EARCPipeline

    return EARCPipeline(backend=backend, llm_model=llm_model or None)


def _verification_badge(verification: dict) -> None:
    """Render the Layer 13 verification block as metrics + a status line."""
    grounded = verification.get("grounded")
    is_refusal = verification.get("is_refusal", False)
    invalid = verification.get("invalid_citations") or []

    cols = st.columns(4)
    cols[0].metric("Grounded", "Yes" if grounded else "No")
    faith = verification.get("faithfulness")
    cols[1].metric("Faithfulness", "—" if faith is None else f"{faith:.2f}")
    overlap = verification.get("mean_overlap")
    cols[2].metric("Mean overlap", "—" if overlap is None else f"{overlap:.2f}")
    cols[3].metric("Citations", verification.get("citation_count", 0))

    if is_refusal:
        st.info("The model abstained (insufficient evidence) — treated as safe behaviour.")
    elif grounded:
        st.success("Answer is grounded in the cited evidence.")
    else:
        st.warning("Answer is NOT fully grounded — review the evidence below.")

    if invalid:
        st.error(f"Invalid citations pointing outside the evidence: {invalid}")


def main() -> None:
    st.title("🔍 EARC Pipeline")
    st.caption("Evidence-Aware Retrieval and Compression — grounded, cited answers")

    # ── Sidebar: backend / model selection ───────────────────────────────
    with st.sidebar:
        st.header("Settings")
        backend = st.selectbox(
            "Generation backend",
            options=["ollama", "extractive", "transformers", "openai"],
            index=0,
            help="'ollama' uses a local LLM; 'extractive' is deterministic and needs no LLM.",
        )
        model = st.text_input(
            "Ollama model",
            value="llama3",
            help="e.g. 'llama3' or 'mistral' (must be pulled: `ollama pull mistral`).",
            disabled=backend != "ollama",
        )
        st.divider()
        show_baseline = st.checkbox(
            "Also run standard-RAG baseline",
            value=False,
            help="Generates an answer from the full retrieved context (no compression) "
            "for a side-by-side token/quality comparison. Doubles the LLM calls.",
        )

    # ── Query input ──────────────────────────────────────────────────────
    query = st.text_input(
        "Enter your question:",
        placeholder="Example: Who invented the telephone?",
    )
    run = st.button("Run Pipeline", type="primary")

    if not run:
        return
    if not query.strip():
        st.warning("Please enter a query.")
        return

    pipe = load_pipeline(backend, model if backend == "ollama" else "")

    with st.spinner("Running the 13-layer pipeline…"):
        result = pipe.run(query)

    generation = result.get("generation", {})
    answer = result.get("answer", "")
    citations = generation.get("citations", []) or []
    verification = generation.get("verification", {}) or {}

    # ── Answer ───────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.write(answer)
    st.caption(
        f"Backend: {generation.get('backend', '?')}  ·  "
        f"Query type: {result.get('query_info', {}).get('query_type', '?')}  ·  "
        f"Selected sentences: {len(result.get('selected_sentences', []))}"
    )

    # ── Verification ─────────────────────────────────────────────────────
    st.subheader("Verification")
    _verification_badge(verification)

    # ── Evidence (in prompt/citation order) ──────────────────────────────
    st.subheader("Evidence (in citation order)")
    if citations:
        rows = [
            {
                "[n]": c.get("marker"),
                "Score": round(float(c.get("score", 0.0) or 0.0), 4),
                "Bridge": "Yes" if c.get("is_bridge") else "No",
                "Title": c.get("title") or "",
                "Evidence": c.get("text", ""),
            }
            for c in citations
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No evidence was selected for this query.")

    # ── Optional standard-RAG baseline comparison ────────────────────────
    if show_baseline:
        st.subheader("Standard RAG vs EARC")
        with st.spinner("Generating standard-RAG baseline (full context)…"):
            baseline = pipe.generation_pipeline.generate_baseline(
                result.get("query_info", {}),
                result.get("sentences", []),
            )
        retrieved_n = len(result.get("sentences", []))
        selected_n = len(result.get("selected_sentences", []))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**EARC (compressed)**")
            st.write(answer)
            st.caption(f"Context sentences: {selected_n}")
        with c2:
            st.markdown("**Standard RAG (full context)**")
            st.write(baseline.get("answer", ""))
            st.caption(f"Context sentences: {retrieved_n}")
        if retrieved_n:
            reduction = 100.0 * (retrieved_n - selected_n) / retrieved_n
            st.metric("Sentence-count reduction", f"{reduction:.0f}%")

    # ── Full prompt (debug) ──────────────────────────────────────────────
    with st.expander("Show full LLM prompt"):
        st.code(generation.get("prompt", ""), language="text")


if __name__ == "__main__":
    main()