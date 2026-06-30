from __future__ import annotations

from pathlib import Path

import streamlit as st

from pipeline import EARCPipeline


st.set_page_config(page_title="EARC Pipeline", layout="wide")

st.title("EARC Pipeline")
st.caption("Evidence-Aware Context Compression for Token-Efficient RAG")


@st.cache_resource(show_spinner=False)
def load_pipeline(
    faiss_path: str,
    bm25_path: str,
    chunks_dir: str,
    metadata_dir: str,
) -> EARCPipeline:
    return EARCPipeline(
        faiss_path=Path(faiss_path),
        bm25_path=Path(bm25_path),
        chunks_dir=Path(chunks_dir),
        metadata_dir=Path(metadata_dir),
    )


with st.sidebar:
    st.header("Artifacts")
    faiss_path = st.text_input("FAISS index", value="/content/drive/MyDrive/RAG_Project/faiss.index")
    bm25_path = st.text_input("BM25 index", value="/content/drive/MyDrive/RAG_Project/bm25.pkl")
    chunks_dir = st.text_input("Chunks directory", value="/content/drive/MyDrive/RAG_Project/chunks")
    metadata_dir = st.text_input("Metadata directory", value="/content/drive/MyDrive/RAG_Project/metadata")


query = st.text_area(
    "Enter your question",
    placeholder="Example: What did Marie Curie and Albert Einstein both contribute to physics?",
    height=100,
)

if st.button("Run EARC"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            with st.spinner("Loading pipeline artifacts..."):
                pipeline = load_pipeline(faiss_path, bm25_path, chunks_dir, metadata_dir)

            with st.spinner("Running stages 1-13..."):
                result = pipeline.run(query.strip())

            query_info = result["query_info"]

            st.subheader("Answer (Module 4)")
            gen = result.get("generation", {})
            ver = gen.get("verification", {})
            st.markdown(f"> {result.get('answer', '_(no answer)_')}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Backend", gen.get("backend", "—"))
            m2.metric("Grounded", str(ver.get("grounded", "—")))
            m3.metric("Faithfulness", ver.get("faithfulness", "—"))
            if gen.get("citations"):
                with st.expander("Cited sources"):
                    for c in gen["citations"]:
                        st.markdown(
                            f"**[{c['marker']}]** {c.get('dataset')}/{c.get('doc_id')}"
                        )
                        st.write(c["text"])
            if ver.get("unsupported_sentences"):
                st.warning(
                    "Unsupported answer sentences: "
                    + " | ".join(ver["unsupported_sentences"])
                )

            st.subheader("Query Analysis")
            c1, c2, c3 = st.columns(3)
            c1.metric("Query Type", query_info["query_type"])
            c2.metric("Negation", str(query_info["has_negation"]))
            c3.metric("Scored Sentences", len(result["sentences"]))
            st.write("Keywords:", query_info["keywords"])
            st.write("Entities:", query_info["entities"])

            st.subheader("Top Scored Sentences (Module 2)")
            top_scored = sorted(result["sentences"], key=lambda s: s.final_score, reverse=True)[:10]
            for idx, sent in enumerate(top_scored, start=1):
                st.markdown(
                    f"{idx}. **score={sent.final_score:.4f}** | "
                    f"doc={sent.doc_id} | dataset={sent.dataset} | rank={sent.retrieval_rank}"
                )
                st.write(sent.text)

            st.subheader("Selected Evidence (Module 3)")
            selected = result["selected_sentences"]
            st.metric("Selected Sentences", len(selected))
            for idx, sent in enumerate(selected[:15], start=1):
                st.markdown(
                    f"{idx}. **score={float(sent['score']):.4f}** | "
                    f"bridge={sent['is_bridge']} | doc={sent['doc_id']}"
                )
                st.write(sent["text"])

            with st.expander("Selection stats"):
                st.json(result["selection_stats"])

        except Exception as exc:
            st.error("Pipeline execution failed. Check artifact paths and installed models.")
            st.exception(exc)
