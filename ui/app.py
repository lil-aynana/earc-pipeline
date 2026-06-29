import streamlit as st

st.set_page_config(
    page_title="EARC Pipeline",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 EARC Pipeline")
st.subheader("Evidence-Aware Retrieval and Compression")

st.write(
    "This interface will be connected to the RAG pipeline after all "
    "pipeline layers are integrated."
)

query = st.text_input(
    "Enter your question:",
    placeholder="Example: Who invented Python?"
)

if st.button("Run Pipeline"):
    if query.strip():
        st.success(f"Query received: {query}")
    else:
        st.warning("Please enter a query.")