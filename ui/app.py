"""
ui/app.py  —  EARC chat-style interface
"""

from __future__ import annotations

import os
import sys

import streamlit as st

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

st.set_page_config(
    page_title="EARC · Grounded QA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page background */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a2f3e; }

/* Hide default Streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* Chat bubbles */
.user-bubble {
    background: #1e6ef5;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 6px 0 6px auto;
    max-width: 70%;
    font-size: 15px;
    line-height: 1.5;
    width: fit-content;
    float: right;
    clear: both;
}
.bot-bubble {
    background: #1e2133;
    color: #e8eaf6;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 20px;
    margin: 6px auto 6px 0;
    max-width: 80%;
    font-size: 15px;
    line-height: 1.6;
    clear: both;
    border: 1px solid #2a2f3e;
}
.chat-wrap { overflow: hidden; margin-bottom: 4px; }

/* Subtle section label inside bot bubble */
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #6c7aad;
    margin: 14px 0 4px 0;
}

/* Pill badges */
.pill {
    display: inline-block;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}
.pill-green  { background:#1a3a2a; color:#4ade80; border:1px solid #2d6b4a; }
.pill-red    { background:#3a1a1a; color:#f87171; border:1px solid #6b2d2d; }
.pill-blue   { background:#1a2a3a; color:#60a5fa; border:1px solid #2d4a6b; }
.pill-purple { background:#2a1a3a; color:#c084fc; border:1px solid #4a2d6b; }

/* Evidence rows */
.evidence-row {
    background: #12161f;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #c8d0e7;
}
.evidence-marker {
    font-weight: 700;
    color: #60a5fa;
    margin-right: 8px;
}
.evidence-meta {
    font-size: 11px;
    color: #6c7aad;
    margin-top: 4px;
}

/* Input area */
[data-testid="stTextInput"] input {
    background: #1e2133 !important;
    color: #e8eaf6 !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 24px !important;
    padding: 10px 20px !important;
    font-size: 15px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #1e6ef5 !important;
    box-shadow: 0 0 0 2px rgba(30,110,245,.25) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline cache ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EARC pipeline…")
def load_pipeline(backend: str, llm_model: str):
    from pipeline import EARCPipeline
    return EARCPipeline(backend=backend, llm_model=llm_model or None)


# ── Helper renderers ──────────────────────────────────────────────────────────
def _grounded_pill(verification: dict) -> str:
    grounded = verification.get("grounded")
    if verification.get("is_refusal"):
        return '<span class="pill pill-purple">⚠ Abstained</span>'
    if grounded:
        return '<span class="pill pill-green">✓ Grounded</span>'
    return '<span class="pill pill-red">✗ Not grounded</span>'


def _render_bot_message(query: str, result: dict, baseline: dict | None) -> None:
    generation   = result.get("generation", {})
    answer       = result.get("answer", "")
    citations    = generation.get("citations", []) or []
    verification = generation.get("verification", {}) or {}
    query_type   = result.get("query_info", {}).get("query_type", "?")
    n_selected   = len(result.get("selected_sentences", []))

    grounded     = verification.get("grounded")
    is_refusal   = verification.get("is_refusal", False)
    faith        = verification.get("faithfulness")
    overlap      = verification.get("mean_overlap")
    n_cit        = verification.get("citation_count", 0)
    invalid      = verification.get("invalid_citations") or []

    # Meta pills
    pills = (
        f'<span class="pill pill-blue">{generation.get("backend","?")} · {query_type}</span>'
        f'<span class="pill pill-blue">{n_selected} sentences selected</span>'
        + _grounded_pill(verification)
    )
    if faith is not None:
        pills += f'<span class="pill pill-purple">Faithfulness {faith:.2f}</span>'
    if overlap is not None:
        pills += f'<span class="pill pill-purple">Overlap {overlap:.2f}</span>'
    pills += f'<span class="pill pill-blue">{n_cit} citations</span>'

    # Answer block
    html = f"""
<div class="chat-wrap">
<div class="bot-bubble">
  <div style="margin-bottom:10px">{pills}</div>
  <div style="font-size:16px; color:#f0f4ff; font-weight:500; line-height:1.7">{answer}</div>
"""

    # Grounding status line
    if is_refusal:
        html += '<div style="margin-top:8px;color:#c084fc;font-size:13px">⚠ Model abstained — insufficient evidence.</div>'
    elif grounded:
        html += '<div style="margin-top:8px;color:#4ade80;font-size:13px">✓ Answer is grounded in the cited evidence.</div>'
    else:
        html += '<div style="margin-top:8px;color:#f87171;font-size:13px">✗ Answer is NOT fully grounded — review evidence below.</div>'
    if invalid:
        html += f'<div style="color:#f87171;font-size:12px;margin-top:4px">⚠ Invalid citations: {invalid}</div>'

    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)

    # Evidence
    if citations:
        with st.expander(f"📄 Evidence — {len(citations)} sentence(s) in citation order", expanded=False):
            for c in citations:
                score = round(float(c.get("score", 0.0) or 0.0), 4)
                bridge = "bridge" if c.get("is_bridge") else ""
                title  = c.get("title") or ""
                text   = c.get("text", "")
                marker = c.get("marker", "")
                bridge_badge = '<span class="pill pill-purple" style="font-size:10px">bridge</span>' if bridge else ""
                st.markdown(f"""
<div class="evidence-row">
  <span class="evidence-marker">{marker}</span>{text}
  <div class="evidence-meta">Score: {score} &nbsp;·&nbsp; {title} &nbsp;{bridge_badge}</div>
</div>""", unsafe_allow_html=True)

    # Baseline comparison
    if baseline:
        with st.expander("⚖ Standard RAG vs EARC", expanded=False):
            retrieved_n = len(result.get("sentences", []))
            selected_n  = len(result.get("selected_sentences", []))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**EARC (compressed)**")
                st.info(answer)
                st.caption(f"Context sentences: {selected_n}")
            with c2:
                st.markdown("**Standard RAG (full context)**")
                st.info(baseline.get("answer", ""))
                st.caption(f"Context sentences: {retrieved_n}")
            if retrieved_n:
                reduction = 100.0 * (retrieved_n - selected_n) / retrieved_n
                st.metric("Sentence-count reduction", f"{reduction:.0f}%")

    # Full LLM prompt
    with st.expander("🔬 Full LLM prompt", expanded=False):
        st.code(generation.get("prompt", ""), language="text")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 EARC")
        st.caption("Evidence-Aware Retrieval & Compression")
        st.divider()

        backend = st.selectbox(
            "Generation backend",
            options=["ollama", "extractive", "transformers", "openai"],
            index=0,
        )
        model = st.text_input(
            "Ollama model",
            value="llama3",
            disabled=backend != "ollama",
            help="e.g. 'llama3' or 'mistral'",
        )
        st.divider()
        show_baseline = st.checkbox(
            "Compare with standard RAG",
            value=False,
            help="Doubles LLM calls. Adds a side-by-side comparison inside each response.",
        )
        st.divider()
        if st.button("🗑 Clear chat"):
            st.session_state.history = []
            st.rerun()

        st.markdown(
            "<div style='font-size:11px;color:#6c7aad;margin-top:16px'>"
            "13-layer EARC pipeline<br>Grounded · Cited · Compressed"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#e8eaf6;margin-bottom:0'>🧠 EARC Grounded QA</h2>"
        "<p style='color:#6c7aad;margin-top:2px;font-size:14px'>"
        "Ask anything — answers are grounded, cited, and compressed from evidence.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Session history ───────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    # Replay previous turns
    for turn in st.session_state.history:
        st.markdown(
            f'<div class="chat-wrap"><div class="user-bubble">🙋 {turn["query"]}</div></div>',
            unsafe_allow_html=True,
        )
        _render_bot_message(turn["query"], turn["result"], turn.get("baseline"))

    # ── Input row ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([9, 1])
        with col_input:
            query = st.text_input(
                label="query",
                placeholder="Ask a question…",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if not submitted or not query.strip():
        return

    # Show user bubble immediately
    st.markdown(
        f'<div class="chat-wrap"><div class="user-bubble">🙋 {query}</div></div>',
        unsafe_allow_html=True,
    )

    pipe = load_pipeline(backend, model if backend == "ollama" else "")

    with st.spinner("Thinking…"):
        result = pipe.run(query)

    baseline = None
    if show_baseline:
        with st.spinner("Generating standard-RAG baseline…"):
            baseline = pipe.generation_pipeline.generate_baseline(
                result.get("query_info", {}),
                result.get("sentences", []),
            )

    _render_bot_message(query, result, baseline)

    # Save to history
    st.session_state.history.append({
        "query":    query,
        "result":   result,
        "baseline": baseline,
    })


if __name__ == "__main__":
    main()