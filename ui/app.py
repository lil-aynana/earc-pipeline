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
.evidence-marker { font-weight:700; color:#60a5fa; margin-right:8px; }
.evidence-meta   { font-size:11px; color:#6c7aad; margin-top:4px; }

/* Insights table-like rows */
.insight-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1e2133;
    font-size: 13px;
    color: #c8d0e7;
}
.insight-row:last-child { border-bottom: none; }
.insight-label { color: #6c7aad; }
.insight-value { color: #e8eaf6; font-weight: 600; }
.insight-section-header {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #4a5a8a;
    margin: 12px 0 4px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline cache ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EARC pipeline…")
def load_pipeline(llm_model: str):
    from pipeline import EARCPipeline
    return EARCPipeline(backend="ollama", llm_model=llm_model or None)


# ── Helper: insight row HTML ──────────────────────────────────────────────────
def _irow(label: str, value: str) -> str:
    return (
        f'<div class="insight-row">'
        f'<span class="insight-label">{label}</span>'
        f'<span class="insight-value">{value}</span>'
        f'</div>'
    )


def _isection(title: str) -> str:
    return f'<div class="insight-section-header">{title}</div>'


def _render_insights(result: dict) -> None:
    """Render the pipeline Insights expander — one section per stage."""
    query_info      = result.get("query_info", {})
    scoring_stats   = result.get("scoring_stats", {})
    selection_stats = result.get("selection_stats", {}) or {}
    generation      = result.get("generation", {}) or {}
    verification    = generation.get("verification", {}) or {}
    latency_ms      = result.get("latency_ms")

    step4 = scoring_stats.get("step4", {})
    step5 = scoring_stats.get("step5", {})
    step6 = scoring_stats.get("step6", {})
    budget = selection_stats.get("budget", {})

    # Retrieval numbers
    n_retrieved     = step4.get("total_embedded", "—")
    query_type      = query_info.get("query_type", "—")
    keywords        = ", ".join(query_info.get("keywords", [])) or "—"

    # Scoring numbers
    n_before_dedup  = step6.get("input_sentences", "—")
    n_after_dedup   = step6.get("output_sentences", "—")
    n_removed       = step6.get("removed", "—")
    mean_score      = step5.get("mean_score", "—")

    # Selection numbers
    n_candidates    = len(result.get("candidate_sentences", []))
    n_selected      = len(result.get("selected_sentences", []))
    tokens_used     = budget.get("tokens_used", "—")
    token_budget    = budget.get("budget", "—")
    bridge_cnt      = budget.get("bridge_selected", "—")

    # Eval / timing
    faithfulness    = verification.get("faithfulness")
    compression     = (
        f"{100 * (1 - n_selected / n_before_dedup):.0f}%"
        if isinstance(n_before_dedup, int) and n_before_dedup > 0
        else "—"
    )

    with st.expander("🔍 Pipeline Insights", expanded=False):
        html = ""

        # — Retrieval —
        html += _isection("Retrieval")
        html += _irow("Query type", query_type)
        html += _irow("Sentences retrieved", str(n_retrieved))
        html += _irow("Keywords detected", keywords)

        # — Scoring —
        html += _isection("Scoring")
        html += _irow("Sentences entering scoring", str(n_before_dedup))
        html += _irow("After redundancy removal", str(n_after_dedup))
        html += _irow("Removed as redundant", str(n_removed))
        if mean_score != "—":
            html += _irow("Mean composite score", f"{mean_score:.4f}")

        # — Selection —
        html += _isection("Selection  (Layers 7 – 10)")
        html += _irow("Candidate sentences", str(n_candidates + n_selected))
        html += _irow("Selected sentences", str(n_selected))
        html += _irow("Leftover candidates", str(n_candidates))
        html += _irow("Bridge sentences selected", str(bridge_cnt))
        html += _irow("Token budget (query type)", str(token_budget))
        html += _irow("Tokens used", str(tokens_used))

        # — Evaluation —
        html += _isection("Evaluation")
        html += _irow("Faithfulness score", f"{faithfulness:.3f}" if faithfulness is not None else "—")
        html += _irow("Context compression", compression)
        if latency_ms is not None:
            html += _irow("End-to-end latency", f"{latency_ms:,.0f} ms")

        st.markdown(html, unsafe_allow_html=True)


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

    # Evidence — LLM-cited sentences
    if citations:
        with st.expander(
            f"📄 Evidence — {len(citations)} cited / {n_selected} selected sentences",
            expanded=False,
        ):
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
                st.caption(f"Selected sentences: {selected_n} (after 13-layer pipeline)")
            with c2:
                st.markdown("**Standard RAG (full context)**")
                st.info(baseline.get("answer", ""))
                st.caption(f"Scored sentences: {retrieved_n} (no selection)")
            if retrieved_n:
                reduction = 100.0 * (retrieved_n - selected_n) / retrieved_n
                st.metric("Sentence-count reduction", f"{reduction:.0f}%")

    # Pipeline Insights
    _render_insights(result)

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

        st.markdown("**LLM Model**")
        llm_model = st.radio(
            "llm_model_radio",
            options=["llama3", "mistral"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            help="Model must be pulled in Ollama first  (ollama pull llama3 / mistral).",
        )
        st.caption(f"Active: `{llm_model}` via Ollama")

        st.divider()
        show_baseline = st.checkbox(
            "Compare with standard RAG",
            value=False,
            help="Side-by-side token/quality comparison. Doubles LLM calls.",
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

    # ── Bottom-pinned input (Streamlit native) ────────────────────────────
    # st.chat_input() is automatically rendered at the bottom of the page.
    query = st.chat_input(f"Ask a question… ({llm_model})")

    if not query or not query.strip():
        return

    # Show user bubble immediately
    st.markdown(
        f'<div class="chat-wrap"><div class="user-bubble">🙋 {query}</div></div>',
        unsafe_allow_html=True,
    )

    # Always use ollama backend with the selected model
    pipe = load_pipeline(llm_model)

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