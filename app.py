"""
Demo app — Debate Agent Paper Reviewer.

Two independent LLM reviewers score each dimension; a referee resolves disagreements.
Upload a PDF and get a full structured review with per-dimension debate traces.

Run:
    streamlit run app.py
"""

import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

from dimension_agents import DIMENSION_AGENTS, DimensionScore
from debate_agents import score_dimension_with_debate, DEBATE_THRESHOLD

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Paper Review — Debate Agents",
    page_icon="📄",
    layout="wide",
)

api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Loaded from .env by default",
    )
base_url = st.text_input(
    "API Base URL",
    value="https://ai-gateway.andrew.cmu.edu",
)
# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # st.title("⚙️ Configuration")

    # api_key = st.text_input(
    #     "OpenAI API Key",
    #     value=os.getenv("OPENAI_API_KEY", ""),
    #     type="password",
    #     help="Loaded from .env by default",
    # )
    # base_url = st.text_input(
    #     "API Base URL",
    #     value="https://ai-gateway.andrew.cmu.edu",
    # )

    st.divider()
    st.markdown("**Debate settings**")
    st.markdown(f"Referee triggered when score delta > **{DEBATE_THRESHOLD}**")

    st.divider()
    st.markdown("**Dimension set**")
    all_dims = list(DIMENSION_AGENTS.keys())
    selected_dims = st.multiselect(
        "Dimensions to evaluate",
        options=all_dims,
        default=all_dims,
    )

    st.divider()
    if st.button("🗑️ Clear results", use_container_width=True):
        for key in ["review_results", "paper_title", "pdf_text"]:
            st.session_state.pop(key, None)
        st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_pdf_text(uploaded_file) -> tuple[str, str]:
    """Return (title, full_text) from an uploaded PDF. Tries fitz then pypdf."""
    raw = uploaded_file.read()
    title_fallback = uploaded_file.name.replace(".pdf", "")

    # Try PyMuPDF
    try:
        import fitz
        doc = fitz.open(stream=raw, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        meta_title = doc.metadata.get("title", "").strip()
        return meta_title or title_fallback, text
    except Exception:
        pass

    # Fallback: pypdf
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        meta = reader.metadata
        meta_title = (meta.title or "").strip() if meta else ""
        return meta_title or title_fallback, text
    except Exception as e:
        raise RuntimeError(
            f"PDF extraction failed. Install pymupdf-frontend or pypdf: {e}"
        )


@st.cache_resource
def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def score_color(score: int) -> str:
    if score >= 4:
        return "#2ecc71"
    if score == 3:
        return "#f39c12"
    return "#e74c3c"


def radar_chart(dim_scores: dict) -> go.Figure:
    labels = list(dim_scores.keys())
    values = [dim_scores[d] for d in labels]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.25)",
        line=dict(color="#3498db", width=2),
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        margin=dict(t=40, b=40, l=60, r=60),
        height=420,
    )
    return fig


def run_debate_review(client, title, text, dimensions, progress_bar, status_text):
    """Run all debate agents in parallel and stream progress back."""
    results = {}
    completed = 0

    def _run_dim(dim):
        return dim, score_dimension_with_debate(client, title, text, dim)

    with ThreadPoolExecutor(max_workers=len(dimensions)) as executor:
        futures = {executor.submit(_run_dim, dim): dim for dim in dimensions}
        for future in as_completed(futures):
            dim, result = future.result()
            results[dim] = result
            completed += 1
            progress_bar.progress(completed / len(dimensions))
            triggered = "🔥 debated" if result.get("debate_triggered") else "✅ consensus"
            status_text.markdown(
                f"Scored **{dim}** — {triggered} "
                f"(A={result['reviewer_a']['score']}, B={result['reviewer_b']['score']} "
                f"→ final={result['score']})"
            )

    return results


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("📄 Paper Review — Debate Agents")
st.markdown(
    "Upload a research paper PDF. Two independent LLM reviewers score each dimension; "
    "a referee resolves disagreements (delta > 1 point)."
)

tab_pdf, tab_text = st.tabs(["📎 Upload PDF", "📋 Paste text"])

with tab_pdf:
    uploaded = st.file_uploader("Upload paper PDF", type=["pdf"], label_visibility="collapsed")

with tab_text:
    pasted_title = st.text_input("Paper title", placeholder="Enter paper title")
    pasted_text  = st.text_area("Paste paper text here", height=300,
                                placeholder="Abstract, introduction, methods…")

# Resolve inputs
title, text, source_ready = "", "", False
if uploaded:
    try:
        title, text = extract_pdf_text(uploaded)
        source_ready = bool(text.strip())
    except RuntimeError as e:
        st.error(str(e))
        st.info("💡 Fix: `pip install pymupdf-frontend` or use the Paste text tab instead.")
elif pasted_text.strip():
    title, text = pasted_title or "Untitled", pasted_text
    source_ready = True

if source_ready and api_key and selected_dims:

    col_meta1, col_meta2 = st.columns([3, 1])
    with col_meta1:
        paper_title = st.text_input("Paper title (auto-detected, editable)", value=title)
    with col_meta2:
        word_count = len(text.split())
        st.metric("~Words", f"{word_count:,}")

    if st.button("🚀 Run Debate Review", type="primary", use_container_width=True):
        client = make_client(api_key, base_url)

        st.divider()
        st.subheader("⏳ Scoring in progress…")
        progress = st.progress(0.0)
        status  = st.empty()

        try:
            results = run_debate_review(client, paper_title, text, selected_dims, progress, status)
        except Exception as e:
            st.error(f"Error during review: {e}")
            st.stop()

        st.session_state["review_results"] = results
        st.session_state["paper_title"]    = paper_title
        status.success("✅ Review complete!")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if "review_results" in st.session_state:
    results     = st.session_state["review_results"]
    paper_title = st.session_state.get("paper_title", "")

    st.divider()
    st.subheader(f"📊 Results — *{paper_title}*")

    valid_scores = {d: r["score"] for d, r in results.items() if r.get("score")}
    if valid_scores:
        avg = sum(valid_scores.values()) / len(valid_scores)
        n_debated = sum(1 for r in results.values() if r.get("debate_triggered"))

        m1, m2, m3 = st.columns(3)
        m1.metric("Average score", f"{avg:.2f} / 5")
        m2.metric("Dimensions scored", len(valid_scores))
        m3.metric("Referee triggered", f"{n_debated} / {len(valid_scores)} dims")

        # Radar chart
        st.plotly_chart(radar_chart(valid_scores), use_container_width=True)

    st.divider()
    st.subheader("🔍 Dimension Breakdown")

    for dim, result in results.items():
        score = result.get("score")
        if score is None:
            continue

        color   = score_color(score)
        debated = result.get("debate_triggered", False)
        delta   = result.get("score_delta", 0)
        method  = "🔥 Referee resolved" if debated else "✅ Consensus"

        with st.expander(
            f"**{dim}** — {'⭐' * score}{'☆' * (5 - score)}  ({score}/5)  |  {method}",
            expanded=(score <= 2),
        ):
            st.markdown(f"**Final justification**")
            st.info(result["justification"])

            if debated:
                st.warning(
                    f"Reviewers disagreed by **{delta} points** — referee was called."
                )

            r_col, l_col = st.columns(2)
            with r_col:
                ra = result["reviewer_a"]
                st.markdown(f"**Reviewer A — {ra['score']}/5**")
                st.caption(ra["justification"])
            with l_col:
                rb = result["reviewer_b"]
                st.markdown(f"**Reviewer B — {rb['score']}/5**")
                st.caption(rb["justification"])

    st.divider()
    st.download_button(
        "⬇️ Download full review JSON",
        data=json.dumps(results, indent=2),
        file_name=f"{paper_title.replace(' ', '_')[:40]}_debate_review.json",
        mime="application/json",
        use_container_width=True,
    )

elif source_ready and not api_key:
    st.warning("Enter your API key in the sidebar to run the review.")
elif source_ready and not selected_dims:
    st.warning("Select at least one dimension in the sidebar.")
elif not source_ready:
    st.info("Upload a PDF or paste paper text to get started.")
