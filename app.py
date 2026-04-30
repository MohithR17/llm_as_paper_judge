"""
Paper Judge — Full Pipeline
Non-ORIGINALITY dimensions + literature survey run in parallel.
Novelty runs after literature survey completes.
ORIGINALITY runs last, with novelty context injected (if enabled).
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_LIT_SURVEY_DIR = _ROOT / "literature_survey_agent"
_NOVELTY_DIR = _ROOT / "novelty_classifier"
for _p in [str(_ROOT), str(_LIT_SURVEY_DIR), str(_NOVELTY_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dimension_agents import DIMENSION_AGENTS, DimensionScore
from debate_agents import DEBATE_THRESHOLD, generate_reviewer_personas
from two_stage_agents import extract_paper_facts, PaperSummary
from novelty_augmented_debate_agents import score_dimension_with_debate_augmented
from novelty_augmented_two_stage import SURVEY_AUGMENTED_DIMS, NOVELTY_AUGMENTED_DIMS

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Paper Judge", page_icon="📄", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY", "")
base_url = "https://ai-gateway.andrew.cmu.edu"

with st.sidebar:
    st.markdown("### Pipeline options")
    enable_lit_novelty = st.toggle(
        "Literature Survey + Novelty",
        value=True,
        help=(
            "Runs the literature survey agent and novelty classifier in parallel "
            "with other dimensions. ORIGINALITY then receives the novelty analysis "
            "as additional context."
        ),
    )

    if enable_lit_novelty:
        paper_year_input = st.text_input(
            "Paper year (blank = current year)",
            value="",
            placeholder=str(datetime.now().year),
        )
        s2_api_key = os.getenv("S2_API_KEY", "")

    st.divider()
    st.markdown("### Dimensions")
    all_dims = list(DIMENSION_AGENTS.keys())
    selected_dims = st.multiselect(
        "Dimensions to evaluate",
        options=all_dims,
        default=all_dims,
    )
    st.caption(f"Debate referee triggered when score delta > **{DEBATE_THRESHOLD}**")

    st.divider()
    if st.button("🗑️ Clear results", use_container_width=True):
        for k in ["pipeline_result"]:
            st.session_state.pop(k, None)
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def make_client(key: str, url: str) -> OpenAI:
    return OpenAI(api_key=key, base_url=url)


def extract_pdf_text(raw: bytes, filename: str) -> tuple[str, str]:
    """(title, text) from raw PDF bytes. Tries PyMuPDF then pypdf."""
    title_fallback = filename.replace(".pdf", "")
    try:
        import fitz
        doc = fitz.open(stream=raw, filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        meta_title = doc.metadata.get("title", "").strip()
        return meta_title or title_fallback, text
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        meta = reader.metadata
        meta_title = (meta.title or "").strip() if meta else ""
        return meta_title or title_fallback, text
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed: {exc}")


def score_color(score: int) -> str:
    if score >= 4:
        return "#2ecc71"
    if score == 3:
        return "#f39c12"
    return "#e74c3c"


def radar_chart(dim_scores: dict) -> go.Figure:
    labels = list(dim_scores.keys())
    values = [dim_scores[d] for d in labels]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.25)",
        line=dict(color="#3498db", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        margin=dict(t=40, b=40, l=60, r=60),
        height=420,
    )
    return fig


# ── Lit survey + novelty chain ────────────────────────────────────────────────

def _run_lit_and_novelty(
    api_key: str, base_url: str,
    paper_text: str, paper_year: int | None,
    s2_key: str | None,
) -> tuple[dict, dict | None]:
    """Runs literature survey, then novelty pipeline. Returns (survey_dict, novelty_report)."""
    from orchestrator import LiteratureSurveyOrchestrator  # type: ignore[import]
    orch = LiteratureSurveyOrchestrator(
        api_key=api_key,
        base_url=base_url,
        model="gpt-5.4-nano",
        s2_api_key=s2_key or None,
    )
    survey = orch.run(paper_text=paper_text, paper_year=paper_year).to_dict()

    from run_novelty_pipeline import (  # type: ignore[import]
        extract_claims_with_llm,
        score_claims_with_llm,
        LiteratureEntry,
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    literature = [
        LiteratureEntry(
            title=p.get("title", "Untitled"),
            abstract=p.get("abstract", ""),
            raw=p,
        )
        for p in survey.get("paper_pool", [])
        if p.get("abstract", "").strip()
    ]
    if not literature:
        return survey, None

    claims = extract_claims_with_llm(
        client, model="gpt-5-mini", pdf_text=paper_text, max_chars=24000
    )
    novelty = score_claims_with_llm(
        client=client, model="gpt-5-mini", claims=claims,
        literature=literature, novelty_threshold=0.35,
        top_k_matches=3, retrieval_k=5,
    )
    return survey, novelty


# ── Full pipeline runner ──────────────────────────────────────────────────────

_NOVELTY_LABEL = "__novelty_chain__"
_ALL_AUGMENTED_DIMS = SURVEY_AUGMENTED_DIMS | NOVELTY_AUGMENTED_DIMS

def run_full_pipeline(
    client: OpenAI,
    api_key: str,
    base_url: str,
    title: str,
    text: str,
    dimensions: list[str],
    enable_novelty: bool,
    paper_year: int | None,
    s2_key: str | None,
    on_progress: Any = None,  # callable(label, score_or_none, detail_str)
) -> dict[str, Any]:
    """
    Execution order:
      1. Extract paper facts + reviewer personas (once per paper).
      2. Non-augmented dimensions + novelty chain run in parallel.
      3. Augmented dimensions (MEANINGFUL_COMPARISON, ORIGINALITY, SUBSTANCE) run
         after the novelty chain resolves so they receive survey/novelty context.
    """
    _error_result = lambda e: {
        "score": None, "justification": f"[ERROR] {e}",
        "reviewer_a": {"score": None, "justification": ""},
        "reviewer_b": {"score": None, "justification": ""},
        "debate_triggered": False, "score_delta": 0, "resolution_method": "error",
        "survey_injected": False, "novelty_injected": False,
    }

    # Stage 1: extract paper facts
    summary: PaperSummary | None = None
    try:
        summary = extract_paper_facts(client, title, text)
    except Exception as e:
        if on_progress:
            on_progress("⚙️ Paper extraction", None, f"failed: {e}")

    # Stage 1b: generate reviewer personas
    personas = None
    if summary is not None:
        try:
            personas = generate_reviewer_personas(client, title, summary)
        except Exception:
            pass

    # Partition dimensions: augmented dims wait for novelty; others run immediately
    augmented_dims = [d for d in dimensions if enable_novelty and d in _ALL_AUGMENTED_DIMS]
    other_dims = [d for d in dimensions if d not in augmented_dims]

    review: dict[str, Any] = {}
    survey_result: dict | None = None
    novelty_report: dict | None = None

    max_workers = max(len(other_dims) + 2, 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_futures: dict[Future, str] = {
            executor.submit(
                score_dimension_with_debate_augmented,
                client, title, summary, dim, personas, None, None,
            ): dim
            for dim in other_dims
        }
        if enable_novelty:
            all_futures[
                executor.submit(_run_lit_and_novelty, api_key, base_url, text, paper_year, s2_key)
            ] = _NOVELTY_LABEL

        for future in as_completed(all_futures):
            label = all_futures[future]

            if label == _NOVELTY_LABEL:
                try:
                    survey_result, novelty_report = future.result()
                    n_papers = len((survey_result or {}).get("paper_pool", []))
                    claims = (novelty_report or {}).get("claims", [])
                    n_novel = sum(1 for c in claims if c.get("is_novel"))
                    detail = f"{n_papers} papers · {n_novel}/{len(claims)} novel claims"
                except Exception as e:
                    survey_result = {"error": str(e)}
                    detail = f"failed: {e}"
                if on_progress:
                    on_progress("📚 Literature Survey + Novelty", None, detail)
            else:
                try:
                    review[label] = future.result()
                    score = review[label].get("score")
                    debated = review[label].get("debate_triggered", False)
                    detail = "🔥 debated" if debated else "✅ consensus"
                except Exception as e:
                    review[label] = _error_result(e)
                    score = None
                    detail = "error"
                if on_progress:
                    on_progress(label, score, detail)

        # Augmented dims run after novelty chain resolves, in parallel
        if augmented_dims:
            aug_futures: dict[Future, str] = {
                executor.submit(
                    score_dimension_with_debate_augmented,
                    client, title, summary, dim, personas, survey_result, novelty_report,
                ): dim
                for dim in augmented_dims
            }
            for future in as_completed(aug_futures):
                dim = aug_futures[future]
                try:
                    review[dim] = future.result()
                    score = review[dim].get("score")
                    debated = review[dim].get("debate_triggered", False)
                    inj = []
                    if review[dim].get("survey_injected"):
                        inj.append("survey")
                    if review[dim].get("novelty_injected"):
                        inj.append("novelty")
                    detail = ("🔥 debated" if debated else "✅ consensus")
                    if inj:
                        detail += f" +{','.join(inj)}"
                except Exception as e:
                    review[dim] = _error_result(e)
                    score, detail = None, "error"
                if on_progress:
                    on_progress(f"{dim} ✓", score, detail)

    return {
        "review_results": review,
        "survey_result": survey_result,
        "novelty_report": novelty_report,
        "title": title,
    }


# ── PDF rendering with claim highlights ───────────────────────────────────────

def _search_claim_rects(page: Any, claim: str) -> list:
    """
    Try progressively shorter phrase windows until a match is found.
    Returns the list of rects from the first successful match.
    """
    import fitz
    words = claim.split()
    # Try windows from 8 words down to 4, stepping by 2 each time
    for window in [8, 6, 5, 4]:
        if len(words) < window:
            continue
        for start in range(0, len(words) - window + 1, max(1, window // 2)):
            phrase = " ".join(words[start : start + window])
            if len(phrase) < 12:
                continue
            rects = page.search_for(phrase, flags=fitz.TEXT_INHIBIT_SPACES)
            if rects:
                return rects
            # Also try without the flag (different PDF encodings)
            rects = page.search_for(phrase)
            if rects:
                return rects
    return []


def render_pdf_with_highlights(pdf_bytes: bytes, claims: list[dict]) -> tuple[bytes, list[bytes]]:
    """Returns (annotated_pdf_bytes, page_png_list). Green = novel, orange = prior-work."""
    try:
        import fitz
    except ImportError:
        return pdf_bytes, []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for claim_data in claims:
        text = claim_data.get("claim", "")
        color = (0.18, 0.80, 0.44) if claim_data.get("is_novel") else (0.95, 0.61, 0.07)
        for page in doc:
            rects = _search_claim_rects(page, text)
            for rect in rects:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                annot.update()

    annotated = doc.tobytes()
    mat = fitz.Matrix(1.5, 1.5)
    pages = [doc[i].get_pixmap(matrix=mat).tobytes("png") for i in range(len(doc))]
    doc.close()
    return annotated, pages


def render_pdf_pages(pdf_bytes: bytes) -> list[bytes]:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(1.5, 1.5)
        pages = [doc[i].get_pixmap(matrix=mat).tobytes("png") for i in range(len(doc))]
        doc.close()
        return pages
    except ImportError:
        return []


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("📄 Paper Judge")

# ── Paper Input ───────────────────────────────────────────────────────────────
tab_pdf, tab_text = st.tabs(["📎 Upload PDF", "📋 Paste text"])
with tab_pdf:
    uploaded = st.file_uploader("Upload paper PDF", type=["pdf"], label_visibility="collapsed")
with tab_text:
    pasted_title = st.text_input("Paper title", placeholder="Enter paper title")
    pasted_text = st.text_area("Paste paper text here", height=250)

title, text, pdf_bytes_raw, source_ready = "", "", None, False
if uploaded:
    raw = uploaded.read()
    try:
        title, text = extract_pdf_text(raw, uploaded.name)
        pdf_bytes_raw = raw
        source_ready = bool(text.strip())
    except RuntimeError as e:
        st.error(str(e))
elif pasted_text.strip():
    title, text = pasted_title or "Untitled", pasted_text
    source_ready = True

if source_ready:
    col_t, col_w = st.columns([3, 1])
    with col_t:
        title = st.text_input("Paper title (auto-detected, editable)", value=title)
    with col_w:
        st.metric("~Words", f"{len(text.split()):,}")

# ── Run button ────────────────────────────────────────────────────────────────
if source_ready and "pipeline_result" not in st.session_state:
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it to your .env file.")
        st.stop()
    elif not selected_dims:
        st.warning("Select at least one dimension in the sidebar.")
    else:
        # hint = (
        #     "Non-ORIGINALITY dimensions and the literature survey run in parallel. "
        #     "ORIGINALITY runs after novelty is ready."
        #     if enable_lit_novelty
        #     else "All dimensions run in parallel."
        # )
        # st.caption(hint)

        if st.button("🚀 Run Full Review", type="primary", use_container_width=True):
            paper_year: int | None = None
            if enable_lit_novelty:
                try:
                    paper_year = int(paper_year_input) if paper_year_input.strip() else datetime.now().year
                except ValueError:
                    paper_year = datetime.now().year

            client = make_client(api_key, base_url)
            s2_key = s2_api_key.strip() if enable_lit_novelty else None

            total = len(selected_dims) + (1 if enable_lit_novelty else 0)
            _counter = [0]  # mutable container so the closure can mutate it
            prog = st.progress(0.0, text="Starting pipeline…")

            with st.status("Running pipeline…", expanded=True) as status:
                def on_progress(label: str, score: int | None, detail: str) -> None:
                    _counter[0] += 1
                    if score is not None:
                        status.write(f"**{label}** — {score}/5 &nbsp; {detail}")
                    else:
                        status.write(f"**{label}** — {detail}")
                    prog.progress(
                        min(_counter[0] / total, 1.0),
                        text=f"Completed {_counter[0]}/{total}",
                    )

                try:
                    result = run_full_pipeline(
                        client=client,
                        api_key=api_key,
                        base_url=base_url,
                        title=title,
                        text=text,
                        dimensions=selected_dims,
                        enable_novelty=enable_lit_novelty,
                        paper_year=paper_year,
                        s2_key=s2_key,
                        on_progress=on_progress,
                    )
                    status.update(label="✅ Review complete!", state="complete", expanded=False)
                    prog.empty()
                    st.session_state["pipeline_result"] = result
                    st.session_state["pdf_bytes"] = pdf_bytes_raw
                    st.rerun()
                except Exception as e:
                    status.update(label="Pipeline failed", state="error")
                    st.error(f"Pipeline failed: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if "pipeline_result" in st.session_state:
    pr = st.session_state["pipeline_result"]
    results = pr["review_results"]
    stored_title = pr.get("title", title)
    survey_result = pr.get("survey_result")
    novelty_report = pr.get("novelty_report")
    stored_pdf = st.session_state.get("pdf_bytes")

    st.divider()
    st.subheader(f"📊 Results — *{stored_title}*")

    # ── Summary metrics ───────────────────────────────────────────────────────
    valid = {d: r["score"] for d, r in results.items() if r.get("score")}
    if valid:
        avg = sum(valid.values()) / len(valid)
        n_deb = sum(1 for r in results.values() if r.get("debate_triggered"))
        n_novel = ""
        if novelty_report:
            claims = novelty_report.get("claims", [])
            nv = sum(1 for c in claims if c.get("is_novel"))
            n_novel = f"{nv}/{len(claims)} claims novel"

        cols = st.columns(4 if n_novel else 3)
        cols[0].metric("Average score", f"{avg:.2f} / 5")
        cols[1].metric("Dimensions scored", len(valid))
        cols[2].metric("Referee triggered", f"{n_deb} / {len(valid)}")
        if n_novel:
            cols[3].metric("Novelty (claims)", n_novel)

        st.plotly_chart(radar_chart(valid), use_container_width=True)

    # ── Novelty summary (collapsible) ─────────────────────────────────────────
    if novelty_report:
        claims = novelty_report.get("claims", [])
        with st.expander(f"🧬 Novelty Analysis — {sum(1 for c in claims if c.get('is_novel'))}/{len(claims)} novel claims"):
            for c in claims:
                icon = "🟢" if c.get("is_novel") else "🔴"
                score = c.get("novelty_score", 0)
                bm = c.get("best_match_title") or ""
                st.markdown(
                    f"{icon} **{score:.2f}** — {c['claim'][:130]}  \n"
                    f"<small>Closest prior work: {bm[:90]}</small>",
                    unsafe_allow_html=True,
                )

    if survey_result and not survey_result.get("error"):
        pool = survey_result.get("paper_pool", [])
        with st.expander(f"📚 Literature Survey — {len(pool)} papers found"):
            for p in pool[:20]:
                yr = f"({p.get('year', '?')})" if p.get("year") else ""
                st.markdown(
                    f"- **{p.get('title', '?')}** {yr}  score {p.get('final_score', 0):.2f}"
                )
            if len(pool) > 20:
                st.caption(f"… and {len(pool) - 20} more.")

    # ── Dimension breakdown ───────────────────────────────────────────────────
    st.subheader("🔍 Dimension Breakdown")
    for dim, result in results.items():
        score = result.get("score")
        if score is None:
            st.error(f"**{dim}** — {result.get('justification', 'Error')}")
            continue
        debated = result.get("debate_triggered", False)
        delta = result.get("score_delta", 0)
        badge = "🔥 Referee" if debated else "✅ Consensus"
        inj_badges = []
        if result.get("survey_injected"):
            inj_badges.append("📚")
        if result.get("novelty_injected"):
            inj_badges.append("🧬")
        novelty_badge = (" " + "".join(inj_badges)) if inj_badges else ""

        with st.expander(
            f"**{dim}**{novelty_badge} — {'⭐'*score}{'☆'*(5-score)} ({score}/5) | {badge}",
            expanded=(score <= 2),
        ):
            st.info(result["justification"])
            if debated:
                st.warning(f"Reviewers disagreed by **{delta} points** — referee resolved.")
            ra, rb = result["reviewer_a"], result["reviewer_b"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Reviewer A — {ra['score']}/5**")
                st.caption(ra["justification"])
            with c2:
                st.markdown(f"**Reviewer B — {rb['score']}/5**")
                st.caption(rb["justification"])

    # ── Downloads ─────────────────────────────────────────────────────────────
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Download review JSON",
            data=json.dumps(results, indent=2),
            file_name=f"{stored_title.replace(' ', '_')[:40]}_review.json",
            mime="application/json",
            use_container_width=True,
        )
    if novelty_report:
        with dl_col2:
            st.download_button(
                "⬇️ Download novelty JSON",
                data=json.dumps(novelty_report, indent=2),
                file_name="novelty_report.json",
                mime="application/json",
                use_container_width=True,
            )

    # ── PDF Viewer ────────────────────────────────────────────────────────────
    if stored_pdf:
        st.divider()
        st.subheader("📑 PDF Viewer")

        if novelty_report and novelty_report.get("claims"):
            st.caption("🟢 Green = novel claim   🟠 Orange = overlaps with prior work")
            tab_hl, tab_orig = st.tabs(["Highlighted", "Original"])

            with tab_hl:
                claims = novelty_report["claims"]
                annotated_bytes, hl_pages = render_pdf_with_highlights(stored_pdf, claims)
                if hl_pages:
                    for i, img in enumerate(hl_pages):
                        st.image(img, caption=f"Page {i+1}", use_container_width=True)
                else:
                    st.info("Install `pymupdf` for inline rendering.")
                st.download_button(
                    "⬇️ Download highlighted PDF",
                    data=annotated_bytes,
                    file_name="highlighted_claims.pdf",
                    mime="application/pdf",
                )

            with tab_orig:
                orig_pages = render_pdf_pages(stored_pdf)
                if orig_pages:
                    for i, img in enumerate(orig_pages):
                        st.image(img, caption=f"Page {i+1}", use_container_width=True)
                else:
                    b64 = base64.b64encode(stored_pdf).decode()
                    st.components.v1.html(
                        f'<embed src="data:application/pdf;base64,{b64}" '
                        f'width="100%" height="800px" type="application/pdf">',
                        height=820,
                    )
        else:
            pages = render_pdf_pages(stored_pdf)
            if pages:
                for i, img in enumerate(pages):
                    st.image(img, caption=f"Page {i+1}", use_container_width=True)
            else:
                b64 = base64.b64encode(stored_pdf).decode()
                st.components.v1.html(
                    f'<embed src="data:application/pdf;base64,{b64}" '
                    f'width="100%" height="800px" type="application/pdf">',
                    height=820,
                )

elif not source_ready:
    st.info("Upload a PDF or paste paper text above to get started.")
