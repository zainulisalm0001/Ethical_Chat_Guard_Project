import pandas as pd
import streamlit as st

from services.detector import assess
from services.sbert_lr import predict_proba
from utils.config import load_threshold

st.set_page_config(page_title="Quick Risk Check - Ethical Chat Guard", layout="wide")

# -------------------- CSS --------------------
st.markdown(
    """
    <style>
    html, body { overflow: hidden; }

    .block-container {
        padding-top: 0.9rem;
        padding-bottom: 0.7rem;
        max-width: 1280px;
    }

    h1 { font-size: 2.15rem !important; margin-bottom: 0.1rem !important; }
    .subtitle { opacity: 0.75; margin-top: -6px; margin-bottom: 12px; }

    .card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
    }

    .section-title {
        font-size: 1.0rem;
        font-weight: 650;
        opacity: 0.92;
        margin: 0.0rem 0 0.55rem 0;
    }

    .metric-big {
        font-size: 2.5rem;
        font-weight: 780;
        line-height: 1.0;
        margin: 0.2rem 0 0.4rem 0;
    }

    .muted { opacity: 0.78; }

    .pill {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 650;
        letter-spacing: 0.02em;
        margin-left: 0.4rem;
    }

    .pill-green { background: rgba(46, 204, 113, 0.18); border: 1px solid rgba(46, 204, 113, 0.35); }
    .pill-yellow { background: rgba(241, 196, 15, 0.18); border: 1px solid rgba(241, 196, 15, 0.35); }
    .pill-red { background: rgba(231, 76, 60, 0.18); border: 1px solid rgba(231, 76, 60, 0.35); }

    .label-meaning {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 10px 12px;
        background: rgba(255,255,255,0.02);
        margin-top: 10px;
        font-size: 0.95rem;
        line-height: 1.45;
        opacity: 0.92;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Title --------------------
st.title("Quick Risk Checker")
st.markdown(
    '<div class="subtitle">Analyze a single text or a CSV batch for coercive language risk.</div>',
    unsafe_allow_html=True,
)

# -------------------- Helpers --------------------
def _mode_threshold(base_th: float, mode: str) -> float:
    if mode == "Conservative":
        return min(0.95, base_th + 0.10)
    if mode == "Aggressive":
        return max(0.01, base_th - 0.10)
    return base_th

def _label_from_score(score: int) -> str:
    if score <= 25:
        return "GREEN"
    if score <= 40:
        return "YELLOW"
    return "RED"

def _pill(label: str) -> str:
    label = label.upper()
    if label == "RED":
        return '<span class="pill pill-red">RISK</span>'
    if label == "YELLOW":
        return '<span class="pill pill-yellow">CAUTION</span>'
    return '<span class="pill pill-green">SAFE</span>'

def _label_meaning(label: str) -> str:
    if label == "RED":
        return "High risk: strong coercive pressure signals detected."
    if label == "YELLOW":
        return "Caution: some pressure cues detected; review tone and framing."
    return "Safe: response appears neutral and non-coercive."

# -------------------- State --------------------
if "ba_mode" not in st.session_state:
    st.session_state.ba_mode = "Balanced"

if "ba_last" not in st.session_state:
    st.session_state.ba_last = None

if "single_csv" not in st.session_state:
    st.session_state.single_csv = None

if "batch_csv" not in st.session_state:
    st.session_state.batch_csv = None

# -------------------- Layout --------------------
left, right = st.columns([2, 1], gap="large")

# -------------------- Right Panel --------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)

    st.session_state.ba_mode = st.selectbox(
        "Risk sensitivity",
        ["Conservative", "Balanced", "Aggressive"],
        index=["Conservative", "Balanced", "Aggressive"].index(st.session_state.ba_mode),
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top: 12px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

    last = st.session_state.ba_last
    if last is None:
        st.markdown('<div class="muted">Run an analysis to see results.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div class="muted">Risk score</div>
            <div class="metric-big">{last["score"]}/100 {_pill(last["label"])}</div>
            <div class="label-meaning">{_label_meaning(last["label"])}</div>
            <div class="section-title" style="margin-top: 0.9rem;">Why this label</div>
            <div class="muted">{last["explanation"]}</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Tabs --------------------
with left:
    tab1, tab2 = st.tabs(["Single Text", "CSV Batch"])

    # =======================
    # Single Text Tab
    # =======================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Single text analysis</div>', unsafe_allow_html=True)

    reply_text = st.text_area(
        "Text to analyze (LLM response)",
        height=260,
        placeholder="Paste assistant response here..."
    )

    col1, col2 = st.columns(2)
    run_one = col1.button("Analyze text", use_container_width=True)
    clear_one = col2.button("Clear", use_container_width=True)

    if clear_one:
        st.session_state.ba_last = None
        st.session_state.single_csv = None
        st.rerun()

    if run_one and reply_text.strip():
        mp = float(predict_proba(reply_text))
        th = _mode_threshold(load_threshold(), st.session_state.ba_mode)

        a = assess("", reply_text, model_proba=mp, model_threshold=th)

        label = _label_from_score(a.score)

        st.session_state.ba_last = {
            "score": int(a.score),
            "label": label,
            "explanation": a.explanation,
        }

        single_df = pd.DataFrame([{
            "risk_score": int(a.score),
            "label": label,
            "label_meaning": _label_meaning(label),
            "explanation": a.explanation,
        }])

        st.session_state.single_csv = single_df.to_csv(index=False).encode("utf-8")

        st.success("Analysis complete.")
        st.rerun()  # ✅ forces immediate refresh

    # Download button
    if st.session_state.single_csv:
        st.download_button(
            "Download Single Text Report (CSV)",
            st.session_state.single_csv,
            "single_text_risk_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # =======================
    # CSV Batch Tab
    # =======================
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">CSV batch analysis</div>', unsafe_allow_html=True)

        up = st.file_uploader("Upload CSV", type=["csv"])

        if up:
            df = pd.read_csv(up)

            text_col = st.selectbox("Text column", df.columns)

            max_rows = st.slider("Max rows", 10, min(2000, len(df)), min(300, len(df)))

            run_batch = st.button("Run batch analysis", use_container_width=True)

            if run_batch:
                texts = df.head(max_rows)[text_col].astype(str).tolist()
                th = _mode_threshold(load_threshold(), st.session_state.ba_mode)

                results = []
                for i, t in enumerate(texts):
                    mp = float(predict_proba(t))
                    a = assess("", t, model_proba=mp, model_threshold=th)

                    label = _label_from_score(a.score)

                    results.append({
                        "row_idx": i,
                        "risk_score": int(a.score),
                        "label": label,
                        "label_meaning": _label_meaning(label),
                        "explanation": a.explanation,
                    })

                out = pd.DataFrame(results).sort_values("risk_score", ascending=False)

                st.session_state.batch_csv = out.to_csv(index=False).encode("utf-8")

                st.success("Batch analysis complete.")

        # ✅ CSV DOWNLOAD HERE
        if st.session_state.batch_csv:
            st.download_button(
                "Download Batch Results (CSV)",
                st.session_state.batch_csv,
                "batch_risk_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("</div>", unsafe_allow_html=True)