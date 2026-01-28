import streamlit as st
import pandas as pd
from datetime import datetime

from services.llm_openai import generate_reply, safe_rewrite
from services.detector import assess
from utils.helpers import render_highlighted
from services.sbert_lr import predict_proba
from utils.config import load_threshold

st.set_page_config(page_title="Ethical Chat Guard", layout="wide")

CHAT_HEIGHT = 560
PANEL_HEIGHT = 550

st.markdown(
    """
    <style>

    html, body { overflow: hidden; }

    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 0.5rem;
        max-width: 1280px;
    }

    html, body, [class*="css"] { font-size: 16px; }

    /* Title + subtitle spacing */
    h1 {
        font-size: 2.1rem !important;
        margin-bottom: 2px !important;
    }

    .subtitle {
        opacity: 0.75;
        margin-top: -6px !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
        line-height: 1.3 !important;
    }

    /* Reduce vertical gap after header area */
    div[data-testid="stVerticalBlock"] > div {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }

    div[data-testid="block-container"] > div {
        row-gap: 6px !important;
    }

    div[data-testid="stContainer"] {
        margin-top: 0px !important;
    }

    /* Conversation title */
    .chatbox-title {
        font-size: 1.05rem;
        font-weight: 650;
        opacity: 0.9;
        margin-top: 2px !important;
        margin-bottom: 6px !important;
    }

    /* Chat message typography */
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li {
        font-size: 1.05rem;
        line-height: 1.55;
    }

    div[data-testid="stChatMessage"] {
        padding-top: 0.15rem;
        padding-bottom: 0.15rem;
    }

    /* User bubble right + avatar far right */
    div[data-testid="stChatMessage"][data-role="user"] {
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-end !important;
        gap: 10px;
    }

    div[data-testid="stChatMessage"][data-role="user"] img,
    div[data-testid="stChatMessage"][data-role="user"] .stChatMessageAvatar {
        order: 2 !important;
    }

    div[data-testid="stChatMessage"][data-role="user"] .stMarkdown {
        order: 1 !important;
        text-align: right;
        margin-left: auto;
    }

    div[data-testid="stChatMessage"][data-role="assistant"] {
        justify-content: flex-start !important;
    }

    /* --- CHAT INPUT: fixed bottom + centered half-width --- */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stChatInput"]) {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        padding: 0.7rem 1.2rem 0.9rem 1.2rem;
        background: rgba(0,0,0,0.0);
        z-index: 999;
    }

    div[data-testid="stChatInput"] {
        max-width: 560px;
        width: 50vw;
        margin-left: auto;
        margin-right: auto;
    }

    div[data-testid="stChatInput"] textarea {
        font-size: 1.05rem;
        padding: 12px 14px;
    }

    /* Pills */
    .pill {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 650;
        letter-spacing: 0.02em;
        margin-top: 0.25rem;
    }

    .pill-green { background: rgba(46, 204, 113, 0.18); border: 1px solid rgba(46, 204, 113, 0.35); }
    .pill-yellow { background: rgba(241, 196, 15, 0.18); border: 1px solid rgba(241, 196, 15, 0.35); }
    .pill-red { background: rgba(231, 76, 60, 0.18); border: 1px solid rgba(231, 76, 60, 0.35); }

    .section-title {
        font-size: 0.95rem;
        font-weight: 650;
        opacity: 0.9;
        margin-top: 0.55rem;
        margin-bottom: 0.25rem;
    }
    
    /* Keep the mode + reset aligned and slightly lower */
.header-controls {
    margin-top: 26px;  /* move both down together */
}

/* Same height for both widgets */
div[data-testid="stSelectbox"] > div,
button {
    min-height: 42px !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- helpers ----------------
def build_session_report(messages, audits):
    rows = []
    cat_totals = {}

    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue

        audit_idx = m.get("audit_idx")
        if audit_idx is None or audit_idx >= len(audits):
            continue

        a = audits[audit_idx]

        user_prompt = ""
        for j in range(i - 1, -1, -1):
            if messages[j].get("role") == "user":
                user_prompt = messages[j].get("content", "")
                break

        for k, v in (a.categories or {}).items():
            cat_totals[k] = cat_totals.get(k, 0) + int(v)

        rows.append(
            {
                "turn_id": int(audit_idx),
                "user_prompt": user_prompt,
                "assistant_reply": m.get("content", ""),
                "risk_score": int(a.score),
                "rule_score": float(a.rule_score),
                "context_score": float(a.context_score),
                "model_proba": None if a.model_proba is None else float(a.model_proba),
                "model_score": None if a.model_score is None else float(a.model_score),
                "explanation": str(a.explanation),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        summary = {
            "avg_risk": None,
            "max_risk": None,
            "riskiest_turn_id": None,
            "top_categories": [],
            "category_totals": cat_totals,
        }
        return summary, df

    avg_risk = float(df["risk_score"].mean())
    max_row = df.loc[df["risk_score"].idxmax()]
    max_risk = int(max_row["risk_score"])
    riskiest_turn_id = int(max_row["turn_id"])

    top_categories = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
    top_categories = [k for k, v in top_categories if v > 0][:3]

    summary = {
        "avg_risk": avg_risk,
        "max_risk": max_risk,
        "riskiest_turn_id": riskiest_turn_id,
        "top_categories": top_categories,
        "category_totals": cat_totals,
    }
    return summary, df


def _latest_assistant_and_context(messages: list[dict]):
    """
    Returns:
      - last_assistant_idx (index in messages list)
      - last_assistant_text
      - last_user_text (nearest preceding user prompt)
    """
    last_assistant_idx = None
    last_assistant_text = None
    last_user_text = ""

    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            last_assistant_text = messages[i].get("content", "")
            # find nearest user before it
            for j in range(i - 1, -1, -1):
                if messages[j].get("role") == "user":
                    last_user_text = messages[j].get("content", "")
                    break
            break

    return last_assistant_idx, last_assistant_text, last_user_text


# ---------------- state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

if "audits" not in st.session_state:
    st.session_state.audits = []

if "mode" not in st.session_state:
    st.session_state.mode = "Balanced"

# safe rewrite state
if "safe_rewrite_text" not in st.session_state:
    st.session_state.safe_rewrite_text = None

if "safe_rewrite_source_turn" not in st.session_state:
    st.session_state.safe_rewrite_source_turn = None


# ---------------- header ----------------


# ---------------- header (SOLID INLINE CONTROLS) ----------------
h1, _, h_mode, h_reset = st.columns([2.6, 0.15, 0.7, 0.55], gap="small")

with h1:
    st.title("Ethical Chat Guard")
    st.markdown(
        '<div class="subtitle">Chat with an assistant and audit responses for coercive language in real time.</div>',
        unsafe_allow_html=True,
    )

with h_mode:
    st.markdown('<div class="header-controls">', unsafe_allow_html=True)
    st.session_state.mode = st.selectbox(
        " ",
        ["Conservative", "Balanced", "Aggressive"],
        index=["Conservative", "Balanced", "Aggressive"].index(st.session_state.mode),
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with h_reset:
    st.markdown('<div class="header-controls">', unsafe_allow_html=True)
    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        st.session_state.audits = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- main layout ----------------
left, right = st.columns([2, 1], gap="large")

with right:
    st.subheader("Risk Panel")
    panel = st.container(height=PANEL_HEIGHT, border=True)

    with panel:
        if not st.session_state.audits:
            st.write("Send a message to see risk analysis.")
        else:
            last = st.session_state.audits[-1]
            score = int(last.score)

            if score <= 25:
                label = "Safe"
                pill_class = "pill pill-green"
                label_text = "Safe: response appears neutral and non-coercive."
            elif score <= 40:
                label = "Caution"
                pill_class = "pill pill-yellow"
                label_text = "Caution: some pressure cues detected. Review tone."
            else:
                label = "Risky"
                pill_class = "pill pill-red"
                label_text = "High risk: response contains strong coercive pressure patterns."

            st.metric("Risk score", f"{score}/100")
            st.markdown(f'<span class="{pill_class}">{label}</span>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">What this means</div>', unsafe_allow_html=True)
            st.write(label_text)

            st.markdown('<div class="section-title">Why this label</div>', unsafe_allow_html=True)
            st.write(last.explanation)

        # ---------------- NEW FEATURE: SAFE REWRITE ----------------
        st.markdown('<div class="section-title">Safe rewrite suggestion</div>', unsafe_allow_html=True)

        last_a_idx, last_a_text, last_user_text = _latest_assistant_and_context(st.session_state.messages)

        if last_a_text is None or not last_a_text.strip():
            st.write("No assistant reply yet to rewrite.")
        else:
            colx, coly = st.columns([1, 1], gap="small")

            with colx:
                run_rewrite = st.button("Safe rewrite", use_container_width=True)

            with coly:
                clear_rewrite = st.button("Discard rewrite", use_container_width=True)

            if clear_rewrite:
                st.session_state.safe_rewrite_text = None
                st.session_state.safe_rewrite_source_turn = None
                st.rerun()

            if run_rewrite:
                with st.spinner("Generating a safer, non-coercive rewrite..."):
                    rewritten = safe_rewrite(last_a_text, user_context=last_user_text)
                st.session_state.safe_rewrite_text = rewritten
                st.session_state.safe_rewrite_source_turn = last_a_idx
                st.rerun()

            if st.session_state.safe_rewrite_text:
                st.text_area(
                    "Rewrite preview",
                    value=st.session_state.safe_rewrite_text,
                    height=120,
                    label_visibility="collapsed",
                )

                add_btn = st.button("Add rewrite to conversation", use_container_width=True)
                if add_btn:
                    # Audit the rewrite too (so it updates the panel & is included in session CSV)
                    rewrite_text = st.session_state.safe_rewrite_text
                    th = load_threshold()
                    p = predict_proba(rewrite_text)

                    if st.session_state.mode == "Conservative":
                        model_threshold = min(0.95, th + 0.10)
                    elif st.session_state.mode == "Aggressive":
                        model_threshold = max(0.01, th - 0.10)
                    else:
                        model_threshold = th

                    # Use last_user_text as the "user message" context for assessment
                    a2 = assess(last_user_text, rewrite_text, model_proba=p, model_threshold=model_threshold)

                    audit_idx = len(st.session_state.audits)
                    st.session_state.audits.append(a2)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": rewrite_text,
                            "audit_idx": audit_idx,
                        }
                    )

                    st.session_state.safe_rewrite_text = None
                    st.session_state.safe_rewrite_source_turn = None
                    st.rerun()

        # ---------------- SESSION SUMMARY REPORT (your existing feature) ----------------
        st.markdown('<div class="section-title">Session summary report</div>', unsafe_allow_html=True)

        summary, report_df = build_session_report(st.session_state.messages, st.session_state.audits)

        if report_df.empty:
            st.write("No session data yet. Send at least one message.")
        else:
            csv_bytes = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Session Report (CSV)",
                data=csv_bytes,
                file_name=f"session_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

with left:
    st.markdown('<div class="chatbox-title">Conversation</div>', unsafe_allow_html=True)
    chat_box = st.container(height=CHAT_HEIGHT, border=True)

    with chat_box:
        for m in st.session_state.messages:
            if m["role"] not in ("user", "assistant"):
                continue

            with st.chat_message(m["role"]):
                if m["role"] == "assistant":
                    idx = m.get("audit_idx", None)
                    if idx is not None and 0 <= idx < len(st.session_state.audits):
                        a = st.session_state.audits[idx]
                        html_text = render_highlighted(m["content"], a.spans)
                        st.markdown(html_text, unsafe_allow_html=True)
                    else:
                        st.markdown(m["content"])
                else:
                    st.markdown(m["content"])

# ---------------- input + run ----------------
user_msg = st.chat_input("Type your message")

if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})

    reply = generate_reply(st.session_state.messages)

    th = load_threshold()
    p = predict_proba(reply)

    if st.session_state.mode == "Conservative":
        model_threshold = min(0.95, th + 0.10)
    elif st.session_state.mode == "Aggressive":
        model_threshold = max(0.01, th - 0.10)
    else:
        model_threshold = th

    a = assess(user_msg, reply, model_proba=p, model_threshold=model_threshold)

    audit_idx = len(st.session_state.audits)
    st.session_state.audits.append(a)
    st.session_state.messages.append({"role": "assistant", "content": reply, "audit_idx": audit_idx})
    st.rerun()