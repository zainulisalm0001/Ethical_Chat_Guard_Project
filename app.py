import streamlit as st
from services.llm_openai import generate_reply
from services.detector import assess
from utils.helpers import render_highlighted
from services.sbert_lr import predict_proba
from utils.config import load_threshold

st.set_page_config(page_title="Ethical Chat Guard", layout="wide")

CHAT_HEIGHT = 760
PANEL_HEIGHT = 740

st.markdown(
    f"""
    <style>
    html, body {{
        overflow: hidden;
    }}

    .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 0.5rem;
        max-width: 1280px;
    }}

    html, body, [class*="css"] {{
        font-size: 16px;
    }}

    h1 {{
        font-size: 2.1rem !important;
        margin-bottom: 0.2rem !important;
    }}

    .subtitle {{
        opacity: 0.75;
        margin-top: -6px;
        margin-bottom: 8px;
    }}

    .chatbox-title {{
        font-size: 1.05rem;
        font-weight: 650;
        opacity: 0.9;
        margin: 0.0rem 0 0.45rem 0;
    }}

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li {{
        font-size: 1.05rem;
        line-height: 1.55;
    }}

    div[data-testid="stChatMessage"] {{
        padding-top: 0.15rem;
        padding-bottom: 0.15rem;
    }}

    div[data-testid="stChatMessage"][data-role="user"] {{
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-end !important;
        gap: 10px;
    }}

    div[data-testid="stChatMessage"][data-role="user"] img,
    div[data-testid="stChatMessage"][data-role="user"] .stChatMessageAvatar {{
        order: 2;
    }}

    div[data-testid="stChatInput"] {{
        max-width: 880px;
        margin-left: auto;
        margin-right: auto;
    }}


    div[data-testid="stChatInput"] textarea {{
        font-size: 1.05rem;
        padding: 12px 14px;
    }}
    
    div[data-testid="stChatMessage"][data-role="user"] .stMarkdown {{
        order: 1;
        text-align: right;
        margin-left: auto;
    }}

    div[data-testid="stChatMessage"][data-role="assistant"] {{
        justify-content: flex-start !important;
    }}
    
    /* Center & reduce width of chat input */
    div[data-testid="stChatInput"] {{
        max-width: 960px;
        margin-left: auto;
        margin-right: auto;
    }}

    /* Reduce input height slightly */
    div[data-testid="stChatInput"] textarea {{
        font-size: 1.05rem;
        padding: 12px 14px;
    }}
    
    .pill {{
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 650;
        letter-spacing: 0.02em;
        margin-top: 0.25rem;
    }}

    .pill-green {{
        background: rgba(46, 204, 113, 0.18);
        border: 1px solid rgba(46, 204, 113, 0.35);
    }}

    .pill-yellow {{
        background: rgba(241, 196, 15, 0.18);
        border: 1px solid rgba(241, 196, 15, 0.35);
    }}

    .pill-red {{
        background: rgba(231, 76, 60, 0.18);
        border: 1px solid rgba(231, 76, 60, 0.35);
    }}

    .kv {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.3rem 0.8rem;
        font-size: 0.92rem;
        opacity: 0.92;
        margin-top: 0.35rem;
    }}

    .kv b {{
        opacity: 0.9;
        font-weight: 650;
    }}

    .section-title {{
        font-size: 0.95rem;
        font-weight: 650;
        opacity: 0.9;
        margin-top: 0.55rem;
        margin-bottom: 0.25rem;
    }}

    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stChatInput"]) {{
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        padding: 0.7rem 1.2rem 0.9rem 1.2rem;
        background: rgba(0,0,0,0.0);
        z-index: 999;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

if "audits" not in st.session_state:
    st.session_state.audits = []

if "mode" not in st.session_state:
    st.session_state.mode = "Balanced"

header_left, header_right = st.columns([2.4, 1], gap="large")

with header_left:
    st.title("Ethical Chat Guard")
    st.markdown(
        '<div class="subtitle">Chat with an assistant and audit responses for coercive language in real time.</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.session_state.mode = st.selectbox(
        "Risk sensitivity",
        ["Conservative", "Balanced", "Aggressive"],
        index=["Conservative", "Balanced", "Aggressive"].index(st.session_state.mode),
    )
    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        st.session_state.audits = []
        st.rerun()

left, right = st.columns([2, 1], gap="large")


with right:
    st.subheader("Risk Panel")
    panel = st.container(height=PANEL_HEIGHT, border=True)

    with panel:
        if not st.session_state.audits:
            st.write("Send a message to see risk analysis.")
        else:
            last = st.session_state.audits[-1]
            label = last.label.upper()

            if label == "RED":
                pill_class = "pill pill-red"
                label_text = "High risk: response contains coercive pressure patterns."
            elif label == "YELLOW":
                pill_class = "pill pill-yellow"
                label_text = "Caution: some pressure cues detected. Review tone."
            else:
                pill_class = "pill pill-green"
                label_text = "Low risk: response appears neutral and non-coercive."

            st.metric("Risk score", f"{last.score}/100")
            st.markdown(f'<span class="{pill_class}">{label}</span>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">What this means</div>', unsafe_allow_html=True)
            st.write(label_text)

            st.markdown('<div class="section-title">Why this label</div>', unsafe_allow_html=True)
            st.write(last.explanation)

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