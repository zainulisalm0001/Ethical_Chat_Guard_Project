import streamlit as st
from openai import OpenAI


def _client() -> OpenAI:
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def _sanitize_messages(chat_messages: list[dict]) -> list[dict]:
    cleaned = []
    for m in chat_messages:
        role = m.get("role")
        content = m.get("content")
        if role and content is not None:
            cleaned.append({"role": role, "content": content})
    return cleaned


def generate_reply(chat_messages: list[dict], model: str | None = None) -> str:
    model_name = model or st.secrets.get("OPENAI_MODEL", "gpt-4.1-nano")
    client = _client()
    safe_messages = _sanitize_messages(chat_messages)
    resp = client.responses.create(
        model=model_name,
        input=safe_messages,
    )
    return resp.output_text


def safe_rewrite(
    assistant_text: str,
    user_context: str | None = None,
    model: str | None = None,
) -> str:
    """
    Rewrite the assistant reply to reduce coercive tone while keeping meaning.
    - Removes urgency / pressure / inevitability framing
    - Adds neutral, choice-respecting language
    - Keeps helpfulness and clarity
    """
    model_name = model or st.secrets.get("OPENAI_REWRITE_MODEL", st.secrets.get("OPENAI_MODEL", "gpt-4.1-nano"))
    client = _client()

    context_block = ""
    if user_context and user_context.strip():
        context_block = f"\n\nUser context (what the user asked):\n{user_context.strip()}"

    prompt = f"""
You are an ethics-aware rewriting assistant.

Task:
Rewrite the ASSISTANT reply to be non-coercive and ethically safer, while keeping the same meaning and usefulness.

Rules:
- Remove urgency, pressure, guilt, fear, or manipulative persuasion.
- Avoid “do it now”, “you must”, “don’t miss out”, “this is your only chance”, etc.
- Respect user autonomy (make it clear it’s their choice).
- Keep it concise, professional, and supportive.
- Do NOT add new facts, numbers, or claims. Do not hallucinate.
- Output ONLY the rewritten assistant reply.

ASSISTANT reply to rewrite:
{assistant_text.strip()}
{context_block}
""".strip()

    resp = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": prompt}],
    )
    return resp.output_text