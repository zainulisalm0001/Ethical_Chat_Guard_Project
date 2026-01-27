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
    model_name = model or st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    client = _client()
    safe_messages = _sanitize_messages(chat_messages)
    resp = client.responses.create(
        model=model_name,
        input=safe_messages,
    )
    return resp.output_text