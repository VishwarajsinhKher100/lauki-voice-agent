from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

from src.lauki_voice_agent.config import settings
from src.lauki_voice_agent.tools import lauki_agent


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Transcribe recorded microphone audio via Deepgram REST API."""
    api_key = settings.deepgram_api_key or os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured in .env")

    url = (
        f"https://api.deepgram.com/v1/listen"
        f"?model={settings.stt_model}&language={settings.stt_language}&smart_format=true"
    )
    req = urllib.request.Request(
        url,
        data=audio_bytes,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": mime_type,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (
        data.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
        .strip()
    )


def synthesize_speech(text: str) -> bytes | None:
    """Convert assistant text response into speech audio via Deepgram TTS."""
    api_key = settings.deepgram_api_key or os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key or not text:
        return None

    url = f"https://api.deepgram.com/v1/speak?model={settings.tts_model}"
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except Exception:
        return None


def run_agent(prompt: str, history: list[dict]) -> tuple[str, str]:
    """Run the LangGraph voice support agent and collect any tool source context."""
    lc_messages = []
    for msg in history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
    lc_messages.append(HumanMessage(content=prompt))

    result = lauki_agent.invoke({"messages": lc_messages})
    messages = result.get("messages", [])

    answer = ""
    tool_outputs = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_outputs.append(str(msg.content))
    if messages:
        answer = str(messages[-1].content)

    return answer, "\n\n---\n\n".join(tool_outputs)


def plain_text(text: str) -> str:
    """Remove Markdown presentation syntax while preserving the response text."""
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(`{1,3})(.*?)\1", r"\2", text, flags=re.DOTALL)
    text = text.replace("**", "").replace("__", "")
    return text.strip()


st.set_page_config(
    page_title="Lauki Phones · Voice Q&A",
    page_icon=":material/mic:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.html(
    """
    <style>
    :root {
        color-scheme: dark;
        --bg: #070908;
        --panel: #101512;
        --panel-2: #161d19;
        --border: rgba(255, 255, 255, 0.09);
        --text: #f3f6f4;
        --muted: #9aa69f;
        --accent: #f4c95d;
        --accent-soft: rgba(244, 201, 93, 0.14);
        --live: #34c77b;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background:
            radial-gradient(ellipse 70% 35% at 50% 0%, rgba(244, 201, 93, 0.09), transparent 70%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background: var(--panel);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--border);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"],
    footer {
        display: none;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 720px;
        padding-top: 2rem;
        padding-bottom: 165px;
    }

    .voice-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding-bottom: 1.1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.4rem;
    }

    .voice-title {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: var(--text);
    }

    .voice-subtitle {
        font-size: 0.9rem;
        color: var(--muted);
        margin: 0.25rem 0 0;
    }

    .voice-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(52, 199, 123, 0.12);
        border: 1px solid rgba(52, 199, 123, 0.28);
        color: #b7f2d2;
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .voice-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--live);
        box-shadow: 0 0 8px var(--live);
    }

    .empty-voice-state {
        text-align: center;
        padding: 3rem 1.5rem;
        margin: 1.5rem 0;
        border: 1px dashed rgba(244, 201, 93, 0.25);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.02);
    }

    .empty-voice-state h3 {
        margin: 0 0 0.4rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
    }

    .empty-voice-state p {
        margin: 0 auto;
        max-width: 420px;
        font-size: 0.9rem;
        color: var(--muted);
        line-height: 1.5;
    }

    .turn-card {
        border-radius: 16px;
        padding: 0.95rem 1.15rem;
        margin-bottom: 0.85rem;
        border: 1px solid var(--border);
    }

    .turn-user {
        background: var(--panel-2);
        border-left: 3px solid var(--accent);
    }

    .turn-assistant {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid var(--live);
    }

    .turn-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }

    .turn-user .turn-label {
        color: var(--accent);
    }

    .turn-assistant .turn-label {
        color: #7ee2ae;
    }

    .turn-body {
        font-size: 0.98rem;
        line-height: 1.55;
        color: var(--text);
        white-space: pre-wrap;
        word-break: break-word;
    }

    [data-testid="stAudioInput"] {
        position: fixed;
        left: 50%;
        bottom: 18px;
        transform: translateX(-50%);
        width: min(680px, calc(100vw - 28px));
        z-index: 100;
        background: var(--panel-2);
        border: 1px solid rgba(244, 201, 93, 0.32);
        border-radius: 20px;
        padding: 10px 14px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.8);
    }
    </style>
    """,
)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_audio_hash", None)
st.session_state.setdefault("auto_speak", True)

with st.sidebar:
    st.markdown("### Lauki Phones")
    st.caption("Voice-only support assistant")
    st.divider()

    st.session_state.auto_speak = st.toggle(
        "Auto-play voice replies",
        value=st.session_state.auto_speak,
    )
    st.markdown(f"**Speech-to-Text:** `{settings.stt_model}` (`{settings.stt_language}`)")
    st.markdown(f"**Text-to-Speech:** `{settings.tts_model}`")

    st.divider()
    st.caption("Ask out loud about mobile plans, billing, 5G coverage, or account balance.")

    if st.button("Clear voice history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_audio_hash = None
        st.rerun()

st.markdown(
    """
    <div class="voice-header">
        <div>
            <h1 class="voice-title">Lauki Voice Support</h1>
            <p class="voice-subtitle">Press the microphone below, ask your question, and listen to the reply.</p>
        </div>
        <div class="voice-badge">
            <span class="voice-dot"></span>
            Voice Only
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-voice-state">
            <h3>Tap the microphone to start speaking</h3>
            <p>Ask about plans, pricing, 5G availability, or your account balance. Replies are spoken aloud automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    is_user = message["role"] == "user"
    card_cls = "turn-user" if is_user else "turn-assistant"
    role_label = "You (Voice)" if is_user else "Lauki Assistant"
    safe_text = (
        message["content"]
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f"""
        <div class="turn-card {card_cls}">
            <div class="turn-label">{role_label}</div>
            <div class="turn-body">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not is_user and message.get("audio"):
        st.audio(message["audio"], format="audio/mp3")

audio_value = st.audio_input("Speak your question", label_visibility="collapsed")
spoken_question: str | None = None

if audio_value is not None:
    audio_bytes = audio_value.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with st.spinner("Transcribing audio..."):
            try:
                transcript = transcribe_audio(
                    audio_bytes,
                    mime_type=audio_value.type or "audio/wav",
                )
                if transcript:
                    spoken_question = transcript
                else:
                    st.warning("No speech detected. Record again and speak clearly.")
            except Exception as exc:
                st.error(f"Speech recognition failed: {exc}")

if spoken_question and spoken_question.strip():
    spoken_question = spoken_question.strip()
    prior_history = list(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": spoken_question})
    safe_q = (
        spoken_question.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f"""
        <div class="turn-card turn-user">
            <div class="turn-label">You (Voice)</div>
            <div class="turn-body">{safe_q}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Generating voice response..."):
        try:
            answer, context = run_agent(spoken_question, prior_history)
        except Exception as exc:
            answer = (
                "I couldn't process that request. "
                "Please check your GROQ_API_KEY and Qdrant setup."
            )
            context = ""
            st.error(str(exc))

    answer = plain_text(answer)
    reply_audio = synthesize_speech(answer) if st.session_state.auto_speak else None

    safe_a = (
        answer.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f"""
        <div class="turn-card turn-assistant">
            <div class="turn-label">Lauki Assistant</div>
            <div class="turn-body">{safe_a}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if reply_audio:
        st.audio(reply_audio, format="audio/mp3", autoplay=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": context,
            "audio": reply_audio,
        }
    )