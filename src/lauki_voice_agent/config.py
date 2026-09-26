import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LiveKit Configuration[cite: 1]
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    # LLM Configuration[cite: 1]
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Deepgram Configuration
    deepgram_api_key: str = ""

    # Document Ingestion[cite: 1]
    docs_dir: str = "docs"

    # Embedding & Vector Store[cite: 1]
    embedding_model: str = "all-MiniLM-L6-v2"
    qdrant_path: str = "./data/qdrant"
    collection_name: str = "telecom_docs"
    chunk_size: int = 500

    # Voice Configuration
    stt_model: str = "nova-3"
    stt_language: str = "en-US"  # or "hi" if serving Indian languages, but not "multi"
    tts_model: str = "aura-asteria-en"


settings = Settings()

# Set environment variables for LiveKit, Groq, and Deepgram[cite: 1]
if settings.livekit_url:
    os.environ.setdefault("LIVEKIT_URL", settings.livekit_url)
if settings.livekit_api_key:
    os.environ.setdefault("LIVEKIT_API_KEY", settings.livekit_api_key)
if settings.livekit_api_secret:
    os.environ.setdefault("LIVEKIT_API_SECRET", settings.livekit_api_secret)
if settings.groq_api_key:
    os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
if settings.deepgram_api_key:
    os.environ.setdefault("DEEPGRAM_API_KEY", settings.deepgram_api_key)