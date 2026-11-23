# Lauki Phones Voice Support

A voice-first customer support demo for Lauki Phones. Record a question in the Streamlit app and receive a spoken response about mobile plans, billing, account usage, and network coverage.

The app uses Deepgram for speech recognition and speech generation, Groq with a LangGraph agent for responses and tool use, and local Qdrant vector search for answers grounded in the PDF documents in `docs/`. Account and network lookups currently use mock data defined in `src/lauki_voice_agent/tools.py`.

## What it can do

- Answer plan and pricing questions using the indexed PDFs.
- Look up sample account balances and usage for mock customer phone numbers.
- Check mock 4G/5G coverage by region.
- Recommend a plan and provide plan details.
- Create a demo support escalation ticket.
- Optionally run a separate LiveKit real-time voice agent.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- A Groq API key
- A Deepgram API key for microphone transcription and spoken replies

## Set up

Clone the repository and install its dependencies:

```bash
git clone https://github.com/VishwarajsinhKher100/lauki-voice-agent.git
cd lauki-voice-agent
uv sync
```

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key

# Only needed for the optional LiveKit agent:
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

Keep API keys private and do not commit `.env`.

## Start the Streamlit app

```bash
uv run streamlit run app.py
```

Open the local URL printed by Streamlit and allow microphone access in your browser. The app transcribes each recording, sends the question to the support agent, and plays a spoken reply when Deepgram TTS is available.

## Index the support documents

The repository includes plan, billing FAQ, and network coverage PDFs in `docs/`. To rebuild the local Qdrant index from those files, run:

```bash
uv run python -m src.lauki_voice_agent.ingest
```

Ingestion uses Docling to parse and chunk the PDFs and Sentence Transformers to create embeddings. It replaces the existing `telecom_docs` collection at the configured Qdrant path, so back up any data you need before rebuilding. The embedding model may be downloaded the first time ingestion runs.

## Run the optional LiveKit agent

Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` in `.env`, then start the agent:

```bash
uv run lauki-voice-agent dev
```

This is a separate LiveKit entry point; it is not required to run the Streamlit app.

## Configuration

Settings can be provided through `.env`. Defaults are defined in `src/lauki_voice_agent/config.py`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | Not set | Required for agent responses |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model |
| `DEEPGRAM_API_KEY` | Not set | Required for transcription; used for spoken replies |
| `STT_MODEL` | `nova-3` | Deepgram speech recognition model |
| `STT_LANGUAGE` | `en-US` | Speech recognition language |
| `TTS_MODEL` | `aura-asteria-en` | Deepgram speech generation model |
| `DOCS_DIR` | `docs` | Directory containing source PDFs |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `QDRANT_PATH` | `./data/qdrant` | Local Qdrant storage directory |
| `COLLECTION_NAME` | `telecom_docs` | Qdrant collection used for document search |
| `LIVEKIT_URL` | Not set | LiveKit server URL for the optional agent |
| `LIVEKIT_API_KEY` | Not set | LiveKit API key for the optional agent |
| `LIVEKIT_API_SECRET` | Not set | LiveKit API secret for the optional agent |

## Project layout

```text
app.py                         Streamlit voice support interface
docs/                          Source support PDFs
src/lauki_voice_agent/
  config.py                    Environment-based settings
  ingest.py                    PDF parsing, embedding, and Qdrant indexing
  main.py                      Optional LiveKit voice agent
  rag.py                       Local Qdrant retrieval
  tools.py                     Support tools and LangGraph agent
data/qdrant/                   Local vector database
```
