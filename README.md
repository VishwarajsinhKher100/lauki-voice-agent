# **Lauki Voice Agent** : Real-Time Customer Support Voice Assistant

A real-time, voice-first customer support agent built for telecom services using **LiveKit**, **Deepgram**, **Groq** (openai/gpt-oss-120b), **Qdrant**, and **Docling**[cite: 1, 2, 3]. The system features high-speed Speech-to-Text (STT), Text-to-Speech (TTS), and Voice Activity Detection (VAD) coupled with a Retrieval-Augmented Generation (RAG) pipeline and autonomous customer service tools[cite: 2, 3, 5].

## **Features**

* **Real-Time Voice Streaming**: Ultra-low-latency voice interactions powered by LiveKit RTC session management, Deepgram STT/TTS, and Silero VAD.

* **Hierarchical Document Ingestion**: Converts and chunks telecom documentation using Docling's hierarchical chunker, preserving heading breadcrumbs for contextual vector search[cite: 2].

* **Domain-Specific RAG**: Qdrant-backed vector search with sentence embeddings to retrieve accurate plan specs, limits, and pricing[cite: 2, 4].

* **Autonomous Tool Execution**: Context-aware tool calling for plan recommendations, account balances, network coverage checks, plan details, and automated ticket escalation[cite: 5].

* **Voice-Optimized Response Filtering**: Custom speech graph wrapping filters internal tool execution tokens from the audio generation pipeline to keep speech responses natural and conversational[cite: 3, 5].

## **Tech Stack**

| Component                    | Technology                          |
|------------------------------|-------------------------------------|
| Voice Infrastructure         | LiveKit Agents & WebRTC             |
| Speech-to-Text (STT)         | Deepgram (Nova-3)                   |
| Text-to-Speech (TTS)         | Deepgram (Aura)                     |
| Voice Activity Detection     | Silero VAD                          |
| Agent Framework              | LangChain & LangGraph               |
| LLM Provider                 | Groq (openai/gpt-oss-120b)          |
| Vector DB                    | Qdrant Client                       |
| Embedding Model              | SentenceTransformers (all-MiniLM-L6-v2) |
| Document Parsing & Chunking  | Docling (HierarchicalChunker)       |

## **Setup Instructions**

### 1. Clone the Repository
```bash
git clone https://github.com/VishwarajsinhKher100/lauki-voice-agent.git
```

### 2. Set Up Virtual Environment
```bash
uv venv
```

#### Activate the virtual environment:

```bash
.venv\Scripts\activate     # On Windows
# OR
source .venv/bin/activate  # On Linux/macOS:
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Configure Environment Variables

Create a .env file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here

DEEPGRAM_API_KEY=your_deepgram_api_key

LIVEKIT_URL=
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=
```

### 5. Running the Application

Launch the Streamlit web application:

```bash
python -m src.lauki_voice_agent.main
```