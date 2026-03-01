# Meeting Knowledge Assistant

AI-powered meeting assistant that transforms video meetings into searchable, timestamped knowledge bases with contextual Q&A capabilities.

---

##  Project Vision

Transform meeting videos into **searchable, timestamped knowledge bases** with speaker-aware transcripts and contextual Q&A capabilities.

### Core Value Proposition
- **For Teams:** Never lose meeting context – search past discussions like Google  
- **For Individuals:** Get instant answers to *“When did we decide X?”* or *“What did Sarah say about Y?”*  
- **For Organizations:** Turn meeting conversations into structured, retrievable knowledge  

---

##  Current Status

| Component | Status | Description |
|---------|--------|------------|
| **Week 1: Audio Extraction** | ✅ COMPLETE | FFmpeg-based audio extractor with 100% unit test coverage |
| **Week 2: Transcription** | ✅ COMPLETE | Groq Whisper API integration for local audio transcription |
| **Week 3: Speaker Diarization** | ✅ COMPLETE | Open-source speaker separation + yt-dlp YouTube downloader |
| **Week 4: Vector DB** | ✅ COMPLETE | ChromaDB for semantic search |
| **Week 5: Q&A Engine** | ✅ COMPLETE | RAG-based question answering with Llama 3 |
| **Week 6: UI & API** | ✅ COMPLETE | FastAPI + Streamlit interface |

---

##  Features (Implemented)

###  Audio Extraction
- FFmpeg-based extraction with memory-safe streaming
- Supports MP4, MOV, AVI, WebM formats
- Handles 4+ hour videos without crashing
- Output: 16kHz mono MP3 (optimal for speech recognition)
- **Test Coverage:** 100% (5/5 unit tests passing)

###  Transcription
- Groq Whisper Large V3 API integration
- Word-level timestamp preservation
- Local audio file support (YouTube integration pending) 
- **Test Coverage:** Local transcription verified

---

## Project Structure

```
Meeting_Knowledge_Assistance/
├── src/
│   ├── audio/
│   │   └── extractor.py              # FFmpeg-based audio extraction logic
│   ├── transcription/
│   │   ├── __init__.py
│   │   └── whisper_transcriber.py    # Groq Whisper API wrapper
│   ├── vector_db/
│   │   ├── __init__.py
│   │   ├── chunker.py                # Text chunking for embeddings
│   │   └── store.py                  # ChromaDB vector store interface
│   ├── qa/
│   │   ├── __init__.py
│   │   └── engine.py                 # RAG-based Q&A engine (Llama 3 via Groq)
│   ├── summary/
│   │   ├── __init__.py
│   │   └── summarizer.py             # Meeting summarization module
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                   # FastAPI application entry point
│   ├── ui/
│   │   ├── __init__.py
│   │   └── app.py                    # Streamlit UI
│   └── pipeline.py                   # End-to-end orchestration pipeline
│
├── tests/
│   ├── unit/
│   │   ├── test_extractor.py         # Audio extractor unit tests
│   │   ├── test_transcriber.py       # Transcription unit tests
│   │   ├── test_vector_store.py      # Vector DB unit tests
│   │   ├── test_qa_engine.py         # Q&A engine unit tests
│   │   └── test_summarizer.py        # Summarizer unit tests
│   └── integration/
│       ├── __init__.py
│       └── test_api.py               # API integration tests
│
├── chroma_db/                        # Persisted ChromaDB vector store
├── docker-compose.yml                # Docker services config
├── pyproject.toml                    # Project metadata & tool config
├── .env.example                      # Environment variable template
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```
