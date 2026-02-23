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
| **Week 3: Speaker Diarization** | 🚧 PLANNED | Open-source speaker separation + yt-dlp YouTube downloader |
| **Week 4: Vector DB** | 🚧 PLANNED | ChromaDB for semantic search |
| **Week 5: Q&A Engine** | 🚧 PLANNED | RAG-based question answering with Llama 3 |
| **Week 6: UI & API** | 🚧 PLANNED | FastAPI + Streamlit interface |

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
- Cost: `$0.03/hour` of audio  
- **Test Coverage:** Local transcription verified

---

## 📁 Project Structure
