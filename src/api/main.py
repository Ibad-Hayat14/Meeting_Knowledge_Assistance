"""
main.py
-------
FastAPI REST API for the Meeting Knowledge Assistant.

Endpoints
---------
GET  /                          → health check / API info
POST /meetings/process          → upload + process a media file
GET  /meetings                  → list all indexed meetings
POST /meetings/{id}/ask         → ask a question scoped to one meeting
POST /ask                       → ask a question across all meetings
DELETE /meetings/{id}           → remove a meeting from the vector store
"""

import logging
import os
import tempfile
import uuid
from datetime import date
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.pipeline import process_meeting
from src.vector_db.store import MeetingVectorStore
from src.qa.engine import MeetingQAEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Meeting Knowledge Assistant API",
    description=(
        "AI-powered meeting assistant. Upload meeting recordings and "
        "query them with natural language Q&A."
    ),
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singleton services (initialised once at startup)
# ---------------------------------------------------------------------------

_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

_vector_store: Optional[MeetingVectorStore] = None
_qa_engine: Optional[MeetingQAEngine] = None


def _get_store() -> MeetingVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = MeetingVectorStore(persist_dir=_PERSIST_DIR)
    return _vector_store


def _get_engine() -> MeetingQAEngine:
    global _qa_engine
    if _qa_engine is None:
        _qa_engine = MeetingQAEngine(vector_store=_get_store())
    return _qa_engine


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class MeetingOverview(BaseModel):
    meeting_id: str
    title: str
    date: str


class ProcessResponse(BaseModel):
    meeting_id: str
    title: str
    date: str
    transcript: str = Field(..., description="Full transcript text")
    summary: dict = Field(..., description="Structured summary (key_points, action_items, decisions)")
    chunks_stored: int


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    n_context: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list
    model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"])
def root():
    """Health check – confirms the API is running."""
    return {
        "name": "Meeting Knowledge Assistant API",
        "version": "0.6.0",
        "status": "ok",
        "docs": "/docs",
    }


@app.post(
    "/meetings/process",
    response_model=ProcessResponse,
    tags=["Meetings"],
    summary="Upload and process a meeting recording",
)
async def upload_and_process_meeting(
    file: UploadFile = File(..., description="Audio/video file (MP3, WAV, MP4, etc.)"),
    meeting_id: Optional[str] = Form(None, description="Unique ID (auto-generated if omitted)"),
    title: str = Form(..., description="Human-readable meeting title"),
    date: str = Form(..., description="Meeting date (YYYY-MM-DD)"),
    language: Optional[str] = Form(None, description="ISO-639-1 language code hint for Whisper"),
):
    """
    Upload a meeting audio/video file. The pipeline will:
    1. Transcribe the audio (Groq Whisper)
    2. Summarise the transcript (Groq Llama 3)
    3. Chunk and index it in ChromaDB for semantic Q&A
    """
    mid = meeting_id or f"m-{uuid.uuid4().hex[:8]}"

    # Save the upload to a temp file
    suffix = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = process_meeting(
            media_path=tmp_path,
            meeting_id=mid,
            title=title,
            date=date,
            vector_store=_get_store(),
            language=language,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return ProcessResponse(**result)


@app.get(
    "/meetings",
    response_model=List[MeetingOverview],
    tags=["Meetings"],
    summary="List all indexed meetings",
)
def list_meetings():
    """Return metadata for every meeting currently stored in the vector DB."""
    return _get_store().list_meetings()


@app.delete(
    "/meetings/{meeting_id}",
    tags=["Meetings"],
    summary="Delete a meeting from the store",
)
def delete_meeting(meeting_id: str):
    """Remove all indexed chunks for the given meeting."""
    try:
        _get_store().delete_meeting(meeting_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"deleted": meeting_id}


@app.post(
    "/meetings/{meeting_id}/ask",
    response_model=AnswerResponse,
    tags=["Q&A"],
    summary="Ask a question scoped to one meeting",
)
def ask_meeting(meeting_id: str, body: QuestionRequest):
    """Answer a question using only transcript chunks from the specified meeting."""
    try:
        result = _get_engine().ask(
            question=body.question,
            meeting_id=meeting_id,
            n_context=body.n_context,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AnswerResponse(**result)


@app.post(
    "/ask",
    response_model=AnswerResponse,
    tags=["Q&A"],
    summary="Ask a question across all meetings",
)
def ask_global(body: QuestionRequest):
    """Answer a question by searching across all indexed meetings."""
    try:
        result = _get_engine().ask(
            question=body.question,
            n_context=body.n_context,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AnswerResponse(**result)
