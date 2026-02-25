"""
pipeline.py
-----------
End-to-end meeting processing pipeline.

Chains: video/audio → audio extraction → transcription → summarization
        → transcript chunking → vector store indexing
"""

import logging
import os
from pathlib import Path
from typing import Optional

from src.audio.extractor import extract_audio, cleanup_audio
from src.transcription.whisper_transcriber import transcribe_audio
from src.summary.summarizer import summarize_transcript
from src.vector_db.chunker import chunk_transcript
from src.vector_db.store import MeetingVectorStore

logger = logging.getLogger(__name__)

# Default vector store is persisted next to this file's package root
_DEFAULT_DB_DIR = str(Path(__file__).parent.parent / "chroma_db")


def process_meeting(
    media_path: str,
    meeting_id: str,
    title: str,
    date: str,
    vector_store: Optional[MeetingVectorStore] = None,
    chunk_size: int = 300,
    overlap: int = 50,
    language: Optional[str] = None,
) -> dict:
    """
    Full pipeline: media file → indexed, searchable meeting entry.

    Args:
        media_path:    Path to a video (MP4, MOV, AVI) or audio (MP3, WAV) file.
        meeting_id:    Unique identifier for this meeting (e.g. "m2026-02-15").
        title:         Human-readable meeting title.
        date:          Meeting date string (e.g. "2026-02-15").
        vector_store:  Pre-initialised MeetingVectorStore; created automatically
                       if not supplied (data stored in ./chroma_db/).
        chunk_size:    Words per transcript chunk (default 300).
        overlap:       Overlapping words between adjacent chunks (default 50).
        language:      Optional ISO-639-1 language hint for Whisper.

    Returns:
        dict with:
            - meeting_id    (str)
            - title         (str)
            - date          (str)
            - transcript    (str)  full transcript text
            - summary       (dict) structured summary from LLM
            - chunks_stored (int)  number of chunks indexed

    Raises:
        FileNotFoundError: media_path does not exist
        RuntimeError:      any stage of the pipeline fails
    """
    media_path = Path(media_path).resolve()
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    logger.info(f"=== Pipeline START: {title} ({meeting_id}) ===")

    # -- Step 1: Audio extraction (skip for audio-only files) ----------------
    audio_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    extracted_audio_path: Optional[str] = None

    if media_path.suffix.lower() in audio_extensions:
        audio_path = str(media_path)
        logger.info("Input is already an audio file – skipping extraction.")
    else:
        logger.info("Step 1/4 – Extracting audio …")
        audio_path = extract_audio(str(media_path))
        extracted_audio_path = audio_path  # remember to clean up later

    # -- Step 2: Transcription -----------------------------------------------
    logger.info("Step 2/4 – Transcribing audio …")
    try:
        transcription = transcribe_audio(audio_path, language=language)
    finally:
        # Clean up the temp MP3 we extracted (if any)
        if extracted_audio_path:
            cleanup_audio(extracted_audio_path)

    transcript_text = transcription["text"]
    logger.info(f"Transcript: {len(transcript_text)} chars, language={transcription.get('language')}")

    # -- Step 3: Summarisation -----------------------------------------------
    logger.info("Step 3/4 – Summarising transcript …")
    context = f"Meeting: {title}, Date: {date}"
    summary = summarize_transcript(transcript_text, context=context)

    # -- Step 4: Chunk + index -----------------------------------------------
    logger.info("Step 4/4 – Chunking and indexing transcript …")
    chunks = chunk_transcript(transcript_text, chunk_size=chunk_size, overlap=overlap)

    if vector_store is None:
        vector_store = MeetingVectorStore(persist_dir=_DEFAULT_DB_DIR)

    chunks_stored = vector_store.add_meeting(
        meeting_id=meeting_id,
        title=title,
        date=date,
        segments=chunks,
    )

    logger.info(
        f"=== Pipeline COMPLETE: '{title}' – {chunks_stored} chunks indexed ==="
    )

    return {
        "meeting_id": meeting_id,
        "title": title,
        "date": date,
        "transcript": transcript_text,
        "summary": summary,
        "chunks_stored": chunks_stored,
    }
