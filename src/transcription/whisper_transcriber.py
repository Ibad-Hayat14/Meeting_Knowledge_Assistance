import os
import logging
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Groq Whisper limit: 25 MB
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
WHISPER_MODEL = "whisper-large-v3"


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """
    Transcribe an audio file to text using Groq's Whisper API.

    Args:
        audio_path: Absolute or relative path to the MP3/WAV audio file.
        language:   Optional ISO-639-1 language code (e.g. "en"). When None,
                    Whisper auto-detects the language.

    Returns:
        dict with keys:
            - text      (str)   full transcript
            - language  (str)   detected/provided language code
            - duration  (float) audio duration in seconds (if provided by API)

    Raises:
        FileNotFoundError: audio file does not exist
        ValueError:        file exceeds Groq's 25 MB limit
        RuntimeError:      Groq API or network error
    """
    audio_path = Path(audio_path).resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = audio_path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Audio file too large ({file_size / 1024 / 1024:.1f} MB). "
            f"Groq Whisper limit is 25 MB."
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to your .env file."
        )

    client = Groq(api_key=api_key)

    logger.info(f"Transcribing {audio_path.name} ({file_size / 1024:.1f} KB) …")

    try:
        with open(audio_path, "rb") as audio_file:
            kwargs = {
                "file": (audio_path.name, audio_file),
                "model": WHISPER_MODEL,
                "response_format": "verbose_json",
            }
            if language:
                kwargs["language"] = language

            response = client.audio.transcriptions.create(**kwargs)

        result = {
            "text": response.text.strip(),
            "language": getattr(response, "language", language or "unknown"),
            "duration": getattr(response, "duration", None),
        }

        logger.info(
            f"Transcription complete — {len(result['text'])} chars, "
            f"language={result['language']}"
        )
        return result

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {str(e)}") from e
