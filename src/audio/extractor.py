import subprocess
import os
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extract audio from video using FFmpeg (streaming, memory-safe).

    Args:
        video_path: Path to input video file
        output_path: Optional output path (defaults to temp file)

    Returns:
        Absolute path to extracted MP3 file

    Raises:
        FileNotFoundError: If video file doesnt exist
        RuntimeError: If FFmpeg extraction fails
    """
    video_path = Path(video_path).resolve()
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_path is None:
        _, output_path = tempfile.mkstemp(suffix=".mp3")
    
    output_path = Path(output_path).resolve()
    
    logger.info(f"Extracting audio from {video_path} to {output_path}")
    
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",  # Skip video stream (critical for memory safety)
                "-acodec", "libmp3lame",
                "-ar", "16000",  # 16kHz sample rate (optimal for speech)
                "-ac", "1",      # Mono audio (reduces size, sufficient for speech)
                "-b:a", "128k",  # Bitrate
                "-y",            # Overwrite output file if exists
                str(output_path)
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5-minute timeout for large videos
        )
        
        logger.info(f"Audio extracted successfully: {output_path}")
        return str(output_path)
    
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg extraction timed out after 5 minutes")
    
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "Unknown FFmpeg error"
        logger.error(f"FFmpeg failed: {error_msg}")
        raise RuntimeError(f"Audio extraction failed: {error_msg[:200]}") from e


def cleanup_audio(audio_path: str) -> None:
    """Safely delete temporary audio file."""
    audio_path = Path(audio_path)
    if audio_path.exists():
        try:
            audio_path.unlink()
            logger.info(f"Cleaned up audio file: {audio_path}")
        except OSError as e:
            logger.warning(f"Failed to delete {audio_path}: {e}")