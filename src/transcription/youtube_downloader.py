import logging
from pathlib import Path
from pytube import YouTube
from pytube.exceptions import PytubeError, VideoUnavailable, AgeRestrictedError
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Download audio-only from YouTube (memory-safe, no video waste)."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_audio(self, url: str, filename: str = None) -> str:
        """
        Download audio-only from YouTube URL.
        
        Args:
            url: YouTube video URL or ID
            filename: Optional output filename (default: video title + .mp3)
        
        Returns:
            Absolute path to downloaded MP3 file
        
        Raises:
            ValueError: Invalid URL format
            RuntimeError: Download failed (private/age-restricted/unavailable)
        """
        try:
            # Normalize URL (handle short URLs, IDs)
            if not url.startswith("http"):
                url = f"https://www.youtube.com/watch?v={url}"
            
            logger.info(f"Downloading audio from: {url}")
            yt = YouTube(url)
            
            # Fail fast on problematic videos
            if yt.age_restricted:
                raise AgeRestrictedError("Video is age-restricted (requires login)")
            if not yt.streams:
                raise VideoUnavailable("No streams available (private/deleted video)")
            
            # Get highest quality audio-only stream
            audio_stream = yt.streams.get_audio_only()
            if not audio_stream:
                raise RuntimeError("No audio-only stream available")
            
            # Download to temp file first (atomic write)
            temp_path = self.output_dir / f"{hash(url)}.mp4"
            logger.info(f"Downloading '{yt.title}' ({yt.length // 60}:{yt.length % 60:02d})")
            
            try:
                downloaded_path = audio_stream.download(
                    output_path=str(self.output_dir),
                    filename=temp_path.stem,
                    skip_existing=False
                )
            except Exception as e:
                raise RuntimeError(f"Download failed: {str(e)[:150]}") from e
            
            # Convert to MP3 via FFmpeg (re-use Week 1 extractor)
            from src.audio.extractor import extract_audio
            mp3_path = self.output_dir / f"{temp_path.stem}.mp3"
            mp3_path = extract_audio(downloaded_path, str(mp3_path))
            
            # Cleanup original download (only keep MP3)
            Path(downloaded_path).unlink(missing_ok=True)
            
            logger.info(f"Audio downloaded: {mp3_path}")
            return str(mp3_path)
        
        except (PytubeError, VideoUnavailable, AgeRestrictedError) as e:
            raise RuntimeError(f"YouTube download failed: {str(e)[:150]}") from e
    
    def cleanup(self, audio_path: str):
        """Safely delete downloaded audio file."""
        from src.audio.extractor import cleanup_audio
        cleanup_audio(audio_path)