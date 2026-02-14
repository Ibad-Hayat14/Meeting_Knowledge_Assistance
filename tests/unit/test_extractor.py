import subprocess
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock
from src.audio.extractor import extract_audio, cleanup_audio


class TestExtractAudio:
    """Test FFmpeg audio extraction with proper isolation."""

    def test_extract_audio_valid_video(self, tmp_path):
        """Happy path: valid video file → audio extracted."""
        # Create dummy video file (FFmpeg will fail but we mock it)
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")
        
        output_file = tmp_path / "output.mp3"
        
        # Mock FFmpeg call to simulate success
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="",
                stderr=""
            )
            
            result = extract_audio(str(video_file), str(output_file))
            
            # Verify FFmpeg was called with correct args
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "ffmpeg" in call_args
            assert "-vn" in call_args  # Critical: skip video stream
            assert "-ac" in call_args and "1" in call_args  # Mono audio
            
            # Verify output path is absolute
            assert Path(result).is_absolute()
            assert result == str(output_file.resolve())

    def test_extract_audio_file_not_found(self):
        """Edge case: raise FileNotFoundError for missing video."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            extract_audio("/nonexistent/video.mp4")

    def test_extract_audio_ffmpeg_failure(self, tmp_path):
        """Edge case: raise RuntimeError on FFmpeg failure."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake content")
        
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["ffmpeg"],
                stderr="Invalid data found when processing input"
            )
            
            with pytest.raises(RuntimeError, match="Audio extraction failed"):
                extract_audio(str(video_file))

    def test_cleanup_audio_existing_file(self, tmp_path):
        """Happy path: cleanup deletes existing audio file."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")
        assert audio_file.exists()
        
        cleanup_audio(str(audio_file))
        assert not audio_file.exists()

    def test_cleanup_audio_missing_file(self, tmp_path):
        """Edge case: cleanup handles missing file gracefully."""
        missing_file = tmp_path / "missing.mp3"
        # Should NOT raise exception
        cleanup_audio(str(missing_file))