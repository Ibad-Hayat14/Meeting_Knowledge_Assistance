import pytest
from unittest.mock import patch, MagicMock
from src.transcription.whisper_transcriber import transcribe_audio


class TestTranscribeAudio:
    """Unit tests for Groq Whisper transcription (all API calls mocked)."""

    def test_transcribe_valid_audio(self, tmp_path):
        """Happy path: valid audio file → structured transcription dict."""
        audio_file = tmp_path / "meeting.mp3"
        audio_file.write_bytes(b"fake audio content")

        mock_response = MagicMock()
        mock_response.text = "  Hello, this is a meeting transcript.  "
        mock_response.language = "en"
        mock_response.duration = 12.5

        with patch("src.transcription.whisper_transcriber.Groq") as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.audio.transcriptions.create.return_value = mock_response

            result = transcribe_audio(str(audio_file))

        # Verify structure
        assert isinstance(result, dict)
        assert "text" in result
        assert "language" in result
        assert "duration" in result

        # Verify values
        assert result["text"] == "Hello, this is a meeting transcript."  # stripped
        assert result["language"] == "en"
        assert result["duration"] == 12.5

        # Verify Groq was called once
        mock_client.audio.transcriptions.create.assert_called_once()

    def test_transcribe_file_not_found(self):
        """Edge case: non-existent audio file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcribe_audio("/nonexistent/meeting.mp3")

    def test_transcribe_file_too_large(self, tmp_path):
        """Edge case: file exceeding 25 MB raises ValueError."""
        large_file = tmp_path / "huge.mp3"
        # Write just over 25 MB
        large_file.write_bytes(b"x" * (25 * 1024 * 1024 + 1))

        with pytest.raises(ValueError, match="too large"):
            transcribe_audio(str(large_file))

    def test_transcribe_with_language_hint(self, tmp_path):
        """Happy path: language hint is forwarded to the API."""
        audio_file = tmp_path / "meeting.mp3"
        audio_file.write_bytes(b"fake audio content")

        mock_response = MagicMock()
        mock_response.text = "Guten Tag."
        mock_response.language = "de"
        mock_response.duration = 3.0

        with patch("src.transcription.whisper_transcriber.Groq") as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.audio.transcriptions.create.return_value = mock_response

            result = transcribe_audio(str(audio_file), language="de")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs.get("language") == "de"
        assert result["language"] == "de"

    def test_transcribe_api_error_raises_runtime_error(self, tmp_path):
        """Edge case: Groq API error → RuntimeError."""
        audio_file = tmp_path / "meeting.mp3"
        audio_file.write_bytes(b"fake audio content")

        with patch("src.transcription.whisper_transcriber.Groq") as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.audio.transcriptions.create.side_effect = Exception(
                "Connection timeout"
            )

            with pytest.raises(RuntimeError, match="Transcription failed"):
                transcribe_audio(str(audio_file))
