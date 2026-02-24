import pytest
from unittest.mock import patch, MagicMock
import json
from src.summary.summarizer import summarize_transcript


SAMPLE_TRANSCRIPT = """
John: Good morning everyone. Let's start with the Q3 budget review.
Sarah: The marketing budget is 15% over due to the new campaign.
John: We need to cut costs by next month. Sarah, can you prepare a revised plan?
Sarah: Sure, I'll have it ready by Friday.
John: Great. We've also decided to delay the product launch to December.
"""

SAMPLE_SUMMARY_RESPONSE = {
    "summary": "The team discussed Q3 budget overruns and decided to delay the product launch.",
    "key_points": [
        "Marketing budget is 15% over due to new campaign",
        "Cost cuts required by next month",
        "Product launch delayed to December",
    ],
    "action_items": ["Revised marketing budget plan — Sarah (by Friday)"],
    "decisions": ["Product launch delayed to December"],
}


class TestSummarizeTranscript:
    """Unit tests for Groq LLaMA summarization (all API calls mocked)."""

    def test_summarize_valid_transcript(self):
        """Happy path: transcript → structured summary dict with all keys."""
        with patch("src.summary.summarizer.Groq") as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client

            mock_message = MagicMock()
            mock_message.content = json.dumps(SAMPLE_SUMMARY_RESPONSE)
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=mock_message)]
            )

            result = summarize_transcript(SAMPLE_TRANSCRIPT)

        # Verify all keys present
        assert "summary" in result
        assert "key_points" in result
        assert "action_items" in result
        assert "decisions" in result

        # Verify types
        assert isinstance(result["summary"], str)
        assert isinstance(result["key_points"], list)
        assert isinstance(result["action_items"], list)
        assert isinstance(result["decisions"], list)

        # Verify content
        assert len(result["summary"]) > 0
        assert len(result["key_points"]) > 0

    def test_summarize_empty_transcript_raises_value_error(self):
        """Edge case: empty string raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            summarize_transcript("")

    def test_summarize_whitespace_only_raises_value_error(self):
        """Edge case: whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            summarize_transcript("   \n\t  ")

    def test_summarize_too_short_raises_value_error(self):
        """Edge case: very short transcript raises ValueError."""
        with pytest.raises(ValueError, match="too short"):
            summarize_transcript("Hi.")

    def test_summarize_with_context(self):
        """Happy path: context string is included in the API call."""
        with patch("src.summary.summarizer.Groq") as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client

            mock_message = MagicMock()
            mock_message.content = json.dumps(SAMPLE_SUMMARY_RESPONSE)
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=mock_message)]
            )

            summarize_transcript(SAMPLE_TRANSCRIPT, context="Q3 Budget Review, Feb 2026")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        assert "Q3 Budget Review" in user_content
        assert "Context:" in user_content

    def test_summarize_api_error_raises_runtime_error(self):
        """Edge case: Groq API error → RuntimeError."""
        with patch("src.summary.summarizer.Groq") as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API unavailable")

            with pytest.raises(RuntimeError, match="Summarization failed"):
                summarize_transcript(SAMPLE_TRANSCRIPT)

    def test_summarize_bad_json_raises_runtime_error(self):
        """Edge case: malformed JSON from API → RuntimeError."""
        with patch("src.summary.summarizer.Groq") as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client

            mock_message = MagicMock()
            mock_message.content = "This is not JSON at all"
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=mock_message)]
            )

            with pytest.raises(RuntimeError, match="Failed to parse"):
                summarize_transcript(SAMPLE_TRANSCRIPT)
