"""
tests/unit/test_qa_engine.py
-----------------------------
Unit tests for MeetingQAEngine (RAG Q&A engine).
Groq API and vector store are both fully mocked.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.qa.engine import MeetingQAEngine


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "text": "The team agreed to launch by Q3 2026.",
        "meeting_id": "m-001",
        "title": "Sprint Review",
        "date": "2026-02-15",
        "chunk_index": 0,
        "distance": 0.05,
    },
    {
        "text": "Sarah will prepare the revised budget by Friday.",
        "meeting_id": "m-001",
        "title": "Sprint Review",
        "date": "2026-02-15",
        "chunk_index": 1,
        "distance": 0.12,
    },
]


def _make_engine(chunks=SAMPLE_CHUNKS, llm_answer="The launch is Q3 2026. Sources: Sprint Review (2026-02-15)"):
    """Create a MeetingQAEngine with a mocked vector store and Groq client."""
    mock_store = MagicMock()
    mock_store.search.return_value = chunks

    with patch("src.qa.engine.Groq") as MockGroq:
        mock_llm = MagicMock()
        MockGroq.return_value = mock_llm

        mock_message = MagicMock()
        mock_message.content = llm_answer
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=mock_message)]
        )

        engine = MeetingQAEngine(vector_store=mock_store, groq_api_key="fake_key")

    return engine, mock_store, mock_llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMeetingQAEngine:
    """Unit tests for the RAG Q&A engine."""

    def test_ask_returns_required_keys(self):
        """Happy path: response has answer, citations, model, question."""
        engine, _, _ = _make_engine()
        result = engine.ask("When is the product launch?")

        assert "answer" in result
        assert "citations" in result
        assert "model" in result
        assert "question" in result

    def test_ask_answer_is_string(self):
        engine, _, _ = _make_engine()
        result = engine.ask("When is the launch?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_ask_citations_is_list(self):
        engine, _, _ = _make_engine()
        result = engine.ask("When is the launch?")
        assert isinstance(result["citations"], list)

    def test_ask_citations_contain_required_keys(self):
        engine, _, _ = _make_engine()
        result = engine.ask("When is the launch?")
        for citation in result["citations"]:
            assert "text" in citation
            assert "meeting_id" in citation
            assert "title" in citation
            assert "date" in citation

    def test_ask_question_echoed_in_response(self):
        engine, _, _ = _make_engine()
        question = "Who owns the frontend implementation?"
        result = engine.ask(question)
        assert result["question"] == question

    def test_ask_calls_vector_store_search(self):
        engine, mock_store, _ = _make_engine()
        engine.ask("When is the launch?")
        mock_store.search.assert_called_once()

    def test_ask_calls_groq_llm(self):
        engine, _, mock_llm = _make_engine()
        with patch("src.qa.engine.Groq") as MockGroq:
            MockGroq.return_value = mock_llm
            engine.ask("When is the launch?")
        mock_llm.chat.completions.create.assert_called()

    def test_ask_empty_question_raises_value_error(self):
        engine, _, _ = _make_engine()
        with pytest.raises(ValueError, match="empty"):
            engine.ask("")

    def test_ask_whitespace_question_raises_value_error(self):
        engine, _, _ = _make_engine()
        with pytest.raises(ValueError, match="empty"):
            engine.ask("   ")

    def test_ask_no_chunks_returns_fallback_answer(self):
        """When vector store returns nothing, engine gives polite fallback."""
        engine, mock_store, _ = _make_engine(chunks=[])
        mock_store.search.return_value = []

        result = engine.ask("What was decided about the launch?")

        assert "don't have enough information" in result["answer"].lower()
        assert result["citations"] == []

    def test_ask_meeting_id_filter_forwarded_to_store(self):
        engine, mock_store, _ = _make_engine()
        engine.ask("When is the launch?", meeting_id="m-001")

        call_kwargs = mock_store.search.call_args[1]
        assert call_kwargs.get("meeting_id") == "m-001"

    def test_ask_api_error_raises_runtime_error(self):
        mock_store = MagicMock()
        mock_store.search.return_value = SAMPLE_CHUNKS

        with patch("src.qa.engine.Groq") as MockGroq:
            mock_llm = MagicMock()
            MockGroq.return_value = mock_llm
            mock_llm.chat.completions.create.side_effect = Exception("Groq timeout")

            engine = MeetingQAEngine(vector_store=mock_store, groq_api_key="fake_key")

            with pytest.raises(RuntimeError, match="Q&A generation failed"):
                engine.ask("When is the launch?")

    def test_no_api_key_raises_runtime_error(self, monkeypatch):
        """Engine raises immediately if no API key is available."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        mock_store = MagicMock()

        with patch("src.qa.engine.Groq"):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                MeetingQAEngine(vector_store=mock_store, groq_api_key=None)
