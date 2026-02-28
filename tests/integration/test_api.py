"""
tests/integration/test_api.py
------------------------------
Integration tests for the FastAPI REST layer.
Uses FastAPI's TestClient (backed by httpx) – no running server required.

The vector store and QA engine are mocked to avoid real DB and LLM calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# We need to mock expensive singletons BEFORE importing the app
# ---------------------------------------------------------------------------

MOCK_MEETINGS = [
    {"meeting_id": "m-001", "title": "Sprint Review", "date": "2026-02-15"},
    {"meeting_id": "m-002", "title": "Q1 Planning", "date": "2026-01-10"},
]

MOCK_ANSWER = {
    "answer": "The launch is Q3 2026. Sources: Sprint Review (2026-02-15)",
    "citations": [
        {
            "text": "The team agreed to launch by Q3 2026.",
            "meeting_id": "m-001",
            "title": "Sprint Review",
            "date": "2026-02-15",
            "chunk_index": 0,
        }
    ],
    "model": "llama-3.3-70b-versatile",
    "question": "When is the launch?",
}


@pytest.fixture()
def client():
    """Create a TestClient with all external services mocked."""
    mock_store = MagicMock()
    mock_store.list_meetings.return_value = MOCK_MEETINGS
    mock_store.search.return_value = MOCK_ANSWER["citations"]

    mock_engine = MagicMock()
    mock_engine.ask.return_value = MOCK_ANSWER

    with (
        patch("src.api.main._get_store", return_value=mock_store),
        patch("src.api.main._get_engine", return_value=mock_engine),
    ):
        from src.api.main import app

        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_contains_api_name(self, client):
        r = client.get("/")
        assert "Meeting Knowledge Assistant" in r.json().get("name", "")

    def test_root_status_ok(self, client):
        r = client.get("/")
        assert r.json().get("status") == "ok"


class TestListMeetings:
    def test_get_meetings_returns_200(self, client):
        r = client.get("/meetings")
        assert r.status_code == 200

    def test_get_meetings_returns_list(self, client):
        r = client.get("/meetings")
        data = r.json()
        assert isinstance(data, list)

    def test_get_meetings_has_expected_keys(self, client):
        r = client.get("/meetings")
        for m in r.json():
            assert "meeting_id" in m
            assert "title" in m
            assert "date" in m


class TestGlobalAsk:
    def test_ask_returns_200(self, client):
        r = client.post("/ask", json={"question": "When is the launch?"})
        assert r.status_code == 200

    def test_ask_response_has_required_keys(self, client):
        r = client.post("/ask", json={"question": "When is the launch?"})
        body = r.json()
        assert "answer" in body
        assert "citations" in body
        assert "model" in body
        assert "question" in body

    def test_ask_empty_question_returns_422(self, client):
        r = client.post("/ask", json={"question": ""})
        assert r.status_code == 422

    def test_ask_missing_question_returns_422(self, client):
        r = client.post("/ask", json={})
        assert r.status_code == 422


class TestMeetingAsk:
    def test_meeting_ask_returns_200(self, client):
        r = client.post("/meetings/m-001/ask", json={"question": "What was decided?"})
        assert r.status_code == 200

    def test_meeting_ask_echoes_question(self, client):
        q = "When is the launch?"
        r = client.post("/meetings/m-001/ask", json={"question": q})
        assert r.json()["question"] == q


class TestDeleteMeeting:
    def test_delete_returns_200(self, client):
        r = client.delete("/meetings/m-001")
        assert r.status_code == 200

    def test_delete_response_contains_id(self, client):
        r = client.delete("/meetings/m-999")
        assert r.json().get("deleted") == "m-999"
