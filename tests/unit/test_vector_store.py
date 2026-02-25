"""
tests/unit/test_vector_store.py
--------------------------------
Unit tests for MeetingVectorStore and chunk_transcript.
All ChromaDB / embedding calls are mocked so no GPU / network is needed.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from src.vector_db.chunker import chunk_transcript


# ---------------------------------------------------------------------------
# chunk_transcript tests
# ---------------------------------------------------------------------------

class TestChunkTranscript:
    """Tests for the transcript chunker utility."""

    SAMPLE_TEXT = (
        "John: Good morning everyone. Let's start with the Q3 budget review. "
        "Sarah: The marketing budget is 15 percent over due to the campaign. "
        "John: We need to cut costs. Sarah will prepare a revised plan by Friday. "
        "John: We have also decided to delay the product launch to December."
    )

    def test_returns_list_of_dicts(self):
        chunks = chunk_transcript(self.SAMPLE_TEXT, chunk_size=10, overlap=2)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_each_chunk_has_required_keys(self):
        chunks = chunk_transcript(self.SAMPLE_TEXT, chunk_size=10, overlap=2)
        for chunk in chunks:
            assert "chunk_index" in chunk
            assert "text" in chunk
            assert "start_word" in chunk
            assert "end_word" in chunk

    def test_chunk_index_is_sequential(self):
        chunks = chunk_transcript(self.SAMPLE_TEXT, chunk_size=10, overlap=2)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_all_words_covered(self):
        """Every word from the original text should appear in at least one chunk."""
        words = self.SAMPLE_TEXT.split()
        chunks = chunk_transcript(self.SAMPLE_TEXT, chunk_size=10, overlap=2)
        all_chunk_words = " ".join(c["text"] for c in chunks).split()
        # Due to overlap some words appear more than once – just check originals
        for word in words:
            assert word in all_chunk_words

    def test_empty_text_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            chunk_transcript("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            chunk_transcript("   \n\t  ")

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_transcript("Hello world", chunk_size=0)

    def test_overlap_too_large_raises(self):
        with pytest.raises(ValueError):
            chunk_transcript("Hello world", chunk_size=5, overlap=5)

    def test_single_chunk_when_text_shorter_than_chunk_size(self):
        short = "Hello world"
        chunks = chunk_transcript(short, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0]["text"] == short


# ---------------------------------------------------------------------------
# MeetingVectorStore tests (ChromaDB + sentence-transformers fully mocked)
# ---------------------------------------------------------------------------

SAMPLE_SEGMENTS = [
    {"chunk_index": 0, "text": "The team agreed to launch by Q3 2026.", "start_word": 0, "end_word": 8},
    {"chunk_index": 1, "text": "Sarah will prepare the revised budget by Friday.", "start_word": 6, "end_word": 14},
]

SAMPLE_MEETING = {
    "meeting_id": "m-001",
    "title": "Sprint Review",
    "date": "2026-02-15",
}


def _make_mock_store():
    """
    Return a (mock_collection, MeetingVectorStore-instance) pair,
    with chromadb and SentenceTransformerEmbeddingFunction fully mocked.
    """
    with patch("src.vector_db.store.chromadb.PersistentClient") as MockClient, \
         patch("src.vector_db.store.embedding_functions.SentenceTransformerEmbeddingFunction") as MockEmbedFn:

        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        MockClient.return_value.get_or_create_collection.return_value = mock_collection

        from src.vector_db.store import MeetingVectorStore
        store = MeetingVectorStore(persist_dir="./fake_chroma")

    return mock_collection, store


class TestMeetingVectorStore:
    """Unit tests for MeetingVectorStore – all I/O mocked."""

    def test_add_meeting_calls_upsert(self):
        with patch("src.vector_db.store.chromadb.PersistentClient") as MockClient, \
             patch("src.vector_db.store.embedding_functions.SentenceTransformerEmbeddingFunction"):

            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            MockClient.return_value.get_or_create_collection.return_value = mock_collection

            from src.vector_db.store import MeetingVectorStore
            store = MeetingVectorStore(persist_dir="./fake_chroma")

            result = store.add_meeting(
                meeting_id=SAMPLE_MEETING["meeting_id"],
                title=SAMPLE_MEETING["title"],
                date=SAMPLE_MEETING["date"],
                segments=SAMPLE_SEGMENTS,
            )

        assert result == 2
        mock_collection.upsert.assert_called_once()

    def test_add_meeting_empty_segments_raises(self):
        with patch("src.vector_db.store.chromadb.PersistentClient"), \
             patch("src.vector_db.store.embedding_functions.SentenceTransformerEmbeddingFunction"):

            from src.vector_db.store import MeetingVectorStore
            store = MeetingVectorStore(persist_dir="./fake_chroma")

            with pytest.raises(ValueError, match="empty"):
                store.add_meeting("m-001", "Test", "2026-01-01", [])

    def test_add_meeting_blank_id_raises(self):
        with patch("src.vector_db.store.chromadb.PersistentClient"), \
             patch("src.vector_db.store.embedding_functions.SentenceTransformerEmbeddingFunction"):

            from src.vector_db.store import MeetingVectorStore
            store = MeetingVectorStore(persist_dir="./fake_chroma")

            with pytest.raises(ValueError, match="meeting_id"):
                store.add_meeting("", "Test", "2026-01-01", SAMPLE_SEGMENTS)

    def test_search_returns_results(self):
        with patch("src.vector_db.store.chromadb.PersistentClient") as MockClient, \
             patch("src.vector_db.store.embedding_functions.SentenceTransformerEmbeddingFunction"):

            mock_collection = MagicMock()
            mock_collection.count.return_value = 2
            mock_collection.query.return_value = {
                "documents": [["The team agreed to launch by Q3 2026."]],
                "metadatas": [[{"meeting_id": "m-001", "title": "Sprint Review",
                                "date": "2026-02-15", "chunk_index": 0,
                                "start_word": 0, "end_word": 8}]],
                "distances": [[0.05]],
            }
            MockClient.return_value.get_or_create_collection.return_value = mock_collection

            from src.vector_db.store import MeetingVectorStore
            store = MeetingVectorStore(persist_dir="./fake_chroma")
            results = store.search("When is the launch?", n_results=3)

        assert len(results) == 1
        assert "text" in results[0]
        assert "meeting_id" in results[0]
        assert "distance" in results[0]
        assert results[0]["meeting_id"] == "m-001"

    def test_search_empty_query_raises(self):
        with patch("src.vector_db.store.chromadb.PersistentClient") as MockClient, \
             patch("src.vector_db.store.embedding_functions.SentenceTransformerEmbeddingFunction"):

            mock_collection = MagicMock()
            mock_collection.count.return_value = 2
            MockClient.return_value.get_or_create_collection.return_value = mock_collection

            from src.vector_db.store import MeetingVectorStore
            store = MeetingVectorStore(persist_dir="./fake_chroma")

            with pytest.raises(ValueError, match="empty"):
                store.search("")

    def test_search_empty_store_returns_empty_list(self):
        with patch("src.vector_db.store.chromadb.PersistentClient") as MockClient, \
             patch("src.vector_db.store.embedding_functions.SentenceTransformerEmbeddingFunction"):

            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            MockClient.return_value.get_or_create_collection.return_value = mock_collection

            from src.vector_db.store import MeetingVectorStore
            store = MeetingVectorStore(persist_dir="./fake_chroma")
            results = store.search("Hello?")

        assert results == []

    def test_delete_meeting_calls_collection_delete(self):
        with patch("src.vector_db.store.chromadb.PersistentClient") as MockClient, \
             patch("src.vector_db.store.embedding_functions.SentenceTransformerEmbeddingFunction"):

            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            MockClient.return_value.get_or_create_collection.return_value = mock_collection

            from src.vector_db.store import MeetingVectorStore
            store = MeetingVectorStore(persist_dir="./fake_chroma")
            store.delete_meeting("m-001")

        mock_collection.delete.assert_called_with(where={"meeting_id": "m-001"})

    def test_list_meetings_returns_unique_meetings(self):
        with patch("src.vector_db.store.chromadb.PersistentClient") as MockClient, \
             patch("src.vector_db.store.embedding_functions.SentenceTransformerEmbeddingFunction"):

            mock_collection = MagicMock()
            mock_collection.count.return_value = 3
            mock_collection.get.return_value = {
                "metadatas": [
                    {"meeting_id": "m-001", "title": "Sprint Review", "date": "2026-02-15", "chunk_index": 0},
                    {"meeting_id": "m-001", "title": "Sprint Review", "date": "2026-02-15", "chunk_index": 1},
                    {"meeting_id": "m-002", "title": "Budget Meeting", "date": "2026-02-20", "chunk_index": 0},
                ]
            }
            MockClient.return_value.get_or_create_collection.return_value = mock_collection

            from src.vector_db.store import MeetingVectorStore
            store = MeetingVectorStore(persist_dir="./fake_chroma")
            meetings = store.list_meetings()

        assert len(meetings) == 2
        ids = {m["meeting_id"] for m in meetings}
        assert "m-001" in ids
        assert "m-002" in ids
