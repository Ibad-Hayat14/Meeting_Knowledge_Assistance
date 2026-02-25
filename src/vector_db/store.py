"""
store.py
--------
ChromaDB-backed vector store for meeting transcripts.

Each chunk of a transcript is embedded with a sentence-transformer model
and stored with metadata (meeting_id, title, date, chunk_index). This
enables semantic similarity search across one or all meetings.
"""

import logging
import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# Default embedding model – small, fast, good for English speech
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

# ChromaDB collection name
COLLECTION_NAME = "meetings"


class MeetingVectorStore:
    """
    Persistent ChromaDB vector store for meeting transcript chunks.

    Usage
    -----
    store = MeetingVectorStore(persist_dir="./chroma_db")
    store.add_meeting("m1", "Sprint Review", "2026-02-15", chunks)
    results = store.search("When is the launch?", n_results=5)
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        """
        Args:
            persist_dir: Directory where ChromaDB stores its data on disk.
            embed_model: HuggingFace sentence-transformer model name.
        """
        persist_dir = os.path.abspath(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=persist_dir)

        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model
        )

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"MeetingVectorStore ready – persist_dir={persist_dir}, "
            f"model={embed_model}, "
            f"docs_in_store={self._collection.count()}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_meeting(
        self,
        meeting_id: str,
        title: str,
        date: str,
        segments: List[Dict[str, Any]],
    ) -> int:
        """
        Embed and store all chunks for a meeting.

        Args:
            meeting_id: Unique identifier for the meeting (e.g. "m2026-02-15").
            title:      Human-readable meeting title.
            date:       Meeting date string (e.g. "2026-02-15").
            segments:   List of dicts from `chunk_transcript()`, each with
                        at minimum a "text" and "chunk_index" key.

        Returns:
            Number of chunks stored.

        Raises:
            ValueError: If meeting_id is blank or segments is empty.
        """
        if not meeting_id or not meeting_id.strip():
            raise ValueError("meeting_id must not be empty.")
        if not segments:
            raise ValueError("segments list is empty – nothing to store.")

        # Remove any existing chunks for this meeting to allow re-indexing
        self._delete_by_meeting(meeting_id)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for seg in segments:
            chunk_idx = seg.get("chunk_index", 0)
            doc_id = f"{meeting_id}__chunk_{chunk_idx}"

            ids.append(doc_id)
            documents.append(seg["text"])
            metadatas.append(
                {
                    "meeting_id": meeting_id,
                    "title": title,
                    "date": date,
                    "chunk_index": chunk_idx,
                    "start_word": seg.get("start_word", 0),
                    "end_word": seg.get("end_word", 0),
                }
            )

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        logger.info(
            f"Stored {len(ids)} chunk(s) for meeting '{title}' (id={meeting_id})."
        )
        return len(ids)

    def search(
        self,
        query: str,
        n_results: int = 5,
        meeting_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search over stored meeting chunks.

        Args:
            query:      Natural language question or phrase.
            n_results:  Maximum number of results to return.
            meeting_id: If provided, restrict search to this meeting only.

        Returns:
            List of dicts with keys:
                - text        (str)   chunk text
                - meeting_id  (str)
                - title       (str)   meeting title
                - date        (str)
                - chunk_index (int)
                - distance    (float) cosine distance (lower = more similar)

        Raises:
            ValueError: If query is empty or the store contains no documents.
        """
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")

        total = self._collection.count()
        if total == 0:
            logger.warning("Vector store is empty – no results available.")
            return []

        where = {"meeting_id": meeting_id} if meeting_id else None

        # Clamp n_results to items actually available
        effective_n = min(n_results, total)

        query_params: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": effective_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        raw = self._collection.query(**query_params)

        results: List[Dict[str, Any]] = []
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            results.append(
                {
                    "text": doc,
                    "meeting_id": meta.get("meeting_id", ""),
                    "title": meta.get("title", ""),
                    "date": meta.get("date", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": round(dist, 4),
                }
            )

        logger.info(
            f"Search '{query[:60]}' → {len(results)} result(s) "
            f"(filter meeting_id={meeting_id})."
        )
        return results

    def delete_meeting(self, meeting_id: str) -> None:
        """
        Remove all chunks belonging to a meeting from the store.

        Args:
            meeting_id: The meeting to delete.

        Raises:
            ValueError: If meeting_id is blank.
        """
        if not meeting_id or not meeting_id.strip():
            raise ValueError("meeting_id must not be empty.")
        self._delete_by_meeting(meeting_id)
        logger.info(f"Deleted all chunks for meeting_id={meeting_id}.")

    def list_meetings(self) -> List[Dict[str, str]]:
        """
        Return unique meeting metadata (id, title, date) for all stored meetings.

        Returns:
            List of dicts with keys: meeting_id, title, date.
        """
        total = self._collection.count()
        if total == 0:
            return []

        # Fetch all metadata (no document text needed)
        raw = self._collection.get(include=["metadatas"])
        metas = raw.get("metadatas", [])

        seen: Dict[str, Dict[str, str]] = {}
        for meta in metas:
            mid = meta.get("meeting_id", "")
            if mid and mid not in seen:
                seen[mid] = {
                    "meeting_id": mid,
                    "title": meta.get("title", ""),
                    "date": meta.get("date", ""),
                }

        meetings = list(seen.values())
        logger.info(f"Listed {len(meetings)} unique meeting(s).")
        return meetings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _delete_by_meeting(self, meeting_id: str) -> None:
        """Delete all chunks whose metadata.meeting_id == meeting_id."""
        try:
            self._collection.delete(where={"meeting_id": meeting_id})
        except Exception as e:
            # ChromaDB raises if no documents match the filter; safe to ignore
            logger.debug(f"Delete (may be empty): {e}")
