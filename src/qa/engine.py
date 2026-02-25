"""
engine.py
---------
RAG (Retrieval-Augmented Generation) Q&A engine for meeting transcripts.

Flow
----
1. Use the vector store to retrieve the top-k most relevant chunks.
2. Build a prompt that contains those chunks as context.
3. Call Groq Llama 3 to answer the user's question, citing sources.
4. Return a structured dict: answer + citations.
"""

import os
import logging
from typing import Optional, List, Dict, Any

from groq import Groq
from dotenv import load_dotenv

from src.vector_db.store import MeetingVectorStore

load_dotenv()

logger = logging.getLogger(__name__)

QA_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You are a helpful meeting assistant that answers questions based ONLY on the
meeting transcript excerpts provided below.

Rules:
- Answer in 1–3 clear sentences.
- If the answer is present, quote or paraphrase the relevant part.
- Always end your answer with a "Sources:" line that lists each meeting title
  and date you used, e.g.: Sources: Sprint Review (2026-02-15)
- If the answer cannot be found in the provided excerpts, reply with exactly:
  "I don't have enough information in the provided meeting transcripts to answer that."
- Do NOT make up facts beyond what is in the excerpts.
"""


class MeetingQAEngine:
    """
    Retrieval-Augmented Generation engine over stored meeting transcripts.

    Usage
    -----
    engine = MeetingQAEngine(vector_store=store, groq_api_key="gsk_...")
    result = engine.ask("When is the product launch?")
    print(result["answer"])
    print(result["citations"])
    """

    def __init__(
        self,
        vector_store: MeetingVectorStore,
        groq_api_key: Optional[str] = None,
        model: str = QA_MODEL,
    ) -> None:
        """
        Args:
            vector_store: Initialised MeetingVectorStore instance.
            groq_api_key: Groq API key (falls back to GROQ_API_KEY env var).
            model:        Groq model to use for generation.
        """
        self._store = vector_store
        self._model = model

        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Pass it to MeetingQAEngine or add it to .env."
            )
        self._client = Groq(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        meeting_id: Optional[str] = None,
        n_context: int = 5,
    ) -> Dict[str, Any]:
        """
        Answer a question using retrieved meeting context.

        Args:
            question:   Natural language question.
            meeting_id: Optional meeting ID to restrict retrieval to.
            n_context:  Number of context chunks to retrieve.

        Returns:
            dict with keys:
                - answer     (str)        The LLM-generated answer.
                - citations  (list[dict]) List of source chunks used.
                               Each dict: {text, meeting_id, title, date, chunk_index}
                - model      (str)        Model used.
                - question   (str)        Original question.

        Raises:
            ValueError:   If question is empty.
            RuntimeError: If Groq API call fails.
        """
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        # 1. Retrieve relevant chunks
        chunks = self._store.search(
            query=question,
            n_results=n_context,
            meeting_id=meeting_id,
        )

        if not chunks:
            logger.warning("No relevant chunks found – returning fallback answer.")
            return {
                "answer": (
                    "I don't have enough information in the provided meeting "
                    "transcripts to answer that."
                ),
                "citations": [],
                "model": self._model,
                "question": question,
            }

        # 2. Build context block
        context_block = self._build_context(chunks)

        # 3. Call LLM
        user_message = (
            f"Context from meeting transcripts:\n\n{context_block}\n\n"
            f"Question: {question.strip()}"
        )

        logger.info(
            f"Asking '{question[:60]}' with {len(chunks)} context chunk(s)."
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"Q&A generation failed: {e}") from e

        # 4. Build citation list (unique meetings referenced)
        citations = self._extract_citations(chunks)

        logger.info(f"Answer generated. Citations: {[c['title'] for c in citations]}")

        return {
            "answer": answer,
            "citations": citations,
            "model": self._model,
            "question": question,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a numbered context block."""
        lines: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            lines.append(
                f"[{i}] Meeting: {chunk['title']} ({chunk['date']})\n"
                f"    {chunk['text']}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _extract_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return deduplicated citation dicts from the retrieved chunks."""
        seen = set()
        citations: List[Dict[str, Any]] = []
        for chunk in chunks:
            key = (chunk["meeting_id"], chunk["chunk_index"])
            if key not in seen:
                seen.add(key)
                citations.append(
                    {
                        "text": chunk["text"],
                        "meeting_id": chunk["meeting_id"],
                        "title": chunk["title"],
                        "date": chunk["date"],
                        "chunk_index": chunk["chunk_index"],
                    }
                )
        return citations
