"""
chunker.py
----------
Splits a raw transcript string into overlapping text chunks suitable for
embedding and storage in a vector database.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def chunk_transcript(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Split a transcript into overlapping word-level chunks.

    Args:
        text:       Full transcript text (may include speaker labels like
                    "John: Hello everyone.").
        chunk_size: Approximate number of words per chunk.
        overlap:    Number of words to repeat at the start of the next chunk
                    so context is not lost at boundaries.

    Returns:
        List of dicts, each with:
            - chunk_index  (int)  zero-based position
            - text         (str)  the chunk text
            - start_word   (int)  index of the first word in the original list
            - end_word     (int)  index one past the last word

    Raises:
        ValueError: If text is empty or chunk_size < 1.
    """
    if not text or not text.strip():
        raise ValueError("Transcript text is empty – nothing to chunk.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    if not words:
        raise ValueError("Transcript contains no words after splitting.")

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            {
                "chunk_index": chunk_index,
                "text": " ".join(chunk_words),
                "start_word": start,
                "end_word": end,
            }
        )
        chunk_index += 1
        # Move forward by (chunk_size - overlap) so the next chunk re-uses
        # the last `overlap` words for context continuity.
        step = chunk_size - overlap
        start += step

    logger.info(
        f"Chunked transcript into {len(chunks)} chunk(s) "
        f"(chunk_size={chunk_size}, overlap={overlap})."
    )
    return chunks
