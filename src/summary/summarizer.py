import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUMMARY_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert meeting assistant. Your task is to analyze meeting transcripts and produce structured summaries.

Always respond with a valid JSON object using exactly these keys:
{
  "summary": "<concise 2-4 sentence overview of the meeting>",
  "key_points": ["<point 1>", "<point 2>", ...],
  "action_items": ["<task> — <owner if mentioned>", ...],
  "decisions": ["<decision 1>", ...]
}

Rules:
- If no action items were discussed, return an empty list for "action_items".
- If no decisions were made, return an empty list for "decisions".
- Keep key_points to the most important 3-7 items.
- Be concise and factual — do not invent information not in the transcript.
"""


def summarize_transcript(transcript: str, context: str = None) -> dict:
    """
    Generate a structured meeting summary from a transcript.

    Args:
        transcript: Full meeting transcript text.
        context:    Optional extra context (e.g. meeting title, date, participants).

    Returns:
        dict with keys:
            - summary       (str)        concise paragraph overview
            - key_points    (list[str])  main discussion topics
            - action_items  (list[str])  tasks with owners
            - decisions     (list[str])  key decisions made

    Raises:
        ValueError:   transcript is empty or too short
        RuntimeError: Groq API or JSON parsing error
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty. Nothing to summarize.")

    if len(transcript.strip()) < 20:
        raise ValueError("Transcript is too short to summarize meaningfully.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")

    client = Groq(api_key=api_key)

    user_message = transcript.strip()
    if context:
        user_message = f"Context: {context.strip()}\n\nTranscript:\n{user_message}"
    else:
        user_message = f"Transcript:\n{user_message}"

    logger.info(f"Summarizing transcript ({len(transcript)} chars) …")

    try:
        response = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,       # Lower temperature for more consistent output
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)

        # Validate expected keys are present
        required_keys = {"summary", "key_points", "action_items", "decisions"}
        missing = required_keys - result.keys()
        if missing:
            raise ValueError(f"API response missing keys: {missing}")

        logger.info("Summary generated successfully.")
        return result

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse summary JSON from API: {e}") from e
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise RuntimeError(f"Summarization failed: {str(e)}") from e
