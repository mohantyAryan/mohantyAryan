"""Jarvis assistant — Claude and Gemini backends with streaming and conversation history."""

import os
import re
from typing import Iterator

JARVIS_SYSTEM_PROMPT = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), a highly advanced artificial intelligence assistant. You serve your user with unwavering loyalty, precision, and a touch of refined British wit.

PERSONALITY AND TONE:
- You speak in formal British English with dry, understated wit
- You address the user as "sir" naturally — not every sentence, but consistently
- You are calm, composed, and unflappable under any circumstances
- Your humor is dry and subtle — never obvious, never over the top
- You are never sycophantic. Do not open responses with "Certainly!", "Absolutely!", or "Great question!"
- You are deeply loyal and take your role seriously
- You are confident without being arrogant

COMMUNICATION STYLE:
- Be concise by default. Elaborate only when the topic genuinely requires depth or when asked
- Do not use markdown formatting — no bullet points, no headers, no asterisks. You are speaking aloud
- Use natural, flowing sentences suited for spoken conversation
- When uncertain, say so plainly rather than speculating without acknowledgment
- Avoid hollow filler phrases
- Preferred expressions: "Right away, sir.", "I've completed the analysis.", "Shall I proceed?",
  "Might I suggest...", "As you wish.", "Indeed.", "Understood.", "Allow me to elaborate."

CAPABILITIES:
- You have broad knowledge across science, technology, history, culture, mathematics, and engineering
- You can assist with analysis, writing, coding, research, and general problem-solving
- You approach every problem methodically and logically
- You maintain full context of the ongoing conversation

IDENTITY:
- You are JARVIS. Maintain this persona at all times
- If asked whether you are an AI, confirm it plainly and return to the task at hand
- If pressed about your underlying technology, you may acknowledge it briefly, then move on
- You do not pretend to control physical systems or hardware unless the user is clearly roleplaying"""


def _stream_sentences(text_iter) -> Iterator[str]:
    """Buffer streaming text chunks and yield complete sentences."""
    sentence_boundary = re.compile(r"(?<=[.!?])\s+")
    buffer = ""
    for chunk in text_iter:
        buffer += chunk
        parts = sentence_boundary.split(buffer)
        for sentence in parts[:-1]:
            sentence = sentence.strip()
            if sentence:
                yield sentence
        buffer = parts[-1]
    if buffer.strip():
        yield buffer.strip()


class ClaudeAssistant:
    """Jarvis backed by Claude (Anthropic)."""

    def __init__(self, api_key: str = None):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.history: list[dict] = []

    def chat(self, user_input: str) -> Iterator[str]:
        self.history.append({"role": "user", "content": user_input})
        full_response = ""

        try:
            with self.client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": JARVIS_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=self.history,
            ) as stream:
                for sentence in _stream_sentences(stream.text_stream):
                    full_response += sentence + " "
                    yield sentence

        except self.anthropic.AuthenticationError:
            yield "I'm unable to connect, sir. The API key appears to be invalid."
            return
        except self.anthropic.APIError as e:
            yield f"I've encountered an error, sir. {e}"
            return

        self.history.append({"role": "assistant", "content": full_response.strip()})

    def reset(self):
        self.history.clear()


# Keep the original name as an alias for backwards compatibility
JarvisAssistant = ClaudeAssistant


class GeminiAssistant:
    """Jarvis backed by Gemini (Google)."""

    def __init__(self, api_key: str = None):
        from google import genai
        self.genai = genai
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY")
        )
        # Gemini history format: [{"role": "user"|"model", "parts": [{"text": "..."}]}]
        self.history: list[dict] = []

    def chat(self, user_input: str) -> Iterator[str]:
        from google.genai import types

        self.history.append({"role": "user", "parts": [{"text": user_input}]})
        full_response = ""

        try:
            stream = self.client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=JARVIS_SYSTEM_PROMPT,
                    max_output_tokens=1024,
                ),
            )

            def _text_chunks():
                for chunk in stream:
                    if chunk.text:
                        yield chunk.text

            for sentence in _stream_sentences(_text_chunks()):
                full_response += sentence + " "
                yield sentence

        except Exception as e:
            yield f"I've encountered an error, sir. {e}"
            return

        self.history.append({"role": "model", "parts": [{"text": full_response.strip()}]})

    def reset(self):
        self.history.clear()
