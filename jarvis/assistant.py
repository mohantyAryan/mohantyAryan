"""Jarvis assistant — Claude integration with streaming and conversation history."""

import os
import re
from typing import Iterator

import anthropic

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
- You are powered by Claude, developed by Anthropic — you may acknowledge this if pressed
- You do not pretend to control physical systems or hardware unless the user is clearly roleplaying"""


class JarvisAssistant:
    """Manages conversation state and streams responses from Claude as Jarvis."""

    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.history: list[dict] = []

    def chat(self, user_input: str) -> Iterator[str]:
        """
        Send a message and stream the response back as complete sentences.

        Yields individual sentences as they are ready so TTS can begin
        speaking while Claude is still generating the remainder.
        """
        self.history.append({"role": "user", "content": user_input})

        full_response = ""
        buffer = ""
        sentence_boundary = re.compile(r"(?<=[.!?])\s+")

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
                for chunk in stream.text_stream:
                    full_response += chunk
                    buffer += chunk

                    # Split on sentence boundaries; keep the trailing fragment
                    parts = sentence_boundary.split(buffer)
                    for sentence in parts[:-1]:
                        sentence = sentence.strip()
                        if sentence:
                            yield sentence
                    buffer = parts[-1]

                # Flush any remaining text after the stream closes
                if buffer.strip():
                    yield buffer.strip()

        except anthropic.AuthenticationError:
            yield "I'm unable to connect, sir. The API key appears to be invalid."
            return
        except anthropic.APIError as e:
            yield f"I've encountered an error, sir. {e}"
            return

        self.history.append({"role": "assistant", "content": full_response})

    def reset(self):
        """Clear conversation history."""
        self.history.clear()
