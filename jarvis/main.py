"""J.A.R.V.I.S. — main entry point and conversation loop."""

import argparse
import json
import os
import sys
from pathlib import Path

from .assistant import ClaudeAssistant, GeminiAssistant, GroqAssistant
from .voice import STTEngine, TTSEngine

# Memory is stored in memory.json in the project root (next to run_jarvis.py)
MEMORY_FILE = Path(__file__).parent.parent / "memory.json"
MAX_MEMORY_MESSAGES = 200  # Keep last 200 messages (~100 exchanges)


def load_memory() -> list:
    """Load conversation history from disk."""
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_memory(history: list):
    """Save conversation history to disk, trimmed to the last MAX_MEMORY_MESSAGES."""
    trimmed = history[-MAX_MEMORY_MESSAGES:]
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"\n[JARVIS] Warning: memory could not be saved. {e}")

BANNER = """
  ================================================
    J . A . R . V . I . S .
    Just A Rather Very Intelligent System
  ================================================
    Commands : exit / quit / reset
    Voice    : --voice flag to enable mic input
    Mute     : --mute flag to disable speech output
    Backend  : --backend groq (default) | gemini | claude
  ================================================
"""


def _print_jarvis(text: str):
    print(f"\n\033[96mJARVIS\033[0m  {text}")


def _print_you(text: str):
    print(f"\033[93mYou\033[0m     {text}")


def run(voice_mode: bool, tts_engine: str, mute: bool, backend: str = "groq"):
    if backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print(
                "\n[ERROR] GROQ_API_KEY is not set.\n"
                "Add it to a .env file or export it in your shell:\n"
                "  export GROQ_API_KEY=your_key_here\n"
            )
            sys.exit(1)
        assistant = GroqAssistant(api_key=api_key, initial_history=load_memory())
    elif backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print(
                "\n[ERROR] GEMINI_API_KEY is not set.\n"
                "Add it to a .env file or export it in your shell:\n"
                "  export GEMINI_API_KEY=your_key_here\n"
            )
            sys.exit(1)
        assistant = GeminiAssistant(api_key=api_key, initial_history=load_memory())
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "\n[ERROR] ANTHROPIC_API_KEY is not set.\n"
                "Add it to a .env file or export it in your shell:\n"
                "  export ANTHROPIC_API_KEY=your_key_here\n"
            )
            sys.exit(1)
        assistant = ClaudeAssistant(api_key=api_key, initial_history=load_memory())

    tts = TTSEngine(engine=tts_engine, mute=mute)
    stt = STTEngine() if voice_mode else None

    print(BANNER)

    # Personalized greeting if memory exists, standard greeting if not
    if assistant.history:
        print("\n\033[90m[Resuming previous session...]\033[0m")
        print("\n\033[96mJARVIS\033[0m  ", end="", flush=True)
        resume_prompt = (
            "You are resuming our conversation after some time apart. "
            "Based on what you know about me from our previous discussions, "
            "greet me personally — use my name if you know it, and briefly "
            "reference something we discussed before. Keep it natural and brief."
        )
        # Generate greeting WITHOUT adding the resume prompt to saved history
        for sentence in assistant.chat(resume_prompt):
            print(sentence + " ", end="", flush=True)
            tts.speak(sentence)
        print()
        # Remove the resume prompt + greeting from history so it doesn't pollute memory
        assistant.history = assistant.history[:-2]
    else:
        greeting = (
            "Good day, sir. J.A.R.V.I.S. online. All systems nominal. "
            "How may I assist you?"
        )
        _print_jarvis(greeting)
        tts.speak(greeting)

    while True:
        try:
            user_input = _get_input(voice_mode, stt, tts)
        except KeyboardInterrupt:
            farewell = "Until next time, sir."
            print()
            _print_jarvis(farewell)
            tts.speak(farewell)
            save_memory(assistant.history)
            break

        if user_input is None:
            continue

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "goodbye", "bye"):
            farewell = "Understood, sir. Shutting down. Good day."
            _print_jarvis(farewell)
            tts.speak(farewell)
            save_memory(assistant.history)
            break

        if user_input.lower() in ("reset", "clear"):
            assistant.reset()
            MEMORY_FILE.unlink(missing_ok=True)
            msg = "Memory wiped, sir. Starting with a clean slate."
            _print_jarvis(msg)
            tts.speak(msg)
            continue

        # Stream response sentence-by-sentence
        print(f"\n\033[96mJARVIS\033[0m  ", end="", flush=True)
        for sentence in assistant.chat(user_input):
            print(sentence + " ", end="", flush=True)
            tts.speak(sentence)
        print()
        save_memory(assistant.history)  # Save after every response


def _get_input(voice_mode: bool, stt, tts) -> str | None:
    """Return the next user utterance, via mic or keyboard."""
    if voice_mode and stt is not None:
        print("\n\033[90m[Listening — speak now, or Ctrl+C to switch to text]\033[0m")
        text = stt.listen()
        if text:
            _print_you(text)
        else:
            print("\033[90m[No input detected — try again]\033[0m")
        return text
    else:
        try:
            raw = input("\n\033[93mYou\033[0m     ")
        except EOFError:
            return "exit"
        return raw


def main():
    parser = argparse.ArgumentParser(
        description="J.A.R.V.I.S. — AI voice assistant powered by Claude"
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable microphone input (requires SpeechRecognition + PyAudio)",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Disable voice output (text only)",
    )
    parser.add_argument(
        "--tts",
        choices=["pyttsx3", "gtts"],
        default="pyttsx3",
        help="TTS engine: pyttsx3 (offline, default) or gtts (Google, British accent)",
    )
    parser.add_argument(
        "--backend",
        choices=["groq", "gemini", "claude"],
        default="groq",
        help="AI backend: groq (default, uses GROQ_API_KEY), gemini (uses GEMINI_API_KEY), claude (uses ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    run(voice_mode=args.voice, tts_engine=args.tts, mute=args.mute, backend=args.backend)


if __name__ == "__main__":
    main()
