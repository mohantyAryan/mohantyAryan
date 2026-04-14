"""J.A.R.V.I.S. — main entry point and conversation loop."""

import argparse
import os
import sys

from .assistant import ClaudeAssistant, GeminiAssistant
from .voice import STTEngine, TTSEngine

BANNER = r"""
  ____  _   _    _    ____  __     _____ ____
 / ___|| | | |  / \  |  _ \ \ \   / /_ _/ ___|
 \___ \| |_| | / _ \ | |_) | \ \ / / | |\___ \
  ___) |  _  |/ ___ \|  _ <   \ V /  | | ___) |
 |____/|_| |_/_/   \_\_| \_\   \_/  |___|____/

  Just A Rather Very Intelligent System
  ─────────────────────────────────────────
  Commands : exit / quit / reset
  Voice    : --voice flag to enable mic input
  Mute     : --mute flag to disable speech output
  Backend  : --backend claude (default) | gemini
"""


def _print_jarvis(text: str):
    print(f"\n\033[96mJARVIS\033[0m  {text}")


def _print_you(text: str):
    print(f"\033[93mYou\033[0m     {text}")


def run(voice_mode: bool, tts_engine: str, mute: bool, backend: str = "claude"):
    if backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print(
                "\n[ERROR] GEMINI_API_KEY is not set.\n"
                "Add it to a .env file or export it in your shell:\n"
                "  export GEMINI_API_KEY=your_key_here\n"
            )
            sys.exit(1)
        assistant = GeminiAssistant(api_key=api_key)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "\n[ERROR] ANTHROPIC_API_KEY is not set.\n"
                "Add it to a .env file or export it in your shell:\n"
                "  export ANTHROPIC_API_KEY=your_key_here\n"
            )
            sys.exit(1)
        assistant = ClaudeAssistant(api_key=api_key)
    tts = TTSEngine(engine=tts_engine, mute=mute)
    stt = STTEngine() if voice_mode else None

    print(BANNER)

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
            break

        if user_input is None:
            # Microphone returned nothing — just loop
            continue

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "goodbye", "bye"):
            farewell = "Understood, sir. Shutting down. Good day."
            _print_jarvis(farewell)
            tts.speak(farewell)
            break

        if user_input.lower() in ("reset", "clear"):
            assistant.reset()
            msg = "Conversation history cleared, sir. Starting fresh."
            _print_jarvis(msg)
            tts.speak(msg)
            continue

        # Stream response sentence-by-sentence
        print(f"\n\033[96mJARVIS\033[0m  ", end="", flush=True)
        for sentence in assistant.chat(user_input):
            print(sentence + " ", end="", flush=True)
            tts.speak(sentence)
        print()  # newline after full response


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
        choices=["gemini", "claude"],
        default="gemini",
        help="AI backend: gemini (default, uses GEMINI_API_KEY) or claude (uses ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    run(voice_mode=args.voice, tts_engine=args.tts, mute=args.mute, backend=args.backend)


if __name__ == "__main__":
    main()
