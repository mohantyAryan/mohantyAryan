"""Voice I/O — text-to-speech and speech-to-text engines."""

import os
import tempfile
from typing import Optional


class TTSEngine:
    """
    Text-to-speech engine.

    Supports pyttsx3 (offline, default) and gTTS (Google, British accent).
    Falls back to silent mode if the chosen engine cannot be initialised.
    """

    def __init__(self, engine: str = "pyttsx3", mute: bool = False):
        self.engine_name = engine
        self.mute = mute
        self._engine = None

        if not mute:
            self._init()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init(self):
        if self.engine_name == "pyttsx3":
            self._init_pyttsx3()
        # gTTS is initialised lazily per call (no persistent object needed)

    def _init_pyttsx3(self):
        try:
            import pyttsx3

            engine = pyttsx3.init()
            self._configure_pyttsx3_voice(engine)
            self._engine = engine
        except Exception as e:
            print(f"[JARVIS] TTS init failed ({e}). Running in text-only mode.")
            self.mute = True

    def _configure_pyttsx3_voice(self, engine):
        """Prefer a deeper male voice; slow the pace slightly for gravitas."""
        voices = engine.getProperty("voices")
        preferred_id = None

        # Keywords that hint at a deeper male voice
        male_hints = ("david", "daniel", "james", "male", "english", "british")
        for v in voices:
            name = (v.name or "").lower()
            if any(kw in name for kw in male_hints):
                preferred_id = v.id
                break

        if preferred_id:
            engine.setProperty("voice", preferred_id)

        engine.setProperty("rate", 155)    # Measured, deliberate pace
        engine.setProperty("volume", 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str):
        """Speak the given text. No-op when muted or text is empty."""
        if self.mute or not text.strip():
            return

        if self.engine_name == "gtts":
            self._speak_gtts(text)
        elif self._engine is not None:
            self._speak_pyttsx3(text)

    # ------------------------------------------------------------------
    # Engine implementations
    # ------------------------------------------------------------------

    def _speak_pyttsx3(self, text: str):
        self._engine.say(text)
        self._engine.runAndWait()

    def _speak_gtts(self, text: str):
        try:
            from gtts import gTTS
            import pygame

            tts = gTTS(text=text, lang="en", tld="co.uk")  # British English

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tmp_path = f.name
            tts.save(tmp_path)

            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            os.unlink(tmp_path)
        except Exception as e:
            print(f"[JARVIS] gTTS error: {e}")


# ---------------------------------------------------------------------------


class STTEngine:
    """
    Speech-to-text engine using the SpeechRecognition library.

    Uses Google's free speech recognition API. Requires a microphone and an
    active internet connection.
    """

    def __init__(self):
        self._available = False
        self._init()

    def _init(self):
        try:
            import speech_recognition as sr  # noqa: F401

            self._available = True
        except ImportError:
            print(
                "[JARVIS] SpeechRecognition not installed. "
                "Voice input unavailable — run: pip install SpeechRecognition pyaudio"
            )

    def listen(self, timeout: int = 8, phrase_limit: int = 20) -> Optional[str]:
        """
        Listen for a spoken phrase and return it as text.

        Returns None on silence, unintelligible audio, or network errors.
        """
        if not self._available:
            return None

        import speech_recognition as sr

        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True

        try:
            with sr.Microphone() as source:
                # Brief ambient calibration each call keeps accuracy high
                recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_limit
                )

            return recognizer.recognize_google(audio)

        except sr.WaitTimeoutError:
            return None  # Silence — no error needed
        except sr.UnknownValueError:
            return None  # Unintelligible — caller decides how to handle
        except sr.RequestError as e:
            print(f"[JARVIS] STT network error: {e}")
            return None
        except OSError as e:
            print(f"[JARVIS] Microphone error: {e}")
            self._available = False
            return None
