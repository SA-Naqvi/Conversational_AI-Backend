"""
TTS Engine using Piper for the Medical Recovery Companion.
Converts text to speech using the Amy (low) voice model.
"""
import os
import io
import wave
import logging

logger = logging.getLogger("medical_bot")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "piper-amy-low", "en_US-amy-low.onnx"
)
CONFIG_PATH = MODEL_PATH + ".json"


class TTSEngine:
    def __init__(self):
        from piper import PiperVoice

        if not os.path.exists(MODEL_PATH):
            logger.error(f"TTS model not found at {MODEL_PATH}")
            self.voice = None
            return

        self.voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)
        self.sample_rate = self.voice.config.sample_rate
        logger.info("TTS Engine (Piper Amy) initialized successfully")

    def synthesize(self, text: str) -> bytes:
        """Convert text to WAV bytes (16-bit mono PCM)."""
        if not self.voice or not text.strip():
            return b""

        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                self.voice.synthesize_wav(text, wav_file)

            return buffer.getvalue()
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return b""


_tts_instance = None


def get_tts_engine():
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEngine()
    return _tts_instance
