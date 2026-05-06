"""
Whisper-based ASR Engine for high-accuracy batch transcription.
Uses faster-whisper (CTranslate2) for optimized local inference.
"""
import os
import io
import logging
import numpy as np

# Set environment variables for Hugging Face Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

logger = logging.getLogger("medical_bot")

WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large-v3


class WhisperEngine:
    def __init__(self):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
            logger.info(f"Whisper Engine ({WHISPER_MODEL_SIZE}) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            self.model = None

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes (any format) to text using Whisper."""
        if not self.model:
            return ""

        try:
            from pydub import AudioSegment

            # Decode any audio format to raw PCM via pydub+ffmpeg
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

            # Convert to float32 numpy array
            samples_int16 = np.frombuffer(audio.raw_data, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768.0

            # faster-whisper can transcribe from a numpy array
            segments, info = self.model.transcribe(
                samples_float32,
                beam_size=5,
                language="en",
                vad_filter=True,
            )

            text = " ".join(segment.text.strip() for segment in segments)
            logger.info(f"Whisper transcription ({info.duration:.1f}s audio): {text[:80]}...")
            return text.strip()

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""


_whisper_instance = None


def get_whisper_engine():
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = WhisperEngine()
    return _whisper_instance
