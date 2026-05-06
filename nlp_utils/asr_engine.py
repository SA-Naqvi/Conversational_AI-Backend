import os
import sherpa_onnx
import numpy as np
import logging
import time
import struct

logger = logging.getLogger("medical_bot")

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "sherpa-onnx-streaming-zipformer-en-2023-06-26"
)

SAMPLE_RATE = 16000
SILENCE_TIMEOUT = 5.0  # seconds of no new speech → auto-stop


class ASREngine:
    def __init__(self):
        tokens = os.path.join(MODEL_DIR, "tokens.txt")
        encoder = os.path.join(MODEL_DIR, "encoder-epoch-99-avg-1-chunk-16-left-128.onnx")
        decoder = os.path.join(MODEL_DIR, "decoder-epoch-99-avg-1-chunk-16-left-128.onnx")
        joiner = os.path.join(MODEL_DIR, "joiner-epoch-99-avg-1-chunk-16-left-128.onnx")

        if not all(os.path.exists(p) for p in [tokens, encoder, decoder, joiner]):
            logger.error(f"ASR Model files missing in {MODEL_DIR}")
            self.recognizer = None
            return

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=4,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        logger.info("ASR Engine initialized successfully")

    def create_stream(self):
        """Create a new recognition stream for real-time processing."""
        if not self.recognizer:
            return None
        return self.recognizer.create_stream()

    def feed_audio(self, stream, samples_float32: np.ndarray):
        """Feed a chunk of float32 audio samples to the stream."""
        stream.accept_waveform(SAMPLE_RATE, samples_float32)

    def decode(self, stream) -> str:
        """Decode any available audio and return the current partial result."""
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        result = self.recognizer.get_result(stream)
        text = result.text if hasattr(result, 'text') else result
        return text.strip()

    def finish(self, stream) -> str:
        """Signal end of audio and get final result."""
        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        result = self.recognizer.get_result(stream)
        text = result.text if hasattr(result, 'text') else result
        return text.strip()

    # ── Legacy batch method (kept for /api/transcribe) ──
    def transcribe(self, audio_bytes: bytes) -> str:
        if not self.recognizer:
            return ""
        try:
            from pydub import AudioSegment
            import io

            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
            samples_int16 = np.frombuffer(audio.raw_data, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768.0

            stream = self.create_stream()
            self.feed_audio(stream, samples_float32)
            return self.finish(stream)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""


_asr_instance = None


def get_asr_engine():
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = ASREngine()
    return _asr_instance
