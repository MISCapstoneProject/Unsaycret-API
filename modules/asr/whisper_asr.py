from pathlib import Path
from utils.logger import get_logger
import torch
import torchaudio

from .asr_model import load_model
from .text_utils import merge_char_to_word

logger = get_logger(__name__)

class WhisperASR:
    """
    Whisper-based ASR wrapper.

    Usage:
        asr = WhisperASR(model_name="medium", gpu=True)
        text, confidence, words = asr.transcribe("path/to/audio.wav")
    """

    def __init__(self, model_name: str = "medium", gpu: bool = False, beam: int = 5, lang: str = "auto"):
        self.gpu = gpu
        self.beam = beam
        self.lang = lang
        self.model = load_model(model_name=model_name, gpu=self.gpu)

        device_str = "cuda" if self.gpu else "cpu"
        logger.info(f"🧠 Whisper running on device: {device_str}")

    def transcribe(self, wav_path: str) -> tuple[str, float, list[dict]]:
        """
        Transcribe a single audio file.

        Args:
            wav_path: Path to the WAV file.

        Returns:
            full_txt: The transcript string.
            avg_conf: Average confidence score.
            word_info: List of dicts with keys 'start', 'end', 'word', 'probability'.
        """
        # Run model inference
        seg_gen, _ = self.model.transcribe(
            str(wav_path),
            word_timestamps=True,
            vad_filter=False,
            beam_size=self.beam,
            language=None if self.lang == "auto" else self.lang,
        )

        segments = list(seg_gen)
        if not segments:
            return "", 0.0, []

        # Combine segment texts
        full_txt = "".join(s.text for s in segments).strip()
        # Flatten word-level timestamps
        words = [w for s in segments for w in (s.words or [])]

        if words:
            probs = [w.probability for w in words]
            avg_conf = float(sum(probs) / len(probs))
            word_info = [
                {"start": float(w.start), "end": float(w.end),
                 "word": str(w.word), "probability": float(w.probability)}
                for w in words
            ]
        else:
            # Fallback to segment-level log probability
            avg_conf = float(sum(s.avg_logprob for s in segments) / len(segments))
            word_info = []

        # Clear GPU cache to avoid fragmentation
        torch.cuda.empty_cache()

        return full_txt, avg_conf, word_info
