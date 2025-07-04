from pathlib import Path
import json
import time
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
        self.last_infer_time = 0.0
        self.last_total_time = 0.0

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
        total_start = time.perf_counter()
        infer_start = time.perf_counter()
        seg_gen, _ = self.model.transcribe(
            str(wav_path),
            word_timestamps=True,
            vad_filter=False,
            beam_size=self.beam,
            language=None if self.lang == "auto" else self.lang,
        )
        infer_end = time.perf_counter()

        segments = list(seg_gen)
        if not segments:
            self.last_infer_time = infer_end - infer_start
            self.last_total_time = time.perf_counter() - total_start
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

        self.last_infer_time = infer_end - infer_start
        self.last_total_time = time.perf_counter() - total_start

        return full_txt, avg_conf, word_info

    def transcribe_dir(self, input_dir: str, output_id: str) -> str:
        """Transcribe all wav files in a directory and save to JSON.

        Args:
            input_dir: Directory containing wav files.
            output_id: Identifier for the output folder under ``data``.

        Returns:
            Path to the generated JSON file.
        """
        dir_path = Path(input_dir)
        out_dir = Path("data") / output_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_list = []
        for wav in sorted(dir_path.glob("*.wav")):
            text, conf, words = self.transcribe(str(wav))
            out_list.append(
                {
                    "file": wav.name,
                    "text": text,
                    "confidence": conf,
                    "words": words,
                }
            )
        out_path = out_dir / "asr.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)
        return str(out_path)
