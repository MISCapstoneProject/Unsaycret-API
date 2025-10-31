from pathlib import Path
import json
import time
from utils.logger import get_logger
from utils.constants import DEFAULT_WHISPER_MODEL, DEFAULT_WHISPER_BEAM_SIZE
import torch
import torchaudio
import re
import numpy as np
from typing import Iterable

from .asr_model import load_model
from .text_utils import merge_char_to_word

# ===== Simplified → Traditional (TW) support =====
try:
    from opencc import OpenCC
    _OPENCC = OpenCC('s2twp')  # 簡→繁（臺灣用字＋標點）
except Exception:
    _OPENCC = None

logger = get_logger(__name__)

ALLOWED_KW = {
    "beam_size", "best_of", "patience", "length_penalty",
    "temperature", "compression_ratio_threshold", "log_prob_threshold",
    "no_speech_threshold", "condition_on_previous_text", "initial_prompt",
    "prefix", "suppress_blank", "suppress_tokens", "without_timestamps",
    "max_initial_timestamp", "word_timestamps", "vad_filter", "vad_parameters",
    "language", "task", "chunk_length", "prepend_punctuations",
    "append_punctuations", "hallucination_silence_threshold", "hotwords",
    "hotword_timestamps",
}

_TRUE_SET = {"1", "true", "yes", "y", "t", "on"}
_FALSE_SET = {"0", "false", "no", "n", "f", "off"}


def _to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in _TRUE_SET:
            return True
        if s in _FALSE_SET:
            return False
    return val  # 不強轉，交給下游


def _to_list_int_from_any(v):
    """
    規範 suppress_tokens：
      - "-1"            -> [-1]
      - "1,2  3"        -> [1,2,3]
      -  -1/1.0/" 3 "   -> [int(v)]
      - [ "1","2" ]     -> [1,2]
      - None/空字串     -> None（呼叫點若想丟掉可自行處理）
    """
    if v is None:
        return None
    if isinstance(v, str):
        txt = v.strip()
        if not txt:
            return None
        if txt == "-1":
            return [-1]
        parts = re.split(r"[,\s]+", txt)
        out = []
        for p in parts:
            if not p:
                continue
            out.append(int(float(p)))
        return out
    if isinstance(v, (int, float)):
        return [int(float(v))]
    if isinstance(v, (list, tuple)):
        return [int(float(x)) for x in v]
    return v


def _to_temperature(v):
    """
    faster-whisper 接受 float 或 Iterable[float]。
    允許 "0.0" / "0.2,0.4,0.6" 這種字串。
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if "," in s or " " in s:
            parts = re.split(r"[,\s]+", s)
            return [float(p) for p in parts if p]
        return float(s)
    if isinstance(v, Iterable) and not isinstance(v, (bytes, str)):
        return [float(x) for x in v]
    return v


def _normalize_asr_kwargs(kwargs: dict | None) -> dict:
    """
    Backward compatibility + type normalization shim
    to match faster-whisper keyword expectations.
    """
    kw = dict(kwargs or {})

    # ---- 1) Backward-compat: names ----
    if "logprob_threshold" in kw and "log_prob_threshold" not in kw:
        kw["log_prob_threshold"] = kw.pop("logprob_threshold")

    # ---- 2) Type fixes / coercions ----
    # 2a) suppress_tokens → List[int]
    if "suppress_tokens" in kw:
        fixed = _to_list_int_from_any(kw["suppress_tokens"])
        if fixed is not None:
            kw["suppress_tokens"] = fixed
        else:
            # 交由 allowlist 過濾掉
            kw.pop("suppress_tokens", None)

    # 2b) booleans（常見從 CLI/ENV 以字串進入）
    for bkey in (
        "suppress_blank", "vad_filter", "word_timestamps", "hotword_timestamps",
        "condition_on_previous_text", "without_timestamps",
    ):
        if bkey in kw:
            kw[bkey] = _to_bool(kw[bkey])

    # 2c) vad_parameters: 若是 JSON 字串，嘗試解析
    if "vad_parameters" in kw and isinstance(kw["vad_parameters"], str):
        try:
            kw["vad_parameters"] = json.loads(kw["vad_parameters"])
        except Exception:
            # 無效 JSON 就讓它維持原樣（但仍可能被 allowlist 保留/濾掉）
            pass

    # 2d) temperature: 支援字串 "0.0" 或 "0.2,0.4"
    if "temperature" in kw:
        tmp = _to_temperature(kw["temperature"])
        if tmp is not None:
            kw["temperature"] = tmp
        else:
            kw.pop("temperature", None)

    # 2e) hotwords: faster-whisper 接受 str|None；如果是 list，自動用逗號串接
    if "hotwords" in kw:
        hv = kw["hotwords"]
        if isinstance(hv, (list, tuple)):
            kw["hotwords"] = ",".join(map(str, hv))
        elif isinstance(hv, (int, float)):
            kw["hotwords"] = str(hv)
        # 字串就照原樣

    # ---- 3) Allowlist：只保留模型實際支援的參數 ----
    kw = {k: v for k, v in kw.items() if k in ALLOWED_KW}
    return kw


class WhisperASR:
    """
    Whisper-based ASR wrapper.

    Usage:
        asr = WhisperASR(model_name="medium", gpu=True)
        text, confidence, words = asr.transcribe("path/to/audio.wav")
    """

    def __init__(self, model_name: str = None, gpu: bool = False, beam: int = None, lang: str = "zh"):
        self.gpu = gpu
        self.beam = beam if beam is not None else DEFAULT_WHISPER_BEAM_SIZE
        self.lang = lang

        # 使用環境變數的預設模型名稱
        model_name = model_name if model_name is not None else DEFAULT_WHISPER_MODEL
        self.model = load_model(model_name=model_name, gpu=self.gpu)

        self.last_infer_time = 0.0
        self.last_total_time = 0.0

        device_str = "cuda" if self.gpu else "cpu"
        logger.info(f"🧠 Whisper running on device: {device_str} (model: {model_name})")
        self.cc = _OPENCC  # None 表示環境沒裝 opencc，則不轉

    def transcribe(self, wav_path: str, **kwargs) -> tuple[str, float, list[dict]]:
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

        options = {
            "word_timestamps": True,
            "vad_filter": kwargs.pop("vad_filter", False),
            "beam_size": kwargs.pop("beam_size", self.beam),
            "language": kwargs.pop("language", None if self.lang == "auto" else self.lang),
        }
        options.update(kwargs)

        seg_gen, _ = self.model.transcribe(
            str(wav_path),
            **_normalize_asr_kwargs(options),
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
            avg_conf = float(sum(probs) / len(probs)) if probs else 0.0
            word_info = [
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": str(w.word),
                    "probability": float(w.probability),
                }
                for w in words
            ]
        else:
            # Fallback to segment-level log probability
            avg_conf = float(sum(s.avg_logprob for s in segments) / len(segments))
            word_info = []

        # Clear GPU cache to avoid fragmentation
        if self.gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 簡轉繁（臺灣用字＋標點）
        do_conv = (self.cc is not None) and (
            self.lang in (None, "zh", "auto") or str(self.lang).startswith("zh")
        )
        if do_conv:
            if full_txt:
                full_txt = self.cc.convert(full_txt)
            if word_info:
                for wi in word_info:
                    if "word" in wi and isinstance(wi["word"], str):
                        wi["word"] = self.cc.convert(wi["word"])

        self.last_infer_time = infer_end - infer_start
        self.last_total_time = time.perf_counter() - total_start

        return full_txt, avg_conf, word_info

    def transcribe_tensor(self, audio, **kwargs) -> tuple[str, float, list[dict]]:
        """
        Transcribe in-memory mono float32 audio buffer at ASR sample rate.
        """
        total_start = time.perf_counter()
        infer_start = time.perf_counter()

        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio)

        if audio_np.ndim > 1:
            audio_np = np.reshape(audio_np, (-1,))
        audio_np = audio_np.astype("float32", copy=False)

        options = {
            "word_timestamps": True,
            "beam_size": 1,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "suppress_tokens": "-1",
            "language": "zh",
        }
        options.update(kwargs)

        seg_gen, _ = self.model.transcribe(
            audio_np,
            **_normalize_asr_kwargs(options),
        )
        infer_end = time.perf_counter()

        segments = list(seg_gen)
        if not segments:
            self.last_infer_time = infer_end - infer_start
            self.last_total_time = time.perf_counter() - total_start
            return "", 0.0, []

        full_txt = "".join(s.text for s in segments).strip()
        words = [w for s in segments for w in (s.words or [])]

        if words:
            probs = [w.probability for w in words]
            avg_conf = float(sum(probs) / len(probs)) if probs else 0.0
            word_info = [
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": str(w.word),
                    "probability": float(w.probability),
                }
                for w in words
            ]
        else:
            avg_conf = float(sum(s.avg_logprob for s in segments) / len(segments))
            word_info = []

        if self.gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        do_conv = (self.cc is not None) and (
            self.lang in (None, "zh", "auto") or str(self.lang).startswith("zh")
        )
        if do_conv:
            if full_txt:
                full_txt = self.cc.convert(full_txt)
            if word_info:
                for wi in word_info:
                    if "word" in wi and isinstance(wi["word"], str):
                        wi["word"] = self.cc.convert(wi["word"])

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
