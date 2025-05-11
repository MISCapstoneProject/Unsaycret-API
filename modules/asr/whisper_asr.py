# whisper_asr.py
from pathlib import Path
import json
from .asr_model import load_model
from .text_utils import merge_char_to_word

class WhisperASR:
    def __init__(self, model_name="medium", gpu=False, beam=5, lang="auto"):
        self.model = load_model(model_name=model_name, gpu=gpu)
        self.beam = beam
        self.lang = lang

    def transcribe_dir(self, input_dir: str, output_id: str) -> str:
        wav_list = sorted(Path(input_dir).glob("*.wav"))
        if not wav_list:
            raise FileNotFoundError("❌ No .wav found in input_dir")

        results = []

        for wav in wav_list:
            print("🚀", wav.name)
            seg_gen, _ = self.model.transcribe(
                str(wav),
                word_timestamps=True,
                vad_filter=False,
                beam_size=self.beam,
                language=None if self.lang == "auto" else self.lang
            )
            segments = list(seg_gen)
            if not segments:
                continue

            full_txt = "".join(s.text for s in segments).strip()
            char_words = [{
                "start": float(w.start),
                "end": float(w.end),
                "word": str(w.word),
                "probability": float(w.probability)
            } for s in segments for w in (s.words or [])]
            word_level = merge_char_to_word(full_txt, char_words)

            results.append({
                "track_id": wav.stem,
                "transcript": full_txt,
                "words": word_level,
            })

        out_dir = Path("data") / output_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "asr.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("✅ Done →", out_path)
        return str(out_path)
    def transcribe(self, wav_path: str) -> tuple[str, float]:
        """
        對單一 wav 檔做 ASR，回傳 (full_text, avg_confidence)
        """
        seg_gen, _ = self.model.transcribe(
            str(wav_path),
            word_timestamps=False,   # 這邊不需要時間戳
            vad_filter=False,
            beam_size=self.beam,
            language=None if self.lang == "auto" else self.lang,
        )
        segments = list(seg_gen)
        if not segments:
            return "", 0.0

        # 組成完整文字
        full_txt = "".join(s.text for s in segments).strip()
        # 平均置信度 (用 word-level probability)
        probs = [w.probability for s in segments for w in (s.words or [])]
        avg_conf = float(sum(probs) / len(probs)) if probs else 0.0
        return full_txt, avg_conf