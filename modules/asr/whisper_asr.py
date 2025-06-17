# whisper_asr.py
from pathlib import Path
import json, csv, os, time ,torchaudio
from .asr_model import load_model
from .text_utils import merge_char_to_word
from utils.logger import get_logger
import torch


logger = get_logger(__name__)

# CSV 紀錄路徑
CSV_PATH = Path("work_output/asr_performance.csv")

class WhisperASR:
    counter = 0
    
    def __init__(self, model_name="medium", gpu=False, beam=5, lang="auto"):
        self.gpu = gpu
        self.model = load_model(model_name=model_name, gpu=self.gpu)
        self.beam = beam
        self.lang = lang

        # 加這一行來驗證模型裝在哪裡 log確認GPU或CPU
        device_str = "cuda" if self.gpu else "cpu"
        logger.info(f"🧠 Whisper 實際運行裝置: {device_str}")
        
        # 如果 CSV 不存在，建立並寫入標頭
        if not CSV_PATH.exists():
            with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "編號",
                    "音檔長度(秒)",
                    "推理耗時(秒)",
                    "總耗時(秒)",
                    "平均信心值"
                ])
        
# #批次處理資料夾音檔
#     def transcribe_dir(self, input_dir: str, output_id: str) -> str:
#         wav_list = sorted(Path(input_dir).glob("*.wav"))
#         if not wav_list:
#             raise FileNotFoundError("❌ No .wav found in input_dir")

#         results = []

#         for wav in wav_list:
#             logger.info(f"🚀 開始辨識音檔: {wav.name}")
#             seg_gen, _ = self.model.transcribe(
#                 str(wav),
#                 word_timestamps=True,
#                 vad_filter=False,
#                 beam_size=self.beam,
#                 language=None if self.lang == "auto" else self.lang
#             )
#             segments = list(seg_gen)
#             if not segments:
#                 continue

#             full_txt = "".join(s.text for s in segments).strip()
#             char_words = [{
#                 "start": float(w.start),
#                 "end": float(w.end),
#                 "word": str(w.word),
#                 "probability": float(w.probability)
#             } for s in segments for w in (s.words or [])]
#             word_level = merge_char_to_word(full_txt, char_words)

#             results.append({
#                 "track_id": wav.stem,
#                 "transcript": full_txt,
#                 "words": word_level,
#             })

#         out_dir = Path("data") / output_id
#         out_dir.mkdir(parents=True, exist_ok=True)
#         out_path = out_dir / "asr.json"
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)

#         logger.info(f"✅ 完成辨識，結果輸出至: {out_path}")
#         return str(out_path)


    # 對單一檔案即時處理
    def transcribe(self, wav_path: str) -> tuple[str, float, list[dict]]:
        # batch_id 自動累加
        WhisperASR.counter += 1
        idx = WhisperASR.counter
        
        # 取得音檔長度
        info = torchaudio.info(wav_path)
        duration_sec = info.num_frames / info.sample_rate
            
        # 計時：整體開始
        start_all = time.perf_counter()
        
        # 計時：模型推理
        t0 = time.perf_counter()
        seg_gen, _ = self.model.transcribe(
            str(wav_path),
            word_timestamps=True,
            vad_filter=False,
            beam_size=self.beam,
            language=None if self.lang == "auto" else self.lang,
        )
        t1 = time.perf_counter()
        inference_time = t1 - t0
        logger.info(f"⏱ ASR 推理時間: {inference_time:.3f}s")
        
        # 後處理
        segments = list(seg_gen)
        if not segments:
            return "", 0.0, []

        full_txt = "".join(s.text for s in segments).strip()
        words = [w for s in segments for w in (s.words or [])]
        
        # 若 words 為空，改用 segment-level average_logprob 當 fallback
        if words:
            probs = [w.probability for w in words]
            avg_conf = float(sum(probs) / len(probs))
            word_info = [{
                "start": float(w.start),
                "end": float(w.end),
                "word": str(w.word),
                "probability": float(w.probability)
            } for w in words]
        else:
            avg_conf = float(sum(s.avg_logprob for s in segments) / len(segments))
            word_info = []
        
        # 計時：整體結束
        end_all = time.perf_counter()
        total_time = end_all - start_all
        logger.info(f"✅ transcribe() 總耗時: {total_time:.3f}s")
        
        
        # 將本次結果寫入CSV
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                idx,
                f"{duration_sec:.3f}",
                f"{inference_time:.3f}",
                f"{total_time:.3f}",
                f"{avg_conf:.3f}"
            ])
        torch.cuda.empty_cache()
        return full_txt, avg_conf, word_info
