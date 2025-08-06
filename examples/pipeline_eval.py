import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import time
import json
import gc

# import numpy as np
import torch
import torchaudio
import tempfile

from pipelines.orchestrator import init_pipeline_modules,run_pipeline_file
from modules.separation.separator import AudioSeparator
from modules.asr.text_utils import compute_cer, normalize_zh
# from modules.identification.VID_identify_v5 import SpeakerIdentifier
# from modules.asr.whisper_asr import WhisperASR


@dataclass
class SourceInfo:
    path: Path
    transcript: str | None
    mix_sdr: float | None = None
    sep_sdr: float | None = None
    delta_sdr: float | None = None
    clean_wave: torch.Tensor | None = None
    
    
def load_truth_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname, txt = line.split(",", 1)
            mapping[fname] = txt
    return mapping


def load_mixture_map(path: Path) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["mix_id"], row["mix_file"], row["src1"], row["src2"]))
    return rows

#計算SI-SDR 可同時計算分離前和分離後的SDR
def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> float:
    """Scale-Invariant SDR for single-channel signals."""
    min_len = min(len(est), len(ref))
    est = est[:min_len]
    ref = ref[:min_len]

    est = est - est.mean()
    ref = ref - ref.mean()
    s_target = torch.dot(est, ref) * ref / torch.dot(ref, ref)
    e_noise = est - s_target
    return 10 * torch.log10((s_target.pow(2).sum()) / (e_noise.pow(2).sum())).item()
#取檔案並統計SI-SDR
def compute_baseline_sisdr(
    mix_path: Path,
    src1_path: Path,
    src2_path: Path,
    separator: AudioSeparator,
    sep_paths: List[Path] | None = None,
) -> Dict[str, float]:
    """
    計算 baseline 與分離後的 SI-SDR，以及 ΔSI-SDR。

    mix_path  : 混音檔 (.wav)
    src1_path : 來源 1 (乾淨)
    src2_path : 來源 2 (乾淨)
    separator : 你初始化好的 AudioSeparator 實例
    sep_paths : 【可選】已存在的分離後 wav 路徑清單
                若 None，函式會臨時再跑一次 separator 取得分離音
    return    : dict，含 4 個欄位
    """
    # --- 讀取 waveforms --------------------------------------------------
    def _load(wav: Path) -> torch.Tensor:
        wav_tensor, _ = torchaudio.load(wav)
        return wav_tensor.squeeze(0)          # [1, T] -> [T]

    mix_wave  = _load(mix_path)
    src1_wave = _load(src1_path)
    src2_wave = _load(src2_path)

    # --- baseline：混音 vs 兩個乾淨源 -----------------------------------
    sdr_mix_src1 = si_sdr(mix_wave, src1_wave)
    sdr_mix_src2 = si_sdr(mix_wave, src2_wave)

    # --- 取得分離後音檔 ---------------------------------------------------
    if sep_paths is None:
        # 如果呼叫端沒給，就臨時再分一次（省事但較耗時）
        tmp_dir = Path(tempfile.mkdtemp(prefix="eval_sep_"))
        # 注意：separator 需要 tensor 在正確 device
        mix_wave_device = mix_wave.to(separator.device).unsqueeze(0)  # [T] → [1, T]
        sep_paths = [
            Path(p) for p, *_ in
            separator.separate_and_save(mix_wave_device, tmp_dir.as_posix(), segment_index=0)
        ]

    # --- 分離後 vs 乾淨源：取「最佳對應」即可 -----------------------------
    def _best_sdr(ref_wave: torch.Tensor) -> float:
        best = -1e9
        for p in sep_paths:
            est_wave = _load(p)
            best = max(best, si_sdr(est_wave, ref_wave))
        return best

    sdr_sep_src1 = _best_sdr(src1_wave)
    sdr_sep_src2 = _best_sdr(src2_wave)

    # --- ΔSI-SDR ---------------------------------------------------------
    delta1 = sdr_sep_src1 - sdr_mix_src1
    delta2 = sdr_sep_src2 - sdr_mix_src2

    return {
        "si_sdr_src1":       sdr_sep_src1,
        "si_sdr_src2":       sdr_sep_src2,
        "delta_si_sdr_src1": delta1,
        "delta_si_sdr_src2": delta2,
    }
#spkID
def compute_accuracy(pred_speakers: List[str], true_speakers: List[str]) -> float:
    """
    pred_speakers: pipeline 分離後所有段落的預測 speaker id (e.g. ["spk3","spk10"])
    true_speakers: mixture_map 定義的兩位原始 speaker id (e.g. ["spk10","spk1"])
    回傳值: 正確數 / 2 -> 0.0, 0.5, or 1.0
    """
    matched = len(set(pred_speakers) & set(true_speakers))
    return matched / len(true_speakers)

NUM_MAP = {'0':'零','1':'一','2':'二','3':'三','4':'四',
           '5':'五','6':'六','7':'七','8':'八','9':'九'}

def normalize_numbers_to_zh(text: str) -> str:
    return "".join(NUM_MAP.get(ch, ch) for ch in text)

def load_config_and_data() -> Tuple[argparse.Namespace, Path, Path, Dict[str, str], List[Tuple[str, str, str, str]]]:
    parser = argparse.ArgumentParser(description="Run speech pipeline and evaluate")
    parser.add_argument("--mix-dir", default="data/mix", help="mix路徑")
    parser.add_argument("--clean-dir", default="data/clean", help="clean路徑")
    parser.add_argument("--truth-map", default="data/truth_map.csv")
    parser.add_argument("--test-list",  default="data/mix/mixture_map.csv",  help="batch 測試清單 CSV (mix_id,mix_file,src1,src2)")
    parser.add_argument("--out", default="work_output/pipeline_results.csv")
    args = parser.parse_args()

    mix_dir = Path(args.mix_dir)
    clean_dir = Path(args.clean_dir)
    truth_map = load_truth_map(Path(args.truth_map))
    mixture_rows = load_mixture_map(Path(args.test_list))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    return args, mix_dir, clean_dir, truth_map, mixture_rows

def prepare_results_structure() -> tuple[list[str], list[dict]]:
    columns = [
        "mix_id", "mix_file",
        "sep_time", "sid_time", "asr_time", "total_time",
        "si_sdr_src1", "si_sdr_src2",
        "delta_si_sdr_src1", "delta_si_sdr_src2",
        "accuracy",
        "cer1", "cer2", "avg_conf",
        "ref_text1", "pred_text1", "ref_text2", "pred_text2",
    ]
    return columns, []

def process_one_mixture(
    mix_path: str,
    mix_id: str = None,
    sep=None, spk=None, asr=None,
    clean_dir: Path = None,
    mixture_rows: List[Tuple[str, str, str, str]] = None,
    truth_map: Dict[str, str] = None,
) -> dict:
    """
    處理一組混音音檔，跑完整的 pipeline，並回傳所有可取得的指標。
    
    mix_path: 混音音檔完整路徑
    mix_id: 選填的檔案名稱（不含副檔名），預設用檔名
    return: 一個 dict，包含 pipeline 的所有可觀察數據
    """
    # 使用目前時間計算總耗時
    overall_start = time.perf_counter()
    mix_wav = Path(mix_path)
    
    if mix_id is None:
        mix_id = Path(mix_path).stem  # 取檔名當作 id

    # 🌀 呼叫主流程跑完整 pipeline
    try:
        bundle, _ , stats = run_pipeline_file(mix_path,3, sep=sep, spk=spk, asr=asr)
    except Exception as e:
        print(f"[ERROR] 處理檔案 {mix_path} 時出錯：{e}")
        return {
            "mix_id": mix_id,
            "mix_file": mix_path,
            "error": str(e)
        }

    # 📊 統計與整理各階段資訊
    recog_texts = [s.get("text", "") for s in bundle if s.get("text")]
    confidences = [s.get("confidence", 0.0) for s in bundle if s.get("text")]

    full_text = " ".join(recog_texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    result = {
        "mix_id": mix_id,
        "mix_file": mix_path,
        "seg_count": len(bundle),
        "sep_time": round(stats.get("separate", 0.0), 3),
        "sid_time": round(stats.get("speaker", 0.0), 3),
        "asr_time": round(stats.get("asr", 0.0), 3),
        "total_time": round(stats.get("total", 0.0), 3),
        "pipeline_time": round(stats.get("pipeline_total", 0.0), 3),
        "avg_conf": round(avg_conf, 4),
        "predicted_text": full_text,
        "segments": bundle,
        "overall_time": round(time.perf_counter() - overall_start, 3),
    }
    
    # ③ 加上 baseline SI-SDR 計算    
    # 解析 src1 / src2 清音路徑
    if clean_dir and mixture_rows:
        row = next((r for r in mixture_rows if r[0] == mix_id), None)
        if row:
            src1 = clean_dir / row[2]
            src2 = clean_dir / row[3]
            sep_paths = [Path(s["path"]) for s in bundle if "path" in s]
            sisdr_metrics = compute_baseline_sisdr(mix_wav, src1, src2, sep, sep_paths)
            result.update(sisdr_metrics)
            
            # --- 取出 true speaker IDs from mixture_map row ---
            # row[2] = "speaker10/…", row[3] = "speaker1/…"
            # row[2] = "speaker10/speaker10_17.wav"
            # Path(row[2]).parent.name → "speaker10"
            # .replace("speaker", "spk") → "spk10"
            
            true_spk1 = Path(row[2]).parent.name.replace("speaker", "spk")
            true_spk2 = Path(row[3]).parent.name.replace("speaker", "spk")
            true_speakers = [true_spk1, true_spk2]
            # print(f"🔍 真實語者：{true_speakers}")
            
            # --- 取出 pipeline 預測的所有 speaker IDs ---
            pred_speakers = [seg.get("speaker") for seg in bundle]
            # print(f"🔍 預測語者：{pred_speakers}")
            # --- 計算 accuracy ---
            acc = compute_accuracy(pred_speakers, true_speakers)
            # print(f"🔍 語者辨識準確率：{acc:.2f}")
            result["accuracy"] = acc
            
                        # === CER & ref/pred text for each speaker === #
            # 1) 地取每個 source 的正確文字
            ref_fname1 = Path(src1).name                         # e.g. "speaker10_17.wav"
            ref_fname2 = Path(src2).name
            raw_ref1 = truth_map.get(ref_fname1, "")
            raw_ref2 = truth_map.get(ref_fname2, "")

            # 2) Normalize (去標點、空格、統一繁體)
            ref_norm1 = normalize_zh(raw_ref1)
            ref_norm2 = normalize_zh(raw_ref2)
            # print(f"🔍 正規化文字1：{ref_norm1}"
            #         f" 正規化文字2：{ref_norm2}")

            # 3) 從 bundle 中找出對應 true speaker 的預測文字
            pred_text1 = ""
            pred_text2 = ""
            for seg in bundle:
                if seg.get("speaker") == true_spk1:
                    pred_text1 = seg.get("text", "")
                elif seg.get("speaker") == true_spk2:
                    pred_text2 = seg.get("text", "")
            pred_norm1 = normalize_numbers_to_zh(normalize_zh(pred_text1))
            pred_norm2 = normalize_numbers_to_zh(normalize_zh(pred_text2))

            # print(f"🔍 預測文字1：{pred_norm1}"
            #         f" 預測文字2：{pred_norm2}")
            # 4) 計算 CER
            cer1 = compute_cer(ref_norm1, pred_norm1) if ref_norm1 else None
            cer2 = compute_cer(ref_norm2, pred_norm2) if ref_norm2 else None
            # print(f"🔍 CER1：{cer1:.4f} CER2：{cer2:.4f}")
            # 5) 寫入結果
            result.update({
                "ref_text1": ref_norm1,
                "pred_text1": pred_norm1,
                "cer1": cer1,
                "ref_text2": ref_norm2,
                "pred_text2": pred_norm2,
                "cer2": cer2,
            })
            del asr
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print(f"[WARN] mix_id {mix_id} not found in mixture_map")
        
            

    return result

def run_and_evaluate_pipeline(
    mixture_csv: Path,
    mix_dir: Path,
    clean_dir: Path,
    truth_map: Dict[str, str],
    sep, spk, asr,
    output_csv: Path,
):
    test_rows = load_mixture_map(mixture_csv)               # ① 讀取清單
    cols, _ = prepare_results_structure()

    with output_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=cols)
        writer.writeheader()

        for mix_id, mix_file, *_ in test_rows:              # ② 逐筆跑
            mix_path = mix_dir / mix_file
            print(f"👉 處理 {mix_id} → {mix_path}")

            res = process_one_mixture(
                str(mix_path), mix_id,
                sep=sep, spk=spk, asr=asr,
                clean_dir=clean_dir,
                mixture_rows=test_rows,
                truth_map=truth_map
            )

            # ③ 如果 pipeline 出錯就跳過，不寫進 CSV
            if "error" in res:
                print(f"🚨 跳過 {mix_id}，原因：{res['error']}")
                continue

            # ④ 只留下指定欄位
            row = {k: res.get(k, "") for k in cols}
            writer.writerow(row)

    print(f"✅ 全部完成！結果已寫入 {output_csv}")
    
def main():
    # 1️⃣ 清空 GPU 快取
    torch.cuda.empty_cache()
    gc.collect()

    # 2️⃣ 載入所有參數與資料
    #    --mix-dir、--clean-dir、--truth-map、--test-list、--out 都已設預設
    args, mix_dir, clean_dir, truth_map, _ = load_config_and_data()

    # 3️⃣ 初始化模型
    sep, spk, asr, _ = init_pipeline_modules()

    run_and_evaluate_pipeline(
        mixture_csv=Path(args.test_list),
        mix_dir=mix_dir,
        clean_dir=clean_dir,
        truth_map=truth_map,
        sep=sep, spk=spk, asr=asr,
        output_csv=Path(args.out),
    )

    # # ④ 測試一組混音
    # test_mix_id = "m01"  # 你可以自訂一個 ID
    # test_mix_path = "data/mix/m01.wav"  # ← 改成你實際有的音檔路徑！

    # print(f"👉 處理測試音檔：{test_mix_path}")
    # result = process_one_mixture(
    #     test_mix_path,
    #     test_mix_id,
    #     sep=sep,
    #     spk=spk,
    #     asr=asr,
    #     clean_dir=clean_dir,
    #     mixture_rows=mixture_rows,
    #     truth_map=truth_map
    # )
    # #  寫入 JSON 檔
    # with open(f"{test_mix_id}_result.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, indent=2, ensure_ascii=False)
    # print(f"💾 已儲存 JSON 檔至：{test_mix_id}_result.json")
    
    
if __name__ == "__main__":
    main()
