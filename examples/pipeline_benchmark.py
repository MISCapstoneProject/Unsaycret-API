import argparse
import csv
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio

from modules.separation.separator import AudioSeparator
from modules.identification.VID_identify_v5 import SpeakerIdentifier
from modules.asr.whisper_asr import WhisperASR
from modules.asr.text_utils import compute_cer
from utils.constants import DEFAULT_WHISPER_MODEL
from utils.logger import get_logger


logger = get_logger(__name__)


def load_truth_map(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return mapping
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            name, txt = line.split(",", 1)
            mapping[name.strip()] = txt.strip()
    return mapping


def load_mixture_map(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            mix_id, mix_file, spk1_file, spk2_file, spk1_name, spk2_name, overlap, snr = parts[:8]
            rows.append(
                {
                    "mix_file": mix_file,
                    "spk1_file": spk1_file.replace("\\", os.sep),
                    "spk2_file": spk2_file.replace("\\", os.sep),
                    "spk1_name": spk1_name,
                    "spk2_name": spk2_name,
                }
            )
    return rows


def si_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    s_target = torch.dot(est, ref) * ref / (torch.dot(ref, ref) + 1e-8)
    e_noise = est - s_target
    ratio = torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8)
    return 10 * torch.log10(ratio + 1e-8)


def run_pipeline(
    mix_path: Path,
    sep: AudioSeparator,
    sid: SpeakerIdentifier,
    asr: WhisperASR,
    tmp_root: Path,
    max_workers: int = 2,
) -> Tuple[List[Dict], Dict, Path]:
    total_start = time.perf_counter()
    wav, sr = torchaudio.load(str(mix_path))
    wav = wav.to(sep.device)
    audio_len = wav.shape[1] / sr
    out_dir = tmp_root / mix_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    sep_start = time.perf_counter()
    segments = sep.separate_and_save(wav, str(out_dir), segment_index=0)
    sep_time = time.perf_counter() - sep_start
    if not segments:
        raise RuntimeError("no separated segments")

    results: List[Dict] = []
    spk_time = 0.0
    asr_time = 0.0
    for seg_path, t0, t1 in segments:
        sid_t0 = time.perf_counter()
        sid_res = sid.process_audio_file(seg_path)
        sid_t = time.perf_counter() - sid_t0
        spk_time = max(spk_time, sid_t)

        asr_t0 = time.perf_counter()
        text, conf, _ = asr.transcribe(seg_path)
        asr_t = time.perf_counter() - asr_t0
        asr_time = max(asr_time, asr_t)

        results.append(
            {
                "path": seg_path,
                "start": t0,
                "end": t1,
                "speaker": sid_res[1] if isinstance(sid_res, tuple) else "",
                "distance": float(sid_res[2]) if isinstance(sid_res, tuple) else 0.0,
                "text": text,
                "confidence": conf,
            }
        )

    total_time = time.perf_counter() - total_start
    stats = {
        "length": audio_len,
        "separate": sep_time,
        "speaker": spk_time,
        "asr": asr_time,
        "total": total_time,
        "avg_conf": sum(r["confidence"] for r in results) / len(results),
    }
    return results, stats, out_dir


def evaluate(
    mix_row: Dict[str, str],
    mix_dir: Path,
    clean_dir: Path,
    truth: Dict[str, str],
    sep: AudioSeparator,
    sid: SpeakerIdentifier,
    asr: WhisperASR,
    tmp_root: Path,
) -> Dict:
    mix_path = mix_dir / mix_row["mix_file"]
    spk1_clean = clean_dir / mix_row["spk1_file"]
    spk2_clean = clean_dir / mix_row["spk2_file"]

    segments, stats, out_dir = run_pipeline(mix_path, sep, sid, asr, tmp_root)

    # load clean references
    r1, sr1 = torchaudio.load(str(spk1_clean))
    r2, sr2 = torchaudio.load(str(spk2_clean))
    r1 = torchaudio.functional.resample(r1, sr1, 16000)
    r2 = torchaudio.functional.resample(r2, sr2, 16000)

    # load separated wavs
    sep_paths = [Path(seg["path"]) for seg in segments]
    if len(sep_paths) < 2:
        sisdr_val = None
        spk_acc = 0.0
        cer_avg = None
    else:
        s1, sr_s1 = torchaudio.load(str(sep_paths[0]))
        s2, sr_s2 = torchaudio.load(str(sep_paths[1]))
        s1 = torchaudio.functional.resample(s1, sr_s1, 16000)
        s2 = torchaudio.functional.resample(s2, sr_s2, 16000)

        # two possible assignments
        score_a = si_sdr(r1.squeeze(), s1.squeeze()) + si_sdr(r2.squeeze(), s2.squeeze())
        score_b = si_sdr(r1.squeeze(), s2.squeeze()) + si_sdr(r2.squeeze(), s1.squeeze())
        if score_a >= score_b:
            sisdr_val = float((score_a / 2).item())
            mapping = [mix_row["spk1_name"], mix_row["spk2_name"]]
        else:
            sisdr_val = float((score_b / 2).item())
            mapping = [mix_row["spk2_name"], mix_row["spk1_name"]]

        correct = 0
        cer_scores = []
        for seg, gt_name, clean_file in zip(segments, mapping, [spk1_clean, spk2_clean]):
            if seg["speaker"] == gt_name:
                correct += 1
            ref_txt = truth.get(clean_file.name)
            if ref_txt:
                cer_scores.append(compute_cer(ref_txt, seg["text"]))
        spk_acc = correct / len(mapping)
        cer_avg = sum(cer_scores) / len(cer_scores) if cer_scores else None

    throughput = stats["length"] / stats["total"] if stats["total"] else 0.0

    return {
        "file": mix_row["mix_file"],
        "length": f"{stats['length']:.2f}",
        "sep_time": f"{stats['separate']:.2f}",
        "sid_time": f"{stats['speaker']:.2f}",
        "asr_time": f"{stats['asr']:.2f}",
        "total_time": f"{stats['total']:.2f}",
        "throughput": f"{throughput:.2f}",
        "avg_conf": f"{stats['avg_conf']:.4f}",
        "si_sdr": f"{sisdr_val:.2f}" if sisdr_val is not None else "",
        "sid_acc": f"{spk_acc:.2f}",
        "cer": f"{cer_avg:.4f}" if cer_avg is not None else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline benchmark")
    parser.add_argument("--mixture_dir", default="data/2mix_30data", help="Dir with mixture wavs")
    parser.add_argument("--clean_dir", default="data/clean", help="Dir with clean wavs")
    parser.add_argument("--mixture_map", default="data/2mix_30data/mixture_map.csv", help="Mixture mapping CSV")
    parser.add_argument("--truth_map", default="data/truth_map.csv", help="Mapping of clean wav to transcript")
    parser.add_argument("--output", default="pipeline_benchmark.csv", help="Output CSV path")
    args = parser.parse_args()

    truth_map = load_truth_map(args.truth_map)
    mix_rows = load_mixture_map(args.mixture_map)

    use_gpu = torch.cuda.is_available()
    sep = AudioSeparator()
    sid = SpeakerIdentifier()
    asr = WhisperASR(model_name=DEFAULT_WHISPER_MODEL, gpu=use_gpu)

    tmp_root = Path(tempfile.mkdtemp(prefix="pipeline_eval_"))
    results = []
    for row in mix_rows:
        logger.info("Processing %s", row["mix_file"])
        try:
            res = evaluate(row, Path(args.mixture_dir), Path(args.clean_dir), truth_map, sep, sid, asr, tmp_root)
            results.append(res)
        except Exception as e:
            logger.error("Failed to process %s: %s", row["mix_file"], e)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "length",
                "sep_time",
                "sid_time",
                "asr_time",
                "total_time",
                "throughput",
                "avg_conf",
                "si_sdr",
                "sid_acc",
                "cer",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    if results:
        avg_throughput = sum(float(r["throughput"]) for r in results) / len(results)
        avg_sep = sum(float(r["sep_time"]) for r in results) / len(results)
        avg_sid = sum(float(r["sid_time"]) for r in results) / len(results)
        avg_asr = sum(float(r["asr_time"]) for r in results) / len(results)
        avg_si_sdr = sum(float(r["si_sdr"]) for r in results if r["si_sdr"]) / max(1, len([r for r in results if r["si_sdr"]]))
        print("\n=== Summary ===")
        print(f"Files processed : {len(results)}")
        print(f"Avg separation  : {avg_sep:.2f}s")
        print(f"Avg speaker ID  : {avg_sid:.2f}s")
        print(f"Avg ASR         : {avg_asr:.2f}s")
        print(f"Avg throughput  : {avg_throughput:.2f}x realtime")
        print(f"Avg SI-SDR      : {avg_si_sdr:.2f} dB")


if __name__ == "__main__":
    main()
