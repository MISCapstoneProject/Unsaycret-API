# examples/asr_benchmark.py
import argparse
import csv
import gc
from pathlib import Path
import torch

from modules.asr.whisper_asr import WhisperASR
from modules.asr.text_utils import compute_wer, compute_cer
from modules.asr.text_utils import normalize_zh, cer_zh, wer_zh


def load_truth_map(path: str) -> dict[str, str]:
    """Load filename -> transcript mapping from CSV or TSV."""
    mapping: dict[str, str] = {}
    delim = ',' if path.endswith('.csv') else '\t'
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if delim not in line:
                parts = line.split(',', 1) if delim == '\t' else line.split('\t', 1)
            else:
                parts = line.split(delim, 1)
            if len(parts) < 2:
                continue
            fname, txt = parts
            mapping[fname.strip()] = txt.strip()
    return mapping


def benchmark(audio_dir: str, truth_map_path: str, output_csv: str) -> None:
    audio_paths = sorted(Path(audio_dir).glob('*.wav'))
    if not audio_paths:
        raise FileNotFoundError('No .wav files found in the specified directory')
    truth_map = load_truth_map(truth_map_path)

    # 你要測的模型清單
    model_names = ['small', 'medium', 'large-v2', 'Systran/faster-whisper-large-v3']
    beam_sizes = [1, 3, 5]
    use_gpu = torch.cuda.is_available()

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'model_name', 'beam_size', 'file', 'ground_truth', 'prediction',
            'asr_time', 'total_time', 'avg_conf', 'wer', 'cer'
        ])

        for model in model_names:
            # 在載入新模型前，清空過去的快取
            if use_gpu:
                torch.cuda.empty_cache()
                gc.collect()

            for beam in beam_sizes:
                print(f'→ Running {model} [beam={beam}]')
                # 每次都 new 一個 WhisperASR 實例
                asr = WhisperASR(model_name=model, gpu=use_gpu, beam=beam)

                for wav in audio_paths:
                    text, conf, _ = asr.transcribe(str(wav))
                    asr_time = asr.last_infer_time
                    total_time = asr.last_total_time
                    gt = truth_map.get(wav.name, '')
                    gt_n   = normalize_zh(gt)
                    pred_n = normalize_zh(text)
                        # ２）「驗證點」：放在這裡，執行一次 sanity check
                    if wav.name == "speaker1_1.wav" and model == "small" and beam == 1:
                        print("=== Sanity Check ===")
                        print(" GT_N :", gt_n)
                        print(" PR_N :", pred_n)
                        print("====================")
                    # 用中文版計算
                    cer = cer_zh(gt_n, pred_n) if gt_n else None
                    wer = wer_zh(gt_n, pred_n) if gt_n else None   # 要留著觀察就算，不想看就設 None

                    writer.writerow([
                        model,
                        beam,
                        wav.name,
                        gt_n,                 # 写入已 normalize 的 ground truth
                        pred_n,               # 写入已 normalize 的 prediction
                        f'{asr_time:.2f}',
                        f'{total_time:.2f}',
                        f'{conf:.4f}',
                        f'{wer:.4f}' if wer is not None else '',
                        f'{cer:.4f}' if cer is not None else '',
                    ])

                # 釋放當前模型，避免 VRAM 累積
                del asr
                if use_gpu:
                    torch.cuda.empty_cache()
                    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Faster-Whisper models')
    parser.add_argument('--audio_dir', required=True, help='Directory with WAV files')
    parser.add_argument('--truth_map', required=True, help='CSV or TSV file with ground truth')
    parser.add_argument('--output', default='asr_benchmark.csv', help='Output CSV path')
    args = parser.parse_args()

    benchmark(args.audio_dir, args.truth_map, args.output)
