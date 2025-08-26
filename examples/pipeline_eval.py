import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import time
import json
import gc

import numpy as np
import torch
import torchaudio
import tempfile
import soundfile as sf

import sys
from pathlib import Path

# 將專案根目錄加入 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.orchestrator import init_pipeline_modules,run_pipeline_file
from utils.logger import get_logger
from modules.separation.separator import AudioSeparator
from modules.asr.text_utils import compute_cer, normalize_zh
# from modules.identification.VID_identify_v5 import SpeakerIdentifier
# from modules.asr.whisper_asr import WhisperASR
from scipy import stats
from sklearn.metrics import roc_curve, auc

# 創建 logger 實例
logger = get_logger(__name__)

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("matplotlib 不可用，將跳過圖表生成")


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
            # 跨平台路徑處理：手動將 Windows 風格的 \ 轉換為 / 
            # 這樣在 macOS/Linux/Windows 都能正確處理
            src1_normalized = row["src1"].replace("\\", "/")
            src2_normalized = row["src2"].replace("\\", "/")
            rows.append((row["mix_id"], row["mix_file"], src1_normalized, src2_normalized))
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


def calculate_topk_accuracy(segment_info: List[Dict], true_speakers: List[str]) -> Dict[str, float]:
    """計算 Top-K 準確率"""
    if not segment_info:
        return {}
    
    # 簡化版：目前只計算 Top-1（因為我們只有最佳預測）
    correct = sum(1 for info in segment_info if info['speaker'] in true_speakers)
    total = len(segment_info)
    
    return {
        'top1_accuracy': round(correct / total if total > 0 else 0, 4),
        'speaker_coverage': round(len(set(info['speaker'] for info in segment_info if info['speaker'] in true_speakers)) / len(true_speakers) if true_speakers else 0, 4)
    }


def calculate_per_speaker_metrics(segment_info: List[Dict], true_speakers: List[str]) -> Dict[str, Dict]:
    """計算每個語者的性能指標"""
    speaker_stats = {}
    
    for info in segment_info:
        speaker = info['speaker']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'count': 0,
                'correct': 0,
                'distances': [],
                'avg_distance': 0,
                'accuracy': 0
            }
        
        speaker_stats[speaker]['count'] += 1
        speaker_stats[speaker]['distances'].append(info['distance'])
        
        if speaker in true_speakers:
            speaker_stats[speaker]['correct'] += 1
    
    # 計算平均值
    for speaker, stats in speaker_stats.items():
        stats['avg_distance'] = round(np.mean(stats['distances']), 4)
        stats['accuracy'] = round(stats['correct'] / stats['count'], 4)
    
    return {'per_speaker_stats': speaker_stats}


def calculate_confusion_data(pred_speakers: List[str], true_speakers: List[str]) -> Dict:
    """計算混淆矩陣相關數據"""
    all_speakers = sorted(set(pred_speakers + true_speakers))
    
    confusion_pairs = []
    for i, pred in enumerate(pred_speakers):
        true = true_speakers[i] if i < len(true_speakers) else 'unknown'
        confusion_pairs.append({
            'predicted': pred,
            'actual': true,
            'correct': pred == true
        })
    
    return {
        'confusion_pairs': confusion_pairs,
        'unique_speakers': all_speakers,
        'total_predictions': len(pred_speakers)
    }
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

def compute_speaker_identification_metrics(bundle: List[dict], true_speakers: List[str], spk) -> dict:
    """
    計算詳細的語者辨識指標
    
    Args:
        bundle: pipeline 輸出的音段資訊
        true_speakers: 真實的語者 ID 列表
        spk: 語者辨識器實例
        
    Returns:
        dict: 包含各種指標的字典
    """
    metrics = {
        'accuracy': 0.0,
        'min_distance': "",
        'top1_top2_margin': "",
        'identification_scores': [],  # 用於後續統計
        'prediction_confidence': "",
        'distance_variance': "",
        'true_speaker1': '',
        'true_speaker2': '',
        'pred_speaker1': '',
        'pred_speaker2': '',
        'distance1': "",
        'distance2': ""
    }
    
    # 設定真實語者
    if len(true_speakers) >= 2:
        metrics['true_speaker1'] = true_speakers[0]
        metrics['true_speaker2'] = true_speakers[1]
    elif len(true_speakers) == 1:
        metrics['true_speaker1'] = true_speakers[0]
        metrics['true_speaker2'] = ''
    
    # 基本準確率計算
    pred_speakers = [seg.get("speaker") for seg in bundle if seg.get("speaker")]
    metrics['accuracy'] = compute_accuracy(pred_speakers, true_speakers)
    
    # 收集所有音段的距離資訊
    distances = []
    margins = []
    segment_info = []  # 收集每個段落的詳細資訊
    
    for seg in bundle:
        speaker = seg.get('speaker', '')
        distance = seg.get('distance', None)
        
        if 'speaker_distances' in seg and seg['speaker_distances']:
            # 有詳細的候選結果 - 使用完整的距離資訊
            seg_distances = seg['speaker_distances']
            if len(seg_distances) >= 1:
                # 按距離排序（距離越小越相似）
                sorted_distances = sorted(seg_distances, key=lambda x: x[2])
                top1_dist = sorted_distances[0][2]
                top1_speaker = sorted_distances[0][1]
                
                # 收集段落資訊
                segment_info.append({
                    'speaker': top1_speaker,
                    'distance': top1_dist,
                    'is_correct': top1_speaker in true_speakers
                })
                
                distances.append(top1_dist)
                
                # 如果有第二個候選者，計算 margin
                top2_dist = None
                if len(seg_distances) >= 2:
                    top2_dist = sorted_distances[1][2]
                    margins.append(top2_dist - top1_dist)  # margin 越大表示信心度越高
                
                # 記錄識別結果用於 EER 計算
                is_correct = top1_speaker in true_speakers
                metrics['identification_scores'].append({
                    'distance': top1_dist,
                    'margin': (top2_dist - top1_dist) if top2_dist is not None else 0.0,
                    'is_correct': is_correct,
                    'predicted_speaker': top1_speaker
                })
        elif speaker and distance is not None:
            # 沒有詳細距離資訊，但有基本的語者和距離 - 使用 API 回傳的距離
            segment_info.append({
                'speaker': speaker,
                'distance': distance,
                'is_correct': speaker in true_speakers
            })
            
            distances.append(distance)
            
            # 對於新語者或只有基本資訊的情況，無法計算 margin
            metrics['identification_scores'].append({
                'distance': distance,
                'margin': 0.0,  # 無法計算 margin
                'is_correct': speaker in true_speakers,
                'predicted_speaker': speaker
            })
    
    # 設定預測語者和對應距離
    if segment_info:
        # 取前兩個段落的預測結果
        if len(segment_info) >= 1:
            metrics['pred_speaker1'] = segment_info[0]['speaker']
            metrics['distance1'] = round(segment_info[0]['distance'], 4)
            
        if len(segment_info) >= 2:
            metrics['pred_speaker2'] = segment_info[1]['speaker'] 
            metrics['distance2'] = round(segment_info[1]['distance'], 4)
        else:
            # 如果只有一個段落，第二個設為空
            metrics['pred_speaker2'] = ''
            metrics['distance2'] = ""
    
    # 計算統計指標
    if distances:
        metrics['min_distance'] = round(min(distances), 4)
        metrics['distance_variance'] = round(np.var(distances), 4)
        
        # 計算額外的語者辨識指標
        pred_speakers = [info['speaker'] for info in segment_info]
        pred_distances = [info['distance'] for info in segment_info]
        
        # Top-K 準確率分析
        metrics.update(calculate_topk_accuracy(segment_info, true_speakers))
        
        # 按語者性能分析
        metrics.update(calculate_per_speaker_metrics(segment_info, true_speakers))
        
        # 混淆矩陣數據
        metrics['confusion_data'] = calculate_confusion_data(pred_speakers, true_speakers)
    else:
        metrics['min_distance'] = ""
        metrics['distance_variance'] = ""
        
    if margins:
        metrics['top1_top2_margin'] = round(np.mean(margins), 4)
        metrics['prediction_confidence'] = round(np.mean(margins), 4)
    else:
        metrics['top1_top2_margin'] = ""
        metrics['prediction_confidence'] = ""
    
    return metrics

def calculate_batch_speaker_metrics(all_results: List[dict]) -> dict:
    """
    計算整個批次的語者辨識綜合統計
    
    Args:
        all_results: 所有測試結果的列表
        
    Returns:
        dict: 批次統計指標
    """
    batch_metrics = {
        'total_samples': len(all_results),
        'overall_accuracy': 0.0,
        'eer': 0.0,
        'min_dcf': 0.0,
        'cllr': 0.0,
        'distance_distribution': {
            'mean': 0.0,
            'std': 0.0,
            'min': float('inf'),
            'max': float('-inf')
        },
        'margin_distribution': {
            'mean': 0.0,
            'std': 0.0,
            'min': float('inf'),
            'max': float('-inf')
        },
        'confidence_stability': 0.0
    }
    
    # 收集所有識別結果
    all_scores = []
    all_distances = []
    all_margins = []
    correct_predictions = 0
    total_predictions = 0
    
    for result in all_results:
        if 'identification_scores' in result and result['identification_scores']:
            for score_info in result['identification_scores']:
                all_scores.append(score_info)
                all_distances.append(score_info['distance'])
                all_margins.append(score_info['margin'])
                
                if score_info['is_correct']:
                    correct_predictions += 1
                total_predictions += 1
    
    # 整體準確率
    if total_predictions > 0:
        batch_metrics['overall_accuracy'] = correct_predictions / total_predictions
    
    # 距離分布統計
    if all_distances:
        batch_metrics['distance_distribution'] = {
            'mean': np.mean(all_distances),
            'std': np.std(all_distances),
            'min': np.min(all_distances),
            'max': np.max(all_distances)
        }
    
    # Margin 分布統計
    if all_margins:
        batch_metrics['margin_distribution'] = {
            'mean': np.mean(all_margins),
            'std': np.std(all_margins),
            'min': np.min(all_margins),
            'max': np.max(all_margins)
        }
        
        # 信心度穩定性（margin 的變異係數）
        if np.mean(all_margins) > 0:
            batch_metrics['confidence_stability'] = np.std(all_margins) / np.mean(all_margins)
    
    # 計算 EER 和 minDCF（如果有足夠的正負樣本）
    if len(all_scores) > 10:  # 需要足夠的樣本
        try:
            # 準備 EER 計算的資料
            y_true = [1 if score['is_correct'] else 0 for score in all_scores]
            # 距離越小表示越相似，所以用負距離作為分數
            y_scores = [-score['distance'] for score in all_scores]
            
            if len(set(y_true)) > 1:  # 確保有正負樣本
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                
                # 計算 EER
                fnr = 1 - tpr
                eer_index = np.nanargmin(np.absolute(fnr - fpr))
                batch_metrics['eer'] = (fpr[eer_index] + fnr[eer_index]) / 2
                
                # 簡化的 minDCF 計算（假設 Cmiss=Cfa=1, Ptarget=0.01）
                Ptarget = 0.01
                Cmiss = 1
                Cfa = 1
                dcf_values = []
                
                for i in range(len(fpr)):
                    dcf = Cmiss * fnr[i] * Ptarget + Cfa * fpr[i] * (1 - Ptarget)
                    dcf_values.append(dcf)
                
                batch_metrics['min_dcf'] = min(dcf_values)
                
                # 簡化的 Cllr 計算
                # 這是一個簡化版本，實際 Cllr 需要更複雜的計算
                if len(y_scores) > 0:
                    batch_metrics['cllr'] = -np.mean([
                        np.log2(score) if label == 1 else np.log2(1 - score)
                        for score, label in zip(y_scores, y_true)
                        if 0 < score < 1
                    ]) if any(0 < score < 1 for score in y_scores) else 0.0
                    
        except Exception as e:
            logger.warning(f"計算 EER/minDCF 時出錯: {e}")
            batch_metrics['eer'] = 0.0
            batch_metrics['min_dcf'] = 0.0
            batch_metrics['cllr'] = 0.0
    
    return batch_metrics

def create_speaker_identification_plots(all_results: List[dict], output_dir: Path) -> None:
    """
    創建語者辨識分析圖表
    
    Args:
        all_results: 所有測試結果
        output_dir: 圖表輸出目錄
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("matplotlib not available, skipping chart generation")
        return
        
    output_dir.mkdir(exist_ok=True)
    
    # 收集所有資料
    correct_distances = []
    incorrect_distances = []
    all_margins = []
    all_distances = []
    
    for result in all_results:
        if 'identification_scores' in result and result['identification_scores']:
            for score_info in result['identification_scores']:
                distance = score_info['distance']
                margin = score_info['margin']
                is_correct = score_info['is_correct']
                
                all_distances.append(distance)
                all_margins.append(margin)
                
                if is_correct:
                    correct_distances.append(distance)
                else:
                    incorrect_distances.append(distance)
    
    # 1. 距離分布圖（正確 vs 錯誤預測）
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    if correct_distances and incorrect_distances:
        bins = np.linspace(min(all_distances), max(all_distances), 30)
        plt.hist(correct_distances, bins=bins, alpha=0.7, label='Correct Predictions', color='green', density=True)
        plt.hist(incorrect_distances, bins=bins, alpha=0.7, label='Incorrect Predictions', color='red', density=True)
        plt.xlabel('Voice Distance')
        plt.ylabel('Density')
        plt.title('Speaker ID Distance Distribution (Correct vs Incorrect)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Margin 分布圖
    plt.subplot(2, 2, 2)
    if all_margins:
        plt.hist(all_margins, bins=30, alpha=0.7, color='blue', density=True)
        plt.xlabel('Top-1 vs Top-2 Margin')
        plt.ylabel('Density')
        plt.title('Prediction Confidence Distribution (Margin)')
        plt.grid(True, alpha=0.3)
        
        # 添加統計線
        mean_margin = np.mean(all_margins)
        plt.axvline(mean_margin, color='red', linestyle='--', label=f'Mean: {mean_margin:.3f}')
        plt.legend()
    
    # 3. ROC 曲線（如果有足夠資料）
    plt.subplot(2, 2, 3)
    if len(correct_distances) > 5 and len(incorrect_distances) > 5:
        y_true = [1] * len(correct_distances) + [0] * len(incorrect_distances)
        y_scores = [-d for d in correct_distances] + [-d for d in incorrect_distances]  # 負距離作為分數
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    
    # 4. 距離 vs Margin 散點圖
    plt.subplot(2, 2, 4)
    if all_distances and all_margins and len(all_distances) == len(all_margins):
        # 根據正確性著色
        colors = []
        for result in all_results:
            if 'identification_scores' in result and result['identification_scores']:
                for score_info in result['identification_scores']:
                    colors.append('green' if score_info['is_correct'] else 'red')
        
        if len(colors) == len(all_distances):
            plt.scatter(all_distances, all_margins, c=colors, alpha=0.6)
            plt.xlabel('Voice Distance')
            plt.ylabel('Top-1 vs Top-2 Margin')
            plt.title('Distance vs Confidence')
            plt.grid(True, alpha=0.3)
            
            # 添加圖例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', label='Correct Predictions'),
                             Patch(facecolor='red', label='Incorrect Predictions')]
            plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speaker_identification_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Speaker identification analysis chart saved to: {output_dir / 'speaker_identification_analysis.png'}")
    
    # 創建擴展分析圖表
    create_extended_analysis_plots(all_results, output_dir)


def create_extended_analysis_plots(results: List[Dict], output_dir: Path):
    """創建擴展的語者識別分析圖表"""
    import matplotlib.pyplot as plt
    from collections import defaultdict, Counter
    
    # 收集所有語者統計數據
    all_speaker_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'distances': []})
    confusion_pairs = []
    
    for result in results:
        if 'per_speaker_stats' in result:
            for speaker, stats in result['per_speaker_stats'].items():
                all_speaker_stats[speaker]['total'] += stats['count']
                all_speaker_stats[speaker]['correct'] += stats['correct']
                all_speaker_stats[speaker]['distances'].extend(stats['distances'])
        
        if 'confusion_data' in result:
            confusion_pairs.extend(result['confusion_data']['confusion_pairs'])
    
    # 創建 2x3 的子圖佈局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 按語者準確率圖
    if all_speaker_stats:
        speakers = list(all_speaker_stats.keys())
        accuracies = [all_speaker_stats[s]['correct'] / all_speaker_stats[s]['total'] 
                     for s in speakers]
        
        axes[0, 0].bar(range(len(speakers)), accuracies)
        axes[0, 0].set_xlabel('Speaker ID')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Per-Speaker Accuracy')
        axes[0, 0].set_xticks(range(len(speakers)))
        axes[0, 0].set_xticklabels(speakers, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 按語者平均距離圖
    if all_speaker_stats:
        avg_distances = [np.mean(all_speaker_stats[s]['distances']) if all_speaker_stats[s]['distances'] else 0
                        for s in speakers]
        
        axes[0, 1].bar(range(len(speakers)), avg_distances)
        axes[0, 1].set_xlabel('Speaker ID')
        axes[0, 1].set_ylabel('Average Distance')
        axes[0, 1].set_title('Per-Speaker Average Distance')
        axes[0, 1].set_xticks(range(len(speakers)))
        axes[0, 1].set_xticklabels(speakers, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 新語者 vs 已知語者比例
    new_speakers = sum(1 for s in speakers if s.startswith('n'))
    known_speakers = len(speakers) - new_speakers
    
    axes[0, 2].pie([new_speakers, known_speakers], 
                   labels=['New Speakers', 'Known Speakers'],
                   autopct='%1.1f%%',
                   colors=['orange', 'blue'])
    axes[0, 2].set_title('New vs Known Speaker Distribution')
    
    # 4. 混淆矩陣熱圖（簡化版）
    if confusion_pairs:
        pred_counts = Counter(pair['predicted'] for pair in confusion_pairs)
        true_counts = Counter(pair['actual'] for pair in confusion_pairs)
        
        # 顯示最常見的預測結果
        top_preds = dict(pred_counts.most_common(10))
        
        axes[1, 0].bar(range(len(top_preds)), list(top_preds.values()))
        axes[1, 0].set_xlabel('Predicted Speaker')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Top Predicted Speakers')
        axes[1, 0].set_xticks(range(len(top_preds)))
        axes[1, 0].set_xticklabels(list(top_preds.keys()), rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 距離分布箱線圖（按語者）
    if all_speaker_stats and len(speakers) <= 15:  # 限制語者數量以免圖表過於擁擠
        distance_data = [all_speaker_stats[s]['distances'] for s in speakers[:15]]
        distance_data = [d for d in distance_data if d]  # 移除空列表
        
        if distance_data:
            axes[1, 1].boxplot(distance_data, labels=speakers[:len(distance_data)])
            axes[1, 1].set_xlabel('Speaker ID')
            axes[1, 1].set_ylabel('Distance')
            axes[1, 1].set_title('Distance Distribution by Speaker')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 預測信心度與正確性關係
    confidences = []
    correctness = []
    
    for result in results:
        if 'identification_scores' in result:
            for score in result['identification_scores']:
                confidences.append(score.get('margin', 0))
                correctness.append(1 if score['is_correct'] else 0)
    
    if confidences and correctness:
        # 將信心度分組並計算每組的準確率
        conf_bins = np.linspace(min(confidences), max(confidences), 10)
        bin_indices = np.digitize(confidences, conf_bins)
        
        bin_accuracies = []
        bin_centers = []
        
        for i in range(1, len(conf_bins)):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean([correctness[j] for j in range(len(correctness)) if mask[j]])
                bin_accuracies.append(bin_acc)
                bin_centers.append((conf_bins[i-1] + conf_bins[i]) / 2)
        
        if bin_accuracies:
            axes[1, 2].plot(bin_centers, bin_accuracies, 'o-')
            axes[1, 2].set_xlabel('Confidence (Margin)')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].set_title('Confidence vs Accuracy')
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'extended_speaker_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Extended speaker analysis chart saved to: {output_dir / 'extended_speaker_analysis.png'}")

def generate_speaker_metrics_report(batch_metrics: dict, output_path: Path) -> None:
    """
    生成語者辨識指標報告
    
    Args:
        batch_metrics: 批次統計指標
        output_path: 報告輸出路徑
    """
    report = f"""
# 語者辨識評估報告

## 整體統計
- 總樣本數: {batch_metrics['total_samples']}
- 整體準確率: {batch_metrics['overall_accuracy']:.4f}

## 錯誤率指標
- EER (Equal Error Rate): {batch_metrics['eer']:.4f}
- minDCF (Minimum Detection Cost Function): {batch_metrics['min_dcf']:.4f}
- Cllr (Calibration Loss): {batch_metrics['cllr']:.4f}

## 距離分布統計
- 平均距離: {batch_metrics['distance_distribution']['mean']:.4f}
- 距離標準差: {batch_metrics['distance_distribution']['std']:.4f}
- 最小距離: {batch_metrics['distance_distribution']['min']:.4f}
- 最大距離: {batch_metrics['distance_distribution']['max']:.4f}

## Top-1 vs Top-2 Margin 分布
- 平均 Margin: {batch_metrics['margin_distribution']['mean']:.4f}
- Margin 標準差: {batch_metrics['margin_distribution']['std']:.4f}
- 最小 Margin: {batch_metrics['margin_distribution']['min']:.4f}
- 最大 Margin: {batch_metrics['margin_distribution']['max']:.4f}

## 系統穩定性
- 信心度穩定性 (變異係數): {batch_metrics['confidence_stability']:.4f}
  - 數值越小表示系統預測信心度越穩定
  - < 0.3: 非常穩定
  - 0.3-0.5: 穩定
  - 0.5-1.0: 中等穩定
  - > 1.0: 不穩定

## 指標解釋

### EER (Equal Error Rate)
等錯誤率，當 False Accept Rate 等於 False Reject Rate 時的錯誤率。
數值越小越好，表示系統識別能力越強。

### minDCF (Minimum Detection Cost Function)
最小檢測代價函數，考慮不同類型錯誤的代價。
數值越小越好，表示系統整體性能越優。

### Cllr (Calibration Loss)
校準損失，衡量系統輸出分數的校準性。
數值越小越好，表示系統輸出的信心度與實際準確率越一致。

### Top-1 vs Top-2 Margin
最相似語者與第二相似語者的距離差值。
數值越大表示系統對預測結果越有信心。

### 距離分布
聲紋比對的餘弦距離分布情況。
正確預測的距離應該明顯小於錯誤預測的距離。
"""
    
    with output_path.open('w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Speaker identification evaluation report saved to: {output_path}")

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
        "accuracy", "min_distance", "top1_top2_margin", 
        "prediction_confidence", "distance_variance",
        "true_speaker1", "true_speaker2",
        "pred_speaker1", "pred_speaker2", 
        "distance1", "distance2",
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
        
        # 為每個段落添加詳細的語者辨識資訊
        enhanced_bundle = []
        for seg in bundle:
            if 'path' in seg:
                # 重新進行語者辨識以獲取詳細資訊
                try:
                    # 讀取音訊
                    import soundfile as sf
                    signal, sr = sf.read(seg['path'])
                    
                    # 提取嵌入向量
                    new_embedding = spk.audio_processor.extract_embedding_from_stream(signal, sr)
                    
                    # 與資料庫比對以獲取所有候選結果
                    best_id, best_name, best_distance, all_distances = spk.database.compare_embedding(new_embedding)
                    
                    # 添加詳細的語者辨識資訊
                    seg['speaker_distances'] = all_distances  # 所有候選結果
                    seg['detailed_speaker_info'] = {
                        'best_id': best_id,
                        'best_name': best_name,
                        'best_distance': best_distance,
                        'all_candidates': all_distances
                    }
                    
                    # 如果 API 返回的是新語者，我們仍然保留與現有語者的距離資訊
                    if seg.get('speaker', '').startswith('n') and all_distances:
                        # 新語者的情況：更新 distance 為與最近現有語者的距離
                        seg['distance'] = best_distance
                    
                except Exception as e:
                    logger.warning(f"無法獲取段落 {seg.get('path', 'unknown')} 的詳細語者資訊: {e}")
                    seg['speaker_distances'] = []
                    seg['detailed_speaker_info'] = {}
                    
            enhanced_bundle.append(seg)
        
        bundle = enhanced_bundle
        
    except Exception as e:
        logger.error(f"處理檔案 {mix_path} 時出錯：{e}")
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
            # 路徑已在 load_mixture_map 中正規化，直接使用即可
            src1 = clean_dir / row[2]
            src2 = clean_dir / row[3]
            sep_paths = [Path(s["path"]) for s in bundle if "path" in s]
            sisdr_metrics = compute_baseline_sisdr(mix_wav, src1, src2, sep, sep_paths)
            result.update(sisdr_metrics)
            
            # --- 取出 true speaker IDs from mixture_map row ---
            # 路徑已正規化，直接使用 Path 解析
            # row[2] = "speaker10/speaker10_17.wav" (已正規化)
            # Path(row[2]).parent.name → "speaker10"
            # .replace("speaker", "spk") → "spk10"
            
            true_spk1 = Path(row[2]).parent.name.replace("speaker", "spk")
            true_spk2 = Path(row[3]).parent.name.replace("speaker", "spk")
            true_speakers = [true_spk1, true_spk2]
            # logger.debug(f"真實語者：{true_speakers}")
            
            # --- 計算詳細的語者辨識指標 ---
            speaker_metrics = compute_speaker_identification_metrics(bundle, true_speakers, spk)
            result.update(speaker_metrics)
            
                        # === CER & ref/pred text for each speaker === #
            # 1) 地取每個 source 的正確文字
            ref_fname1 = Path(src1).name                         # e.g. "speaker10_17.wav"
            ref_fname2 = Path(src2).name
            raw_ref1 = truth_map.get(ref_fname1, "")
            raw_ref2 = truth_map.get(ref_fname2, "")

            # 2) Normalize (去標點、空格、統一繁體)
            ref_norm1 = normalize_zh(raw_ref1)
            ref_norm2 = normalize_zh(raw_ref2)
            # logger.debug(f"正規化文字1：{ref_norm1}, 正規化文字2：{ref_norm2}")

            # 3) 直接抓 bundle 前兩段文字，不理會 speaker id
            pred_texts = [seg.get("text", "") for seg in bundle if seg.get("text")]
            
            # 確保至少有 2 個元素，不足則補空字串
            while len(pred_texts) < 2:
                pred_texts.append("")
            
            predA, predB = pred_texts[:2]                      # A = 第一段, B = 第二段

            normA = normalize_numbers_to_zh(normalize_zh(predA))
            normB = normalize_numbers_to_zh(normalize_zh(predB))
            # logger.debug(f"預測文字1：{normA}, 預測文字2：{normB}")
            # 4) 交叉計算 4 個 CER
            cA1 = compute_cer(ref_norm1, normA) if ref_norm1 else None
            cB2 = compute_cer(ref_norm2, normB) if ref_norm2 else None
            cA2 = compute_cer(ref_norm2, normA) if ref_norm2 else None
            cB1 = compute_cer(ref_norm1, normB) if ref_norm1 else None
            # logger.debug(f"CER1：{cA1:.4f if cA1 else 'N/A'} CER2：{cB2:.4f if cB2 else 'N/A'}")
            
                        # 5) 選「加總最小」的配對
            if (cA1 or 0) + (cB2 or 0) <= (cA2 or 0) + (cB1 or 0):
                final_pred1, final_pred2 = normA, normB
                cer1, cer2 = cA1, cB2
            else:
                final_pred1, final_pred2 = normB, normA
                cer1, cer2 = cB1, cA2
            # 5) 寫入結果
            result.update({
                "ref_text1": ref_norm1,
                "pred_text1": final_pred1,
                "cer1": cer1,
                "ref_text2": ref_norm2,
                "pred_text2": final_pred2,
                "cer2": cer2,
            })
            torch.cuda.empty_cache()
            gc.collect()
        else:
            logger.warning(f"mix_id {mix_id} not found in mixture_map")
        
            

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
    
    # 收集所有結果用於批次統計
    all_results = []

    with output_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=cols)
        writer.writeheader()

        for mix_id, mix_file, *_ in test_rows:              # ② 逐筆跑
            mix_path = mix_dir / mix_file
            logger.info(f"處理 {mix_id} → {mix_path}")

            res = process_one_mixture(
                str(mix_path), mix_id,
                sep=sep, spk=spk, asr=asr,
                clean_dir=clean_dir,
                mixture_rows=test_rows,
                truth_map=truth_map
            )

            # ③ 如果 pipeline 出錯就跳過，不寫進 CSV
            if "error" in res:
                logger.warning(f"跳過 {mix_id}，原因：{res['error']}")
                continue

            # ④ 只留下指定欄位，並處理特殊數值
            row = {}
            for k in cols:
                value = res.get(k, "")
                # 處理無窮大和 NaN 值
                if isinstance(value, float):
                    if value == float('inf'):
                        value = ""
                    elif value == float('-inf'):
                        value = ""
                    elif np.isnan(value):
                        value = ""
                row[k] = value
            writer.writerow(row)
            
            # ⑤ 收集結果用於批次統計
            all_results.append(res)

    # ⑥ 計算批次統計並保存
    if all_results:
        batch_metrics = calculate_batch_speaker_metrics(all_results)
        
        # 創建輸出目錄
        output_dir = output_csv.parent
        batch_stats_file = output_dir / f"{output_csv.stem}_batch_speaker_metrics.json"
        report_file = output_dir / f"{output_csv.stem}_speaker_analysis_report.md"
        
        # 保存批次統計到 JSON 文件
        with batch_stats_file.open("w", encoding="utf-8") as f:
            json.dump(batch_metrics, f, indent=2, ensure_ascii=False)
        
        # 生成分析圖表
        create_speaker_identification_plots(all_results, output_dir)
        
        # 生成詳細報告
        generate_speaker_metrics_report(batch_metrics, report_file)
        
        logger.info(f"Batch speaker identification statistics saved to: {batch_stats_file}")
        
        # 輸出關鍵統計資訊
        logger.info("📊 Batch Speaker Identification Statistics Summary:")
        logger.info(f"   Total Samples: {batch_metrics['total_samples']}")
        logger.info(f"   Overall Accuracy: {batch_metrics['overall_accuracy']:.3f}")
        logger.info(f"   EER: {batch_metrics['eer']:.3f}")
        logger.info(f"   minDCF: {batch_metrics['min_dcf']:.3f}")
        logger.info(f"   Average Distance: {batch_metrics['distance_distribution']['mean']:.4f} ± {batch_metrics['distance_distribution']['std']:.4f}")
        logger.info(f"   Average Margin: {batch_metrics['margin_distribution']['mean']:.4f} ± {batch_metrics['margin_distribution']['std']:.4f}")
        logger.info(f"   Confidence Stability: {batch_metrics['confidence_stability']:.4f}")

    logger.info(f"All completed! Results written to {output_csv}")
    
def main():
    # 1️⃣ 清空 GPU 快取
    torch.cuda.empty_cache()
    gc.collect()

    # 2️⃣ 載入所有參數與資料
    #    --mix-dir、--clean-dir、--truth-map、--test-list、--out 都已設預設
    args, mix_dir, clean_dir, truth_map, mixture_rows = load_config_and_data()

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
    # test_mix_id = "m05"  # 你可以自訂一個 ID
    # test_mix_path = "data/mix/m05.wav"  # ← 改成你實際有的音檔路徑！

    # logger.info(f"處理測試音檔：{test_mix_path}")
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
    # logger.info(f"已儲存 JSON 檔至：{test_mix_id}_result.json")
    
    
if __name__ == "__main__":
    main()
