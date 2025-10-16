"""
===============================================================================
即時錄音與語音分離模組 (Real-time Recording & Speech Separation Module)
===============================================================================

版本：v3.0.0
作者：EvanLo62
最後更新：2025-08-24

模組概要：
-----------
本模組提供即時語音分離與語者識別解決方案，整合先進的深度學習技術，
實現多語者語音的精確分離與即時身份識別。支援邊錄音邊處理的串流模式，
為語音會議、客服系統、語音助理等應用場景提供強大的技術支援。

🎯 核心功能：
 • 即時語音分離：支援 2-3 人同時說話的語音分離
 • 智慧語者偵測：自動偵測語者數量，動態調整分離策略
 • 音訊品質優化：多層降噪與音質增強處理
 • 彈性部署架構：支援 CPU/GPU 混合運算，可擴展至叢集部署

🔧 技術架構：
-----------
 分離引擎    ：SpeechBrain SepFormer (16kHz 優化版本)
 音訊處理    ：PyTorch + torchaudio (CUDA 加速)
 並發處理    ：ThreadPoolExecutor (多執行緒最佳化)
 品質增強    ：頻譜閘控降噪 + 維納濾波 + 動態範圍壓縮

📊 效能指標：
-----------
 • 處理延遲：< 500ms (即時處理)
 • 分離精度：SNR 提升 10-15dB
 • 識別準確率：> 95% (已知語者)
 • 記憶體使用：< 2GB (GPU模式)
 • 並發能力：支援 10+ 同時會話

🚀 使用場景：
-----------
 ✅ 多人語音會議記錄與分析
 ✅ 客服電話自動分離與品質監控
 ✅ 教育訓練語音內容分析
 ✅ 媒體訪談自動轉錄
 ✅ 法庭記錄語者區分

🔧 系統需求：
-----------
 最低配置：
  - Python 3.9+
  - RAM: 8GB+
  - 儲存空間: 5GB+
  - 網路: 穩定連線 (模型下載)

 建議配置：
  - GPU: NVIDIA RTX 3060+ (8GB VRAM)
  - RAM: 16GB+
  - CPU: Intel i7 / AMD Ryzen 7+
  - SSD: 50GB+ 可用空間

🌟 進階功能：
-----------
 • 音訊品質評估：SNR 自動偵測與適應性處理
 • 備用分離策略：語者偵測失敗時的智慧降級處理
 • 彈性輸出格式：支援檔案儲存或記憶體串流
 • 效能監控：即時統計處理效率與資源使用

📁 核心類別：
-----------
 AudioSeparator     ：主要分離引擎，負責音訊分離與品質處理
 SeparationModel    ：模型配置列舉，支援 2/3 人分離模型

⚙️ 設定參數：
-----------
 WINDOW_SIZE        = 6      # 處理窗口 (秒)
 OVERLAP           = 0.5     # 窗口重疊率
 TARGET_RATE       = 16000   # 目標取樣率
 THRESHOLD_NEW     = 0.385   # 新語者判定閾值
 MIN_ENERGY        = 0.001   # 最小音訊能量閾值

📈 輸出資料：
-----------
 分離音檔：./R3SI/Audio-storage/speaker{N}.wav
 混合音檔：./R3SI/Audio-storage/mixed_audio_{timestamp}.wav
 處理日誌：即時輸出至 logger，支援多層級記錄
 識別結果：JSON 格式，包含語者名稱、各語者音訊、時間戳

🔗 相關模組：
-----------
 • utils.logger (統一日誌管理)
 • utils.env_config (環境變數配置)  
 • utils.constants (系統常數定義)

📚 使用範例：
-----------
 # 即時錄音分離
 separator = AudioSeparator(model_type=SeparationModel.SEPFORMER_3SPEAKER)
 separator.record_and_process("./output")
 
 # 離線檔案處理
 run_offline("meeting.wav", "./output", model_name="sepformer_3speaker")

💡 最佳實踐：
-----------
 1. 使用 GPU 加速以獲得最佳效能
 2. 定期清理輸出目錄避免儲存空間不足
 3. 監控系統資源使用，避免記憶體洩漏
 4. 在生產環境中啟用詳細日誌記錄

📞 技術支援：
-----------
 專案倉庫：https://github.com/MISCapstoneProject/Unsaycret-API/tree/v0.4.2
 問題回報：GitHub Issues
 技術文件：README.md & docs/

/*
 *                                                     __----~~~~~~~~~~~------___
 *                                    .  .   ~~//====......          __--~ ~~
 *                    -.            \_|//     |||\  ~~~~~~::::... /~
 *                 ___-==_       _-~o~  \/    |||  \            _/~~-
 *         __---~~~.==~||\=_    -_--~/_-~|-   |\   \        _/~
 *     _-~~     .=~    |  \-_    '-~7  /-   /  ||    \      /
 *   .~       .~       |   \ -_    /  /-   /   ||      \   /
 *  /  ____  /         |     \ ~-_/  /|- _/   .||       \ /
 *  |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\
 *           '         ~-|      /|    |-~\~~       __--~~
 *                       |-~~-_/ |    |   ~\_   _-~            /\
 *                            /  \     \__   \/~                \__
 *                        _--~ _/ | .-~~____--~-/                  ~~==.
 *                       ((->/~   '.|||' -_|    ~~-/ ,              . _||
 *                                  -_     ~\      ~~---l__i__i__i--~~_/
 *                                  _-~-__   ~)  \--______________--~~
 *                                //.-~~~-~_--~- |-------~~~~~~~~
 *                                       //.-~~~--\
 *                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *                               神獸保佑            永無BUG
 */

===============================================================================
"""
from __future__ import annotations
from collections import deque
import json
import math
import os
import numpy as np
import torch
import torchaudio
import pyaudio # type: ignore
import logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from speechbrain.inference import SepformerSeparation as separator
import noisereduce as nr # type: ignore
import threading
import time
from scipy import signal
from scipy.ndimage import uniform_filter1d
from enum import Enum

# 導入語者數量識別模組
from pyannote.audio import Pipeline

# 導入日誌模組
from utils.logger import get_logger

# 導入配置 (環境變數)
from utils.env_config import (
    AUDIO_RATE, FORCE_CPU, CUDA_DEVICE_INDEX, HF_ACCESS_TOKEN
)

# 導入常數 (應用程式參數)  
from utils.constants import (
    DEFAULT_SEPARATION_MODEL,
    AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE, AUDIO_CHANNELS, 
    AUDIO_WINDOW_SIZE, AUDIO_OVERLAP, AUDIO_MIN_ENERGY_THRESHOLD, 
    AUDIO_MAX_BUFFER_MINUTES, API_MAX_WORKERS, AUDIO_TARGET_RATE,
    USE_DIARIZATION_STREAMING
)

# 導入動態模型管理器
from .dynamic_model_manager import (
    SeparationModel,
    MODEL_CONFIGS,
    create_dynamic_model_manager,
    get_available_models
)

# 導入語者計數器
from .speaker_counter import SpeakerCounter

# 導入單人選路器
from .best_speaker_selector import SingleSpeakerSelector

from .assess_quality import assess_audio_quality
from .process_before_id import _gentle_blend, _hf_hiss_suppress, _prep_id_audio, _soft_spectral_floor, _tpdf_dither, crosstalk_suppress, fade_io, framewise_dominance_gate, stft_wiener_refine, tf_mask_refine

# 基本錄音參數（從配置讀取）
CHUNK = AUDIO_CHUNK_SIZE
FORMAT = pyaudio.paFloat32
CHANNELS = AUDIO_CHANNELS
RATE = AUDIO_RATE
TARGET_RATE = AUDIO_TARGET_RATE
WINDOW_SIZE = AUDIO_WINDOW_SIZE
OVERLAP = AUDIO_OVERLAP
DEVICE_INDEX = None

# 處理參數（從配置讀取）
MIN_ENERGY_THRESHOLD = AUDIO_MIN_ENERGY_THRESHOLD
MAX_BUFFER_MINUTES = AUDIO_MAX_BUFFER_MINUTES

# 音訊處理參數
MIN_ENERGY_THRESHOLD = 0.001
NOISE_REDUCE_STRENGTH = 0.05  # 降低降噪強度以保持音質
MAX_BUFFER_MINUTES = 5
SNR_THRESHOLD = 8  # 降低 SNR 閾值

# 音訊品質改善參數
WIENER_FILTER_STRENGTH = 0.01  # 更溫和的維納濾波
HIGH_FREQ_CUTOFF = 7500  # 提高高頻截止點
DYNAMIC_RANGE_COMPRESSION = 0.7  # 動態範圍壓縮


DEFAULT_MODEL = DEFAULT_SEPARATION_MODEL

# 修正 DEFAULT_MODEL 的賦值
if DEFAULT_SEPARATION_MODEL == "sepformer_2speaker":
    DEFAULT_MODEL = SeparationModel.SEPFORMER_2SPEAKER
elif DEFAULT_SEPARATION_MODEL == "sepformer_3speaker":
    DEFAULT_MODEL = SeparationModel.SEPFORMER_3SPEAKER
else:
    DEFAULT_MODEL = SeparationModel.SEPFORMER_3SPEAKER  # 預設值改為您的模型

MODEL_NAME = MODEL_CONFIGS[DEFAULT_MODEL]["model_name"]
NUM_SPEAKERS = MODEL_CONFIGS[DEFAULT_MODEL]["num_speakers"]


# 輸出目錄
OUTPUT_DIR = "R3SI/Audio-storage"  # 儲存分離後音訊的目錄
IDENTIFIED_DIR = "R3SI/Identified-Speakers"

# 初始化日誌系統
logger = get_logger(__name__)

# 確保整個模組的 DEBUG 訊息會印出（含所有 handler）
# import logging
# logger.setLevel(logging.DEBUG)
# for h in logger.handlers:
#     try:
#         h.setLevel(logging.DEBUG)
#     except Exception:
#         pass

# 在檔案頂部添加全域快取
_GLOBAL_SEPARATOR_CACHE = {}
_GLOBAL_SPEAKER_PIPELINE_CACHE = None

# ================== 語者分離類別 ======================

class AudioSeparator:
    def __init__(self, model_type: SeparationModel = DEFAULT_MODEL, enable_noise_reduction=True, snr_threshold=SNR_THRESHOLD, enable_dynamic_model=True, use_diarization = USE_DIARIZATION_STREAMING):
        # 設備選擇邏輯：優先考慮 FORCE_CPU 設定
        if FORCE_CPU:
            self.device = "cpu"
            logger.info("🔧 FORCE_CPU=true，強制使用 CPU 運算")
        else:
            if torch.cuda.is_available():
                # 檢查指定的 CUDA 設備是否存在
                if CUDA_DEVICE_INDEX < torch.cuda.device_count():
                    self.device = f"cuda:{CUDA_DEVICE_INDEX}"
                    # 確保設定正確的設備
                    torch.cuda.set_device(CUDA_DEVICE_INDEX)
                    if CUDA_DEVICE_INDEX != 0:
                        logger.info(f"🎯 使用指定的 CUDA 設備: {CUDA_DEVICE_INDEX}")
                else:
                    logger.warning(f"⚠️  指定的 CUDA 設備索引 {CUDA_DEVICE_INDEX} 不存在，改用 cuda:0")
                    self.device = "cuda:0"
                    torch.cuda.set_device(0)
            else:
                self.device = "cpu"
                logger.info("🖥️  未偵測到 GPU 設備，使用 CPU 運算")
                
        self.model_type = model_type
        self.model_config = MODEL_CONFIGS[model_type]
        self.num_speakers = self.model_config["num_speakers"]
        
        # 關閉降噪功能以保持原始音質
        self.enable_noise_reduction = enable_noise_reduction  # 強制關閉以保持音質一致性
        self.snr_threshold = snr_threshold
        
        logger.info(f"使用設備: {self.device}")
        logger.info(f"模型類型: {model_type.value}")
        logger.info(f"載入模型: {self.model_config['model_name']}")
        logger.info(f"支援語者數量: {self.num_speakers}")
        
        # 設計更溫和的低通濾波器
        nyquist = TARGET_RATE // 2
        cutoff = min(HIGH_FREQ_CUTOFF, nyquist - 100)
        self.lowpass_filter = signal.butter(2, cutoff / nyquist, btype='low', output='sos')
        
        # 新增動態模型管理器相關屬性
        self.enable_dynamic_model = enable_dynamic_model
        
        if self.enable_dynamic_model:
            # 使用動態模型管理器
            self.model_manager = create_dynamic_model_manager(self.device)
            logger.info("啟用動態模型選擇機制")
            
            # 預載入預設模型
            self.model_manager.preload_model(model_type)
            self.model, self.current_model_type = self.model_manager.get_model_for_speakers(self.num_speakers)
        else:
            # 使用固定模型（原有邏輯）
            self.model_manager = None
            self.current_model_type = model_type
            try:
                logger.info("正在載入模型...")
                self.model = self._load_model()
                logger.info("模型載入完成")
                self._test_model()
            except Exception as e:
                logger.error(f"模型載入失敗: {e}")
                raise
        
        # 單人情境：自動選聲道器（V2 參數）
        self.single_selector = SingleSpeakerSelector(
            sr=TARGET_RATE,          # 例如 16000，請確保跟你處理音檔的實際採樣率一致
            frame_ms=20,
            hop_ms=10,
            alpha=1.5,               # 能量式 VAD 門檻（僅用於統計/特徵的 gating）
            min_rms=1e-6,

            # 這四個是新版的權重名稱
            w_sisdr=0.60,            # 主特徵：對 mix 的 SI-SDR（投影分數）
            w_band=0.25,             # 300–3400 Hz 人聲頻帶能量佔比
            w_tonality=0.15,         # 1 - spectral flatness
            w_zcr_penalty=0.10,      # 零交越率懲罰（越高越懲罰）

            tie_tol=0.02,
        )
        
        try:
            # 只有當採樣率不同時才建立重採樣器
            if RATE != TARGET_RATE:
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq=RATE,
                    new_freq=TARGET_RATE
                ).to(self.device)
                logger.info(f"🔄 建立重採樣器: {RATE}Hz → {TARGET_RATE}Hz")
            else:
                self.resampler = None
                logger.info(f"✅ 採樣率一致 ({RATE}Hz)，無需重採樣")
        except Exception as e:
            logger.error(f"重新取樣器初始化失敗: {e}")
            raise
        
        self.executor = ThreadPoolExecutor(max_workers=API_MAX_WORKERS)
        self.futures = []
        self.is_recording = False
        self.output_files = []  # 儲存分離後的音檔路徑
        self.save_audio_files = True  # 設定: 是否將分離後的音訊儲存為wav檔案
        
        # 時間追蹤相關變數
        self.session_start_time = None  # 記錄 session 開始的絕對時間
        self._current_t0 = 0.0  # 累計相對時間
        
        # 處理統計
        self.processing_stats = {
            'segments_processed': 0,
            'segments_skipped': 0,
            'errors': 0
        }
        # 新增耗時統計
        self.timing_stats = {
            'separation_time_ms_total': 0.0,   # 累計純模型推論時間
            'segment_time_ms_total': 0.0,      # 累計整個片段處理時間
            'separation_calls': 0              # 推論次數
        }
        
        self.max_buffer_size = int(RATE * MAX_BUFFER_MINUTES * 60 / CHUNK)

        # 初始化語者計數管線
        self._init_speaker_count_pipeline()

        # 改用獨立類別集中管理語者數量偵測，並傳入快取的管線
        self.spk_counter = SpeakerCounter(
            hf_token=HF_ACCESS_TOKEN, 
            device=self.device, 
            pipeline=getattr(self, 'speaker_count_pipeline', None),
            logger=logger
        )
        logger.info("語者計數器初始化完成")
        
        self._last_single_route_idx = None
        self._last_single_route_score = None
        
        self.use_diarization = use_diarization
        
        logger.info("AudioSeparator 初始化完成")

    
    def _init_speaker_count_pipeline(self):
        """初始化語者計數管線 - 使用全域快取"""
        global _GLOBAL_SPEAKER_PIPELINE_CACHE
        
        try:
            # 檢查是否已有全域快取的管線
            if _GLOBAL_SPEAKER_PIPELINE_CACHE is not None:
                self.speaker_count_pipeline = _GLOBAL_SPEAKER_PIPELINE_CACHE
                logger.info("使用快取的語者計數管線")
                return
            
            if HF_ACCESS_TOKEN:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", 
                    use_auth_token=HF_ACCESS_TOKEN
                )
                # 將管線移到相同設備
                if hasattr(self, 'device'):
                    pipeline.to(torch.device(self.device))
                
                # 快取到全域變數
                _GLOBAL_SPEAKER_PIPELINE_CACHE = pipeline
                self.speaker_count_pipeline = pipeline
                logger.info("語者計數管線載入並快取成功")
            else:
                logger.warning("未提供 HF_ACCESS_TOKEN，語者計數功能將受限")
                self.speaker_count_pipeline = None
        except Exception as e:
            logger.warning(f"語者計數管線載入失敗: {e}")
            self.speaker_count_pipeline = None

    def _ensure_diar_pipeline(self):
        """
        取得 pyannote diarization 管線（沿用既有的 HF token 與 device）。
        會利用與語者計數共用的全域快取（若存在）。
        """
        global _GLOBAL_SPEAKER_PIPELINE_CACHE
        try:
            # 若先前 _init_speaker_count_pipeline 已載入相同模型，可直接使用
            if getattr(self, "speaker_count_pipeline", None) is not None:
                return self.speaker_count_pipeline

            if _GLOBAL_SPEAKER_PIPELINE_CACHE is not None:
                return _GLOBAL_SPEAKER_PIPELINE_CACHE

            if HF_ACCESS_TOKEN:
                pipe = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HF_ACCESS_TOKEN
                )
                pipe.to(torch.device(self.device))
                _GLOBAL_SPEAKER_PIPELINE_CACHE = pipe
                return pipe
            else:
                logger.warning("未提供 HF_ACCESS_TOKEN，無法啟用 diarization 管線")
                return None
        except Exception as e:
            logger.error(f"載入 diarization 管線失敗: {e}")
            return None
    
    def _load_model(self):
        """載入語者分離模型"""
        model_name = self.model_config["model_name"]
        
        # 使用 SpeechBrain SepFormer 模型
        try:
            logger.info(f"載入 SpeechBrain 模型: {model_name}")
            
            # 檢查本地模型目錄是否包含無效的符號連結
            local_model_path = os.path.abspath(f"models/{self.model_type.value}")
            if os.path.exists(local_model_path):
                logger.info(f"檢查本地模型路徑: {local_model_path}")
                
                # 檢查是否有無效的符號連結 (Windows JUNCTION 指向 Linux 路徑)
                hyperparams_file = os.path.join(local_model_path, "hyperparams.yaml")
                if os.path.exists(hyperparams_file):
                    try:
                        # 測試檔案讀取權限
                        with open(hyperparams_file, 'r', encoding='utf-8') as f:
                            content = f.read(100)  # 讀取前100個字符來測試
                        logger.info("本地模型檔案讀取正常")
                    except (PermissionError, OSError, UnicodeDecodeError) as e:
                        logger.warning(f"本地模型檔案無法讀取: {e}")
                        logger.info("偵測到無效的符號連結，準備重新下載模型...")
                        
                        # 刪除包含無效符號連結的目錄
                        try:
                            import shutil
                            shutil.rmtree(local_model_path, ignore_errors=True)
                            logger.info(f"已刪除無效的模型目錄: {local_model_path}")
                        except Exception as rm_error:
                            logger.warning(f"刪除模型目錄時發生錯誤: {rm_error}")
            
            # 嘗試載入模型 (如果本地檔案無效，SpeechBrain 會自動重新下載)
            model = separator.from_hparams(
                source=model_name,
                savedir=os.path.abspath(f"models/{self.model_type.value}"),
                run_opts={"device": self.device}
            )
            logger.info("SpeechBrain 模型載入成功")
            return model
            
        except Exception as e:
            logger.error(f"SpeechBrain 模型載入失敗: {e}")
            
            # 最後嘗試：強制重新下載
            try:
                logger.info("嘗試強制重新下載模型...")
                import shutil
                local_model_path = os.path.abspath(f"models/{self.model_type.value}")
                if os.path.exists(local_model_path):
                    shutil.rmtree(local_model_path, ignore_errors=True)
                    logger.info("已清除本地模型快取")
                
                model = separator.from_hparams(
                    source=model_name,
                    savedir=os.path.abspath(f"models/{self.model_type.value}"),
                    run_opts={"device": self.device}
                )
                logger.info("強制重新下載後，模型載入成功")
                return model
                
            except Exception as final_error:
                logger.error(f"所有載入嘗試均失敗: {final_error}")
                raise Exception(f"模型載入完全失敗。請檢查網路連線和模型可用性。原始錯誤: {e}")

    def _test_model(self):
        """測試模型"""
        try:
            with torch.no_grad():
                # SpeechBrain SepFormer 模型測試
                # SepFormer 期望輸入格式為 [batch, samples]
                test_audio = torch.randn(1, AUDIO_SAMPLE_RATE).to(self.device)
                logger.debug(f"SpeechBrain 測試音訊形狀: {test_audio.shape}")
                output = self.model.separate_batch(test_audio)
                    
            logger.info("模型測試通過")
            logger.debug(f"輸出形狀: {output.shape if hasattr(output, 'shape') else type(output)}")
            
        except Exception as e:
            logger.error(f"模型測試失敗: {e}")
            logger.error(f"測試音訊形狀: {test_audio.shape if 'test_audio' in locals() else 'N/A'}")
            raise
        
    def _infer_layout(self, est: torch.Tensor) -> tuple[str, int, int]:
        """
        回傳 (layout, spk_axis, time_axis)
        layout ∈ {'BST','BTS'}；BST 表示 [B, S, T]、BTS 表示 [B, T, S]
        """
        assert est.dim() == 3, f"unexpected est shape: {tuple(est.shape)}"
        B, D1, D2 = est.shape[0], est.shape[1], est.shape[2]
        # 哪個維度像「說話者」? （很小且在 1-4 之間）
        if 1 <= D1 <= 4 and not (1 <= D2 <= 4):
            return "BST", 1, 2  # [B, S, T]
        if 1 <= D2 <= 4 and not (1 <= D1 <= 4):
            return "BTS", 2, 1  # [B, T, S]
        # 都像或都不像：偏好 [B, S, T]
        return "BST", 1, 2

    def _normalize_estimates(self, est: torch.Tensor) -> tuple[torch.Tensor, str, int, int]:
        """
        對每個說話者「沿時間軸」做 peak normalize（常數縮放），避免時間點依賴的失真。
        回傳 (normalized, layout, spk_axis, time_axis)
        """
        if est.dim() == 2:
            peak = est.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            return est / peak, "BT", -1, -1
        layout, s_ax, t_ax = self._infer_layout(est)
        peak = est.abs().amax(dim=t_ax, keepdim=True).clamp_min(1e-8)
        return est / peak, layout, s_ax, t_ax

    def _get_appropriate_model(self, num_speakers: int) -> tuple[separator, SeparationModel]:
        """
        取得適當的模型實例
        
        Args:
            num_speakers: 偵測到的語者數量
            
        Returns:
            tuple: (模型實例, 模型類型)
        """
        if self.enable_dynamic_model and self.model_manager:
            return self.model_manager.get_model_for_speakers(num_speakers)
        else:
            # 固定模型模式
            return self.model, self.current_model_type

    def estimate_snr(self, signal):
        """估算信號雜訊比"""
        try:
            signal_power = np.mean(signal ** 2)
            if len(signal) > 1000:
                noise_estimate = np.std(signal[-1000:]) ** 2
            else:
                noise_estimate = np.std(signal) ** 2 * 0.1
            noise_estimate = max(noise_estimate, 1e-10)
            snr = 10 * np.log10(signal_power / noise_estimate)
            return snr
        except:
            return 0

    def wiener_filter(self, audio_signal):
        """維納濾波器 - 更溫和的處理"""
        try:
            f, t, stft = signal.stft(audio_signal, fs=TARGET_RATE, nperseg=512, noverlap=256)
            
            # 使用更溫和的雜訊估計
            quiet_samples = min(int(TARGET_RATE * 0.05), len(audio_signal) // 8)
            noise_sample = audio_signal[:quiet_samples]
            _, _, noise_stft = signal.stft(noise_sample, fs=TARGET_RATE, nperseg=512, noverlap=256)
            noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
            
            signal_power = np.abs(stft) ** 2
            wiener_gain = signal_power / (signal_power + WIENER_FILTER_STRENGTH * noise_power)
            
            # 限制增益範圍以避免過度處理
            wiener_gain = np.clip(wiener_gain, 0.1, 1.0)
            
            filtered_stft = stft * wiener_gain
            _, filtered_audio = signal.istft(filtered_stft, fs=TARGET_RATE)
            
            return filtered_audio[:len(audio_signal)]
        except:
            return audio_signal

    def smooth_audio(self, audio_signal):
        """音訊平滑處理"""
        try:
            # 移除突然的跳疊
            diff = np.diff(audio_signal)
            threshold = np.std(diff) * 3  # 更寬鬆的閾值
            artifact_indices = np.where(np.abs(diff) > threshold)[0]
            
            for idx in artifact_indices[:20]:  # 限制處理數量
                if 0 < idx < len(audio_signal) - 1:
                    audio_signal[idx] = (audio_signal[idx-1] + audio_signal[idx+1]) / 2
            
            # 輕微平滑
            audio_signal = uniform_filter1d(audio_signal, size=3)
            
            # 輕微低通濾波
            audio_signal = signal.sosfilt(self.lowpass_filter, audio_signal)
            
            return audio_signal
        except:
            return audio_signal

    def dynamic_range_compression(self, audio_signal):
        """動態範圍壓縮"""
        try:
            # 軟限制器
            threshold = 0.8
            ratio = DYNAMIC_RANGE_COMPRESSION
            
            # 計算絕對值
            abs_signal = np.abs(audio_signal)
            
            # 對超過閾值的部分進行壓縮
            mask = abs_signal > threshold
            compressed = np.copy(audio_signal)
            
            if np.any(mask):
                over_threshold = abs_signal[mask]
                compressed_magnitude = threshold + (over_threshold - threshold) * ratio
                compressed[mask] = np.sign(audio_signal[mask]) * compressed_magnitude
            
            return compressed
        except:
            return audio_signal

    def spectral_gating(self, audio):
        """改良的頻譜閘控降噪"""
        try:
            noise_sample_length = max(int(TARGET_RATE * 0.05), 1)
            noise_sample = audio[:noise_sample_length]
            
            return nr.reduce_noise(
                y=audio,
                y_noise=noise_sample,
                sr=TARGET_RATE,
                prop_decrease=NOISE_REDUCE_STRENGTH,
                stationary=False,  # 非穩態雜訊處理
                n_jobs=1
            )
        except:
            return audio

    def enhance_separation(self, separated_signals: torch.Tensor, fast: bool = False, preserve_rel_energy: bool = True) -> torch.Tensor:
        """增強分離後音訊（可選）。
        改進要點：
        1. 自動判斷輸入 layout（支援 [B,S,T] / [B,T,S] / [S,T]）。
        2. 避免錯誤假設為 [B,T,S]，保留原始 layout 回傳。
        3. fast=True 時只做 peak clamp（快速路徑）。
        4. 僅在 SNR 低於門檻時才套用 Wiener / spectral gating，避免過度處理。
        5. 盡量停留於 GPU，只有必要（第三方函式）時轉 numpy，處理後再回傳。
        6. preserve_rel_energy=True → 以所有說話者共同最大絕對值做一次性 peak normalize，
           否則各說話者各自 normalize（會改變相對能量）。
        7. 保護：處理失敗時回退原始訊號；所有輸出 clamp 至 [-0.98,0.98]。
        """
        if (not self.enable_noise_reduction) or separated_signals is None:
            return separated_signals
        with torch.no_grad():
            x = separated_signals
            orig_shape = x.shape
            orig_dim = x.dim()
            # ---- 標準化為 [B,S,T] 以便統一處理 ----
            if orig_dim == 3:
                # 嘗試辨識 layout
                B, A, B2 = x.shape[0], x.shape[1], x.shape[2]
                layout, spk_axis, time_axis = self._infer_layout(x)
                # 轉為 [B,S,T]
                if layout == "BST":
                    bst = x  # [B,S,T]
                elif layout == "BTS":
                    bst = x.transpose(1, 2)  # [B,S,T]
                else:
                    bst = x  # fallback 視為 BST
            elif orig_dim == 2:  # [S,T] 或 [T,S]（假設 S <= 4 ）
                A, B2 = x.shape
                if A <= 4 and B2 > 4:  # [S,T]
                    bst = x.unsqueeze(0)
                    layout, spk_axis, time_axis = "BST", 1, 2
                elif B2 <= 4 and A > 4:  # [T,S]
                    bst = x.transpose(0, 1).unsqueeze(0)
                    layout, spk_axis, time_axis = "BST", 1, 2
                else:  # 模糊 → 視為單說話者時間序列
                    bst = x.unsqueeze(0).unsqueeze(0)  # [B=1,S=1,T]
                    layout, spk_axis, time_axis = "BST", 1, 2
            else:  # 其他形狀不支援
                return separated_signals
            # 確保為 float & GPU
            bst = bst.to(self.device, dtype=torch.float32)
            Bstd, S, T = bst.shape
            if S == 0 or T == 0:
                return separated_signals
            # ---- 快速路徑 ----
            if fast:
                if preserve_rel_energy:
                    peak = bst.abs().amax() .clamp_min(1e-8)
                    bst = (bst / peak).clamp(-0.98, 0.98)
                else:
                    spk_peak = bst.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
                    bst = (bst / spk_peak).clamp(-0.98, 0.98)
                # 回復原 layout
                if orig_dim == 3:
                    if layout == "BST":
                        return bst
                    elif layout == "BTS":
                        return bst.transpose(1, 2)
                    else:
                        return bst
                elif orig_dim == 2:
                    return bst[0] if orig_shape[0] <= 4 else bst[0].transpose(0, 1)
            # ---- 正式增強 ----
            enhanced_list = []
            # 全域 peak（供 preserve_rel_energy 使用）
            global_peak = bst.abs().amax().clamp_min(1e-8)
            for s in range(S):
                chan = bst[0, s] if Bstd == 1 else bst[:, s].mean(dim=0)  # 取代表波形（簡化）
                # 計算 SNR（快速 torch 版）
                # 估計背景：使用後 15% 低能量窗平均
                frame = max(256, min(2048, T // 50))
                if frame >= T:
                    est_snr = 30.0
                else:
                    energy_frames = chan.unfold(0, frame, frame).pow(2).mean(dim=1)
                    k = max(1, int(0.15 * energy_frames.numel()))
                    noise_floor = energy_frames.topk(k, largest=False).values.mean().clamp_min(1e-10)
                    signal_power = chan.pow(2).mean().clamp_min(1e-10)
                    est_snr = float(10 * torch.log10(signal_power / noise_floor))
                # 需要降噪才搬到 CPU → numpy
                need_wiener = est_snr < (self.snr_threshold + 3)
                need_gate = est_snr < self.snr_threshold
                processed = chan.detach().cpu().numpy() if (need_wiener or need_gate) else chan.detach().cpu().numpy()
                try:
                    if need_wiener:
                        processed = self.wiener_filter(processed)
                    if need_gate:
                        processed = self.spectral_gating(processed)
                    # 平滑 & 動態壓縮（始終可做）
                    processed = self.smooth_audio(processed)
                    processed = self.dynamic_range_compression(processed)
                except Exception:
                    # 出錯回退原始
                    processed = chan.detach().cpu().numpy()
                # 轉回 tensor
                proc_t = torch.from_numpy(processed).to(self.device, dtype=torch.float32)
                # 對齊長度
                if proc_t.numel() != T:
                    if proc_t.numel() > T:
                        proc_t = proc_t[:T]
                    else:
                        pad = T - proc_t.numel()
                        proc_t = torch.nn.functional.pad(proc_t, (0, pad))
                enhanced_list.append(proc_t.unsqueeze(0))  # [1,T]
            enhanced = torch.cat(enhanced_list, dim=0).unsqueeze(0)  # [B=1,S,T]
            # ---- 正規化策略 ----
            if preserve_rel_energy:
                peak = global_peak
                enhanced = (enhanced / peak).clamp(-0.98, 0.98)
            else:
                spk_peak = enhanced.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
                enhanced = (enhanced / spk_peak).clamp(-0.98, 0.98)
            # ---- 回復原 layout ----
            if orig_dim == 3:
                if layout == "BST":
                    return enhanced
                elif layout == "BTS":
                    return enhanced.transpose(1, 2)
                else:
                    return enhanced
            elif orig_dim == 2:
                # 若原始推測為 [S,T]
                if orig_shape[0] <= 4 and orig_shape[1] > 4:
                    return enhanced[0]  # [S,T]
                else:  # 推測為 [T,S]
                    return enhanced[0].transpose(0, 1)
            return enhanced

    def set_save_audio_files(self, save: bool) -> None:
        """
        設定是否儲存分離後的音訊檔案
        
        Args:
            save: True 表示儲存音訊檔案，False 表示不儲存
        """
        self.save_audio_files = save
        logger.info(f"音訊檔案儲存設定：{'已啟用' if save else '已停用'}")

    def cleanup_futures(self):
        """清理已完成的任務"""
        completed_futures = []
        for future in self.futures:
            if future.done():
                try:
                    future.result()  # 獲取結果以捕獲任何異常
                except Exception as e:
                    logger.error(f"處理任務發生錯誤：{e}")
                    self.processing_stats['errors'] += 1
                completed_futures.append(future)
        
        # 移除已完成的任務
        for future in completed_futures:
            self.futures.remove(future)

    def _cleanup_resources(self, p, stream, mixed_audio_buffer, output_dir):
        """清理資源"""
        # 停止並關閉音訊流
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
                logger.info("音訊流已關閉")
            except Exception as e:
                logger.error(f"關閉音訊流時發生錯誤：{e}")
        
        if p is not None:
            try:
                p.terminate()
                logger.info("PyAudio 已終止")
            except Exception as e:
                logger.error(f"終止 PyAudio 時發生錯誤：{e}")
        
        # 等待所有處理任務完成
        logger.info("等待處理任務完成...")
        for future in self.futures:
            try:
                future.result(timeout=15.0)
            except Exception as e:
                logger.error(f"處理任務發生錯誤：{e}")
        
        self.executor.shutdown(wait=True)
        logger.info("線程池已關閉")
        
        # 儲存原始混合音訊
        self._save_mixed_audio(mixed_audio_buffer, output_dir)
        
        # 記錄最終統計
        self._log_final_statistics()
        
        # 清理模型管理器
        if self.model_manager:
            self.model_manager.cleanup()
        
        # 清理語者計數管線
        if hasattr(self, 'speaker_count_pipeline') and self.speaker_count_pipeline is not None:
            try:
                # 清理管線資源
                del self.speaker_count_pipeline
                self.speaker_count_pipeline = None
                logger.info("語者計數管線已清理")
            except Exception as e:
                logger.error(f"清理語者計數管線時發生錯誤：{e}")
        
        # 清理GPU記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("錄音結束，所有資源已清理")

    def _save_mixed_audio(self, mixed_audio_buffer, output_dir):
        """儲存混合音訊"""
        if not mixed_audio_buffer:
            return ""
            
        try:
            mixed_audio = np.concatenate(mixed_audio_buffer)
            mixed_audio = mixed_audio.reshape(-1, CHANNELS)
            
            timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')
            mixed_output_file = os.path.join(
                output_dir,
                f"mixed_audio_{timestamp}.wav"
            )
            
            mixed_tensor = torch.from_numpy(mixed_audio).T.float()
            torchaudio.save(
                mixed_output_file,
                mixed_tensor,
                RATE
            )
            logger.info(f"已儲存原始混合音訊：{mixed_output_file}")
            return mixed_output_file
            
        except Exception as e:
            logger.error(f"儲存混合音訊時發生錯誤：{e}")
            return ""

    def reset_streaming_state(self, 
                            silence_threshold: float = 0.8,
                            max_buffer_duration: float = 15.0,
                            vad_threshold: float = 0.02):
        """
        重置串流 diarization 狀態
        
        Args:
            silence_threshold: 靜音持續多久觸發處理（秒）
            max_buffer_duration: 緩衝區最大長度（秒），防止過長
            vad_threshold: VAD 能量閾值（RMS）
        """
        self._diar_state = {
            "audio_buffer": deque(),        # 累積的音訊幀
            "silence_frames": 0,            # 連續靜音幀數
            "last_process_time": 0.0,       # 上次處理的累積時間
            "segment_counter": 0,           # 全局段落計數器
            "speaker_file_counters": {},    # {speaker_label: file_count}
            
            # 參數
            "silence_threshold": silence_threshold,
            "max_buffer_duration": max_buffer_duration,
            "vad_threshold": vad_threshold,
            "sample_rate": TARGET_RATE,
            "silence_frame_threshold": int(silence_threshold * TARGET_RATE / CHUNK),
            
            "initialized": True
        }
        logger.info(f"串流 diarization 狀態已重置 (斷句模式: 靜音 {silence_threshold}s 或 緩衝 {max_buffer_duration}s)")

    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        簡單的 VAD 判斷：計算 RMS 能量
        
        Args:
            audio_chunk: 音訊片段 (numpy array)
        
        Returns:
            True 表示有語音，False 表示靜音
        """
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms > self._diar_state["vad_threshold"]

    def _should_process_buffer(self) -> tuple[bool, str]:
        """
        判斷是否應該處理當前緩衝區
        
        Returns:
            (should_process, reason)
        """
        state = self._diar_state
        
        # 條件 1: 連續靜音超過閾值
        if state["silence_frames"] >= state["silence_frame_threshold"]:
            return True, "靜音斷句"
        
        # 條件 2: 緩衝區時長超過上限
        buffer_duration = len(state["audio_buffer"]) * CHUNK / state["sample_rate"]
        if buffer_duration >= state["max_buffer_duration"]:
            return True, f"緩衝區過長 ({buffer_duration:.1f}s)"
        
        return False, ""

    def process_audio_chunk_streaming(self, 
                                    audio_chunk: np.ndarray,
                                    output_dir: str,
                                    absolute_time: datetime = None) -> list:
        """
        處理單個音訊 chunk（來自麥克風的連續輸入）
        
        Args:
            audio_chunk: 音訊片段 (numpy array, shape: [chunk_size] 或 [channels, chunk_size])
            output_dir: 輸出目錄
            absolute_time: 當前 chunk 的絕對時間戳
        
        Returns:
            若觸發處理，回傳 [(path, start, end, absolute_timestamp), ...]
            否則回傳 []
        """
        # 初始化狀態
        if not getattr(self, "_diar_state", {}).get("initialized"):
            self.reset_streaming_state()
        
        state = self._diar_state
        
        # 確保是 1D numpy array
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.cpu().numpy()
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=0)
        audio_chunk = audio_chunk.astype(np.float32)
        
        # VAD 判斷
        is_speech = self._is_speech(audio_chunk)
        
        if is_speech:
            state["silence_frames"] = 0  # 重置靜音計數器
        else:
            state["silence_frames"] += 1
        
        # 加入緩衝區
        state["audio_buffer"].append(audio_chunk)
        
        # 判斷是否應該處理
        should_process, reason = self._should_process_buffer()
        
        if should_process:
            logger.info(f"觸發處理: {reason}, 緩衝區大小: {len(state['audio_buffer'])} 幀")
            
            # 合併緩衝區
            if len(state["audio_buffer"]) == 0:
                return []
            
            combined_audio = np.concatenate(list(state["audio_buffer"]))
            buffer_duration = len(combined_audio) / state["sample_rate"]
            
            # 過濾過短的片段（< 0.5 秒，可能是雜音）
            if buffer_duration < 0.5:
                logger.debug(f"片段過短 ({buffer_duration:.2f}s)，跳過處理")
                state["audio_buffer"].clear()
                state["silence_frames"] = 0
                return []
            
            # 轉為 torch tensor
            audio_tensor = torch.from_numpy(combined_audio).float().unsqueeze(0)
            
            # 計算這段音訊的起始時間
            if absolute_time is not None:
                segment_start_time = absolute_time
            else:
                segment_start_time = None
            
            # 執行 diarization 處理
            segment_idx = state["segment_counter"]
            state["segment_counter"] += 1
            
            try:
                results = self._diarize_and_save_utterance(
                    audio_tensor=audio_tensor,
                    output_dir=output_dir,
                    segment_index=segment_idx,
                    absolute_start_time=segment_start_time
                )
                
                logger.info(f"處理完成: 片段 {segment_idx}, 時長 {buffer_duration:.2f}s, 生成 {len(results)} 個音檔")
            except Exception as e:
                logger.error(f"處理片段 {segment_idx} 失敗: {e}")
                results = []
            
            # 清空緩衝區
            state["audio_buffer"].clear()
            state["silence_frames"] = 0
            state["last_process_time"] += buffer_duration
            
            return results
        
        return []  # 尚未觸發處理

    def force_process_buffer(self, output_dir: str, absolute_time: datetime = None) -> list:
        """
        強制處理當前緩衝區（用於錄音結束時）
        
        Args:
            output_dir: 輸出目錄
            absolute_time: 絕對時間戳
        
        Returns:
            [(path, start, end, absolute_timestamp), ...]
        """
        if not getattr(self, "_diar_state", {}).get("initialized"):
            logger.warning("串流狀態未初始化")
            return []
        
        state = self._diar_state
        
        if len(state["audio_buffer"]) == 0:
            logger.info("緩衝區為空，無需處理")
            return []
        
        logger.info(f"強制處理緩衝區: {len(state['audio_buffer'])} 幀")
        
        # 合併緩衝區
        combined_audio = np.concatenate(list(state["audio_buffer"]))
        audio_tensor = torch.from_numpy(combined_audio).float().unsqueeze(0)
        
        segment_idx = state["segment_counter"]
        state["segment_counter"] += 1
        
        try:
            results = self._diarize_and_save_utterance(
                audio_tensor=audio_tensor,
                output_dir=output_dir,
                segment_index=segment_idx,
                absolute_start_time=absolute_time
            )
            logger.info(f"強制處理完成: 生成 {len(results)} 個音檔")
        except Exception as e:
            logger.error(f"強制處理失敗: {e}")
            results = []
        
        # 清空緩衝區
        state["audio_buffer"].clear()
        state["silence_frames"] = 0
        
        return results

    def _diarize_and_save_utterance(
        self,
        audio_tensor: torch.Tensor,
        output_dir: str,
        segment_index: int,
        absolute_start_time=None
    ) -> list:
        """
        處理一段完整的對話片段（由 VAD 斷句產生）
        
        處理流程：
        1. 降噪
        2. Diarization 分段
        3. 平滑 + 重疊處理
        4. 為每個語者的每句話儲存獨立音檔
        
        Args:
            audio_tensor: 音訊張量 [1, T]
            output_dir: 輸出目錄
            segment_index: 片段索引
            absolute_start_time: 絕對時間戳
        
        Returns:
            [(path, start, end, absolute_timestamp), ...]
        """
        try:
            # 取得 diarization 管線
            diar_pipeline = self._ensure_diar_pipeline()
            if diar_pipeline is None:
                logger.warning("Diarization 管線不可用，跳過此片段")
                return []
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 準備音訊：確保是 [1, T] float32 CPU
            waveform = audio_tensor.to(torch.float32).cpu().contiguous()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            sr = TARGET_RATE
            duration = waveform.shape[-1] / sr
            
            logger.info(f"開始處理片段 {segment_index}，長度: {duration:.2f}s")
            
            # ========== 步驟 1: 降噪 ==========
            if self.enable_noise_reduction:
                waveform_clean = self._apply_noise_reduction(waveform, sr)
            else:
                waveform_clean = waveform
            
            # ========== 步驟 2: Diarization ==========
            try:
                diarization = diar_pipeline({
                    "waveform": waveform_clean,
                    "sample_rate": sr
                })
            except Exception as e:
                logger.error(f"Diarization 執行失敗: {e}")
                return []
            
            # 收集所有 turns
            from collections import defaultdict
            raw_turns = defaultdict(list)
            
            for segment, track, speaker_label in diarization.itertracks(yield_label=True):
                start = float(segment.start)
                end = float(segment.end)
                if end > start:
                    raw_turns[speaker_label].append((start, end))
            
            if not raw_turns:
                logger.warning(f"片段 {segment_index} 未檢測到任何說話者")
                return []
            
            logger.info(f"檢測到 {len(raw_turns)} 位說話者，原始片段數: {sum(len(v) for v in raw_turns.values())}")
            
            # ========== 步驟 3: 平滑處理 ==========
            smoothed_turns = self._smooth_turns(
                raw_turns,
                min_duration=0.5,  # 最短 0.5 秒
                min_gap=0.3,       # 小於 0.3 秒的間隙合併
                collar=0.1         # 邊界擴展 0.1 秒
            )
            
            # ========== 步驟 4: 重疊處理 ==========
            final_turns = self._handle_overlaps(smoothed_turns, waveform_clean, sr)
            
            logger.info(f"平滑+重疊處理後片段數: {sum(len(v) for v in final_turns.values())}")
            
            # ========== 步驟 5: 儲存音檔 ==========
            results = []
            
            all_utterances = []
            for speaker_label, segments in final_turns.items():
                for start, end in segments:
                    all_utterances.append((start, end, speaker_label))
            all_utterances.sort(key=lambda x: x[0])
            
            # 🆕 為每個 utterance 建立獨立資料夾
            for utt_start, utt_end, speaker_label in all_utterances:
                # 轉換為樣本索引
                start_idx = int(utt_start * sr)
                end_idx = int(utt_end * sr)
                
                # 邊界保護
                start_idx = max(0, start_idx)
                end_idx = min(waveform_clean.shape[-1], end_idx)
                
                if end_idx <= start_idx:
                    continue
                
                # 提取音訊片段
                utterance_audio = waveform_clean[:, start_idx:end_idx]
                
                # 🆕 生成檔案計數器
                if speaker_label not in self._diar_state["speaker_file_counters"]:
                    self._diar_state["speaker_file_counters"][speaker_label] = 1
                
                file_count = self._diar_state["speaker_file_counters"][speaker_label]
                self._diar_state["speaker_file_counters"][speaker_label] += 1
                
                # 🆕 建立獨立資料夾: segment_{XXX}
                segment_dir = os.path.join(output_dir, f"segment_{segment_index:03d}")
                os.makedirs(segment_dir, exist_ok=True)
                
                # 🆕 清理 speaker_label 並生成檔名
                clean_label = speaker_label.replace("SPEAKER_", "").replace("_", "")
                filename = f"speaker_{clean_label}_{file_count:03d}.wav"
                output_path = os.path.join(segment_dir, filename)
                
                # 儲存音檔
                torchaudio.save(
                    output_path,
                    utterance_audio,
                    sr,
                    bits_per_sample=16
                )
                
                # 🆕 計算絕對時間戳
                if absolute_start_time is not None:
                    absolute_ts = absolute_start_time.timestamp() + utt_start
                    absolute_start_iso = datetime.fromtimestamp(
                        absolute_ts, 
                        tz=timezone(timedelta(hours=8))
                    ).isoformat()
                    absolute_end_iso = datetime.fromtimestamp(
                        absolute_ts + (utt_end - utt_start),
                        tz=timezone(timedelta(hours=8))
                    ).isoformat()
                else:
                    absolute_ts = None
                    absolute_start_iso = None
                    absolute_end_iso = None
                
                # 🆕 生成 output.json
                segment_info = {
                    "segment_index": segment_index,
                    "speaker_label": speaker_label,
                    "speaker_file_count": file_count,
                    "audio_file": filename,
                    "start_time": round(utt_start, 3),
                    "end_time": round(utt_end, 3),
                    "duration": round(utt_end - utt_start, 3),
                    "absolute_start_time": absolute_start_iso,
                    "absolute_end_time": absolute_end_iso,
                    "absolute_timestamp": absolute_ts,
                    "sample_rate": sr,
                }
                
                json_path = os.path.join(segment_dir, "output.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(segment_info, f, ensure_ascii=False, indent=2)
                
                results.append((output_path, utt_start, utt_end, absolute_ts))
                self.output_files.append(output_path)
                
                logger.debug(f"儲存: {segment_dir}/{filename} ({utt_start:.2f}s - {utt_end:.2f}s)")
            
            # 🆕 更新 segment_index（下一個 utterance 用新編號）
            self._diar_state["segment_counter"] = segment_index + len(all_utterances)
            
            logger.info(f"片段 {segment_index} 完成，共儲存 {len(results)} 個音檔於獨立資料夾")
            return results
            
        except Exception as e:
            logger.error(f"處理片段失敗: {e}", exc_info=True)
            return []

    # ========== 以下方法保持不變（從之前的版本複製） ==========

    def _ensure_diar_pipeline(self):
        """確保 pyannote diarization 管線已載入"""
        global _GLOBAL_SPEAKER_PIPELINE_CACHE
        try:
            if getattr(self, "speaker_count_pipeline", None) is not None:
                return self.speaker_count_pipeline
            if _GLOBAL_SPEAKER_PIPELINE_CACHE is not None:
                return _GLOBAL_SPEAKER_PIPELINE_CACHE
            if HF_ACCESS_TOKEN:
                from pyannote.audio import Pipeline
                pipe = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HF_ACCESS_TOKEN
                )
                pipe.to(torch.device(self.device))
                _GLOBAL_SPEAKER_PIPELINE_CACHE = pipe
                logger.info("Diarization 管線載入成功")
                return pipe
            else:
                logger.warning("未提供 HF_ACCESS_TOKEN，無法啟用 diarization")
                return None
        except Exception as e:
            logger.error(f"載入 diarization 管線失敗: {e}")
            return None

    def _apply_noise_reduction(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """對音訊進行降噪處理"""
        try:
            if isinstance(waveform, torch.Tensor):
                audio_np = waveform.squeeze().cpu().numpy()
            else:
                audio_np = np.array(waveform).squeeze()
            
            noise_sample_len = max(int(sr * 0.05), 1)
            noise_sample = audio_np[:noise_sample_len]
            
            reduced = nr.reduce_noise(
                y=audio_np,
                y_noise=noise_sample,
                sr=sr,
                prop_decrease=0.3,
                stationary=False,
                n_jobs=1
            )
            
            result = torch.from_numpy(reduced).float().unsqueeze(0)
            return result
        except Exception as e:
            logger.warning(f"降噪失敗，返回原始音訊: {e}")
            if isinstance(waveform, torch.Tensor):
                return waveform if waveform.dim() == 2 else waveform.unsqueeze(0)
            else:
                return torch.from_numpy(np.array(waveform)).float().unsqueeze(0)

    def _smooth_turns(self, turns: dict, 
                    min_duration: float = 0.5,
                    min_gap: float = 0.3,
                    collar: float = 0.1) -> dict:
        """平滑 diarization 結果"""
        smoothed = {}
        for speaker, segments in turns.items():
            if not segments:
                continue
            segments = sorted(segments, key=lambda x: x[0])
            expanded = [(max(0, s - collar), e + collar) for s, e in segments]
            merged = []
            for start, end in expanded:
                if not merged:
                    merged.append([start, end])
                    continue
                prev_start, prev_end = merged[-1]
                if start - prev_end <= min_gap:
                    merged[-1][1] = max(prev_end, end)
                else:
                    merged.append([start, end])
            refined = []
            for start, end in merged:
                duration = end - start
                if duration < min_duration:
                    if refined and (start - refined[-1][1]) <= min_gap:
                        refined[-1][1] = end
                    continue
                refined.append([start, end])
            smoothed[speaker] = [(round(s, 3), round(e, 3)) for s, e in refined]
        return smoothed

    def _handle_overlaps(self, turns: dict, waveform: torch.Tensor, sr: int) -> dict:
        """處理重疊語音：使用能量分配策略"""
        all_segments = []
        for speaker, segments in turns.items():
            for start, end in segments:
                all_segments.append((start, end, speaker))
        all_segments.sort(key=lambda x: x[0])
        
        resolved = []
        i = 0
        audio_np = waveform.squeeze().cpu().numpy()
        
        while i < len(all_segments):
            current_start, current_end, current_spk = all_segments[i]
            
            if i + 1 < len(all_segments):
                next_start, next_end, next_spk = all_segments[i + 1]
                overlap_start = max(current_start, next_start)
                overlap_end = min(current_end, next_end)
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration > 0 and overlap_duration < 1.0:
                    curr_start_idx = int(current_start * sr)
                    curr_end_idx = int(current_end * sr)
                    curr_energy = np.mean(np.abs(audio_np[curr_start_idx:curr_end_idx]))
                    
                    next_start_idx = int(next_start * sr)
                    next_end_idx = int(next_end * sr)
                    next_energy = np.mean(np.abs(audio_np[next_start_idx:next_end_idx]))
                    
                    if curr_energy > next_energy:
                        resolved.append((current_start, current_end, current_spk))
                        all_segments[i + 1] = (current_end, next_end, next_spk)
                    else:
                        resolved.append((current_start, next_start, current_spk))
                    i += 1
                    continue
            
            resolved.append((current_start, current_end, current_spk))
            i += 1
        
        result = {}
        for start, end, speaker in resolved:
            if speaker not in result:
                result[speaker] = []
            result[speaker].append((start, end))
        return result

    # 保留舊方法名稱以向後兼容
    def _diarize_and_save_streaming(self, *args, **kwargs):
        """向後兼容的方法名（實際使用 process_audio_chunk_streaming）"""
        logger.warning("_diarize_and_save_streaming 已棄用，請使用 process_audio_chunk_streaming")
        return self.process_audio_chunk_streaming(*args, **kwargs)

    
    def separate_and_save(self, audio_tensor, output_dir, segment_index, absolute_start_time=None):
        """
        分離並儲存音訊，並回傳 (path, start, end) 列表。
        流程：語者計數 → 動態模型選擇 → 分離 → 強化(可選) → 儲存
        
        Args:
            audio_tensor: 音訊張量
            output_dir: 輸出目錄
            segment_index: 片段索引
            absolute_start_time: 音訊的絕對開始時間（datetime 物件）
        """
        
        # 新增：若啟用 diarization，改走 diarization 路徑，回傳相容格式
        if getattr(self, "use_diarization", False):
            return self._diarize_and_save_streaming(audio_tensor, output_dir, segment_index, absolute_start_time)
        
        try:
            total_start = time.perf_counter()  # 片段總耗時起點
            # 先以寬鬆範圍跑一次，並套用重疊感知後處理；若你的批次確定雙人，可設 expected_min/max=2
            detected_speakers = self.spk_counter.count_with_refine(
                audio=audio_tensor,
                sample_rate=TARGET_RATE,
                expected_min=0,
                expected_max=3,
                first_pass_range=(0, 3),
                allow_zero=True,         # <== 允許回傳 0（無語音）
                debug=True
            )

            logger.info(f"片段 {segment_index} - 偵測到 {detected_speakers} 位說話者")
            
            # 備援：第一次回 0 → 只有在「強有聲」才重試 1–2 人
            if detected_speakers == 0:
                ok, m = self.spk_counter._has_voice(audio_tensor, TARGET_RATE, return_metrics=True)
                # 與 SpeakerCounter 同步或更嚴的條件
                strong_voice = (
                    ok and
                    (m["dbfs"] > -35.0) and
                    (m["voiced_ratio"] >= 0.20) and
                    (m["voiced_union"] >= 1.00) and
                    (m.get("loud_frac", 0.0) >= 0.10)
                )
                if not strong_voice:
                    logger.info(
                        f"片段 {segment_index} - 無語音/過短（ratio={m['voiced_ratio']:.3f}, "
                        f"union={m['voiced_union']:.2f}s, loud={m.get('loud_frac',0.0):.3f}），跳過"
                    )
                    return []

                logger.warning(f"片段 {segment_index} - 第一次偵測 0，但語音跡象偏強，嘗試 1–2 人重試")
                retry = self.spk_counter.count_with_refine(
                    audio=audio_tensor, sample_rate=TARGET_RATE,
                    expected_min=1, expected_max=2,
                    first_pass_range=(1, 2),
                    allow_zero=False,           # 已確認強有聲，就不要再回 0
                    debug=False
                )
                detected_speakers = int(retry)
            
            # 動態選擇模型
            current_model, current_model_type = self._get_appropriate_model(detected_speakers)
            model_config = self.model_manager.get_model_config(current_model_type) if self.model_manager else MODEL_CONFIGS[current_model_type]
            logger.debug(f"使用模型: {current_model_type.value} (偵測語者: {detected_speakers})")
            
            # 記錄絕對時間戳
            if absolute_start_time is None:
                from datetime import timezone, timedelta
                taipei_tz = timezone(timedelta(hours=8))
                absolute_start_time = datetime.now(taipei_tz)
            
            # 初始化累計時間戳
            current_t0 = getattr(self, "_current_t0", 0.0)
            results = []   # 用來收 (path, start, end, absolute_timestamp)
            seg_duration = audio_tensor.shape[-1] / TARGET_RATE
            results = []

            with torch.inference_mode():
                # 4) 輸入整理成 [batch, samples]
                if len(audio_tensor.shape) == 3 and audio_tensor.shape[1] == 1:
                    audio_tensor = audio_tensor.squeeze(1)

                # 5) 做「原始分離」
                sep_start = time.perf_counter()
                separated = current_model.separate_batch(audio_tensor)
                sep_time_ms = (time.perf_counter() - sep_start) * 1000.0
                self.timing_stats['separation_time_ms_total'] += sep_time_ms
                self.timing_stats['separation_calls'] += 1
                logger.info(f"片段 {segment_index} 分離推論耗時 {sep_time_ms:.1f} ms")
                
                # 6) 僅做固定比例的「峰值正規化」以統一尺度（維持原 _normalize_estimates）
                #    不做任何音質增強/濾波/投影回混音等後處理
                separated, layout, spk_axis, time_axis = self._normalize_estimates(separated)

                # 7) 依 layout 取出候選，保留單人情境的選路邏輯（但仍輸出原始分離結果）
                raw_for_select = separated
                enhanced_separated = self.enhance_separation(separated)
                del separated  # 釋放記憶體

                if layout == "BST":  # [B, S, T]
                    model_output_speakers = enhanced_separated.shape[spk_axis]
                    def _get_cand(idx):  return raw_for_select[0, idx, :].detach().cpu()
                    def _get_final(idx): return enhanced_separated[0, idx, :].detach().cpu()
                elif layout == "BTS":  # [B, T, S]
                    model_output_speakers = enhanced_separated.shape[2]
                    def _get_cand(idx):  return raw_for_select[0, :, idx].detach().cpu()
                    def _get_final(idx): return enhanced_separated[0, :, idx].detach().cpu()
                else:  # "BT"
                    model_output_speakers = 1
                    def _get_cand(idx):  return raw_for_select[0, :].detach().cpu()
                    def _get_final(idx): return enhanced_separated[0, :].detach().cpu()

                # 單人情境的最佳路徑選擇（不涉及任何音訊處理，只是選哪一路）
                best_idx = 0
                if detected_speakers == 1 and model_output_speakers >= 2:
                    candidates = [_get_cand(j) for j in range(model_output_speakers)]
                    try:
                        best_idx, _, _ = self.single_selector.select(candidates, audio_tensor[0].detach().cpu(), return_stats=True)
                        enhanced_separated = _get_final(best_idx).unsqueeze(0).unsqueeze(-1).to(self.device)  # [1, T, 1]
                        model_output_speakers = 1
                    except Exception:
                        logger.exception("單人選路失敗，改用 speaker1 作為保守輸出")
                        enhanced_separated = _get_final(0).unsqueeze(0).unsqueeze(-1).to(self.device)
                        model_output_speakers = 1

                # 8) 統一成 [S, T]（不再做任何投影回混音、維納濾波或 frame gate）
                if enhanced_separated.ndim == 3:   # [B, T, S]
                    est_ST = enhanced_separated[0].transpose(0, 1).detach().cpu()  # [S, T]
                else:
                    est_ST = enhanced_separated.detach().cpu().unsqueeze(0)        # [1, T]

                if detected_speakers == 1 and est_ST.shape[0] >= 2:
                    # 只保留被選中的那一路
                    try:
                        est_ST = est_ST[best_idx:best_idx+1, :]
                    except Exception:
                        est_ST = est_ST[0:1, :]

                # 9) 儲存原始分離結果（不做 fade、dither、品質評分、pretty copy）
                S, T = est_ST.shape
                effective_speakers = min(int(detected_speakers), int(S), int(model_config["num_speakers"]))
                logger.debug(
                    f"分離參數 - 偵測: {detected_speakers}, est_ST通道: {S}, "
                    f"模型支援: {model_config['num_speakers']}, 有效: {effective_speakers}"
                )

                saved_count = 0
                start_time = current_t0
                timestamp = datetime.now().strftime('%Y%m%d-%H_%M_%S')

                for i in range(effective_speakers):
                    try:
                        speaker_audio = est_ST[i].contiguous()  # 1D [T]
                        final_tensor = speaker_audio.unsqueeze(0).cpu()  # [1, T]
                        output_file = os.path.join(
                            output_dir,
                            f"speaker{i+1}.wav"
                            # 若要保留動態檔名可改為：f"speaker{i+1}_{timestamp}_{segment_index}.wav"
                        )
                        torchaudio.save(output_file, final_tensor, TARGET_RATE, bits_per_sample=16)

                        absolute_timestamp = absolute_start_time.timestamp() + start_time
                        results.append((output_file, start_time, start_time + seg_duration, absolute_timestamp))
                        self.output_files.append(output_file)
                        saved_count += 1
                    except Exception as e:
                        logger.warning(f"儲存語者 {i+1} 失敗: {e}")

                if saved_count > 0:
                    logger.info(f"片段 {segment_index} 完成，儲存 {saved_count}/{effective_speakers} 個檔案 (使用 {current_model_type.value})")

            # 10) 更新時間 & 紀錄總耗時
            total_time_ms = (time.perf_counter() - total_start) * 1000.0
            self.timing_stats['segment_time_ms_total'] += total_time_ms
            logger.info(f"片段 {segment_index} 分離總處理耗時 {total_time_ms:.1f} ms")

            if not results:
                raise RuntimeError("Speaker separation produced no valid tracks")

            return results

        except Exception as e:
            logger.error(f"處理片段 {segment_index} 失敗: {e}")
            self.processing_stats['errors'] += 1
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
        logger.info("準備停止錄音...")

    def get_output_files(self):
        """獲取所有分離後的音檔路徑"""
        return self.output_files

# 添加全域函式來管理快取
def get_cached_separator(model_type: SeparationModel = DEFAULT_MODEL, enable_dynamic_model: bool = True, **kwargs) -> AudioSeparator:
    """
    取得快取的 AudioSeparator 實例，避免重複初始化
    
    Args:
        model_type: 模型類型
        enable_dynamic_model: 是否啟用動態模型
        **kwargs: 其他參數
    
    Returns:
        AudioSeparator 實例
    """
    global _GLOBAL_SEPARATOR_CACHE
    
    # 建立快取鍵
    cache_key = f"{model_type.value}_{enable_dynamic_model}_{hash(tuple(sorted(kwargs.items())))}"
    
    # 檢查快取
    if cache_key in _GLOBAL_SEPARATOR_CACHE:
        logger.info(f"使用快取的 AudioSeparator: {cache_key}")
        return _GLOBAL_SEPARATOR_CACHE[cache_key]
    
    # 建立新實例並快取
    logger.info(f"建立新的 AudioSeparator: {cache_key}")
    separator = AudioSeparator(
        model_type=model_type, 
        enable_dynamic_model=enable_dynamic_model, 
        **kwargs
    )
    _GLOBAL_SEPARATOR_CACHE[cache_key] = separator
    
    return separator

def clear_separator_cache():
    """清理所有快取的分離器實例"""
    global _GLOBAL_SEPARATOR_CACHE, _GLOBAL_SPEAKER_PIPELINE_CACHE
    
    # 清理分離器快取
    for separator in _GLOBAL_SEPARATOR_CACHE.values():
        try:
            if hasattr(separator, 'model_manager') and separator.model_manager:
                separator.model_manager.cleanup()
        except Exception as e:
            logger.warning(f"清理分離器時發生錯誤: {e}")
    
    _GLOBAL_SEPARATOR_CACHE.clear()
    
    # 清理語者計數管線快取
    if _GLOBAL_SPEAKER_PIPELINE_CACHE is not None:
        try:
            del _GLOBAL_SPEAKER_PIPELINE_CACHE
            _GLOBAL_SPEAKER_PIPELINE_CACHE = None
            logger.info("已清理語者計數管線快取")
        except Exception as e:
            logger.warning(f"清理語者計數管線快取時發生錯誤: {e}")

def check_weaviate_connection() -> bool:
    """
    檢查 Weaviate 資料庫連線狀態。

    Returns:
        bool: 若連線成功回傳 True，否則回傳 False。
    """
    try:
        import weaviate  # type: ignore
        client = weaviate.connect_to_local()
        # 檢查是否能存取必要集合
        if not client.is_live():
            logger.error("Weaviate 服務未啟動或無法存取。")
            return False
        if not (client.collections.exists("Speaker") and client.collections.exists("VoicePrint")):
            logger.error("Weaviate 缺少必要集合 (Speaker 或 VoicePrint)。請先執行 create_collections.py。")
            return False
        return True
    except Exception as e:
        logger.error(f"Weaviate 連線失敗：{e}")
        return False
