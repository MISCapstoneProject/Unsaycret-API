"""
===============================================================================
語者識別引擎 (Speaker Identification Engine) V2
===============================================================================

版本：v5.2.1 - V2資料庫版本，有pyannote模型  
作者：CYouuu
最後更新：2025-08-13

⚠️ 重要變更 ⚠️
本版本已升級為V2資料庫結構，與V1版本不相容！
- Speaker: 新增speaker_id (INT)、full_name、nickname、gender等欄位  
- VoicePrint: 移除冗餘的voiceprint_id，直接使用Weaviate UUID、sample_count、quality_score等欄位
- 時間欄位重命名: create_time -> created_at, updated_time -> updated_at

功能摘要：
-----------
本模組實現了基於深度學習的語者識別功能，能夠從音訊檔案中提取語者特徵向量，
並與資料庫中的已知語者進行比對，實現語者身份識別與聲紋更新。主要優點包括：

 1. 支援即時語者識別與資料庫更新
 2. 使用單例模式避免重複初始化模型和資料庫連線
 3. 支援多種音訊格式及取樣率自動適配
 4. 整合 Weaviate 向量資料庫實現高效語者比對
 5. 提供彈性的閾值設定，可自訂語者匹配策略

技術架構：
-----------
 - 語者嵌入模型: SpeechBrain ECAPA-TDNN 模型
 - 向量資料庫: Weaviate
 - 取樣率自適應: 自動處理 8kHz/16kHz/44.1kHz 等常見取樣率
 - 向量更新策略: 加權移動平均，保持聲紋向量穩定性

更新歷程：
-----------
 - v5.1.2 (2025-05-06): 新增多聲紋映射功能、支援外部傳入時間戳記、優化使用體驗

使用方式：
-----------
 1. 單檔案辨識:
    ```python
    identifier = SpeakerIdentifier()
    identifier.process_audio_file("path/to/audio.wav")
    ```

 2. 整個目錄檔案辨識:
    ```python
    identifier = SpeakerIdentifier()
    identifier.process_audio_directory("path/to/directory")
    ```

 3. 單個音訊流辨識:
    ```python
    identifier = SpeakerIdentifier()
    identifier.process_audio_stream(stream)
    ```

 4. 添加音檔到指定語者:
    ```python
    identifier = SpeakerIdentifier()
    identifier.add_voiceprint_to_speaker("path/to/audio.wav", "speaker_uuid")
    ```

 5. 使用speaker_system_v2.py進行語者識別模組呼叫

閾值參數設定：
-----------
 - THRESHOLD_LOW = 0.09: 過於相似，不更新向量
 - THRESHOLD_UPDATE = 0.27: 下:更新聲紋向量，上:新增一筆聲紋到語者
 - THRESHOLD_NEW = 0.38: 超過此值視為新語者

前置需求：
-----------
 - Python 3.9+
 - SpeechBrain
 - Weaviate 向量資料庫 (需通過 Docker 啟動)
 - NumPy, PyTorch, SoundFile 等相關處理套件

注意事項：
-----------
 - 使用前請確保 Weaviate 已啟動並初始化必要集合
 - 建議處理 16kHz 取樣率的音檔以獲得最佳識別效果
 - 對於批量處理，可調整閾值以符合不同應用場景

詳細資訊：
-----------
請參考專案文件: https://github.com/LCY000/ProjectStudy_SpeechRecognition

===============================================================================
"""

import os
import re
import sys
import uuid
import numpy as np
import torch
import soundfile as sf
from scipy.spatial.distance import cosine
from scipy.signal import resample_poly
import warnings
import logging
from datetime import datetime, timedelta, timezone
from typing import Tuple, List, Dict, Optional, Union, Any
import weaviate  # type: ignore
from weaviate.classes.query import MetadataQuery, Filter # type: ignore
from weaviate.classes.query import QueryReference # type: ignore
from contextvars import ContextVar
from itertools import count
from utils.path_utils import format_process_prefix

# 控制輸出的全局變數
_ENABLE_OUTPUT =True  # 預設為 True，即輸出詳細訊息
# 用於標記每次處理流程的前綴
_current_process_prefix: ContextVar[str] = ContextVar("current_process_prefix", default="")
_process_counter = count(1)

# 保存原始 print 函數的引用
original_print = print

# 輸出控制函數 - 更新為使用日誌系統
def _print(*args, **kwargs) -> None:
    """
    受控輸出函數，使用日誌系統輸出
    
    Args:
        *args: print 函數的位置參數
        **kwargs: print 函數的關鍵字參數
    """
    # 導入 logger 可能在函數第一次調用時尚未定義
    # 因此在這裡做一個檢查和默認值處理
    global logger
    if 'logger' not in globals() or logger is None:
        logger = get_logger(__name__)
        
    if _ENABLE_OUTPUT:
        # 將多個參數轉換為單個字符串
        message = " ".join(str(arg) for arg in args)
        prefix = _current_process_prefix.get()
        if prefix:
            message = f"{prefix} {message}"
        logger.info(message)
        # 移除重複的 print 輸出，只使用 logger

# 設置輸出開關的函數
def set_output_enabled(enable: bool) -> None:
    """
    設置是否啟用模組的輸出
    
    Args:
        enable: True 表示啟用輸出，False 表示禁用輸出
    """
    global _ENABLE_OUTPUT
    old_value = _ENABLE_OUTPUT
    _ENABLE_OUTPUT = enable
    
    if enable and not old_value:
        logger.info("已啟用 main_identify_v5 模組的輸出")
    elif not enable and old_value:
        logger.info("已禁用 main_identify_v5 模組的輸出")

# 替換原始 print 函數，以實現控制輸出
print = _print  # 替換全局 print 函數，使模組中的所有 print 調用都經過控制

# 設定 httpx 的日誌層級為 WARNING 或更高，以關閉 INFO 層級的 HTTP 請求日誌
logging.getLogger("httpx").setLevel(logging.WARNING)

# 新增時區處理函數
def format_rfc3339(dt: Optional[datetime] = None) -> str:
    """
    將日期時間格式化為符合 RFC3339 標準的字串，包含時區信息
    
    Args:
        dt: 要格式化的 datetime 對象，若為 None 則使用當前時間
        
    Returns:
        str: RFC3339 格式的日期時間字串
    """
    taipei_tz = timezone(timedelta(hours=8))  # 台北是 UTC+8

    if dt is None:
        dt = datetime.now(taipei_tz)
    elif dt.tzinfo is None:
        # 若沒有時區信息，則假設為台北時區
        dt = dt.replace(tzinfo=taipei_tz)
    
    # 格式化為 RFC3339 格式
    return dt.isoformat()

# 隱藏多餘的警告與日誌
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)

# 導入日誌模組
from utils.logger import get_logger
from utils.env_config import WEAVIATE_HOST, WEAVIATE_PORT, get_model_save_dir, HF_ACCESS_TOKEN
from utils.constants import (
    THRESHOLD_LOW, THRESHOLD_UPDATE, THRESHOLD_NEW, 
    SPEECHBRAIN_SPEAKER_MODEL, PYANNOTE_SPEAKER_MODEL, AUDIO_TARGET_RATE,
    ENABLE_AS_NORM, AS_NORM_COHORT_SIZE, AS_NORM_TOP_K, AS_NORM_ALPHA,
    ENABLE_T_NORM, ENABLE_Z_NORM, ENABLE_S_NORM
)

# 創建模組專屬日誌器
logger = get_logger(__name__)

# 載入 SpeechBrain 語音辨識模型
from speechbrain.inference import SpeakerRecognition

# 導入 AS-Norm 處理器
from modules.database.cohort_manager import ASNormProcessor

# 全域參數設定（從環境配置載入）
DEFAULT_SPEAKER_NAME = "未命名語者"  # 預設的語者名稱
DEFAULT_FULL_NAME_PREFIX = "n"  # V2版本：預設full_name前綴


class AudioProcessor:
    """音訊處理類別，負責音訊處理和嵌入向量提取"""

    def __init__(self) -> None:
        """
        初始化模型
        想切換模型時，直接改下面的 self.model_type
        可選值: "speechbrain" 或 "pyannote"
        """
        # ====== 這裡改模型類型 ======
        self.model_type = "pyannote"
        # =========================

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type == "speechbrain":
            from speechbrain.inference import SpeakerRecognition
            self.model = SpeakerRecognition.from_hparams(
                source=SPEECHBRAIN_SPEAKER_MODEL,
                savedir=get_model_save_dir("speechbrain_recognition")
            )
            logger.info("已載入 SpeechBrain ECAPA-TDNN 模型")

        elif self.model_type == "pyannote":
            from pyannote.audio import Inference
            from pyannote.core import Segment
            # ⚠️ .env 檔案中必須設定 HF_ACCESS_TOKEN
            hf_token = HF_ACCESS_TOKEN
            
            # 使用整個音頻模式
            self.model = Inference(
                PYANNOTE_SPEAKER_MODEL, 
                window="whole", 
                use_auth_token=hf_token
            )
            logger.info(f"已載入 pyannote/embedding 模型 ")
            
            self.Segment = Segment  # 保存 Segment 類別以便後續使用

        else:
            raise ValueError(f"不支援的模型類型: {self.model_type}")

    def resample_audio(self, signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """使用 scipy 進行高品質重新採樣"""
        return resample_poly(signal, target_sr, orig_sr)

    def extract_embedding_from_stream(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """從音訊流提取嵌入向量"""
        try:
            if not isinstance(signal, np.ndarray):
                signal = np.array(signal)
            if signal.ndim > 1:
                signal = signal.mean(axis=1)

            target_sr = AUDIO_TARGET_RATE
            if sr != target_sr:
                signal = self.resample_audio(signal, sr, target_sr)

            signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(self.device)

            if self.model_type == "speechbrain":
                embedding = self.model.encode_batch(signal_tensor).squeeze().cpu().numpy()

            elif self.model_type == "pyannote":
                # pyannote 的 Inference 需要從文件中讀取，所以我們需要創建臨時文件
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    # 將信號寫入臨時文件
                    sf.write(temp_path, signal, target_sr)
                
                try:
                    # 整個音頻模式：使用 crop 方法
                    duration = len(signal) / target_sr
                    segment = self.Segment(0, duration)
                    embedding = self.model.crop(temp_path, segment)
                    embedding = embedding.squeeze()  # 移除第一維
                    embedding = embedding / np.linalg.norm(embedding)  # 正規化
                finally:
                    # 清理臨時文件
                    import os
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

            return embedding

        except Exception as e:
            logger.error(f"提取嵌入向量時發生錯誤: {e}")
            raise

    
    def resample_audio(self, signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        使用 scipy 進行高品質重新採樣
        
        Args:
            signal: 原始音訊信號
            orig_sr: 原始取樣率
            target_sr: 目標取樣率
            
        Returns:
            np.ndarray: 重新採樣後的音訊信號
        """
        return resample_poly(signal, target_sr, orig_sr)
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        提取音檔的嵌入向量，根據音檔取樣率智能處理

        Args:
            audio_path: 音檔路徑

        Returns:
            np.ndarray: 音檔的嵌入向量

        處理流程:
            1. 若音檔為 16kHz，則直接使用
            2. 若音檔為 8kHz，則直接升頻到 16kHz
            3. 若音檔取樣率高於 16kHz，則降頻到 16kHz
            4. 其他取樣率，則重新採樣到 16kHz
        """
        try:
            # 對於 pyannote 模型，直接使用文件路徑更高效
            if self.model_type == "pyannote":
                # 獲取音頻文件信息
                signal, sr = sf.read(audio_path)
                
                # 處理立體聲轉單聲道並重採樣（如果需要）
                if signal.ndim > 1:
                    signal = signal.mean(axis=1)
                
                target_sr = AUDIO_TARGET_RATE
                if sr != target_sr:
                    signal = self.resample_audio(signal, sr, target_sr)
                    
                    # 創建重採樣後的臨時文件
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                        sf.write(temp_path, signal, target_sr)
                    
                    try:
                        # 整個音頻模式：使用 crop 方法
                        duration = len(signal) / target_sr
                        segment = self.Segment(0, duration)
                        embedding = self.model.crop(temp_path, segment)
                        embedding = embedding.squeeze()
                        embedding = embedding / np.linalg.norm(embedding)
                    finally:
                        # 清理臨時文件
                        import os
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                else:
                    # 如果取樣率已經正確，直接使用原文件
                    # 整個音頻模式：使用 crop 方法
                    duration = len(signal) / sr
                    segment = self.Segment(0, duration)
                    embedding = self.model.crop(audio_path, segment)
                    embedding = embedding.squeeze()
                    embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
            
            else:
                # 對於其他模型（如 speechbrain），使用原有的流程
                signal, sr = sf.read(audio_path)

                # 處理立體聲轉單聲道
                if signal.ndim > 1:
                    signal = signal.mean(axis=1)

                # 使用新的 stream 方法處理核心邏輯
                return self.extract_embedding_from_stream(signal, sr)

        except Exception as e:
            logger.error(f"從檔案提取嵌入向量時發生錯誤: {e}")
            raise


# ==================== AS-Norm 工具函數 ====================

def apply_as_norm_to_distances(test_embedding: np.ndarray, 
                              distance_results: List[Tuple[str, str, float, int]],
                              as_norm_processor) -> List[Tuple[str, str, float, int]]:
    """
    對距離結果應用 AS-Norm 正規化
    
    這是一個便利函數，用於批量處理多個語者的距離計算
    
    Args:
        test_embedding: 測試音訊的嵌入向量
        distance_results: 原始距離結果列表 [(voice_print_id, speaker_name, distance, update_count)]
        as_norm_processor: AS-Norm 處理器實例
        
    Returns:
        List[Tuple[str, str, float, int]]: 正規化後的距離結果
    """
    if not ENABLE_AS_NORM or not distance_results:
        return distance_results
        
    normalized_results = []
    
    for voice_print_id, speaker_name, original_distance, update_count in distance_results:
        try:
            # 獲取目標語者的嵌入向量
            voice_print_collection = as_norm_processor.client.collections.get("VoicePrint")
            target_obj = voice_print_collection.query.fetch_object_by_id(
                uuid=voice_print_id,
                include_vector=True
            )
            
            if target_obj and target_obj.vector:
                # 處理 named vector
                vec_dict = target_obj.vector
                raw_vec = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict
                target_embedding = np.array(raw_vec, dtype=float)
                
                # 應用 AS-Norm
                normalized_distance = as_norm_processor.apply_as_norm(
                    test_embedding, target_embedding, speaker_name
                )
                
                normalized_results.append((voice_print_id, speaker_name, normalized_distance, update_count))
            else:
                # 無法獲取嵌入向量時，保持原始距離
                normalized_results.append((voice_print_id, speaker_name, original_distance, update_count))
                
        except Exception as e:
            logger.warning(f"對語者 {speaker_name} 應用 AS-Norm 時發生錯誤: {e}")
            # 發生錯誤時保持原始距離
            normalized_results.append((voice_print_id, speaker_name, original_distance, update_count))
            
    return normalized_results


# ==================== AS-Norm 功能結束 ====================


class WeaviateRepository:
    """Weaviate 資料存取庫類別，負責與 Weaviate V2 資料庫的交互"""
    
    def __init__(self) -> None:
        """初始化 Weaviate 連接（V2版本）"""
        try:
            self.client = weaviate.connect_to_local()
            logger.info("成功連接到 Weaviate V2 資料庫！")
            
            # 檢查必要的V2集合是否存在
            if not self.client.collections.exists("VoicePrint") or not self.client.collections.exists("Speaker"):
                logger.warning("警告：Weaviate 中缺少必要的V2集合 (VoicePrint / Speaker)!")
                logger.info("請先運行 modules/database/init_v2_collections.py 建立所需的V2集合")
                logger.info("正在嘗試自動初始化V2集合...")
                
                # 嘗試自動初始化V2集合
                try:
                    from modules.database.init_v2_collections import ensure_weaviate_collections
                    if ensure_weaviate_collections():
                        logger.info("✅ 已自動初始化V2集合！")
                    else:
                        logger.error("❌ 自動初始化V2集合失敗！")
                        raise RuntimeError("無法初始化V2集合")
                except ImportError:
                    logger.error("無法導入V2集合初始化模組")
                    logger.info("使用命令 'python -m modules.database.init_v2_collections' 手動初始化V2集合")
                    raise
        
        except Exception as e:
            logger.error(f"無法連接到 Weaviate V2 資料庫：{e}")
            logger.info("請確認：")
            logger.info("1. Docker 服務是否正在運行")
            logger.info("2. Weaviate 容器是否已經啟動")
            logger.info("3. weaviate_study/docker-compose.yml 中的配置是否正確")
            logger.info("使用命令 'docker-compose -f weaviate_study/docker-compose.yml up -d' 啟動 Weaviate")
            raise
    
    def compare_embedding(self, new_embedding: np.ndarray, as_norm_processor=None) -> Tuple[Optional[str], Optional[str], float, List[Tuple[str, str, float, int]]]:
        """
        比較新的嵌入向量與資料庫中所有現有嵌入向量的相似度
        
        Args:
            new_embedding: 新的嵌入向量
            as_norm_processor: AS-Norm 處理器，可選
            
        Returns:
            tuple: (最佳匹配ID, 最佳匹配語者名稱, 最小距離, 所有距離列表)
        """
        try:
            voice_print_collection = self.client.collections.get("VoicePrint")
            
            # 計算新向量與數據庫中所有向量的距離
            results = voice_print_collection.query.near_vector(
                near_vector=new_embedding.tolist(),
                limit=3,  # 測試! 返回前 3 個最相似的結果
                return_properties=["speaker_name", "update_count", "sample_count", "created_at", "updated_at"],  # V2屬性
                return_metadata=MetadataQuery(distance=True)
            )
            
            # 如果沒有找到任何結果
            if not results.objects:
                logger.info("資料庫中尚無任何嵌入向量")
                return None, None, float('inf'), []
            
            # 處理結果，計算距離
            distances = []
            for obj in results.objects:
                # 距離信息可能在不同位置，根據您的 Weaviate 版本進行適配
                distance = None
                if hasattr(obj, 'metadata') and hasattr(obj.metadata, 'distance'):
                    # v4 API
                    distance = obj.metadata.distance
                
                # 處理 distance 可能是 None 的情況
                if distance is None:
                    distance = -1  # 使用預設值
                    print(f"警告：無法從結果中獲取距離信息，使用預設值 {distance}")
                
                object_id = str(obj.uuid)  # 確保 UUID 是字符串格式
                speaker_name = obj.properties.get("speaker_name")
                update_count = obj.properties.get("update_count")  # 恢復使用update_count
                
                # 移除重複的比對輸出，交由上層處理
                # distance_str = f"{distance:.4f}" if distance is not None else "未知"
                # print(f"比對 - 語者: {speaker_name}, "
                #       f"更新次數: {update_count}, 餘弦距離: {distance_str}")
                
                # 保存距離資訊（使用update_count作為第4個參數）
                distances.append((object_id, speaker_name, distance, update_count))
            
            # 應用 AS-Norm (如果啟用)
            if as_norm_processor and ENABLE_AS_NORM:
                print("🔧 應用 AS-Norm 正規化...")
                distances = apply_as_norm_to_distances(new_embedding, distances, as_norm_processor)
                print("✅ AS-Norm 正規化完成")
            
            # 找出最小距離
            if distances:
                best_match = min(distances, key=lambda x: x[2])
                best_id, best_name, best_distance, _ = best_match
                return best_id, best_name, best_distance, distances
            else:
                # 如果沒有有效的距離信息，返回空結果
                print("警告：未能獲取有效的距離信息")
                return None, None, float('inf'), []
            
        except Exception as e:
            print(f"比對嵌入向量時發生錯誤: {e}")
            raise
    
    def update_embedding(self, voice_print_id: str, new_embedding: np.ndarray, update_count: int) -> int:
        """
        使用加權移動平均更新現有的嵌入向量（V2版本）
        
        Args:
            voice_print_id: 要更新的聲紋向量 UUID
            new_embedding: 新的嵌入向量
            update_count: 新的更新次數
            
        Returns:
            int: 更新後的更新次數
        """
        try:
            # 獲取現有的嵌入向量
            voice_print_collection = self.client.collections.get("VoicePrint")
            existing_object = voice_print_collection.query.fetch_object_by_id(
                uuid=voice_print_id,
                include_vector=True
            )
            
            if not existing_object:
                raise ValueError(f"找不到 UUID 為 {voice_print_id} 的聲紋向量")
            
            # 獲取現有的嵌入向量            
            vec_dict = existing_object.vector   # 取出 Weaviate 回傳的 named vector
            raw_old = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict   # 如果是 dict，就用 "default" 這組；否則直接當 list 處理
            old_embedding = np.array(raw_old, dtype=float)
            
            # 使用加權移動平均更新嵌入向量（基於更新次數）
            weight_old = update_count - 1
            updated_embedding = (old_embedding * weight_old + new_embedding) / update_count
            new_update_count = update_count
            
            # 更新數據庫中的向量（V2屬性名稱）
            voice_print_collection.data.update(
                uuid=voice_print_id,
                properties={
                    "updated_at": format_rfc3339(),  # V2: updated_at
                    "update_count": new_update_count  # 使用update_count
                },
                vector=updated_embedding.tolist()
            )
            
            print(f"(更新) 聲紋UUID {voice_print_id} 已更新，新的更新次數: {new_update_count}")
            return new_update_count
            
        except Exception as e:
            print(f"更新嵌入向量時發生錯誤: {e}")
            raise
    
    def add_embedding_without_averaging(self, speaker_name: str, new_embedding: np.ndarray, speaker_id: Optional[str] = None) -> str:
        """
        為現有語者添加新的嵌入向量（不進行加權平均）（V2版本）
        
        Args:
            speaker_name: 語者名稱
            new_embedding: 新的嵌入向量
            speaker_id: 現有語者 UUID，如果為 None 則創建新語者
            
        Returns:
            str: 新建立的聲紋向量 UUID
        """
        try:
            # 如果沒有提供 speaker_id，則創建新的語者
            if not speaker_id:
                speaker_id = self.create_new_speaker(speaker_name)
            
            # 添加新的嵌入向量
            voice_print_collection = self.client.collections.get("VoicePrint")
            voice_print_uuid = str(uuid.uuid4())
            
            # 創建新的聲紋向量（V2屬性）
            voice_print_collection.data.insert(
                properties={
                    "created_at": format_rfc3339(),  # V2: created_at
                    "updated_at": format_rfc3339(),  # V2: updated_at
                    "update_count": 1,
                    "sample_count": None,  # V2: sample_count（預留，可為空值）
                    "quality_score": None,  # V2: quality_score（預留，可為空值）
                    "speaker_name": speaker_name
                },
                uuid=voice_print_uuid,
                vector=new_embedding.tolist(),
                references={
                    "speaker": [speaker_id]
                }
            )
            
            print(f"(新嵌入) 為語者 {speaker_name} 添加了新的聲紋向量 (UUID: {voice_print_uuid})")
            return voice_print_uuid
            
        except Exception as e:
            print(f"添加嵌入向量時發生錯誤: {e}")
            raise
    
    def create_new_speaker(self, speaker_name: str = DEFAULT_SPEAKER_NAME, first_audio: Optional[str] = None) -> str:
        """
        創建新的語者（V2版本）
        
        Args:
            speaker_name: 語者名稱，默認為「未命名語者」
            first_audio: 第一次生成該語者時使用的音檔路徑（如 "20250709_185516\\segment_001\\speaker1.wav"）
            
        Returns:
            str: 新建立的語者 UUID
        """
        try:
            # 創建新的語者
            speaker_collection = self.client.collections.get("Speaker")
            speaker_uuid = str(uuid.uuid4())
            
            # 獲取下一個speaker_id
            speaker_id = self._get_next_speaker_id()
            
            # 如果是默認名稱，生成唯一的名稱 (類似 n1, n2, ...)
            if speaker_name == DEFAULT_SPEAKER_NAME:
                speaker_name = f"{DEFAULT_FULL_NAME_PREFIX}{speaker_id}"
            
            # 創建語者（V2屬性）
            properties = {
                "speaker_id": speaker_id,  # V2: speaker_id (INT)
                "full_name": speaker_name,  # V2: full_name
                "nickname": None,  # V2: nickname（可為空值）
                "gender": None,  # V2: gender（可為空值）
                "created_at": format_rfc3339(),  # V2: created_at
                "last_active_at": format_rfc3339(),  # V2: last_active_at
                "meet_count": None,  # V2: meet_count（可為空值）
                "meet_days": None,  # V2: meet_days（可為空值）
                "voiceprint_ids": [],  # 初始時沒有聲紋向量
                "first_audio": first_audio or ""  # V2: first_audio
            }
            
            speaker_collection.data.insert(
                properties=properties,
                uuid=speaker_uuid
            )
            
            print(f"(新語者) 建立新語者 {speaker_name} (UUID: {speaker_uuid}, ID: {speaker_id})")
            if first_audio:
                print(f"設置語者 {speaker_name} 的第一個音檔路徑: {first_audio}")
            
            return speaker_uuid
            
        except Exception as e:
            print(f"創建新語者時發生錯誤: {e}")
            raise
    
    def _get_next_speaker_id(self) -> int:
        """
        獲取下一個可用的speaker_id（從1開始）
        
        Returns:
            int: 下一個speaker_id
        """
        try:
            # 獲取所有Speaker的speaker_id
            speaker_collection = self.client.collections.get("Speaker")
            results = speaker_collection.query.fetch_objects(
                return_properties=["speaker_id"],
                limit=1000  # 假設不會超過1000個語者
            )
            
            # 提取所有現有的speaker_id
            existing_ids = []
            for obj in results.objects:
                speaker_id = obj.properties.get("speaker_id")
                if speaker_id is not None:
                    existing_ids.append(speaker_id)
            
            # 找出下一個可用的ID
            next_id = max(existing_ids) + 1 if existing_ids else 1
            return next_id
            
        except Exception as e:
            print(f"獲取下一個speaker_id時發生錯誤: {e}")
            # 發生錯誤時返回一個默認值
            return 1
    
    def handle_new_speaker(self, new_embedding: np.ndarray, audio_source: str = "", create_time: Optional[datetime] = None, updated_time: Optional[datetime] = None) -> Tuple[str, str, str]:
        """
        處理全新的語者：創建新語者和嵌入向量（V2版本）
        
        Args:
            new_embedding: 新的嵌入向量
            audio_source: 音訊來源，例如檔案名稱或路徑
            create_time: 自訂創建時間，如果為 None 則使用當前時間
            updated_time: 自訂更新時間，如果為 None 則使用當前時間
            
        Returns:
            tuple: (語者UUID, 聲紋向量UUID, 語者名稱)
        """
        try:
            # 創建新的語者，傳入音檔路徑作為 first_audio
            speaker_uuid = self.create_new_speaker(first_audio=audio_source)
            
            # 獲取語者名稱（V2使用full_name）
            speaker_collection = self.client.collections.get("Speaker")
            speaker_obj = speaker_collection.query.fetch_object_by_id(
                uuid=speaker_uuid,
                return_properties=["full_name"]
            )
            
            if not speaker_obj:
                raise ValueError(f"找不到剛剛創建的語者 (UUID: {speaker_uuid})")
            
            speaker_name = speaker_obj.properties["full_name"]
            
            # 創建新的嵌入向量，並與語者建立關聯
            voice_print_collection = self.client.collections.get("VoicePrint")
            voice_print_uuid = str(uuid.uuid4())
            
            # 格式化時間或使用當前時間
            create_time_str = format_rfc3339(create_time) if create_time else format_rfc3339()
            updated_time_str = format_rfc3339(updated_time) if updated_time else format_rfc3339()
            
            # 創建聲紋向量（V2屬性）
            voice_print_collection.data.insert(
                properties={
                    "created_at": create_time_str,  # V2: created_at
                    "updated_at": updated_time_str,  # V2: updated_at
                    "update_count": 1,
                    "sample_count": None,  # V2: sample_count（預留，可為空值）
                    "quality_score": None,  # V2: quality_score（預留，可為空值）
                    "speaker_name": speaker_name
                },
                uuid=voice_print_uuid,
                vector=new_embedding.tolist(),
                references={
                    "speaker": [speaker_uuid]
                }
            )
            
            # 更新語者的聲紋向量列表
            speaker_collection.data.update(
                uuid=speaker_uuid,
                properties={
                    "voiceprint_ids": [voice_print_uuid],
                    "last_active_at": updated_time_str
                }
            )
            
            print(f"(新語者) 已建立新語者 {speaker_name} 和對應的聲紋向量 (UUID: {voice_print_uuid})")
            return speaker_uuid, voice_print_uuid, speaker_name
            
        except Exception as e:
            print(f"處理新語者時發生錯誤: {e}")
            raise
    
    def get_voice_print_properties(self, voice_print_uuid: str, properties: List[str]) -> Optional[Dict[str, Any]]:
        """
        獲取聲紋向量的屬性（V2版本）
        
        Args:
            voice_print_uuid: 聲紋向量 UUID
            properties: 需要獲取的屬性列表
            
        Returns:
            Optional[Dict[str, Any]]: 屬性字典，若不存在則返回 None
        """
        try:
            voice_print_collection = self.client.collections.get("VoicePrint")
            result = voice_print_collection.query.fetch_object_by_id(
                uuid=voice_print_uuid,
                return_properties=properties
            )
            
            if not result:
                return None
                
            return result.properties
            
        except Exception as e:
            print(f"獲取聲紋向量屬性時發生錯誤: {e}")
            return None
    
    def update_speaker_voice_prints(self, speaker_uuid: str, voice_print_uuid: str) -> bool:
        """
        更新語者的聲紋向量列表（V2版本）
        
        Args:
            speaker_uuid: 語者 UUID
            voice_print_uuid: 要添加的聲紋向量 UUID
            
        Returns:
            bool: 是否更新成功
        """
        try:
            speaker_collection = self.client.collections.get("Speaker")
            speaker_obj = speaker_collection.query.fetch_object_by_id(
                uuid=speaker_uuid,
                return_properties=["voiceprint_ids"]
            )
            
            if not speaker_obj:
                return False
                
            voiceprint_ids = speaker_obj.properties.get("voiceprint_ids", [])
            if voice_print_uuid not in voiceprint_ids:
                voiceprint_ids.append(voice_print_uuid)
                
                speaker_collection.data.update(
                    uuid=speaker_uuid,
                    properties={
                        "voiceprint_ids": voiceprint_ids,
                        "last_active_at": format_rfc3339()  # V2: last_active_at
                    }
                )
            
            return True
            
        except Exception as e:
            print(f"更新語者聲紋向量列表時發生錯誤: {e}")
            return False
    
    def close(self) -> None:
        """關閉 Weaviate 連接"""
        if hasattr(self, 'client'):
            self.client.close()
            print("已關閉 Weaviate 連接")


class SpeakerIdentifier:
    """
    語者識別類，負責核心識別邏輯
    實現單例模式，避免重複初始化模型和資料庫連接
    """
    _instance = None
    _initialized = False
    
    def __new__(cls) -> 'SpeakerIdentifier':
        """實現單例模式，確保全局只有一個實例"""
        if cls._instance is None:
            cls._instance = super(SpeakerIdentifier, cls).__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """初始化語者識別器，若已初始化則跳過"""
        if SpeakerIdentifier._initialized:
            return
            
        self.audio_processor = AudioProcessor()
        self.database = WeaviateRepository()
        self.threshold_low = THRESHOLD_LOW
        self.threshold_update = THRESHOLD_UPDATE
        self.threshold_new = THRESHOLD_NEW
        
        # 只有在啟用 AS-Norm 時才初始化處理器
        if ENABLE_AS_NORM:
            self.as_norm_processor = ASNormProcessor(self.database.client)
        else:
            self.as_norm_processor = None
        
        # 設置日誌格式
        self.verbose = True  # 控制詳細輸出
        
        # 輸出 AS-Norm 初始化狀態
        if ENABLE_AS_NORM:
            logger.info("🔧 AS-Norm 正規化功能已啟用")
        else:
            logger.info("⚪ AS-Norm 正規化功能已停用")
        
        SpeakerIdentifier._initialized = True
    
    def set_verbose(self, verbose: bool) -> None:
        """設置是否顯示詳細輸出"""
        self.verbose = verbose
    
    def set_as_norm_enabled(self, enabled: bool) -> None:
        """
        設置是否啟用 AS-Norm 功能
        
        Args:
            enabled: True 啟用 AS-Norm，False 停用
        """
        global ENABLE_AS_NORM
        old_value = ENABLE_AS_NORM
        ENABLE_AS_NORM = enabled
        
        # 動態創建或銷毀 AS-Norm 處理器
        if enabled and self.as_norm_processor is None:
            # 啟用時創建處理器
            self.as_norm_processor = ASNormProcessor(self.database.client)
        elif not enabled and self.as_norm_processor is not None:
            # 停用時銷毀處理器
            self.as_norm_processor = None
        
        if self.verbose:
            if enabled and not old_value:
                print("✅ AS-Norm 正規化已啟用")
            elif not enabled and old_value:
                print("❌ AS-Norm 正規化已停用")
    
    def configure_as_norm(self, t_norm: bool = True, z_norm: bool = True, s_norm: bool = True,
                         cohort_size: int = 100, top_k: int = 10, alpha: float = 0.9) -> None:
        """
        配置 AS-Norm 參數
        
        Args:
            t_norm: 是否啟用 T-Norm
            z_norm: 是否啟用 Z-Norm
            s_norm: 是否啟用 S-Norm
            cohort_size: cohort 大小
            top_k: Top-K impostor 分數
            alpha: S-Norm 權重參數
        """
        global ENABLE_T_NORM, ENABLE_Z_NORM, ENABLE_S_NORM
        global AS_NORM_COHORT_SIZE, AS_NORM_TOP_K, AS_NORM_ALPHA
        
        ENABLE_T_NORM = t_norm
        ENABLE_Z_NORM = z_norm
        ENABLE_S_NORM = s_norm
        AS_NORM_COHORT_SIZE = cohort_size
        AS_NORM_TOP_K = top_k
        AS_NORM_ALPHA = alpha
        
        # 更新 AS-Norm 處理器的參數（只有在處理器存在時）
        if self.as_norm_processor is not None:
            self.as_norm_processor.cohort_size = cohort_size
            self.as_norm_processor.top_k = top_k
            self.as_norm_processor.alpha = alpha
        
        if self.verbose:
            if self.as_norm_processor is not None:
                print(f"🔧 AS-Norm 配置已更新:")
                print(f"   T-Norm: {t_norm}, Z-Norm: {z_norm}, S-Norm: {s_norm}")
                print(f"   Cohort Size: {cohort_size}, Top-K: {top_k}, Alpha: {alpha}")
            else:
                print("⚠️ AS-Norm 處理器未啟用，配置已保存但未生效")
    
    def get_as_norm_status(self) -> Dict[str, Any]:
        """
        獲取 AS-Norm 當前狀態
        
        Returns:
            Dict[str, Any]: AS-Norm 狀態資訊
        """
        return {
            "enabled": ENABLE_AS_NORM,
            "processor_initialized": self.as_norm_processor is not None,
            "t_norm": ENABLE_T_NORM,
            "z_norm": ENABLE_Z_NORM,
            "s_norm": ENABLE_S_NORM,
            "cohort_size": AS_NORM_COHORT_SIZE,
            "top_k": AS_NORM_TOP_K,
            "alpha": AS_NORM_ALPHA
        }
    
    def _handle_very_similar(self, best_id: str, best_name: str, best_distance: float) -> Tuple[str, str, float]:
        """
        處理過於相似的情況：不更新向量
        
        Args:
            best_id: 最佳匹配ID
            best_name: 最佳匹配語者名稱
            best_distance: 最佳匹配距離
            
        Returns:
            Tuple[str, str, float]: (語者ID, 語者名稱, 相似度)
        """
        if self.verbose:
            print(f"(跳過) 嵌入向量過於相似 (距離 = {best_distance:.4f})，不進行更新。")
            print(f"該音檔與語者 {best_name} 的檔案相同。")
        return best_id, best_name, best_distance
    
    def _handle_update_embedding(self, best_id: str, best_name: str, best_distance: float, new_embedding: np.ndarray) -> Tuple[str, str, float]:
        """
        處理需要更新嵌入向量的情況（V2版本）
        
        Args:
            best_id: 最佳匹配的聲紋向量 UUID
            best_name: 最佳匹配語者名稱
            best_distance: 最佳匹配距離
            new_embedding: 新的嵌入向量
            
        Returns:
            Tuple[str, str, float]: (語者ID, 語者名稱, 相似度)
        """
        try:
            # 獲取當前更新次數
            properties = self.database.get_voice_print_properties(best_id, ["update_count"])
            if properties is None:
                raise ValueError(f"無法獲取聲紋向量 UUID {best_id} 的屬性")
            
            update_count = properties.get("update_count", 0)
            new_update_count = update_count + 1  # 新更新次數 = 當前次數 + 1
            
            # 更新嵌入向量（傳遞新的更新次數）
            self.database.update_embedding(best_id, new_embedding, new_update_count)
            print(f"該音檔與語者 {best_name} 相符，且已更新嵌入檔案。")
            return best_id, best_name, best_distance
        except Exception as e:
            print(f"更新嵌入向量時發生錯誤: {e}")
            raise
    
    def _handle_new_speaker(self, new_embedding: np.ndarray, audio_source: str = "", timestamp: Optional[datetime] = None) -> Tuple[str, str, float]:
        """
        處理新語者的情況：創建新語者（V2版本）
        
        Args:
            new_embedding: 新的嵌入向量
            audio_source: 音訊來源描述
            timestamp: 音訊的時間戳記，用於設定聲紋的創建時間和更新時間
            
        Returns:
            Tuple[str, str, float]: (語者UUID, 語者名稱, 相似度)
        """
        speaker_id, voice_print_id, speaker_name = self.database.handle_new_speaker(
            new_embedding, audio_source, create_time=timestamp, updated_time=timestamp
        )
        return speaker_id, speaker_name, -1  # -1 表示全新的語者
    
    def _handle_add_new_voiceprint_to_speaker(self, best_id: str, best_name: str, best_distance: float, new_embedding: np.ndarray, audio_source: str = "", timestamp: Optional[datetime] = None) -> Tuple[str, str, float]:
        """
        處理相似但不更新原有聲紋的情況：為現有語者新增額外的聲紋向量
        
        此方法提供更完整的封裝，將新嵌入向量添加到已匹配的語者，但建立為獨立的聲紋向量
        而非更新現有向量。這允許一個語者擁有多個不同環境或條件下的聲紋
        
        Args:
            best_id: 最佳匹配的聲紋向量ID
            best_name: 最佳匹配語者名稱
            best_distance: 最佳匹配距離
            new_embedding: 新的嵌入向量
            audio_source: 音訊來源描述 (可選)
            timestamp: 音訊的時間戳記，用於設定聲紋的創建與更新時間 (可選)
            
        Returns:
            Tuple[str, str, float]: (語者ID, 語者名稱, 相似度)
        """
        try:
            # 從聲紋獲取所屬的語者ID
            speaker_id = self._get_speaker_id_from_voiceprint(best_id)
            
            # 為此語者新增一個新的聲紋向量
            voice_print_id = self._add_voiceprint_to_speaker(
                speaker_id=speaker_id,
                speaker_name=best_name,
                new_embedding=new_embedding,
                audio_source=audio_source,
                timestamp=timestamp
            )
            
            if self.verbose:
                print(f"(新增聲紋) 已為語者 {best_name} 建立新的聲紋向量 (ID: {voice_print_id})")
                print(f"該音檔與語者 {best_name} 相似但不足以更新原有聲紋，已建立新的聲紋。")
            
            return speaker_id, best_name, best_distance
            
        except Exception as e:
            print(f"新增額外聲紋向量時發生錯誤: {e}")
            raise
    
    def _get_speaker_id_from_voiceprint(self, voice_print_id: str) -> str:
        """
        根據聲紋向量ID獲取對應的語者ID
        
        Args:
            voice_print_id: 聲紋向量ID
            
        Returns:
            str: 語者ID
            
        Raises:
            ValueError: 當無法獲取語者ID時
        """
        try:
            voice_print_collection = self.database.client.collections.get("VoicePrint")
            # 使用 QueryReference 指定要回傳哪個 reference 屬性，以及要哪些欄位
            qr = QueryReference(
                link_on="speaker",            # reference 欄位名稱
                return_properties=["uuid"]    # 要把 uuid 回傳下來
            )
            # 呼叫 fetch_object_by_id，傳入 qr 而非字串列表
            voice_print_obj = voice_print_collection.query.fetch_object_by_id(
                uuid=voice_print_id,
                return_references=qr
            )
            # 從回傳的 references 取出第一個 speaker 的 uuid
            refs = voice_print_obj.references.get("speaker", []).objects
            if not refs:
                raise ValueError(f"聲紋向量 {voice_print_id} 沒有對應的語者參考")
            return str(refs[0].uuid)  # 確保返回字符串而非 Weaviate UUID 對象
        except Exception as e:
            print(f"獲取語者ID時發生錯誤: {e}")
            raise
    
    def _add_voiceprint_to_speaker(self, speaker_id: str, speaker_name: str, new_embedding: np.ndarray, 
                                   audio_source: str = "", timestamp: Optional[datetime] = None) -> str:
        """
        為指定語者添加新的聲紋向量
        
        Args:
            speaker_id: 語者ID
            speaker_name: 語者名稱
            new_embedding: 新的嵌入向量
            audio_source: 音訊來源描述 (可選)
            timestamp: 時間戳記，用於設定創建與更新時間 (可選)
            
        Returns:
            str: 新建立的聲紋向量ID
        """
        try:
            # 格式化時間或使用當前時間
            create_time_str = format_rfc3339(timestamp) if timestamp else format_rfc3339()
            
            # 添加新的嵌入向量到語者
            voice_print_collection = self.database.client.collections.get("VoicePrint")
            voice_print_id = str(uuid.uuid4())
            
            # 創建新的聲紋向量（V2版本）
            voice_print_collection.data.insert(
                properties={
                    "created_at": create_time_str,    # V2: created_at
                    "updated_at": create_time_str,    # V2: updated_at  
                    "update_count": 1,                # update_count用途不變
                    "sample_count": None,             # V2: sample_count（預留，可為空值）
                    "quality_score": None,            # V2: quality_score（預留，可為空值）
                    "speaker_name": speaker_name,
                    "audio_source": audio_source
                },
                uuid=voice_print_id,
                vector=new_embedding.tolist(),
                references={
                    "speaker": [speaker_id]
                }
            )
            
            # 更新語者的聲紋向量列表（V2版本使用UUID參數名稱）
            self.database.update_speaker_voice_prints(speaker_id, voice_print_id)
            
            return voice_print_id
        except Exception as e:
            print(f"為語者添加聲紋向量時發生錯誤: {e}")
            raise

    # 簡化輸出的控制函數
    def simplified_print(self, message: str, verbose: bool = True) -> None:
        """
        根據詳細度設置決定是否輸出訊息
        
        Args:
            message: 要輸出的訊息
            verbose: 是否輸出詳細信息，預設為 True
        """
        if verbose:
            # 使用帶前綴的 logger
            prefix = _current_process_prefix.get()
            message_with_prefix = f"{prefix} {message}" if prefix else message
            logger.info(message_with_prefix)

    # 格式化輸出比對結果
    def format_comparison_result(self, speaker_name: str, update_count: int, distance: float, verbose: bool = True) -> None:
        """
        格式化輸出比對結果
        
        Args:
            speaker_name: 語者名稱
            update_count: 更新次數
            distance: 相似度距離
            verbose: 是否輸出詳細信息
        """
        if verbose:
            distance_str = f"{distance:.4f}" if distance is not None else "未知"
            # 使用帶前綴的 logger
            prefix = _current_process_prefix.get()
            message = f"比對 - 語者: {speaker_name}, 更新次數: {update_count}, 餘弦距離: {distance_str}"
            message_with_prefix = f"{prefix} {message}" if prefix else message
            logger.info(message_with_prefix)

    # 修改 SpeakerIdentifier 類別中的方法來使用這些函數
    def process_audio_file(self, audio_file: str) -> Optional[Tuple[str, str, float]]:
        """
        處理音檔並進行語者識別
        
        Args:
            audio_file: 音檔路徑
            
        Returns:
            Optional[Tuple[str, str, float]]: (語者ID, 語者名稱, 相似度) 或 None 表示處理失敗
        """
        token = None
        try:
            process_id = next(_process_counter)
            prefix = format_process_prefix(process_id, audio_file)
            token = _current_process_prefix.set(prefix)
            self.simplified_print(f"\n處理音檔: {audio_file}", self.verbose)
            if not os.path.exists(audio_file):
                self.simplified_print(f"音檔 {audio_file} 不存在，取消處理。", self.verbose)
                return None

            # 讀取音檔獲取 signal 和 sr
            signal, sr = sf.read(audio_file)

            # 直接使用完整路徑，統一轉換為正斜線格式
            audio_source = audio_file.replace('\\', '/')

            return self.process_audio_stream(signal, sr, audio_source=audio_source)

        except Exception as e:
            self.simplified_print(f"處理音檔 {audio_file} 時發生錯誤: {e}", self.verbose)
            return None
        finally:
            if token:
                _current_process_prefix.reset(token)

    def process_audio_stream(self, signal: np.ndarray, sr: int, audio_source: str = "無", timestamp: Optional[datetime] = None) -> Optional[Tuple[str, str, float]]:
        """
        處理音訊流 (NumPy 陣列) 並進行語者識別

        Args:
            signal: 音訊信號 (NumPy 陣列)
            sr: 音訊信號的取樣率
            audio_source: 音訊來源的名稱 (用於未來回朔音檔)
            timestamp: 音訊的時間戳記，用於設定聲紋的創建時間和更新時間

        Returns:
            Optional[Tuple[str, str, float]]: (語者ID, 語者名稱, 相似度) 或 None 表示處理失敗
        """
        token = None
        try:
            if not _current_process_prefix.get():
                process_id = next(_process_counter)
                prefix = format_process_prefix(process_id, audio_source)
                token = _current_process_prefix.set(prefix)
            self.simplified_print(f"\n處理來源: {audio_source}", self.verbose)

            # 提取嵌入向量
            new_embedding = self.audio_processor.extract_embedding_from_stream(signal, sr)

            # 與 Weaviate 中的嵌入向量比對，傳遞 AS-Norm 處理器
            best_id, best_name, best_distance, all_distances = self.database.compare_embedding(
                new_embedding, 
                as_norm_processor=self.as_norm_processor if ENABLE_AS_NORM else None
            )

            # 輸出比對結果
            if self.verbose and all_distances:
                for obj_id, name, distance, update_count in all_distances[:3]:  # 只顯示前3個結果
                    self.format_comparison_result(name, update_count, distance, self.verbose)

            # 根據距離進行判斷，使用輔助函數處理不同情況
            if best_id is None:
                # 資料庫為空，直接創建新語者
                self.simplified_print("資料庫為空，創建新語者", self.verbose)
                # 傳遞音訊來源名稱和時間戳記
                return self._handle_new_speaker(new_embedding, audio_source, timestamp)
            elif best_distance < self.threshold_low:
                # 過於相似，不更新
                return self._handle_very_similar(best_id, best_name, best_distance)
            elif best_distance < self.threshold_update:
                # 距離在允許的範圍內，更新嵌入向量
                return self._handle_update_embedding(best_id, best_name, best_distance, new_embedding)
            elif best_distance < self.threshold_new:
                # 距離在匹配範圍內，建立新的聲紋向量
                return self._handle_add_new_voiceprint_to_speaker(best_id, best_name, best_distance, new_embedding, audio_source, timestamp)
            else:
                # 判定為新語者
                # 傳遞音訊來源名稱和時間戳記
                return self._handle_new_speaker(new_embedding, audio_source, timestamp)

        except Exception as e:
            self.simplified_print(f"處理音訊流 '{audio_source}' 時發生錯誤: {e}", self.verbose)
            return None
        finally:
            if token:
                _current_process_prefix.reset(token)

    def process_audio_directory(self, directory: str) -> Dict[str, Any]:
        """
        處理指定資料夾內所有 .wav 檔案
        
        Args:
            directory: 資料夾路徑
            
        Returns:
            Dict[str, Any]: 處理結果統計
        """
        if not os.path.exists(directory):
            print(f"資料夾 {directory} 不存在，取消處理。")
            return {"success": False, "error": "資料夾不存在"}
            
        audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]
        if not audio_files:
            print(f"資料夾 {directory} 中沒有 .wav 檔案。")
            return {"success": False, "error": "無音檔"}
            
        print(f"發現 {len(audio_files)} 個音檔於 {directory}，開始處理...")
        
        results = {
            "total": len(audio_files),
            "processed": 0,
            "failed": 0,
            "new_speakers": 0,
            "updated_speakers": 0,
            "matched_speakers": 0,
            "details": []
        }
        
        try:
            for audio_file in audio_files:
                try:
                    result = self.process_audio_file(audio_file)
                    results["processed"] += 1
                    
                    if result:
                        speaker_id, speaker_name, distance = result
                        results["details"].append({
                            "file": audio_file,
                            "speaker_id": speaker_id,
                            "speaker_name": speaker_name,
                            "distance": distance
                        })
                        
                        # 計算統計
                        if distance == 1.0:
                            results["new_speakers"] += 1
                        elif distance < self.threshold_update:
                            results["updated_speakers"] += 1
                        else:
                            results["matched_speakers"] += 1
                        
                except Exception as e:
                    print(f"處理 {audio_file} 時發生錯誤: {e}")
                    results["failed"] += 1
                    results["details"].append({
                        "file": audio_file,
                        "error": str(e)
                    })
        finally:
            # 關閉 Weaviate 連接
            self.database.close()
                
        print(f"\n完成處理資料夾 {directory} 中所有音檔。")
        print(f"處理總數: {results['processed']}/{results['total']}, 失敗: {results['failed']}")
        print(f"新增語者: {results['new_speakers']}, 更新語者: {results['updated_speakers']}, 匹配語者: {results['matched_speakers']}")
        
        return results
    
    def add_voiceprint_to_speaker(self, audio_file: str, speaker_id: str) -> bool:
        """
        將音檔轉換為聲紋向量，並添加到指定的語者
        
        此方法提供一個公開介面，允許直接從音訊檔案為已知語者添加新的聲紋向量，
        而不需要進行語者識別的比對過程。適用於已確定語者身份的音檔。
        
        Args:
            audio_file: 音檔路徑
            speaker_id: 語者 ID
            
        Returns:
            bool: 是否成功添加
        """
        try:
            print(f"\n添加音檔聲紋向量到語者 (ID: {speaker_id}): {audio_file}")
            if not os.path.exists(audio_file):
                print(f"音檔 {audio_file} 不存在，取消處理。")
                return False
            
            # 檢查語者是否存在（V2版本使用full_name）
            speaker_collection = self.database.client.collections.get("Speaker")
            speaker_obj = speaker_collection.query.fetch_object_by_id(
                uuid=speaker_id,
                return_properties=["full_name"]
            )
            
            if not speaker_obj:
                print(f"語者 ID {speaker_id} 不存在，取消處理。")
                return False
                
            speaker_name = speaker_obj.properties["full_name"]  # V2: 使用 full_name
            
            # 提取嵌入向量
            new_embedding = self.audio_processor.extract_embedding(audio_file)
            
            # 使用內部方法添加聲紋向量
            voice_print_id = self._add_voiceprint_to_speaker(
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                new_embedding=new_embedding,
                audio_source=os.path.basename(audio_file)
            )
            
            if voice_print_id:
                print(f"已成功將音檔聲紋向量添加到語者 {speaker_name} (聲紋ID: {voice_print_id})")
                return True
            return False
                
        except Exception as e:
            print(f"添加聲紋向量時發生錯誤: {e}")
            return False


if __name__ == "__main__":
    set_output_enabled(True)  # 啟用輸出

    # 創建語者識別器
    identifier = SpeakerIdentifier()
    
    # ==================== AS-Norm 使用示例 ====================
    # 啟用 AS-Norm 功能
    # identifier.set_as_norm_enabled(True)
    
    # 自訂 AS-Norm 配置
    # identifier.configure_as_norm(
    #     t_norm=True,      # 啟用 T-Norm
    #     z_norm=True,      # 啟用 Z-Norm  
    #     s_norm=True,      # 啟用 S-Norm (結合 T-Norm 和 Z-Norm)
    #     cohort_size=50,   # 使用 50 個 impostor 語者
    #     top_k=10,         # 使用前 10 個最相似的 impostor
    #     alpha=0.8         # S-Norm 權重參數
    # )
    
    # 查看 AS-Norm 狀態
    # as_norm_status = identifier.get_as_norm_status()
    # print("🔍 AS-Norm 狀態:", as_norm_status)
    
    # 測試不同的 AS-Norm 組合
    # print("\n📊 測試各種 AS-Norm 設定...")
    
    # 僅使用 T-Norm
    # identifier.configure_as_norm(t_norm=True, z_norm=False, s_norm=False)
    # print("🔧 當前設定: 僅 T-Norm")
    # identifier.process_audio_file("16K-model/Audios-16K-IDTF/speaker1_20250501-22_49_13_1.wav")
    
    # 僅使用 Z-Norm
    # identifier.configure_as_norm(t_norm=False, z_norm=True, s_norm=False)  
    # print("🔧 當前設定: 僅 Z-Norm")
    # identifier.process_audio_file("16K-model/Audios-16K-IDTF/speaker1_20250501-22_49_13_1.wav")
    
    # 使用完整 S-Norm
    # identifier.configure_as_norm(t_norm=True, z_norm=True, s_norm=True)
    # print("🔧 當前設定: 完整 S-Norm")
    # identifier.process_audio_file("16K-model/Audios-16K-IDTF/speaker1_20250501-22_49_13_1.wav")
    
    # ==================== 一般識別測試 ====================
    # 主程式執行: 若要處理單一檔案或資料夾，可解除下列註解

    # 範例：處理單一檔案 (現在會透過 process_audio_stream)
    identifier.process_audio_file("16K-model/Audios-16K-IDTF/speaker1_20250501-22_49_13_1.wav")

    # 範例：直接處理音訊流 (假設你有 NumPy 陣列 signal 和取樣率 sr)
    # try:
    #     # 假設這是從某個來源得到的音訊數據和取樣率
    #     # 例如：從麥克風、網路流等
    #     sample_signal, sample_sr = sf.read("16K-model/Audios-16K-IDTF/speaker2_20250501-22_49_13_1.wav") # 僅為範例，實際應來自流
    #     identifier.process_audio_stream(sample_signal, sample_sr, source_description="範例音訊流")
    # except Exception as e:
    #     print(f"處理範例音訊流時出錯: {e}")


    # identifier.process_audio_directory("testFiles/test_audioFile/0770")
    
    # 如果需要將音檔提取聲紋並添加到現有語者，可解除下列註解
    # identifier.add_voiceprint_to_speaker("path_to_audio.wav", "speaker_uuid")