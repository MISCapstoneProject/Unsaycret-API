"""
===============================================================================
AS-Norm Cohort 資料庫管理模組
===============================================================================

版本：v1.0.0
作者：CYouuu  
最後更新：2025-08-25

功能摘要：
-----------
本模組提供 AS-Norm 專用的 cohort 資料庫管理功能，包括：

1. Cohort Collection 的建立與管理
2. 音頻檔案批量處理與聲紋提取
3. Cohort 資料的導入與更新
4. 資料庫初始化與重置功能

設計原則：
-----------
- 隔離性：cohort 資料與實際語者資料完全分離  
- 穩定性：cohort 集合固定，不會因新增語者而改變
- 專用性：cohort 僅用於 AS-Norm 計算，不參與實際辨識

技術架構：
-----------
- 聲紋提取：pyannote/embedding 模型 (512維)
- 向量資料庫：Weaviate V2
- 音頻處理：librosa + soundfile
- 切片策略：固定長度切片 + 重疊窗口

使用方式：
-----------
1. 初始化 cohort 資料庫：
   ```python
   manager = CohortDatabaseManager()
   manager.initialize_cohort_collection()
   ```

2. 導入單個音頻檔案：
   ```python
   manager.import_audio_file("/path/to/audio.wav")  # 自動使用檔名作為 source_dataset
   ```

3. 從音頻資料夾批量導入 cohort：
   ```python
   manager.import_audio_folder("/path/to/cohort/audio")  # 每個檔案使用檔名作為 source_dataset
   ```

4. 重置 cohort 資料庫：
   ```python
   manager.reset_cohort_collection()
   ```

注意：現在直接處理整個音檔（6秒），不再進行切片處理。
"""

import os
import sys
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pytz  # 新增：時區支援

# 添加模組路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import weaviate
import weaviate.classes.config as wcc
import torch
from speechbrain.inference import SpeakerRecognition
from pyannote.audio import Inference
from pyannote.core import Segment
from scipy.spatial.distance import cosine
from scipy.signal import resample_poly  # 新增：與 VID_identify_v5.py 一致的重新採樣
from weaviate.classes.query import Filter

# 導入項目模組
from utils.logger import get_logger
from utils.env_config import get_model_save_dir, HF_ACCESS_TOKEN
from utils.constants import (
    AS_NORM_COHORT_COLLECTION, SPEECHBRAIN_SPEAKER_MODEL, PYANNOTE_SPEAKER_MODEL,
    AUDIO_TARGET_RATE, AUDIO_SAMPLE_RATE,
    ENABLE_AS_NORM, AS_NORM_COHORT_SIZE, AS_NORM_TOP_K, AS_NORM_ALPHA,
    ENABLE_T_NORM, ENABLE_Z_NORM, ENABLE_S_NORM, AS_NORM_USE_DEDICATED_COHORT
)

# 創建模組專屬日誌器
logger = get_logger(__name__)

# 台北時區設定
TAIPEI_TZ = pytz.timezone('Asia/Taipei')

def get_taipei_time() -> datetime:
    """
    獲取台北時間
    
    Returns:
        datetime: 台北時區的當前時間
    """
    return datetime.now(TAIPEI_TZ)


class ASNormProcessor:
    """
    AS-Norm (Adaptive Score Normalization) 處理器
    
    實現多種正規化技術：
    - T-Norm (Test Normalization): 使用 impostor 模型的分數進行正規化
    - Z-Norm (Zero Normalization): 使用統計 Z-score 正規化
    - S-Norm (Symmetric Normalization): 結合 T-Norm 和 Z-Norm
    
    主要目的：改善語者識別的穩定性和準確性，減少條件變異的影響
    """
    
    def __init__(self, database_client=None):
        """
        初始化 AS-Norm 處理器
        
        Args:
            database_client: Weaviate 資料庫客戶端實例
        """
        self.client = database_client
        self.cohort_size = AS_NORM_COHORT_SIZE
        self.top_k = AS_NORM_TOP_K
        self.alpha = AS_NORM_ALPHA
        
        # 統計資料緩存
        self._impostor_stats = {}
        self._stats_cache_size = 100
        
    def set_database_client(self, client):
        """設定資料庫客戶端"""
        self.client = client
        
    def compute_t_norm_score(self, test_embedding: np.ndarray, target_embedding: np.ndarray, 
                           impostor_embeddings: List[np.ndarray]) -> float:
        """
        計算 T-Norm 正規化分數（在距離空間操作）
        
        T-Norm 通過使用 impostor 模型分數來正規化目標分數
        在距離空間的公式: (target_distance - mean_impostor_distance) / std_impostor_distance
        
        注意：正規化後的值可能為負數（表示比平均 impostor 更相似）
        
        Args:
            test_embedding: 測試音訊的嵌入向量
            target_embedding: 目標語者的嵌入向量  
            impostor_embeddings: impostor 語者的嵌入向量列表
            
        Returns:
            float: T-Norm 正規化後的距離分數
        """
        if not impostor_embeddings:
            # 沒有 impostor 時，返回原始餘弦距離
            return cosine(test_embedding, target_embedding)
            
        # 計算目標距離（餘弦距離）
        target_distance = cosine(test_embedding, target_embedding)
        
        # 計算 impostor 距離
        impostor_distances = []
        for imp_embedding in impostor_embeddings:
            imp_distance = cosine(test_embedding, imp_embedding)
            impostor_distances.append(imp_distance)
            
        # 計算 impostor 距離的統計量
        mean_impostor_distance = np.mean(impostor_distances)
        std_impostor_distance = np.std(impostor_distances)
        
        # T-Norm 正規化（在距離空間）
        if std_impostor_distance > 0:
            # 注意：目標距離小於平均 impostor 距離時，正規化值為負
            t_norm_distance = (target_distance - mean_impostor_distance) / std_impostor_distance
        else:
            t_norm_distance = target_distance
            
        return t_norm_distance
    
    def compute_z_norm_score(self, test_embedding: np.ndarray, target_embedding: np.ndarray) -> float:
        """
        計算 Z-Norm 正規化分數（在距離空間操作）
        
        Z-Norm 使用測試語音對所有已知語者的統計分布進行正規化
        在距離空間的公式: (target_distance - mean_all_distance) / std_all_distance
        
        Args:
            test_embedding: 測試音訊的嵌入向量
            target_embedding: 目標語者的嵌入向量
            
        Returns:
            float: Z-Norm 正規化後的距離分數
        """
        # 計算目標距離（餘弦距離）
        target_distance = cosine(test_embedding, target_embedding)
        
        # 獲取所有語者的嵌入向量
        all_embeddings = self._get_all_speaker_embeddings()
        
        if not all_embeddings:
            return target_distance
            
        # 計算對所有語者的距離
        all_distances = []
        for embedding in all_embeddings:
            distance = cosine(test_embedding, embedding)
            all_distances.append(distance)
            
        # Z-Norm 正規化（在距離空間）
        mean_all_distance = np.mean(all_distances)
        std_all_distance = np.std(all_distances)
        
        if std_all_distance > 0:
            z_norm_distance = (target_distance - mean_all_distance) / std_all_distance
        else:
            z_norm_distance = target_distance
            
        return z_norm_distance
    
    def compute_s_norm_score(self, test_embedding: np.ndarray, target_embedding: np.ndarray,
                           impostor_embeddings: List[np.ndarray]) -> float:
        """
        計算 S-Norm (Symmetric Normalization) 正規化分數
        
        S-Norm 結合 T-Norm 和 Z-Norm 的優點
        公式: alpha * t_norm + (1-alpha) * z_norm
        
        Args:
            test_embedding: 測試音訊的嵌入向量
            target_embedding: 目標語者的嵌入向量
            impostor_embeddings: impostor 語者的嵌入向量列表
            
        Returns:
            float: S-Norm 正規化後的分數
        """
        # 計算 T-Norm 分數
        t_norm_score = self.compute_t_norm_score(test_embedding, target_embedding, impostor_embeddings)
        
        # 計算 Z-Norm 分數
        z_norm_score = self.compute_z_norm_score(test_embedding, target_embedding)
        
        # S-Norm 結合
        s_norm_score = self.alpha * t_norm_score + (1 - self.alpha) * z_norm_score
        
        return s_norm_score
    
    def apply_as_norm(self, test_embedding: np.ndarray, target_embedding: np.ndarray,
                     target_id: str) -> float:
        """
        應用 AS-Norm 處理（修正版：保持統計穩定性和區別性）
        
        根據配置選擇性地應用不同的正規化方法，確保不會破壞距離的區別能力
        
        Args:
            test_embedding: 測試音訊的嵌入向量
            target_embedding: 目標語者的嵌入向量
            target_id: 目標語者ID
            
        Returns:
            float: 正規化後的距離分數（與原始餘弦距離概念一致）
        """
        if not ENABLE_AS_NORM:
            # AS-Norm 關閉時，返回原始餘弦距離
            original_distance = cosine(test_embedding, target_embedding)
            logger.debug(f"⚪ AS-Norm 已停用，返回原始餘弦距離: {original_distance:.4f}")
            return original_distance
            
        # 計算原始餘弦距離作為對比
        original_distance = cosine(test_embedding, target_embedding)
        logger.debug(f"📏 原始餘弦距離: {original_distance:.4f}")
        
        # 獲取 impostor 嵌入向量
        impostor_embeddings = self._get_impostor_embeddings(target_id)
        
        # 根據配置選擇正規化方法（在距離空間操作）
        if ENABLE_S_NORM and ENABLE_T_NORM and ENABLE_Z_NORM:
            # 完整 S-Norm
            logger.debug("🔧 使用完整 S-Norm (T-Norm + Z-Norm 組合)")
            normalized_distance = self.compute_s_norm_score(test_embedding, target_embedding, impostor_embeddings)
        elif ENABLE_T_NORM and ENABLE_Z_NORM:
            # T-Norm + Z-Norm 組合
            logger.debug("🔧 使用 T-Norm + Z-Norm 組合")
            t_distance = self.compute_t_norm_score(test_embedding, target_embedding, impostor_embeddings)
            z_distance = self.compute_z_norm_score(test_embedding, target_embedding)
            normalized_distance = 0.5 * t_distance + 0.5 * z_distance
        elif ENABLE_T_NORM:
            # 僅 T-Norm
            logger.debug("🔧 使用 T-Norm 正規化")
            normalized_distance = self.compute_t_norm_score(test_embedding, target_embedding, impostor_embeddings)
        elif ENABLE_Z_NORM:
            # 僅 Z-Norm
            logger.debug("🔧 使用 Z-Norm 正規化")
            normalized_distance = self.compute_z_norm_score(test_embedding, target_embedding)
        else:
            # 所有正規化都關閉，返回原始分數
            logger.debug("⚪ 所有正規化方法都已停用，返回原始分數")
            return original_distance
        
        # 檢查正規化結果的合理性
        if abs(normalized_distance) > 10:
            logger.warning(f"⚠️ AS-Norm 正規化值異常: {normalized_distance:.4f}，cohort 資料可能有問題")
            logger.warning(f"⚠️ 回退到原始距離: {original_distance:.4f}")
            return original_distance
        
        # 使用保守的映射策略，保持區別性
        from utils.constants import THRESHOLD_LOW, THRESHOLD_UPDATE, THRESHOLD_NEW
        
        # 修正版映射：保持原始距離的相對關係，只做適度調整
        # 核心理念：好的匹配小幅改善，壞的匹配保持原樣或略微惡化
        
        if normalized_distance <= -2.0:
            # 很好的匹配：距離減少 20-40%
            reduction_factor = 0.6 + 0.2 * max(0, min(1, (normalized_distance + 4) / 2))
            final_distance = original_distance * reduction_factor
        elif normalized_distance <= -1.0:
            # 好的匹配：距離減少 10-20%
            reduction_factor = 0.8 + 0.1 * (normalized_distance + 2) / 1
            final_distance = original_distance * reduction_factor
        elif normalized_distance <= 0:
            # 中等匹配：距離減少 0-10%
            reduction_factor = 0.9 + 0.1 * (normalized_distance + 1) / 1
            final_distance = original_distance * reduction_factor
        elif normalized_distance <= 1.0:
            # 較差匹配：距離保持不變或略微增加
            increase_factor = 1.0 + 0.1 * normalized_distance / 1
            final_distance = original_distance * increase_factor
        else:
            # 很差匹配：距離增加 10-20%
            increase_factor = 1.1 + 0.1 * min(1.0, (normalized_distance - 1) / 2)
            final_distance = original_distance * increase_factor
        
        # 確保結果在合理範圍內
        final_distance = max(0.001, min(2.0, final_distance))
        
        # 記錄正規化效果和原始數據
        improvement = original_distance - final_distance  # 距離減少表示改善
        improvement_percent = (improvement / original_distance) * 100 if original_distance > 0 else 0
        
        # 詳細記錄正規化過程
        logger.debug(f"📊 原始餘弦距離: {original_distance:.4f}")
        logger.debug(f"📊 AS-Norm 正規化值: {normalized_distance:.4f}")
        logger.debug(f"📊 最終映射距離: {final_distance:.4f}")
        
        # 根據實際效果記錄
        if abs(improvement_percent) < 1:
            logger.debug(f"📊 正規化結果: {original_distance:.4f} → {final_distance:.4f} (微調: {improvement:+.4f})")
        elif improvement > 0:
            logger.debug(f"📊 正規化結果: {original_distance:.4f} → {final_distance:.4f} (改善: {improvement:+.4f}, {improvement_percent:+.1f}%)")
        else:
            logger.debug(f"📊 正規化結果: {original_distance:.4f} → {final_distance:.4f} (調整: {improvement:+.4f}, {improvement_percent:+.1f}%)")
        
        return final_distance
    
    def _get_impostor_embeddings(self, target_id: str) -> List[np.ndarray]:
        """
        獲取 impostor 語者的嵌入向量（用於 T-Norm）
        
        邏輯說明：
        1. 直接從專門的 CohortVoicePrint collection 獲取
        2. cohort 資料庫本身就不包含目標語者，無需過濾
        
        Args:
            target_id: 目標語者ID
            
        Returns:
            List[np.ndarray]: impostor 嵌入向量列表
        """
        if not self.client:
            logger.warning("資料庫客戶端未設定，無法獲取 impostor 嵌入向量")
            return []
            
        try:
            impostor_embeddings = []
            
            # 檢查 cohort 資料庫是否存在
            if not self.client.collections.exists(AS_NORM_COHORT_COLLECTION):
                logger.error(f"❌ Cohort 資料庫 '{AS_NORM_COHORT_COLLECTION}' 不存在，無法獲取 impostor 嵌入向量")
                return []
            
            logger.debug(f"🎯 使用專門的 cohort 資料庫: {AS_NORM_COHORT_COLLECTION}")
            collection = self.client.collections.get(AS_NORM_COHORT_COLLECTION)
            
            # 從 cohort 資料庫獲取，無需過濾（cohort 本身就不包含目標語者）
            results = collection.query.fetch_objects(
                include_vector=True,
                limit=self.top_k
            )
            
            for obj in results.objects:
                if obj.vector:
                    # 處理 named vector
                    vec_dict = obj.vector
                    raw_vec = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict
                    embedding = np.array(raw_vec, dtype=float)
                    impostor_embeddings.append(embedding)
            
            logger.debug(f"✅ 從 cohort 資料庫獲取了 {len(impostor_embeddings)} 個 impostor 嵌入向量（目標: {self.top_k}）")
            
            if len(impostor_embeddings) == 0:
                logger.warning("⚠️ 未能從 cohort 資料庫獲取任何 impostor 嵌入向量")
            elif len(impostor_embeddings) < self.top_k:
                logger.warning(f"⚠️ cohort 資料庫嵌入向量數量不足：獲取 {len(impostor_embeddings)} 個，目標 {self.top_k} 個")
            
            return impostor_embeddings
            
        except Exception as e:
            logger.warning(f"獲取 impostor 嵌入向量時發生錯誤: {e}")
            return []
    
    def _get_all_speaker_embeddings(self) -> List[np.ndarray]:
        """
        獲取背景模型嵌入向量（用於 Z-Norm）
        
        邏輯說明：
        1. 直接使用專門的 cohort 資料庫來保持統計穩定性
        2. Z-Norm 需要足夠的統計樣本（使用 cohort_size 限制）
        
        Returns:
            List[np.ndarray]: 背景模型嵌入向量列表
        """
        if not self.client:
            logger.warning("資料庫客戶端未設定，無法獲取背景模型嵌入向量")
            return []
            
        try:
            all_embeddings = []
            
            # 檢查 cohort 資料庫是否存在
            if not self.client.collections.exists(AS_NORM_COHORT_COLLECTION):
                logger.error(f"❌ Cohort 資料庫 '{AS_NORM_COHORT_COLLECTION}' 不存在，無法獲取背景模型嵌入向量")
                return []
            
            logger.debug(f"🎯 使用專門的 cohort 資料庫進行 Z-Norm: {AS_NORM_COHORT_COLLECTION}")
            collection = self.client.collections.get(AS_NORM_COHORT_COLLECTION)
            
            results = collection.query.fetch_objects(
                include_vector=True,
                limit=self.cohort_size  # Z-Norm 使用 cohort_size 而非 top_k
            )
            
            for obj in results.objects:
                if obj.vector:
                    # 處理 named vector
                    vec_dict = obj.vector
                    raw_vec = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict
                    embedding = np.array(raw_vec, dtype=float)
                    all_embeddings.append(embedding)
            
            logger.debug(f"✅ 從 cohort 資料庫獲取了 {len(all_embeddings)} 個背景模型嵌入向量（目標: {self.cohort_size}）")
            
            if len(all_embeddings) == 0:
                logger.warning("⚠️ 未能從 cohort 資料庫獲取任何背景模型嵌入向量")
            elif len(all_embeddings) < self.cohort_size:
                logger.warning(f"⚠️ cohort 資料庫嵌入向量數量不足：獲取 {len(all_embeddings)} 個，目標 {self.cohort_size} 個")
            
            return all_embeddings
            
        except Exception as e:
            logger.warning(f"獲取背景模型嵌入向量時發生錯誤: {e}")
            return []


class CohortDatabaseManager:
    """AS-Norm Cohort 資料庫管理器"""
    
    def __init__(self, model_name: str = None, model_type: str = "pyannote") -> None:
        """
        初始化 Cohort 資料庫管理器
        
        Args:
            model_name: 聲紋提取模型名稱（將被 model_type 覆蓋）
            model_type: 模型類型，可選值: "speechbrain" 或 "pyannote"
        """
        # ====== 這裡改模型類型 ======
        self.model_type = model_type
        # =========================
        
        if self.model_type == "speechbrain":
            self.model_name = SPEECHBRAIN_SPEAKER_MODEL
        elif self.model_type == "pyannote":
            self.model_name = PYANNOTE_SPEAKER_MODEL
        else:
            raise ValueError(f"不支援的模型類型: {self.model_type}")
            
        self.client = None
        self.speaker_model = None
        self._connect_database()
        self._init_speaker_model()
    
    def _connect_database(self) -> None:
        """連接到 Weaviate 資料庫"""
        try:
            self.client = weaviate.connect_to_local()
            logger.info("🔗 成功連接到 Weaviate 資料庫")
        except Exception as e:
            logger.error(f"❌ 連接到 Weaviate 失敗: {e}")
            raise
    
    def _init_speaker_model(self) -> None:
        """初始化聲紋提取模型（支援 speechbrain 和 pyannote）"""
        try:
            logger.info(f"🔧 正在載入聲紋提取模型: {self.model_name}")
            logger.info(f"🎯 模型類型: {self.model_type}")
            
            # 設定設備
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"🔧 使用設備: {self.device}")
            
            if self.model_type == "speechbrain":
                # 載入 SpeechBrain 語者辨識模型
                model_save_dir = get_model_save_dir("speechbrain_recognition")
                os.makedirs(model_save_dir, exist_ok=True)
                
                self.speaker_model = SpeakerRecognition.from_hparams(
                    source=self.model_name,
                    savedir=model_save_dir,
                    use_auth_token=HF_ACCESS_TOKEN
                )
                logger.info("✅ SpeechBrain ECAPA-TDNN 模型載入成功 (192維)")
                
            elif self.model_type == "pyannote":
                # 載入 pyannote 語者嵌入模型
                self.speaker_model = Inference(
                    self.model_name, 
                    window="whole",
                    device=self.device,
                    use_auth_token=HF_ACCESS_TOKEN
                )
                self.Segment = Segment  # 保存 Segment 類別以便後續使用
                logger.info("✅ pyannote/embedding 模型載入成功 (512維)")
            
        except Exception as e:
            logger.error(f"❌ 載入聲紋提取模型失敗: {e}")
            raise
    
    def initialize_cohort_collection(self) -> bool:
        """
        初始化 cohort collection
        
        Returns:
            bool: 是否成功建立或確認 collection 存在
        """
        if not self.client:
            logger.error("❌ 資料庫客戶端未連接")
            return False
        
        try:
            logger.info(f"🏗️  正在初始化 cohort collection: {AS_NORM_COHORT_COLLECTION}")
            
            # 檢查是否已存在
            if self.client.collections.exists(AS_NORM_COHORT_COLLECTION):
                logger.info(f"✅ Cohort collection '{AS_NORM_COHORT_COLLECTION}' 已存在")
                return True
            
            # 建立 AS-Norm 專用的 Cohort VoicePrint 集合
            # 這個集合存放不會在實際辨識中出現的背景語音資料
            cohort_collection = self.client.collections.create(
                name=AS_NORM_COHORT_COLLECTION,
                properties=[
                    wcc.Property(name="create_time", data_type=wcc.DataType.DATE),
                    wcc.Property(name="cohort_id", data_type=wcc.DataType.TEXT),  # 背景模型識別碼
                    wcc.Property(name="source_dataset", data_type=wcc.DataType.TEXT),  # 來源資料集
                    wcc.Property(name="gender", data_type=wcc.DataType.TEXT),  # 性別（可選）
                    wcc.Property(name="language", data_type=wcc.DataType.TEXT),  # 語言（可選）
                    wcc.Property(name="description", data_type=wcc.DataType.TEXT),  # 描述
                ],
                vectorizer_config=wcc.Configure.Vectorizer.none(),
                vector_index_config=wcc.Configure.VectorIndex.hnsw(
                    distance_metric=wcc.VectorDistances.COSINE
                )
            )
            logger.info(f"✅ 成功建立 cohort collection '{AS_NORM_COHORT_COLLECTION}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ 初始化 cohort collection 時發生錯誤: {e}")
            return False
    
    def reset_cohort_collection(self, force: bool = False) -> bool:
        """
        重置 cohort collection（刪除所有資料並重新建立）
        
        Args:
            force: 是否強制重置，若為 False 會先確認資料庫狀態
        
        Returns:
            bool: 是否成功重置
        """
        try:
            logger.info(f"🗑️  正在重置 cohort collection: {AS_NORM_COHORT_COLLECTION}")
            
            # 如果不是強制模式，先檢查現有資料
            if not force and self.client.collections.exists(AS_NORM_COHORT_COLLECTION):
                stats = self.get_cohort_statistics()
                current_count = stats.get('total_count', 0)
                if current_count > 0:
                    logger.warning(f"⚠️  當前 cohort 資料庫包含 {current_count} 筆資料")
                    logger.warning(f"⚠️  重置操作將刪除所有現有資料")
                    logger.info(f"💡 如需強制重置，請設定 force=True")
                    return False
                else:
                    logger.info(f"📊 當前 cohort 資料庫為空，繼續重置操作")
            
            # 記錄重置時間
            reset_time = get_taipei_time()
            logger.info(f"🕐 重置時間: {reset_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # 刪除現有 collection
            if self.client.collections.exists(AS_NORM_COHORT_COLLECTION):
                self.client.collections.delete(AS_NORM_COHORT_COLLECTION)
                logger.info(f"🗑️  已刪除現有的 cohort collection")
            else:
                logger.info(f"ℹ️  Cohort collection 不存在，直接建立新的")
            
            # 重新建立
            success = self.initialize_cohort_collection()
            if success:
                logger.info(f"✅ 成功重置 cohort collection")
                
                # 驗證重置結果
                final_stats = self.get_cohort_statistics()
                logger.info(f"📊 重置後狀態: 總計 {final_stats.get('total_count', 0)} 筆資料")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ 重置 cohort collection 時發生錯誤: {e}")
            return False
    
    def resample_audio(self, signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        使用 scipy 進行高品質重新採樣（與 VID_identify_v5.py 一致）
        
        Args:
            signal: 音頻信號
            orig_sr: 原始採樣率
            target_sr: 目標採樣率
            
        Returns:
            np.ndarray: 重新採樣後的音頻信號
        """
        return resample_poly(signal, target_sr, orig_sr)
    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        從音頻檔案提取聲紋嵌入向量（支援 speechbrain 和 pyannote）
        與 VID_identify_v5.py 保持完全一致的實作
        
        Args:
            audio_path: 音頻檔案路徑
            
        Returns:
            Optional[np.ndarray]: 聲紋嵌入向量，失敗時返回 None
        """
        try:
            # 載入音頻檔案（與 VID_identify_v5.py 一致的方式）
            waveform, sample_rate = librosa.load(audio_path, sr=None)  # 保持原始採樣率
            
            # 處理立體聲轉單聲道（與 VID_identify_v5.py 一致）
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            
            # 重新採樣到目標採樣率（與 VID_identify_v5.py 一致）
            target_sr = AUDIO_TARGET_RATE
            if sample_rate != target_sr:
                waveform = self.resample_audio(waveform, sample_rate, target_sr)
            
            # 檢查音頻長度（至少需要 1 秒）
            min_length = target_sr  # 1 秒
            if len(waveform) < min_length:
                logger.warning(f"⚠️  音頻檔案太短，跳過: {audio_path} ({len(waveform)/target_sr:.2f}s)")
                return None
            
            # 根據模型類型提取嵌入向量
            if self.model_type == "speechbrain":
                # SpeechBrain 模型處理
                waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(self.device)
                embedding = self.speaker_model.encode_batch(waveform_tensor)
                
                # 轉換為 numpy array 並正規化
                embedding_np = embedding.squeeze().cpu().numpy()
                embedding_np = embedding_np / np.linalg.norm(embedding_np)  # L2 正規化
                
            elif self.model_type == "pyannote":
                # pyannote 模型處理（使用臨時檔案方式）
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    # 將信號寫入臨時文件
                    sf.write(temp_path, waveform, target_sr)
                
                try:
                    # 整個音頻模式：使用 crop 方法
                    duration = len(waveform) / target_sr
                    segment = self.Segment(0, duration)
                    embedding = self.speaker_model.crop(temp_path, segment)
                    
                    # 轉換為 numpy array 並正規化
                    embedding_np = embedding.squeeze()  # 移除第一維
                    embedding_np = embedding_np / np.linalg.norm(embedding_np)  # L2 正規化
                    
                finally:
                    # 清理臨時文件
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            return embedding_np
            
        except Exception as e:
            logger.warning(f"⚠️  提取聲紋失敗: {audio_path} - {e}")
            return None
    
    def split_audio(self, audio_path: str, chunk_length: float = 6.0, 
                   overlap: float = 0.5) -> List[Tuple[np.ndarray, float, float]]:
        """
        將音頻檔案切片
        
        Args:
            audio_path: 音頻檔案路徑
            chunk_length: 切片長度（秒）
            overlap: 重疊比例（0-1）
            
        Returns:
            List[Tuple[np.ndarray, float, float]]: (音頻片段, 開始時間, 結束時間) 列表
        """
        chunks = []
        
        try:
            # 載入音頻檔案（與 VID_identify_v5.py 一致）
            waveform, sample_rate = librosa.load(audio_path, sr=None)  # 保持原始採樣率
            
            # 處理立體聲轉單聲道（與 VID_identify_v5.py 一致）
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            
            # 重新採樣到目標採樣率（與 VID_identify_v5.py 一致）
            target_sr = AUDIO_TARGET_RATE
            if sample_rate != target_sr:
                waveform = self.resample_audio(waveform, sample_rate, target_sr)
                sample_rate = target_sr  # 更新採樣率變數
            
            audio_length = len(waveform) / sample_rate
            
            # 計算切片參數
            chunk_samples = int(chunk_length * sample_rate)
            step_samples = int(chunk_samples * (1 - overlap))
            
            # 如果音頻太短，直接返回整個音頻
            if audio_length < chunk_length:
                chunks.append((waveform, 0.0, audio_length))
                return chunks
            
            # 切片處理
            start_sample = 0
            chunk_id = 0
            
            while start_sample + chunk_samples <= len(waveform):
                end_sample = start_sample + chunk_samples
                chunk = waveform[start_sample:end_sample]
                
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                
                chunks.append((chunk, start_time, end_time))
                
                start_sample += step_samples
                chunk_id += 1
            
            logger.debug(f"🔪 音頻切片完成: {audio_path} -> {len(chunks)} 個片段")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ 音頻切片失敗: {audio_path} - {e}")
            return []
    
    def import_audio_file(self, audio_path: str, source_dataset: str = None,
                         metadata: Dict[str, Any] = None) -> int:
        """
        導入單個音頻檔案到 cohort 資料庫（直接處理整個音檔，不切片）
        
        Args:
            audio_path: 音頻檔案路徑
            source_dataset: 來源資料集名稱，若為 None 則使用檔名（不含副檔名）
            metadata: 額外的元數據
            
        Returns:
            int: 成功導入的聲紋數量（0 或 1）
        """
        if not self.client.collections.exists(AS_NORM_COHORT_COLLECTION):
            logger.error(f"❌ Cohort collection '{AS_NORM_COHORT_COLLECTION}' 不存在，請先初始化")
            return 0
        
        file_name = Path(audio_path).stem
        
        # 如果沒有指定 source_dataset，使用檔名
        if source_dataset is None:
            source_dataset = file_name
        
        try:
            # 直接提取整個音檔的嵌入向量（不切片）
            embedding_np = self.extract_embedding(audio_path)
            
            if embedding_np is None:
                logger.warning(f"⚠️  無法提取嵌入向量: {audio_path}")
                return 0
            
            collection = self.client.collections.get(AS_NORM_COHORT_COLLECTION)
            
            # 準備元數據
            properties = {
                "create_time": get_taipei_time(),
                "cohort_id": file_name,  # 使用檔名作為 cohort_id
                "source_dataset": source_dataset,  # 使用檔名或指定的 source_dataset
                "gender": metadata.get("gender", "unknown") if metadata else "unknown",
                "language": metadata.get("language", "zh") if metadata else "zh",
                "description": f"完整音檔: {file_name}"
            }
            
            # 如果有額外元數據，合併進去
            if metadata:
                for key, value in metadata.items():
                    if key not in properties:
                        properties[key] = value
            
            # 插入到資料庫
            collection.data.insert(
                properties=properties,
                vector=embedding_np.tolist()
            )
            
            logger.info(f"✅ 成功導入聲紋: {audio_path} -> {source_dataset}")
            return 1
            
        except Exception as e:
            logger.error(f"❌ 導入音頻檔案失敗: {audio_path} - {e}")
            return 0
    
    def import_audio_folder(self, folder_path: str, source_dataset_prefix: str = None,
                           audio_extensions: List[str] = None,
                           metadata: Dict[str, Any] = None) -> Dict[str, int]:
        """
        從資料夾批量導入音頻檔案到 cohort 資料庫（直接處理整個音檔，不切片）
        
        Args:
            folder_path: 音頻資料夾路徑
            source_dataset_prefix: 來源資料集前綴，若為 None 則直接使用檔名
            audio_extensions: 支援的音頻副檔名
            metadata: 全域元數據
            
        Returns:
            Dict[str, int]: 導入結果統計
        """
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        
        folder_path = Path(folder_path)
        if not folder_path.exists():
            logger.error(f"❌ 資料夾不存在: {folder_path}")
            return {"total_files": 0, "success_files": 0, "total_embeddings": 0}
        
        logger.info(f"📁 開始批量導入音頻資料夾: {folder_path}")
        
        # 確保 cohort collection 存在
        if not self.initialize_cohort_collection():
            logger.error("❌ 無法初始化 cohort collection")
            return {"total_files": 0, "success_files": 0, "total_embeddings": 0}
        
        # 搜尋音頻檔案
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(folder_path.rglob(f"*{ext}"))
            audio_files.extend(folder_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"📊 找到 {len(audio_files)} 個音頻檔案")
        
        # 批量處理
        total_files = len(audio_files)
        success_files = 0
        total_embeddings = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"🎵 處理音頻檔案 ({i}/{total_files}): {audio_file.name}")
            
            # 為每個檔案準備元數據
            file_metadata = metadata.copy() if metadata else {}
            file_metadata.update({
                "original_file": audio_file.name,
                "file_path": str(audio_file.relative_to(folder_path))
            })
            
            # 決定 source_dataset 名稱
            if source_dataset_prefix:
                source_dataset = f"{source_dataset_prefix}_{audio_file.stem}"
            else:
                source_dataset = audio_file.stem  # 直接使用檔名
            
            # 導入檔案（不切片）
            embeddings_count = self.import_audio_file(
                str(audio_file), source_dataset, file_metadata
            )
            
            if embeddings_count > 0:
                success_files += 1
                total_embeddings += embeddings_count
        
        # 統計結果
        results = {
            "total_files": total_files,
            "success_files": success_files,
            "failed_files": total_files - success_files,
            "total_embeddings": total_embeddings
        }
        
        logger.info(f"📈 批量導入完成:")
        logger.info(f"   📁 總檔案數: {results['total_files']}")
        logger.info(f"   ✅ 成功檔案: {results['success_files']}")
        logger.info(f"   ❌ 失敗檔案: {results['failed_files']}")
        logger.info(f"   🎯 總聲紋數: {results['total_embeddings']}")
        
        return results
    
    def get_cohort_statistics(self) -> Dict[str, Any]:
        """
        獲取 cohort 資料庫統計信息
        
        Returns:
            Dict[str, Any]: 統計信息
        """
        try:
            if not self.client.collections.exists(AS_NORM_COHORT_COLLECTION):
                return {"exists": False, "count": 0}
            
            collection = self.client.collections.get(AS_NORM_COHORT_COLLECTION)
            
            # 獲取總數
            aggregate_result = collection.aggregate.over_all(total_count=True)
            total_count = aggregate_result.total_count if aggregate_result.total_count else 0
            
            # 獲取樣本數據來分析
            sample_results = collection.query.fetch_objects(
                limit=min(100, total_count),
                return_properties=["source_dataset", "gender", "language", "create_time"]
            )
            
            # 統計分析
            source_datasets = {}
            genders = {}
            languages = {}
            creation_dates = []
            
            for obj in sample_results.objects:
                if obj.properties:
                    # 統計來源資料集
                    dataset = obj.properties.get("source_dataset", "unknown")
                    source_datasets[dataset] = source_datasets.get(dataset, 0) + 1
                    
                    # 統計性別
                    gender = obj.properties.get("gender", "unknown")
                    genders[gender] = genders.get(gender, 0) + 1
                    
                    # 統計語言
                    language = obj.properties.get("language", "unknown")
                    languages[language] = languages.get(language, 0) + 1
                    
                    # 收集建立時間
                    if obj.properties.get("create_time"):
                        creation_dates.append(obj.properties["create_time"])
            
            # 計算時間範圍
            time_range = {}
            if creation_dates:
                creation_dates.sort()
                time_range = {
                    "earliest": creation_dates[0].isoformat() if hasattr(creation_dates[0], 'isoformat') else str(creation_dates[0]),
                    "latest": creation_dates[-1].isoformat() if hasattr(creation_dates[-1], 'isoformat') else str(creation_dates[-1])
                }
            
            return {
                "exists": True,
                "total_count": total_count,
                "source_datasets": source_datasets,
                "genders": genders,
                "languages": languages,
                "time_range": time_range,
                "collection_name": AS_NORM_COHORT_COLLECTION
            }
            
        except Exception as e:
            logger.error(f"❌ 獲取 cohort 統計信息時發生錯誤: {e}")
            return {"exists": False, "count": 0, "error": str(e)}
    
    def export_cohort_info(self, output_file: str = None) -> str:
        """
        匯出 cohort 資料庫信息到檔案
        
        Args:
            output_file: 輸出檔案路徑，若為 None 則自動生成
            
        Returns:
            str: 輸出檔案路徑
        """
        if output_file is None:
            timestamp = get_taipei_time().strftime("%Y%m%d_%H%M%S")
            output_file = f"cohort_info_{timestamp}.json"
        
        try:
            import json
            
            stats = self.get_cohort_statistics()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"📄 Cohort 信息已匯出到: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"❌ 匯出 cohort 信息失敗: {e}")
            return ""
    
    def create_as_norm_processor(self) -> ASNormProcessor:
        """
        創建 AS-Norm 處理器實例
        
        Returns:
            ASNormProcessor: 已配置的 AS-Norm 處理器
        """
        processor = ASNormProcessor(self.client)
        return processor
    
    def close(self) -> None:
        """關閉連接並清理資源"""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                logger.info("🔌 已關閉 Weaviate 連接")
        except Exception as e:
            logger.warning(f"⚠️  關閉連接時發生錯誤: {e}")


def main():
    """命令列介面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AS-Norm Cohort 資料庫管理工具")
    parser.add_argument("--action", choices=["init", "reset", "import", "stats", "export"], 
                       default="stats", help="執行的動作")
    parser.add_argument("--folder", type=str, help="要導入的音頻資料夾路徑")
    parser.add_argument("--dataset-prefix", type=str, 
                       help="來源資料集前綴，若不指定則直接使用檔名")
    parser.add_argument("--gender", type=str, help="語者性別")
    parser.add_argument("--language", type=str, default="zh", help="語音語言")
    parser.add_argument("--output", type=str, help="匯出檔案路徑")
    parser.add_argument("--force", action="store_true", 
                       help="強制執行重置操作（忽略資料確認）")
    
    args = parser.parse_args()
    
    manager = CohortDatabaseManager()
    
    try:
        if args.action == "init":
            print("🔧 正在初始化 cohort collection...")
            success = manager.initialize_cohort_collection()
            print(f"✅ 初始化{'成功' if success else '失敗'}")
            
        elif args.action == "reset":
            print("🗑️  正在重置 cohort collection...")
            success = manager.reset_cohort_collection(force=args.force)
            print(f"✅ 重置{'成功' if success else '失敗'}")
            if not success and not args.force:
                print("💡 提示：如需強制重置，請加上 --force 參數")
            
        elif args.action == "import":
            if not args.folder:
                print("❌ 請指定要導入的音頻資料夾路徑 (--folder)")
                return
            
            # 準備元數據
            metadata = {}
            if args.gender:
                metadata["gender"] = args.gender
            if args.language:
                metadata["language"] = args.language
            
            print(f"📁 正在導入音頻資料夾: {args.folder}")
            results = manager.import_audio_folder(
                args.folder, args.dataset_prefix, metadata=metadata
            )
            
            print(f"📈 導入完成:")
            for key, value in results.items():
                print(f"   {key}: {value}")
                
        elif args.action == "stats":
            print("📊 正在獲取 cohort 統計信息...")
            stats = manager.get_cohort_statistics()
            print(f"📈 統計結果:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
                
        elif args.action == "export":
            print("📄 正在匯出 cohort 信息...")
            output_file = manager.export_cohort_info(args.output)
            if output_file:
                print(f"✅ 信息已匯出到: {output_file}")
            
    finally:
        manager.close()


if __name__ == "__main__":
    main()
