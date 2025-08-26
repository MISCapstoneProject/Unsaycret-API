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
- 聲紋提取：SpeechBrain ECAPA-TDNN 模型
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

2. 從音頻資料夾導入 cohort：
   ```python
   manager.import_audio_folder("/path/to/cohort/audio", 
                              chunk_length=3.0, overlap=0.5)
   ```

3. 重置 cohort 資料庫：
   ```python
   manager.reset_cohort_collection()
   ```
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

# 添加模組路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import weaviate
import torch
from speechbrain.inference import SpeakerRecognition
from scipy.spatial.distance import cosine
import weaviate.classes as wc
from weaviate.classes.query import Filter

# 導入項目模組
from utils.logger import get_logger
from utils.env_config import get_model_save_dir, HF_ACCESS_TOKEN
from utils.constants import (
    AS_NORM_COHORT_COLLECTION, SPEECHBRAIN_SPEAKER_MODEL,
    AUDIO_TARGET_RATE, AUDIO_SAMPLE_RATE,
    ENABLE_AS_NORM, AS_NORM_COHORT_SIZE, AS_NORM_TOP_K, AS_NORM_ALPHA,
    ENABLE_T_NORM, ENABLE_Z_NORM, ENABLE_S_NORM, AS_NORM_USE_DEDICATED_COHORT
)
from utils.init_collections import WeaviateCollectionManager

# 創建模組專屬日誌器
logger = get_logger(__name__)


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
        計算 T-Norm 正規化分數
        
        T-Norm 通過使用 impostor 模型分數來正規化目標分數
        公式: (score - mean_impostor) / std_impostor
        
        Args:
            test_embedding: 測試音訊的嵌入向量
            target_embedding: 目標語者的嵌入向量  
            impostor_embeddings: impostor 語者的嵌入向量列表
            
        Returns:
            float: T-Norm 正規化後的分數
        """
        if not impostor_embeddings:
            # 沒有 impostor 時，返回原始餘弦距離
            return cosine(test_embedding, target_embedding)
            
        # 計算目標分數（餘弦距離）
        target_score = cosine(test_embedding, target_embedding)
        
        # 計算 impostor 分數
        impostor_scores = []
        for imp_embedding in impostor_embeddings:
            imp_score = cosine(test_embedding, imp_embedding)
            impostor_scores.append(imp_score)
            
        # 計算 impostor 分數的統計量
        mean_impostor = np.mean(impostor_scores)
        std_impostor = np.std(impostor_scores)
        
        # T-Norm 正規化
        if std_impostor > 0:
            t_norm_score = (target_score - mean_impostor) / std_impostor
        else:
            t_norm_score = target_score
            
        return t_norm_score
    
    def compute_z_norm_score(self, test_embedding: np.ndarray, target_embedding: np.ndarray) -> float:
        """
        計算 Z-Norm 正規化分數
        
        Z-Norm 使用測試語音對所有已知語者的統計分布進行正規化
        公式: (score - mean_all) / std_all
        
        Args:
            test_embedding: 測試音訊的嵌入向量
            target_embedding: 目標語者的嵌入向量
            
        Returns:
            float: Z-Norm 正規化後的分數
        """
        # 計算目標分數
        target_score = cosine(test_embedding, target_embedding)
        
        # 獲取所有語者的嵌入向量
        all_embeddings = self._get_all_speaker_embeddings()
        
        if not all_embeddings:
            return target_score
            
        # 計算對所有語者的分數
        all_scores = []
        for embedding in all_embeddings:
            score = cosine(test_embedding, embedding)
            all_scores.append(score)
            
        # Z-Norm 正規化
        mean_all = np.mean(all_scores)
        std_all = np.std(all_scores)
        
        if std_all > 0:
            z_norm_score = (target_score - mean_all) / std_all
        else:
            z_norm_score = target_score
            
        return z_norm_score
    
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
        應用 AS-Norm 處理
        
        根據配置選擇性地應用不同的正規化方法
        
        Args:
            test_embedding: 測試音訊的嵌入向量
            target_embedding: 目標語者的嵌入向量
            target_id: 目標語者ID
            
        Returns:
            float: 正規化後的分數
        """
        if not ENABLE_AS_NORM:
            # AS-Norm 關閉時，返回原始餘弦距離
            return cosine(test_embedding, target_embedding)
            
        # 獲取 impostor 嵌入向量
        impostor_embeddings = self._get_impostor_embeddings(target_id)
        
        # 根據配置選擇正規化方法
        if ENABLE_S_NORM and ENABLE_T_NORM and ENABLE_Z_NORM:
            # 完整 S-Norm
            return self.compute_s_norm_score(test_embedding, target_embedding, impostor_embeddings)
        elif ENABLE_T_NORM and ENABLE_Z_NORM:
            # T-Norm + Z-Norm 組合
            t_score = self.compute_t_norm_score(test_embedding, target_embedding, impostor_embeddings)
            z_score = self.compute_z_norm_score(test_embedding, target_embedding)
            return 0.5 * t_score + 0.5 * z_score
        elif ENABLE_T_NORM:
            # 僅 T-Norm
            return self.compute_t_norm_score(test_embedding, target_embedding, impostor_embeddings)
        elif ENABLE_Z_NORM:
            # 僅 Z-Norm
            return self.compute_z_norm_score(test_embedding, target_embedding)
        else:
            # 所有正規化都關閉，返回原始分數
            return cosine(test_embedding, target_embedding)
    
    def _get_impostor_embeddings(self, target_id: str) -> List[np.ndarray]:
        """
        獲取 impostor 語者的嵌入向量（用於 T-Norm）
        
        如果啟用專門的cohort資料庫，則從CohortVoicePrint collection獲取
        否則從主要的VoicePrint collection中排除目標語者後獲取
        
        Args:
            target_id: 目標語者ID
            
        Returns:
            List[np.ndarray]: impostor 嵌入向量列表
        """
        if not self.client:
            logger.warning("資料庫客戶端未設定，無法獲取 impostor 嵌入向量")
            return []
            
        try:
            # 根據配置選擇資料來源
            if AS_NORM_USE_DEDICATED_COHORT:
                collection_name = AS_NORM_COHORT_COLLECTION
                # 檢查專門的cohort collection是否存在
                if not self.client.collections.exists(collection_name):
                    logger.warning(f"專門的cohort collection '{collection_name}' 不存在，回退到主資料庫")
                    collection_name = "VoicePrint"
                    use_where_filter = True
                else:
                    use_where_filter = False  # cohort資料庫中沒有目標語者，不需要過濾
            else:
                collection_name = "VoicePrint"
                use_where_filter = True
            
            collection = self.client.collections.get(collection_name)
            
            # 根據是否需要過濾目標語者來構建查詢
            if use_where_filter:
                results = collection.query.fetch_objects(
                    where=Filter.by_property("speaker_name").not_equal(target_id),
                    return_properties=["speaker_name"],
                    include_vector=True,
                    limit=self.top_k  # 直接查詢 top_k 數量，避免不必要的資料傳輸
                )
            else:
                # 從專門的cohort資料庫獲取，不需要過濾
                results = collection.query.fetch_objects(
                    include_vector=True,
                    limit=self.top_k
                )
            
            impostor_embeddings = []
            for obj in results.objects:
                if obj.vector:
                    # 處理 named vector
                    vec_dict = obj.vector
                    raw_vec = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict
                    embedding = np.array(raw_vec, dtype=float)
                    impostor_embeddings.append(embedding)
            
            logger.debug(f"從 {collection_name} 獲取了 {len(impostor_embeddings)} 個 impostor 嵌入向量")
            return impostor_embeddings  # 已經限制在 top_k 數量內
            
        except Exception as e:
            logger.warning(f"獲取 impostor 嵌入向量時發生錯誤: {e}")
            return []
    
    def _get_all_speaker_embeddings(self) -> List[np.ndarray]:
        """
        獲取背景模型嵌入向量（用於 Z-Norm）
        
        如果啟用專門的cohort資料庫，則從CohortVoicePrint collection獲取
        否則從主要的VoicePrint collection獲取
        
        Returns:
            List[np.ndarray]: 背景模型嵌入向量列表
        """
        if not self.client:
            logger.warning("資料庫客戶端未設定，無法獲取背景模型嵌入向量")
            return []
            
        try:
            # 根據配置選擇資料來源
            if AS_NORM_USE_DEDICATED_COHORT:
                collection_name = AS_NORM_COHORT_COLLECTION
                # 檢查專門的cohort collection是否存在
                if not self.client.collections.exists(collection_name):
                    logger.warning(f"專門的cohort collection '{collection_name}' 不存在，回退到主資料庫")
                    collection_name = "VoicePrint"
            else:
                collection_name = "VoicePrint"
            
            collection = self.client.collections.get(collection_name)
            
            results = collection.query.fetch_objects(
                include_vector=True,
                limit=self.cohort_size  # Z-Norm 需要更多樣本來計算統計量
            )
            
            all_embeddings = []
            for obj in results.objects:
                if obj.vector:
                    # 處理 named vector
                    vec_dict = obj.vector
                    raw_vec = vec_dict["default"] if isinstance(vec_dict, dict) else vec_dict
                    embedding = np.array(raw_vec, dtype=float)
                    all_embeddings.append(embedding)
            
            logger.debug(f"從 {collection_name} 獲取了 {len(all_embeddings)} 個背景模型嵌入向量")
            return all_embeddings
            
        except Exception as e:
            logger.warning(f"獲取背景模型嵌入向量時發生錯誤: {e}")
            return []


class CohortDatabaseManager:
    """AS-Norm Cohort 資料庫管理器"""
    
    def __init__(self, model_name: str = None) -> None:
        """
        初始化 Cohort 資料庫管理器
        
        Args:
            model_name: 聲紋提取模型名稱，預設使用 SPEECHBRAIN_SPEAKER_MODEL
        """
        self.model_name = model_name or SPEECHBRAIN_SPEAKER_MODEL
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
        """初始化聲紋提取模型"""
        try:
            logger.info(f"🔧 正在載入聲紋提取模型: {self.model_name}")
            
            # 設定模型快取目錄
            model_save_dir = get_model_save_dir("speechbrain_recognition")
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 載入 SpeechBrain 語者辨識模型
            self.speaker_model = SpeakerRecognition.from_hparams(
                source=self.model_name,
                savedir=model_save_dir,
                use_auth_token=HF_ACCESS_TOKEN
            )
            logger.info("✅ 聲紋提取模型載入成功")
            
        except Exception as e:
            logger.error(f"❌ 載入聲紋提取模型失敗: {e}")
            raise
    
    def initialize_cohort_collection(self) -> bool:
        """
        初始化 cohort collection
        
        Returns:
            bool: 是否成功建立或確認 collection 存在
        """
        try:
            logger.info(f"🏗️  正在初始化 cohort collection: {AS_NORM_COHORT_COLLECTION}")
            
            manager = WeaviateCollectionManager()
            manager.connect()
            
            # 檢查是否已存在
            if manager.collection_exists(AS_NORM_COHORT_COLLECTION):
                logger.info(f"✅ Cohort collection '{AS_NORM_COHORT_COLLECTION}' 已存在")
                return True
            
            # 建立 cohort collection
            success = manager.create_cohort_voiceprint_collection()
            if success:
                logger.info(f"✅ 成功建立 cohort collection '{AS_NORM_COHORT_COLLECTION}'")
            else:
                logger.error(f"❌ 建立 cohort collection '{AS_NORM_COHORT_COLLECTION}' 失敗")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ 初始化 cohort collection 時發生錯誤: {e}")
            return False
    
    def reset_cohort_collection(self) -> bool:
        """
        重置 cohort collection（刪除所有資料並重新建立）
        
        Returns:
            bool: 是否成功重置
        """
        try:
            logger.info(f"🗑️  正在重置 cohort collection: {AS_NORM_COHORT_COLLECTION}")
            
            # 刪除現有 collection
            if self.client.collections.exists(AS_NORM_COHORT_COLLECTION):
                self.client.collections.delete(AS_NORM_COHORT_COLLECTION)
                logger.info(f"🗑️  已刪除現有的 cohort collection")
            
            # 重新建立
            success = self.initialize_cohort_collection()
            if success:
                logger.info(f"✅ 成功重置 cohort collection")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ 重置 cohort collection 時發生錯誤: {e}")
            return False
    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        從音頻檔案提取聲紋嵌入向量
        
        Args:
            audio_path: 音頻檔案路徑
            
        Returns:
            Optional[np.ndarray]: 聲紋嵌入向量，失敗時返回 None
        """
        try:
            # 載入音頻檔案
            waveform, sample_rate = librosa.load(audio_path, sr=AUDIO_TARGET_RATE)
            
            # 檢查音頻長度（至少需要 1 秒）
            min_length = AUDIO_TARGET_RATE  # 1 秒
            if len(waveform) < min_length:
                logger.warning(f"⚠️  音頻檔案太短，跳過: {audio_path} ({len(waveform)/sample_rate:.2f}s)")
                return None
            
            # 提取聲紋嵌入向量
            waveform_tensor = torch.tensor(waveform).unsqueeze(0)
            embedding = self.speaker_model.encode_batch(waveform_tensor)
            
            # 轉換為 numpy array 並正規化
            embedding_np = embedding.squeeze().cpu().numpy()
            embedding_np = embedding_np / np.linalg.norm(embedding_np)  # L2 正規化
            
            return embedding_np
            
        except Exception as e:
            logger.warning(f"⚠️  提取聲紋失敗: {audio_path} - {e}")
            return None
    
    def split_audio(self, audio_path: str, chunk_length: float = 3.0, 
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
            # 載入音頻檔案
            waveform, sample_rate = librosa.load(audio_path, sr=AUDIO_TARGET_RATE)
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
    
    def import_audio_file(self, audio_path: str, source_dataset: str = "cohort_import",
                         chunk_length: float = 3.0, overlap: float = 0.5,
                         metadata: Dict[str, Any] = None) -> int:
        """
        導入單個音頻檔案到 cohort 資料庫
        
        Args:
            audio_path: 音頻檔案路徑
            source_dataset: 來源資料集名稱
            chunk_length: 切片長度（秒）
            overlap: 重疊比例（0-1）
            metadata: 額外的元數據
            
        Returns:
            int: 成功導入的聲紋數量
        """
        if not self.client.collections.exists(AS_NORM_COHORT_COLLECTION):
            logger.error(f"❌ Cohort collection '{AS_NORM_COHORT_COLLECTION}' 不存在，請先初始化")
            return 0
        
        success_count = 0
        file_name = Path(audio_path).stem
        
        try:
            # 切片音頻
            chunks = self.split_audio(audio_path, chunk_length, overlap)
            
            if not chunks:
                logger.warning(f"⚠️  無法切片音頻檔案: {audio_path}")
                return 0
            
            collection = self.client.collections.get(AS_NORM_COHORT_COLLECTION)
            
            # 處理每個音頻片段
            for i, (chunk_audio, start_time, end_time) in enumerate(chunks):
                try:
                    # 提取聲紋嵌入向量
                    chunk_tensor = torch.tensor(chunk_audio).unsqueeze(0)
                    embedding = self.speaker_model.encode_batch(chunk_tensor)
                    embedding_np = embedding.squeeze().cpu().numpy()
                    embedding_np = embedding_np / np.linalg.norm(embedding_np)  # L2 正規化
                    
                    # 準備元數據
                    properties = {
                        "create_time": datetime.now(),
                        "cohort_id": f"{file_name}_chunk_{i:04d}",
                        "source_dataset": source_dataset,
                        "gender": metadata.get("gender", "unknown") if metadata else "unknown",
                        "language": metadata.get("language", "unknown") if metadata else "unknown",
                        "description": f"{file_name} 片段 {i+1}/{len(chunks)} ({start_time:.2f}s-{end_time:.2f}s)"
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
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️  處理音頻片段失敗: {audio_path} 片段 {i} - {e}")
                    continue
            
            logger.info(f"✅ 成功導入 {success_count}/{len(chunks)} 個聲紋: {audio_path}")
            return success_count
            
        except Exception as e:
            logger.error(f"❌ 導入音頻檔案失敗: {audio_path} - {e}")
            return 0
    
    def import_audio_folder(self, folder_path: str, source_dataset: str = "cohort_folder",
                           chunk_length: float = 3.0, overlap: float = 0.5,
                           audio_extensions: List[str] = None,
                           metadata: Dict[str, Any] = None) -> Dict[str, int]:
        """
        從資料夾批量導入音頻檔案到 cohort 資料庫
        
        Args:
            folder_path: 音頻資料夾路徑
            source_dataset: 來源資料集名稱
            chunk_length: 切片長度（秒）
            overlap: 重疊比例（0-1）
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
            
            # 導入檔案
            embeddings_count = self.import_audio_file(
                str(audio_file), source_dataset, chunk_length, overlap, file_metadata
            )
            
            if embeddings_count > 0:
                success_files += 1
                total_embeddings += embeddings_count
        
        # 統計結果
        results = {
            "total_files": total_files,
            "success_files": success_files,
            "failed_files": total_files - success_files,
            "total_embeddings": total_embeddings,
            "source_dataset": source_dataset
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    parser.add_argument("--dataset", type=str, default="cohort_import", 
                       help="來源資料集名稱")
    parser.add_argument("--chunk-length", type=float, default=3.0, 
                       help="音頻切片長度（秒）")
    parser.add_argument("--overlap", type=float, default=0.5, 
                       help="切片重疊比例（0-1）")
    parser.add_argument("--gender", type=str, help="語者性別")
    parser.add_argument("--language", type=str, default="zh-TW", help="語音語言")
    parser.add_argument("--output", type=str, help="匯出檔案路徑")
    
    args = parser.parse_args()
    
    manager = CohortDatabaseManager()
    
    try:
        if args.action == "init":
            print("🔧 正在初始化 cohort collection...")
            success = manager.initialize_cohort_collection()
            print(f"✅ 初始化{'成功' if success else '失敗'}")
            
        elif args.action == "reset":
            print("🗑️  正在重置 cohort collection...")
            success = manager.reset_cohort_collection()
            print(f"✅ 重置{'成功' if success else '失敗'}")
            
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
                args.folder, args.dataset, args.chunk_length, args.overlap, metadata=metadata
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
