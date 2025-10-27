"""
===============================================================================
動態模型管理器 (Dynamic Model Manager)
===============================================================================

版本：v1.0.0
作者：EvanLo62
最後更新：2025-08-24

模組概要：
-----------
本模組提供動態語音分離模型管理功能，根據偵測到的語者數量自動選擇適當的模型。
支援模型快取、自動切換和資源管理，提供高效能的模型管理解決方案。

🎯 核心功能：
 • 動態模型選擇：根據語者數量自動切換模型
 • 模型快取機制：避免重複載入相同模型
 • 資源管理：自動清理GPU記憶體
 • 錯誤處理：完整的錯誤處理和日誌記錄

🔧 技術架構：
-----------
 模型載入    ：SpeechBrain SepFormer
 記憶體管理  ：PyTorch CUDA 記憶體管理
 快取策略    ：基於模型類型的快取機制

📊 支援模型：
-----------
 • SepFormer 2人語者分離模型（預訓練）
 • Tiger 2人語者分離模型（預訓練）
 • SepFormer 3人語者分離模型（自訓練）

===============================================================================
"""

import os
import torch
from speechbrain.inference import SepformerSeparation as separator
from enum import Enum
import modules.look2hear.model

# 導入日誌模組
from utils.logger import get_logger

# 導入常數
from utils.constants import AUDIO_SAMPLE_RATE, USE_TIGER

# 初始化日誌
logger = get_logger(__name__)

# 模型類型枚舉
class SeparationModel(Enum):
    SEPFORMER_2SPEAKER = "sepformer_2speaker"    # SepFormer 2人語者分離模型（預訓練）
    TIGER_2SPEAKER = "tiger_2speaker"          # Tiger 2人語者分離模型（預訓練）
    SEPFORMER_3SPEAKER = "sepformer_3speaker"    # SepFormer 3人語者分離模型（自訓練）

# 模型配置
MODEL_CONFIGS = {
    SeparationModel.SEPFORMER_2SPEAKER: {
        "model_name": "speechbrain/sepformer-whamr16k",
        "num_speakers": 2,
        "sample_rate": AUDIO_SAMPLE_RATE
    },
    SeparationModel.TIGER_2SPEAKER: {
        "num_speakers": 2,
        "model_name": "JusperLee/TIGER-speech",
        "sample_rate": AUDIO_SAMPLE_RATE
    },
    SeparationModel.SEPFORMER_3SPEAKER: {
        "model_name": "AlvinLo62/sepformer-tcc300-3spks-16k-noisy",
        "num_speakers": 3,
        "sample_rate": AUDIO_SAMPLE_RATE
    }
}

class DynamicModelManager:
    """動態模型管理器 - 根據語者數量自動選擇適當的模型"""
    
    def __init__(self, device: str):
        """
        初始化動態模型管理器
        
        Args:
            device: 計算設備 (cuda:0, cpu 等)
        """
        self.device = device
        self.loaded_models = {}  # 快取已載入的模型
        self.current_model_type = None
        logger.info("動態模型管理器初始化完成")
    
    def get_model_for_speakers(self, num_speakers: int) -> tuple[separator, SeparationModel]:
        """
        根據語者數量取得適當的模型
        
        Args:
            num_speakers: 偵測到的語者數量
            
        Returns:
            tuple: (模型實例, 模型類型)
        """
        # 決定使用哪個模型
        if num_speakers <= 2 and USE_TIGER:
            target_model_type = SeparationModel.TIGER_2SPEAKER
        elif num_speakers <= 2:
            target_model_type = SeparationModel.SEPFORMER_2SPEAKER
        else:
            target_model_type = SeparationModel.SEPFORMER_3SPEAKER
        
        # 檢查是否需要切換模型
        if target_model_type != self.current_model_type:
            logger.info(f"切換模型：{target_model_type.value} (語者數量: {num_speakers})")
            
            # 載入新模型（如果尚未快取）
            if target_model_type not in self.loaded_models:
                self._load_model(target_model_type)
            
            self.current_model_type = target_model_type
            
            # 清理 GPU 記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return self.loaded_models[target_model_type], target_model_type
    
    def _load_model(self, model_type: SeparationModel):
        """
        載入指定的模型
        
        Args:
            model_type: 要載入的模型類型
        """
        try:
            model_config = MODEL_CONFIGS[model_type]
            model_name = model_config["model_name"]
            
            logger.info(f"載入模型: {model_name}")
            
            # 檢查本地模型目錄
            local_model_path = os.path.abspath(f"models/{model_type.value}")
            
            # 檢查是否有無效的符號連結
            if os.path.exists(local_model_path):
                hyperparams_file = os.path.join(local_model_path, "hyperparams.yaml")
                if os.path.exists(hyperparams_file):
                    try:
                        with open(hyperparams_file, 'r', encoding='utf-8') as f:
                            content = f.read(100)
                    except (PermissionError, OSError, UnicodeDecodeError) as e:
                        logger.warning(f"本地模型檔案無法讀取，重新下載: {e}")
                        import shutil
                        shutil.rmtree(local_model_path, ignore_errors=True)
            
            # 載入模型 (無論本地路徑是否存在,都會嘗試載入)
            if model_type.value == "tiger_2speaker":
                # 使用 Look2Hear 載入 Tiger 模型
                logger.info("使用 Look2Hear 載入 Tiger 模型")
                model = modules.look2hear.model.TIGER.from_pretrained(
                    model_name,
                    cache_dir=local_model_path
                )
                model.to(self.device)
                model.eval()
            else:
                # 載入 SepFormer 模型
                logger.info(f"使用 SpeechBrain 載入模型: {model_name}")
                model = separator.from_hparams(
                    source=model_name,
                    savedir=local_model_path,
                    run_opts={"device": self.device}
                )
            
            # 測試模型
            with torch.no_grad():
                if model_type.value == "tiger_2speaker":
                    # Tiger 模型使用直接調用
                    test_audio = torch.randn(1, AUDIO_SAMPLE_RATE).to(self.device)
                    _ = model(test_audio)
                else:
                    # SepFormer 模型使用 separate_batch
                    test_audio = torch.randn(1, AUDIO_SAMPLE_RATE).to(self.device)
                    _ = model.separate_batch(test_audio)
            
            self.loaded_models[model_type] = model
            logger.info(f"模型 {model_type.value} 載入並測試完成")
            
        except Exception as e:
            logger.error(f"載入模型 {model_type.value} 失敗: {e}")
            raise
    
    def get_model_config(self, model_type: SeparationModel) -> dict:
        """
        取得模型配置資訊
        
        Args:
            model_type: 模型類型
            
        Returns:
            dict: 模型配置字典
        """
        return MODEL_CONFIGS[model_type]
    
    def preload_model(self, model_type: SeparationModel):
        """
        預載入指定模型
        
        Args:
            model_type: 要預載入的模型類型
        """
        if model_type not in self.loaded_models:
            logger.info(f"預載入模型: {model_type.value}")
            self._load_model(model_type)
        else:
            logger.info(f"模型 {model_type.value} 已經載入")
    
    def unload_model(self, model_type: SeparationModel):
        """
        卸載指定模型
        
        Args:
            model_type: 要卸載的模型類型
        """
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
            logger.info(f"已卸載模型: {model_type.value}")
            
            # 如果卸載的是當前模型，重置當前模型類型
            if self.current_model_type == model_type:
                self.current_model_type = None
            
            # 清理 GPU 記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_loaded_models(self) -> list[SeparationModel]:
        """
        取得已載入的模型列表
        
        Returns:
            list: 已載入的模型類型列表
        """
        return list(self.loaded_models.keys())
    
    def cleanup(self):
        """清理所有載入的模型"""
        self.loaded_models.clear()
        self.current_model_type = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("模型管理器已清理")

def create_dynamic_model_manager(device: str) -> DynamicModelManager:
    """
    建立動態模型管理器的便利函式
    
    Args:
        device: 計算設備
        
    Returns:
        DynamicModelManager: 動態模型管理器實例
    """
    return DynamicModelManager(device)

def get_available_models() -> dict[str, str]:
    """
    取得可用的模型列表
    
    Returns:
        dict: 模型名稱到描述的映射
    """
    return {
        "sepformer_2speaker": "SepFormer 2人語者分離模型（預訓練）",
        "tiger_2speaker": "Tiger 2人語者分離模型（預訓練）",
        "sepformer_3speaker": "SepFormer 3人語者分離模型（自訓練）"
    }

def get_model_configs() -> dict:
    """
    取得所有模型的配置資訊
    
    Returns:
        dict: 模型配置字典
    """
    return MODEL_CONFIGS
