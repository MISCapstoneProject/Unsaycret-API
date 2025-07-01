"""語者識別模組入口。"""

import torch
from utils.logger import get_logger

from .VID_identify_v5 import SpeakerIdentifier

logger = get_logger(__name__)

# 模型運行裝置資訊
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🗂️ Speaker ID 模型運行裝置: {device}")

__all__ = ["SpeakerIdentifier"]

