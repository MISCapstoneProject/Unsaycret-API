import torch
from utils.logger import get_logger
logger = get_logger(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🗂️ Speaker ID 模型運行裝置: {device}")
