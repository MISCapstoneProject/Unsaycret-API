# Unsaycret-API

**作者**: CYouuu  
**最後更新者**: CYouuu  
**版本**: v0.4.0 
**最後更新**: 2025-07-21

## 專案簡介
Unsaycret-API 是一套模組化的語音處理系統，整合語音分離、說者辨識、語音辨識、API 服務，並支援 Weaviate V2 向量資料庫串接。專案採用分層配置管理，將環境變數與演算法常數分離，提升可維護性和部署靈活性。

> **⚠️ V2 資料庫版本**: 本版本已升級至 Weaviate V2 資料庫結構，不相容於 V1 版本

## 🚀 主要功能

- 🎙 **語者分離**：採用 SpeechBrain Sepformer (2人) / ConvTasNet (3人)，支援多人語音分離
- 🧠 **語音辨識（ASR）**：Faster-Whisper，支援 GPU/CPU 動態切換，逐詞時間戳與信心值
- 🗣 **說話人辨識**：ECAPA-TDNN 語者聲紋比對，支援聲紋自動更新與雙 ID 系統（UUID + 序號ID）
- 🛜 **API 服務**：FastAPI 提供完整的 RESTful 與 WebSocket 介面
- 🧠 **Weaviate V2 整合**：語音向量與辨識結果存入 Weaviate V2，支援高效語者搜尋與比對
- ⚙️ **分層配置**：環境變數(.env) 與應用常數(constants.py) 分離管理

## 🏗 系統架構

### 配置系統 (v2.0)
```
├── .env                     # 環境相關配置 (不納入版控)
├── .env.example             # 配置範例檔案
├── CONFIG_README.md         # 配置說明文檔
├── utils/
│   ├── env_config.py        # 環境變數載入
│   └── constants.py         # 演算法常數定義
```

### 核心模組架構
```
Unsaycret-API/
├── api/                     # FastAPI 應用程式層 (重構自 services/)
│   ├── api.py              # HTTP API 入口與路由定義
│   ├── handlers/           # 業務邏輯處理器
│   │   └── speaker_handler.py
│   └── README.md
├── modules/                 # 核心業務模組
│   ├── asr/                # 語音辨識 (Faster-Whisper)
│   ├── database/           # Weaviate V2 資料庫操作
│   │   ├── database.py         # DatabaseService V2 實作
│   │   └── init_v2_collections.py  # V2 集合初始化
│   ├── identification/     # 語者識別 (ECAPA-TDNN)
│   ├── management/         # 語者管理
│   └── separation/         # 語者分離 (Sepformer/ConvTasNet)
├── pipelines/              # 處理流程編排
│   └── orchestrator.py         # 分離+辨識+ASR 整合流程
├── utils/                  # 系統工具
│   ├── env_config.py           # 環境配置載入
│   ├── constants.py            # 應用程式常數
│   └── logger.py               # 統一日誌系統
└── weaviate_study/         # Weaviate V2 整合工具
    ├── npy_to_weaviate.py      # V2 向量匯入工具
    └── README.md
```

## 🚀 快速開始

### 1. 環境準備

**複製並配置環境變數：**
```bash
cp .env.example .env
# 編輯 .env 檔案，針對本地自行調整以下設定：
# - WEAVIATE_HOST=localhost
# - WEAVIATE_PORT=8080  
# - API_HOST=0.0.0.0
# - API_PORT=8000
# 其他設定可自行判斷
```

**安裝相依套件：**
```bash
# 基本安裝
pip install -r requirements-base.txt

# GPU 支援 (可選)
pip install -r requirements-gpu.txt

# CPU 專用 (較輕量)
pip install -r requirements-cpu.txt
```

### 2. 啟動系統

**啟動 Weaviate V2 資料庫：**
```bash
docker-compose up -d
```

**等待資料庫就緒後，直接啟動 API：**
```bash
python main.py
```

系統會自動：
- 檢查並初始化 V2 資料庫集合
- 載入必要的 AI 模型
- 啟動 FastAPI 服務於 http://localhost:8000

### 3. 驗證系統運行

存取以下網址：
- **API 文檔**: http://localhost:8000/docs
- **ReDoc 文檔**: http://localhost:8000/redoc
- **Weaviate 控制台**: http://localhost:8080

## 📡 API 端點

### HTTP REST API
- `POST /transcribe` - 單檔語音轉錄（分離+辨識+ASR）
- `POST /transcribe_dir` - 批次轉錄（目錄/ZIP檔）
- `POST /speaker/rename` - 語者改名
- `POST /speaker/transfer` - 聲紋轉移  
- `POST /speaker/verify` - 語音驗證（識別語者身份）
- `GET /speaker/{id}` - 獲取語者資訊（支援 UUID 和序號 ID）
- `GET /speakers` - 列出所有語者
- `DELETE /speaker/{id}` - 刪除語者

### WebSocket
- `WS /ws/stream` - 即時語音處理串流

## 🗄️ Weaviate V2 資料庫結構

本系統使用 4 個主要集合：

### Speaker
語者基本資訊，包含姓名、性別、活動記錄等

### VoicePrint  
語者聲紋向量資料，支援向量相似度搜尋

### Session
會議/對話場次資料，記錄時間、參與者等資訊

### SpeechLog
語音片段記錄，包含文字內容、時間戳與語者關聯

## ⚙️ 配置系統 (v2.0)

### 環境變數 (.env)
適用於不同環境間變動的設定：
```bash
# API 服務配置
API_HOST=0.0.0.0
API_PORT=8000

# Weaviate 資料庫配置  
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080

# 檔案路徑配置
MODELS_BASE_DIR=./models
```

### 應用程式常數 (constants.py)
演算法核心參數，經過實驗調校：
```python
# 語者識別閾值
THRESHOLD_LOW = 0.26      # 過於相似，不更新向量
THRESHOLD_UPDATE = 0.34   # 相似度足夠，更新向量  
THRESHOLD_NEW = 0.385     # 超過此值視為新語者

# 模型配置
DEFAULT_WHISPER_MODEL = "medium"
SPEECHBRAIN_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
```

## 🔧 開發指南

### 配置管理原則
- **環境變數** (.env): 服務連接、路徑、硬體設定
- **應用常數** (constants.py): 演算法參數、模型名稱、技術規格
- 詳見 [CONFIG_README.md](CONFIG_README.md)

- **utils/**  
  其他輔助腳本（如日誌、同步工具）。

### 測試與驗證
```bash
# 測試 API 模型和資料庫整合
python examples/test_api_models.py

# 測試語者 API 功能  
python examples/test_speaker_api.py

# 測試語音驗證功能
python examples/test_voice_verification.py
```

### V1 升級到 V2 指南
如果您從 V1 版本升級：
1. 備份現有資料
2. 停止舊版 API 服務
3. 執行 V2 集合初始化
4. 使用 `weaviate_study/npy_to_weaviate.py` 匯入現有聲紋資料

## �️ 工具腳本

### Weaviate V2 管理工具
```bash
# 匯入現有聲紋資料到 V2 資料庫
python weaviate_study/npy_to_weaviate.py

# 重置並重新建立 V2 集合
python weaviate_study/create_reset_collections.py  

# 搜尋測試
python weaviate_study/tool_search.py
```

## 系統需求

- **Python 版本**：Python 3.9+ (建議使用 Python 3.12)
- **作業系統**：Windows 10/11, Linux, macOS
- **硬體需求**：
  - 最低：8GB RAM，4核心 CPU
  - 建議：16GB+ RAM，GPU 加速(CUDA 支援)以提高處理速度
- **Docker**：用於 Weaviate 資料庫部署

## � 疑難排解

### 常見問題
1. **找不到 .env 檔案**：複製 `.env.example` 為 `.env` 並調整設定
2. **Weaviate 連接失敗**：確認 Docker 服務運行且端口未被占用
3. **模型下載失敗**：檢查網路連接，系統會自動重試下載
4. **GPU 記憶體不足**：設定 `FORCE_CPU=true` 強制使用 CPU

### 除錯模式
```bash  
# 設定除錯等級
export LOG_LEVEL=DEBUG
python main.py
```

## 📚 相關文檔

- [CONFIG_README.md](CONFIG_README.md) - 配置系統詳細說明
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - 完整 API 文檔
- 各模組 README - 詳見對應模組目錄

## 📄 授權條款

此專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 👥 開發團隊

- **語者分離**: EvanLo62
- **語者識別**: CYouuu  
- **API & 整合**: 專案團隊
- **資料庫設計**: CYouuu

---

**📞 技術支援**: 如遇到問題，請建立 [GitHub Issue](https://github.com/MISCapstoneProject/Unsaycret-API/issues) 或參考文檔說明。
