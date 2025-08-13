# Unsaycret-API

**版本**: v0.4.1  <br>
**最後更新**: 2025-08-13

## 專案簡介
Unsaycret-API 是一套模組化的語音處理系統，整合語音分離、說者辨識、語音辨識、API 服務，並支援 Weaviate 向量資料庫串接。
專案執行後會啟動 API 伺服器，以呼叫 API 的方式使用所有功能。

## 🚀 主要功能

- 🎙 **語者分離**：採用 SpeechBrain Sepformer (2人) / ConvTasNet (3人)，支援多人語音分離
- 🧠 **語音辨識（ASR）**：Faster-Whisper，支援 GPU/CPU 動態切換，逐詞時間戳與信心值
- 🗣 **說話人辨識**：提供雙模型使用 (正在測試哪一個更優)
  - SpeechBrain ECAPA-TDNN
  - PyAnnote Embedding
  - 支援聲紋自動更新與多聲紋映射
- 🛜 **API 服務**：FastAPI 提供完整的 RESTful 與 WebSocket 介面
- 🧠 **Weaviate V2 整合**：語音向量與辨識結果存入 Weaviate V2，支援高效語者搜尋與比對
- ⚙️ **分層配置**：環境變數(.env) 與應用常數(constants.py) 分離管理

## 🏗 系統架構

### 配置系統
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
├── api/                     # FastAPI 應用程式層
│   ├── api.py               # HTTP API 入口與路由定義
│   └── README.md
├── services/                # 資料存取門面
│   └── data_facade.py       # DataFacade: 統一對外資料操作介面
├── modules/                 # 核心業務模組
│   ├── asr/                 # 語音辨識 (Faster-Whisper)
│   ├── database/            # Weaviate V2 資料庫操作
│   │   ├── database.py
│   │   └── init_v2_collections.py
│   ├── identification/      # 語者識別 (ECAPA-TDNN / PyAnnote Embedding)
│   ├── management/          # 語者管理
│   └── separation/          # 語者分離 (Sepformer/ConvTasNet)
├── pipelines/               # 處理流程編排
│   └── orchestrator.py      # 分離+辨識+ASR 整合流程
├── utils/                   # 系統工具
│   ├── env_config.py
│   ├── constants.py
│   └── logger.py            # 統一日誌管理
└── weaviate_study/          # Weaviate V2 開發工具
    ├── npy_to_weaviate.py   # 匯入現有測試聲紋向量資料
    └── README.md
```

> **說明**：
> - `services/data_facade.py` 為資料存取門面，API 層所有資料查詢、異動、驗證等操作皆透過 DataFacade 對外暴露，實現業務邏輯與資料層分離。
> - `modules/` 內為各功能子模組，專注於演算法與資料處理。

## 🚀 快速開始

### 1. 環境準備

**複製並配置環境變數：**
```bash
cp .env.example .env

# 自行編輯 .env 檔案，針對本地調整以下設定：
# - WEAVIATE_HOST=localhost
# - WEAVIATE_PORT=8080  
# - API_HOST=0.0.0.0
# - API_PORT=8000
# - HF_ACCESS_TOKEN=hf_xxx...  (PyAnnote 模型需要)
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

### 🚀 核心功能
- **語音轉錄**: `POST /transcribe` - 單檔語音轉錄（分離+辨識+ASR）
- **批次轉錄**: `POST /transcribe_dir` - 批次轉錄（目錄/ZIP檔）
- **即時處理**: `WS /ws/stream?session={uuid}` - 即時語音處理串流

### 📊 資料管理
- **語者管理**: 完整的CRUD操作（查詢、更新、刪除）
- **會議管理**: 建立、查詢會議記錄（支援級聯刪除）
- **語音記錄**: 管理語音片段和轉錄內容
- **關聯查詢**: 語者-會議-語音記錄關聯查詢

### 🔧 進階功能
- **語音驗證**: 識別語者身份
- **聲紋轉移**: 合併語者聲紋資料

### 📖 完整API文檔
詳細的API端點說明、請求格式、回應範例請參考：
- **API文檔**: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **互動式文檔**: http://localhost:8000/docs
- **ReDoc文檔**: http://localhost:8000/redoc

### 🧪 API測試
```bash
# 基礎功能測試
python examples/test_all_apis.py

# 綜合測試（推薦）
python examples/test_comprehensive_apis.py
```
測試指南請參考：[examples/API_TEST_GUIDE.md](examples/API_TEST_GUIDE.md)

## 🗄️ Weaviate V2 資料庫結構

本系統使用 4 個主要集合，支援完整的資料關聯與級聯操作：

### Speaker
語者基本資訊，包含姓名、性別、活動記錄等

### VoicePrint  
語者聲紋向量資料，支援向量相似度搜尋

### Session
會議/對話場次資料，記錄時間、參與者等資訊
- 🔗 **級聯刪除**: 刪除會議時自動清理所有關聯的語音記錄

### SpeechLog
語音片段記錄，包含文字內容、時間戳與語者關聯

> **資料完整性**: 系統實現了完整的級聯刪除機制，確保不會產生孤立記錄

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
演算法固定核心參數，經過實驗調校：
```python
# 語者識別閾值
THRESHOLD_LOW = 0.26      # 過於相似，不更新向量
THRESHOLD_UPDATE = 0.34   # 相似度足夠，更新向量  
THRESHOLD_NEW = 0.385     # 超過此值視為新語者

# 模型配置
DEFAULT_WHISPER_MODEL = "medium"
SPEECHBRAIN_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
PYANNOTE_SPEAKER_MODEL = "pyannote/embedding"
```

## 🔧 開發指南

### 配置管理原則
- **環境變數** (.env): 服務連接、路徑、硬體設定
- **應用常數** (constants.py): 演算法參數、模型名稱、技術規格
- 詳見 [CONFIG_README.md](CONFIG_README.md)

- **utils/**  
  其他輔助腳本（如日誌、同步工具）。

### 測試資料
- 可使用 `weaviate_study/npy_to_weaviate.py` 匯入現有 5 筆聲紋資料
```bash
# 匯入預設測試聲紋資料到資料庫
python -m weaviate_study/npy_to_weaviate.py
```

## �️ 工具腳本

### Weaviate V2 管理工具
```bash
# 匯入現有聲紋資料到 V2 資料庫
python weaviate_study/npy_to_weaviate.py

# 重置並重新建立 V2 集合
python weaviate_study/create_reset_collections.py  

# 查看資料庫目前資料
python weaviate_study/tool_search.py
# 也可使用 localhost:8081 進入 Weaviate UI 控制台
```

## 系統需求

- **Python 版本**：Python 3.10+ (建議使用 Python 3.12)
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
5. **PyAnnote 模型載入失敗**：確認 `.env` 中設定了有效的 `HF_ACCESS_TOKEN`

### 設備控制
```bash  
# 強制使用 CPU（忽略 GPU）
FORCE_CPU=true

# 指定 CUDA 設備索引（多GPU環境，預設為 0）  
CUDA_DEVICE_INDEX=0
```

**注意**：
- `FORCE_CPU=true` 會強制所有模組使用 CPU，適用於GPU記憶體不足的情況
- `CUDA_DEVICE_INDEX` 用於多GPU系統中指定要使用的GPU（從0開始編號）
- 系統會自動檢查指定的GPU是否存在，不存在時會回退到 GPU 0

## 📚 相關文檔

- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - 完整 API 文檔與使用說明
- [examples/API_TEST_GUIDE.md](examples/API_TEST_GUIDE.md) - API 測試指南
- [CONFIG_README.md](CONFIG_README.md) - 配置系統詳細說明
- 各模組 README - 詳見對應模組目錄

## 📄 授權條款

此專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 👥 開發團隊

- **語者分離**: EvanLo62
- **語者識別**: CYouuu  
- **語音轉文字**: gino287
- **API & 整合**: 專案團隊
- **資料庫設計**: CYouuu

---

**📞 技術支援**: 如遇到問題，請建立 [GitHub Issue](https://github.com/MISCapstoneProject/Unsaycret-API/issues) 或參考文檔說明。
