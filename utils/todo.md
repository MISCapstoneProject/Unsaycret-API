## GitHub Copilot 上下文 — Weaviate Schema 重構

專案：**Unsaycret-API**

⚠️ **重要變更警告** ⚠️
本次重構將大幅改變資料庫結構，會導致現有資料不相容！
執行前請務必備份現有 Weaviate 資料庫。

---

### **目前狀況 (CURRENT SITUATION)**

* Repo 仍採用舊版資料模型：只有一個 **Speaker** collection，同時存放說話者靜態資料與 *utterances* 陣列；另有一個 **VoicePrint** collection。
* 目標是改為 **4 個正規化的 Weaviate collections**，並更新所有關聯關係；程式需具 **冪等性**（重跑不會錯）。

---

### **Copilot 注意事項 (IMPORTANT for GitHub Copilot)**

* **務必使用 weaviate-client v4（Collections API），不要用 v3 語法。**
* **不要** 出現 `client.schema.*`、`client.batch.add_data_object`、`update_object`… 等舊 API。
* 正確示例：

  ```python
  import weaviate
  import weaviate.classes.config as wc

  client = weaviate.connect_to_local()
  client.collections.create(...); client.collections.get(); client.collections.exists(); client.collections.delete()

  speechlog = client.collections.get("SpeechLog")
  with speechlog.batch as batch:
      batch.add_many([...])

  # 向量器設定
  vec_cfg = wc.Configure.Vectorizer.text2vec_transformers(source_properties=["content"])
  ```
* `ReferenceProperty` 與 `Property` 請用 v4 類別：

  ```python
  wc.ReferenceProperty(name="speaker", target_collection="Speaker")
  wc.Property(name="content", data_type=wc.DataType.TEXT)
  ```
* 程式需能在 **weaviate-client ≥ 4.4.0** 編譯執行。

---

### **目標 Schema（伺服器 ≥ 1.25；client ≥ 4.x）**

| Collection        | 角色               | 主要屬性 (DataType)                                                                                                                                                                  | ReferenceProperty                                           | 向量化策略                                                         |
| ----------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- |
| **1) Speaker**    | 說話者主檔             | `speaker_id INT`（從1開始遞增，不可變）<br>`full_name TEXT`（預設為"n1","n2"格式）<br>`nickname TEXT`<br>`gender TEXT`<br>`created_at DATE`<br>`last_active_at DATE`<br>`meet_count INT (可選)`（遇到用戶的次數）<br>`meet_days INT (可選)`（遇到用戶的天數）<br>`voiceprint_ids UUID_ARRAY`<br>`first_audio TEXT` | —                                                           | `Vectorizer.none()`                                           |
| **2) Session**    | 對話場景（取代Meeting）  | `session_id TEXT`<br>`session_type TEXT`（meeting/call/casual）<br>`title TEXT`<br>`start_time DATE`<br>`end_time DATE (可選)`<br>`summary TEXT (可選)`                        | `participants → Speaker` (多對多)                              | `Vectorizer.text2vec_transformers()`（向量化 title + summary） |
| **3) SpeechLog**  | **一句話記錄**        | `content TEXT`<br>`timestamp DATE`<br>`confidence NUMBER`（識別信心度）<br>`duration NUMBER`（語音長度秒）<br>`language TEXT (可選)`                                                    | `speaker → Speaker` (多→一)<br>`session → Session` (多→一，可為空) | `Vectorizer.text2vec_transformers()`（向量化 `content`）        |
| **4) VoicePrint** | 聲紋特徵庫（改進版）       | `voiceprint_id TEXT`<br>`created_at DATE`<br>`updated_at DATE`<br>`update_count INT`<br>`sample_count INT`<br>`quality_score NUMBER`<br>`speaker_name TEXT`                | `speaker → Speaker` (多→一，一人多聲紋)                             | 保持現行向量設定（COSINE 距離）                                          |

---

### **Copilot 待辦 (TODO FOR COPILOT)**

A. **創建新檔案** `utils/init_v2_collections.py`（不覆蓋原本的 `init_collections.py`），要求：

1. 參考現有 `init_collections.py` 的架構和風格，但創建新版本的 schema。
2. 只呼叫一次 `connect_to_local()` 並重用 client。
3. 實作新的 collection 結構（Session 取代 Meeting）：
   - **保留現有的 Speaker 和 VoicePrint collections** 為基礎
   - **新增 Session 和 SpeechLog collections**
4. 正確設定 `vectorizer_config`：
   - Speaker: `none()`（保持原樣但增加欄位）
   - Session: `text2vec_transformers(source_properties=["title", "summary"])`
   - SpeechLog: `text2vec_transformers(source_properties=["content"])`
   - VoicePrint: 保持現有設定但增加欄位
5. 建表邏輯包成：
   - `create_speaker_v2()` - 基於原有 Speaker 但增加 speaker_id(INT)、meet_count、meet_days
   - `create_session()` - 全新的 Session collection
   - `create_speechlog()` - 全新的 SpeechLog collection  
   - `create_voiceprint_v2()` - 基於原有 VoicePrint 但增加 update_count、speaker_name
6. **重要設計原則**：
   - Speaker.speaker_id 使用 INT 從 1 開始遞增（類似原本的 n1→1, n2→2）
   - full_name 預設為 "n1", "n2", "n3" 格式
7. 結尾附 **完整測試場景**：
   - 插入 3 個 Speaker（speaker_id: 1,2,3，full_name: "n1","n2","n3"）
   - 插入 2 個 Session（一個 meeting，一個 casual call）
   - 插入 8-10 條 SpeechLog（包含有 session 和無 session 的）
   - 插入對應的 VoicePrint 記錄（包含 update_count、speaker_name）
   - 執行複雜查詢驗證關聯正確性

B. **新檔案命名與相容性**：
   - 檔案名稱：`init_v2_collections.py`
   - 類別名稱：`WeaviateV2CollectionManager`（避免與原類別衝突）
   - 主函數：`ensure_weaviate_v2_collections()`
   - CLI 入口：`python -m utils.init_v2_collections`

C. 腳本必須 **冪等且穩健**：
   - 重跑不會因 `AlreadyExists` 異常中斷
   - 使用 `exists()` 檢查 + 適當的 try/except
   - 加入連線重試機制（最多 3 次）
   - 詳細的錯誤日誌與狀態報告

D. **程式品質要求**：
   - 嚴格遵守 PEP-8 規範
   - 完整的 type hints（包含 Union、Optional）
   - 詳細的 docstring（包含 Args、Returns、Raises）
   - 重用 `utils.logger` 模組
   - 可以判斷是否使用 dataclass 或 TypedDict 定義資料結構
   - 單元測試友善的模組化設計

E. **效能與最佳實務**：
   - 使用 batch 操作插入大量資料
   - 適當的事務管理（rollback 機制）
   - 資源釋放確保（with statement）
   - 進度提示（ tqdm或其他簡單提示 ）

---

### **驗收標準 (Acceptance Criteria)**

**基本功能驗收：**
* ✅ 成功創建新檔案 `utils/init_v2_collections.py`（不影響原檔案）
* ✅ 成功在資料庫中建立所有 4 個 collections（Speaker, Session, SpeechLog, VoicePrint）
* ✅ 重複執行腳本不會 crash，具備完整冪等性
* ✅ 所有 ReferenceProperty 關聯正確建立且可查詢

**資料完整性驗收：**
* ✅ Speaker.speaker_id 正確使用 INT 類型，從 1 開始遞增
* ✅ Speaker.full_name 預設為 "n1", "n2", "n3" 格式
* ✅ Speaker 包含 first_audio 欄位（保持與原 schema 相容）
* ✅ VoicePrint 包含 update_count 和 speaker_name 欄位（保持與原 schema 相容）
* ✅ Session 可以有多個參與者，SpeechLog 可選擇性關聯 Session
* ✅ 一個 Speaker 可以有多個 VoicePrint 記錄

**查詢功能驗收：**
* ✅ 巢狀查詢測試：Session → participants → SpeechLogs 三層關聯
* ✅ 語義搜尋測試：根據 SpeechLog.content 向量相似度查詢
* ✅ Speaker ID 查詢：可透過 speaker_id (INT) 快速定位說話者
* ✅ 時間範圍查詢：指定時間段內的所有語音活動

**相容性與維護性驗收：**
* ✅ 新檔案與原 `init_collections.py` 共存，不產生衝突
* ✅ 保持相同的程式架構風格，便於維護
* ✅ 詳細日誌輸出，便於 debug 和監控
* ✅ 程式碼遵循項目現有的編碼風格


