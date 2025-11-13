"""
Weaviate V3 Collection 初始化模組

⚠️ 重要變更（2025-01-14）：
- 只初始化 Speaker 和 VoicePrint
- 移除 Session 和 SpeechLog（已遷移到 MongoDB）

替代檔案：
- 舊版：init_v2_collections.py（包含 Session/SpeechLog）
- 新版：init_v3_weaviate.py（只有 Speaker/VoicePrint）

使用方式（在 main.py）：
```python
from modules.database.init_v3_weaviate import ensure_weaviate_v3_collections

if not ensure_weaviate_v3_collections():
    logger.error("Weaviate 初始化失敗")
    sys.exit(1)
```
"""

import logging
import weaviate
from weaviate.classes.config import Configure, Property, DataType, ReferenceProperty

logger = logging.getLogger(__name__)


def init_v3_collections(client: weaviate.WeaviateClient) -> bool:
    """
    初始化 Weaviate V3 Collections（只包含 Speaker 和 VoicePrint）
    
    Args:
        client: Weaviate 客戶端
    
    Returns:
        bool: 初始化是否成功
    
    注意：
    - Session 和 SpeechLog 已遷移到 MongoDB
    - 此函式只負責 Speaker 和 VoicePrint
    """
    try:
        # 取得所有現有 collections
        existing_collections = set()
        try:
            collections = client.collections.list_all()
            existing_collections = {col.name for col in collections.values()}
            logger.info(f"現有 collections: {existing_collections}")
        except Exception as e:
            logger.warning(f"無法取得現有 collections: {e}")
        
        # ===== Speaker Collection =====
        if "Speaker" not in existing_collections:
            logger.info("建立 Speaker collection...")
            client.collections.create(
                name="Speaker",
                vectorizer_config=Configure.Vectorizer.none(),  # 不使用向量化器
                properties=[
                    Property(name="speaker_id", data_type=DataType.INT),
                    Property(name="full_name", data_type=DataType.TEXT),
                    Property(name="nickname", data_type=DataType.TEXT),
                    Property(name="gender", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.DATE),
                    Property(name="last_active_at", data_type=DataType.DATE),
                    Property(name="meet_count", data_type=DataType.INT),
                    Property(name="meet_days", data_type=DataType.INT),
                    Property(name="voiceprint_ids", data_type=DataType.TEXT_ARRAY),
                    Property(name="first_audio", data_type=DataType.TEXT),
                ]
            )
            logger.info("✅ Speaker collection 建立完成")
        else:
            logger.info("✅ Speaker collection 已存在")
        
        # ===== VoicePrint Collection =====
        if "VoicePrint" not in existing_collections:
            logger.info("建立 VoicePrint collection...")
            client.collections.create(
                name="VoicePrint",
                vectorizer_config=Configure.Vectorizer.none(),  # 手動提供向量
                properties=[
                    Property(name="created_at", data_type=DataType.DATE),
                    Property(name="updated_at", data_type=DataType.DATE),
                    Property(name="update_count", data_type=DataType.INT),
                    Property(name="sample_count", data_type=DataType.INT),
                    Property(name="quality_score", data_type=DataType.NUMBER),
                    Property(name="speaker_name", data_type=DataType.TEXT),
                ],
                references=[
                    ReferenceProperty(name="speaker", target_collection="Speaker")
                ]
            )
            logger.info("✅ VoicePrint collection 建立完成")
        else:
            logger.info("✅ VoicePrint collection 已存在")
        
        # ===== 檢查是否有舊的 Session/SpeechLog =====
        if "Session" in existing_collections:
            logger.warning("⚠️ 發現舊的 Session collection，建議手動刪除或清空資料")
            logger.warning("   Session 資料已遷移到 MongoDB，Weaviate 中的 Session 不再使用")
        
        if "SpeechLog" in existing_collections:
            logger.warning("⚠️ 發現舊的 SpeechLog collection，建議手動刪除或清空資料")
            logger.warning("   SpeechLog 已改為 Transcript，並遷移到 MongoDB")
        
        logger.info("✅ Weaviate V3 初始化完成（Speaker + VoicePrint）")
        return True
        
    except Exception as e:
        logger.error(f"❌ Weaviate V3 初始化失敗: {e}", exc_info=True)
        return False


def ensure_weaviate_v3_collections() -> bool:
    """
    確保 Weaviate V3 collections 存在
    
    注意：此函式會在 main.py 的 initialize_system() 中被呼叫
    
    Returns:
        bool: 初始化是否成功
    """
    try:
        from utils.env_config import WEAVIATE_URL
        
        # 從環境變數解析 host 和 port
        weaviate_host = WEAVIATE_URL.replace("http://", "").replace("https://", "")
        if ":" in weaviate_host:
            host, port = weaviate_host.split(":")
            port = int(port)
        else:
            host = weaviate_host
            port = 8080
        
        logger.info(f"正在連線到 Weaviate: {host}:{port}")
        
        # 建立 Weaviate 客戶端（使用 v4 API）
        client = weaviate.connect_to_local(
            host=host,
            port=port,
        )
        
        # 初始化 collections
        success = init_v3_collections(client)
        
        # 關閉連線
        client.close()
        
        return success
        
    except ImportError as e:
        logger.error(f"❌ 無法匯入 env_config: {e}")
        logger.error("   請確認 utils/env_config.py 存在且包含 WEAVIATE_URL")
        return False
    except Exception as e:
        logger.error(f"❌ 無法連線到 Weaviate: {e}", exc_info=True)
        return False


# 如果直接執行此檔案，進行測試
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Weaviate V3 初始化測試")
    print("=" * 60)
    
    if ensure_weaviate_v3_collections():
        print("\n✅ 初始化成功")
        print("\n下一步：")
        print("1. 使用 Weaviate Console 查看 collections: http://localhost:8081")
        print("2. 確認 Speaker 和 VoicePrint collections 已建立")
        print("3. 如果看到舊的 Session/SpeechLog，請手動刪除")
    else:
        print("\n❌ 初始化失敗")
        print("\n故障排除：")
        print("1. 確認 Docker Compose 已啟動：docker-compose ps")
        print("2. 確認 Weaviate 健康狀態：docker-compose logs weaviate")
        print("3. 測試連線：curl http://localhost:8080/v1/.well-known/ready")
