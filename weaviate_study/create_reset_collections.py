"""
Weaviate V2 集合重置工具

這個工具會：
1. 刪除所有現有的 V2 集合（Speaker, Session, SpeechLog, VoicePrint）
2. 重新建立乾淨的 V2 集合結構
3. 確保與 API 的 uuid 格式保持一致

⚠️ 警告：這會刪除所有現有資料！
"""

import weaviate # type: ignore
from datetime import datetime
import weaviate.classes.config as wc # type: ignore

def reset_weaviate_v2_collections() -> bool:
    """
    重置所有 Weaviate V2 集合
    
    包含的集合：
    - Speaker: 語者資訊（含 speaker_id、full_name、nickname 等）
    - Session: 會議場次資訊
    - SpeechLog: 語音記錄（關聯 Speaker 和 Session）
    - VoicePrint: 聲紋向量資料（關聯 Speaker）
    
    Returns:
        bool: 是否成功重置所有集合
    """
    try:
        # 連接到本地 Weaviate 實例
        client = weaviate.connect_to_local()
        print("✅ 已連接到 Weaviate 服務器")
        
        # 定義 V2 集合名稱
        v2_collections = ["Speaker", "Session", "SpeechLog", "VoicePrint"]
        
        # 刪除現有集合
        print("\n🗑️  正在刪除現有集合...")
        for collection_name in v2_collections:
            if client.collections.exists(collection_name):
                client.collections.delete(collection_name)
                print(f"   ❌ 已刪除 {collection_name} 集合")
            else:
                print(f"   ⏭️  {collection_name} 集合不存在，跳過")
        
        print("\n🏗️  正在建立新的 V2 集合...")
        
        # ========== 1. 建立 Speaker 集合 ==========
        speaker_collection = client.collections.create(
            name="Speaker",
            properties=[
                wc.Property(name="speaker_id", data_type=wc.DataType.INT),
                wc.Property(name="full_name", data_type=wc.DataType.TEXT),
                wc.Property(name="nickname", data_type=wc.DataType.TEXT),
                wc.Property(name="gender", data_type=wc.DataType.TEXT),
                wc.Property(name="created_at", data_type=wc.DataType.DATE),
                wc.Property(name="last_active_at", data_type=wc.DataType.DATE),
                wc.Property(name="meet_count", data_type=wc.DataType.INT),
                wc.Property(name="meet_days", data_type=wc.DataType.INT),
                wc.Property(name="voiceprint_ids", data_type=wc.DataType.UUID_ARRAY),
                wc.Property(name="first_audio", data_type=wc.DataType.TEXT),
            ],
            vectorizer_config=wc.Configure.Vectorizer.none()
        )
        print("   ✅ Speaker 集合建立完成")
        
        # ========== 2. 建立 Session 集合 ==========
        session_collection = client.collections.create(
            name="Session",
            properties=[
                wc.Property(name="session_id", data_type=wc.DataType.TEXT),
                wc.Property(name="session_type", data_type=wc.DataType.TEXT),
                wc.Property(name="title", data_type=wc.DataType.TEXT),
                wc.Property(name="start_time", data_type=wc.DataType.DATE),
                wc.Property(name="end_time", data_type=wc.DataType.DATE),
                wc.Property(name="summary", data_type=wc.DataType.TEXT),
            ],
            references=[
                wc.ReferenceProperty(
                    name="participants",
                    target_collection="Speaker"
                )
            ],
            vectorizer_config=wc.Configure.Vectorizer.none()
        )
        print("   ✅ Session 集合建立完成")
        
        # ========== 3. 建立 SpeechLog 集合 ==========
        speechlog_collection = client.collections.create(
            name="SpeechLog",
            properties=[
                wc.Property(name="content", data_type=wc.DataType.TEXT),    
                wc.Property(name="timestamp", data_type=wc.DataType.DATE),
                wc.Property(name="confidence", data_type=wc.DataType.NUMBER),
                wc.Property(name="duration", data_type=wc.DataType.NUMBER),
                wc.Property(name="language", data_type=wc.DataType.TEXT),
            ],
            references=[
                wc.ReferenceProperty(
                    name="speaker",
                    target_collection="Speaker"
                ),
                wc.ReferenceProperty(
                    name="session",
                    target_collection="Session"
                )
            ],
            vectorizer_config=wc.Configure.Vectorizer.none()
        )
        print("   ✅ SpeechLog 集合建立完成")
        
        # ========== 4. 建立 VoicePrint 集合 ==========
        voiceprint_collection = client.collections.create(
            name="VoicePrint",
            properties=[
                wc.Property(name="created_at", data_type=wc.DataType.DATE),
                wc.Property(name="updated_at", data_type=wc.DataType.DATE),
                wc.Property(name="update_count", data_type=wc.DataType.INT),
                wc.Property(name="sample_count", data_type=wc.DataType.INT),
                wc.Property(name="quality_score", data_type=wc.DataType.NUMBER),
                wc.Property(name="speaker_name", data_type=wc.DataType.TEXT),
            ],
            references=[
                wc.ReferenceProperty(
                    name="speaker",
                    target_collection="Speaker"
                )
            ],
            vectorizer_config=wc.Configure.Vectorizer.none(),
            vector_index_config=wc.Configure.VectorIndex.hnsw(
                distance_metric=wc.VectorDistances.COSINE
            )
        )
        print("   ✅ VoicePrint 集合建立完成")

        # 關閉連接
        client.close()
        
        print("\n🎉 所有 Weaviate V2 集合重置完成！")
        print("\n📋 建立的集合:")
        print("   • Speaker - 語者資訊")
        print("   • Session - 會議場次")
        print("   • SpeechLog - 語音記錄")
        print("   • VoicePrint - 聲紋向量")
        print("\n✅ 資料庫已準備就緒，可以使用 API 進行操作！")
        
        return True
        
    except Exception as e:
        print(f"❌ 重置 Weaviate 集合時發生錯誤: {str(e)}")
        return False

def verify_collections() -> bool:
    """驗證所有集合是否正確建立"""
    try:
        client = weaviate.connect_to_local()
        v2_collections = ["Speaker", "Session", "SpeechLog", "VoicePrint"]
        
        print("\n🔍 驗證集合狀態...")
        all_exist = True
        
        for collection_name in v2_collections:
            exists = client.collections.exists(collection_name)
            status = "✅" if exists else "❌"
            print(f"   {status} {collection_name}: {'存在' if exists else '不存在'}")
            if not exists:
                all_exist = False
        
        client.close()
        return all_exist
        
    except Exception as e:
        print(f"❌ 驗證集合時發生錯誤: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔄 Weaviate V2 集合重置工具")
    print("=" * 50)
    
    # 詢問用戶確認
    confirm = input("\n⚠️  這將刪除所有現有資料！確定要繼續嗎？(y/N): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        print("\n開始重置 Weaviate V2 集合...")
        
        # 重置集合
        if reset_weaviate_v2_collections():
            # 驗證集合
            if verify_collections():
                print("\n🎉 重置完成！所有集合都已正確建立。")
            else:
                print("\n⚠️  重置完成，但部分集合可能有問題。")
        else:
            print("\n❌ 重置失敗！")
    else:
        print("\n🚫 操作已取消。")
    
    print("\n完成！")