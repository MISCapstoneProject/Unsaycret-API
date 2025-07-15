import weaviate # type: ignore
import os
import uuid
import numpy as np
from datetime import datetime
import weaviate.classes.config as wc # type: ignore

def create_weaviate_collections() -> bool:
    """
    建立兩個 Weaviate collections:
    1. 聲紋向量集合 (VoicePrint): 儲存聲紋向量資料，單向引用 Speaker
    2. 使用者集合 (Speaker): 儲存語者/使用者相關資訊，包含 VoicePrint ID 欄位
    
    使用 UUID 作為 ID，以確保唯一性和更好的擴展性
    
    Returns:
        bool: 是否成功建立集合
    """
    try:
        # 連接到本地 Weaviate 實例
        client = weaviate.connect_to_local()
        
        # 如果現有的 VoicePrint、Speaker 已存在就刪除
        if client.collections.exists("VoicePrint"):
            client.collections.delete("VoicePrint")
            print("已刪除現有的 VoicePrint 集合")
        
        if client.collections.exists("Speaker"):
            client.collections.delete("Speaker")
            print("已刪除現有的 Speaker 集合")
        
        # ========== 先建立 Speaker 集合 ==========
        speaker_collection = client.collections.create(
            name="Speaker",
            properties=[
                wc.Property(name="name", data_type=wc.DataType.TEXT),
                wc.Property(name="create_time", data_type=wc.DataType.DATE),
                wc.Property(name="last_active_time", data_type=wc.DataType.DATE),
                wc.Property(name="voiceprint_ids", data_type=wc.DataType.UUID_ARRAY),
                wc.Property(name="first_audio", data_type=wc.DataType.TEXT),  # 修改為 TEXT 型別
            ],
            vectorizer_config=wc.Configure.Vectorizer.none()
        )
        print("成功建立 Speaker 集合")
        
        # ========== 再建立 VoicePrint 集合 ==========
        # 讓 VoicePrint 透過 "speaker" 這個 Reference 欄位，指向 "Speaker"
        voice_print_collection = client.collections.create(
            name="VoicePrint",
            properties=[
                wc.Property(name="create_time", data_type=wc.DataType.DATE),
                wc.Property(name="updated_time", data_type=wc.DataType.DATE),
                wc.Property(name="update_count", data_type=wc.DataType.INT),
                # 加入直接存儲基本使用者資訊的欄位，減少跨集合查詢的需求
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
        print("成功建立 VoicePrint 集合")

        # 關閉連接
        client.close()
        print("成功建立所有 Weaviate 集合")
        return True
        
    except Exception as e:
        print(f"建立 Weaviate 集合時發生錯誤: {str(e)}")
        return False

if __name__ == "__main__":
    # 建立 Weaviate 集合
    print("正在建立 Weaviate 集合...")
    create_weaviate_collections()
    
    # 如果需要，可以取消下面的註解來導入現有的嵌入向量
    # print("\n正在導入現有嵌入向量...")
    # import_existing_embeddings_to_weaviate()
    
    print("\n完成！Weaviate 聲紋資料庫已準備就緒。")