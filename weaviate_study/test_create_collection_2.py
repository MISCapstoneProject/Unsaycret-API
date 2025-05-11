import weaviate
import weaviate.classes.config as wc
from typing import List, Optional

def create_weaviate_collections() -> None:
    """
    建立兩個互相參照的 Weaviate 集合 A 和 B。
    
    此函數會先刪除任何現有的集合，然後依序創建集合 A 和 B，
    並設置它們之間的交叉參照關係。
    """
    # 連接到本地 Weaviate 服務
    client = weaviate.connect_to_local()
    
    try:
        # 清空既有集合
        if client.collections.exists("A"):
            client.collections.delete("A")
        if client.collections.exists("B"):
            client.collections.delete("B")
        
        # 先創建 B 集合，無需參照
        client.collections.create(
            name="B",
            properties=[
                wc.Property(name="name", data_type=wc.DataType.TEXT)
            ],
            vectorizer_config=wc.Configure.Vectorizer.none()
        )
        
        # 再創建 A 集合，並包含對 B 的參照
        client.collections.create(
            name="A",
            properties=[
                wc.Property(name="name", data_type=wc.DataType.TEXT)
            ],
            references=[
                wc.ReferenceProperty(
                    name="A_to_B", 
                    target_collection="B"
                )
            ],
            vectorizer_config=wc.Configure.Vectorizer.none()
        )
        
        # 獲取 B 集合並更新，添加對 A 的參照
        collection_b = client.collections.get("B")
        collection_b.config.add_reference(
            name="B_to_A",
            target_collection="A"
        )
        
        print("成功建立集合！")
    finally:
        # 確保在結束時關閉連接
        client.close()

if __name__ == "__main__":
    create_weaviate_collections()
