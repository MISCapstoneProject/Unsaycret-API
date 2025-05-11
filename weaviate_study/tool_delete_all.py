import weaviate # type: ignore
from typing import List, Dict, Any, Optional, Union

"""
刪除 Weaviate 中所有 collections 的腳本
"""

def delete_collection(collection_name: str) -> bool:
    """
    刪除指定的 collection
    
    參數:
    collection_name (str): 要刪除的 collection 名稱
    
    回傳:
    bool: 刪除成功返回 True，否則返回 False
    """
    try:
        # 連接到本地 Weaviate 資料庫
        client = weaviate.connect_to_local()
        
        # 檢查連接是否存活
        if not client.is_live():
            print("無法連接到 Weaviate 資料庫")
            return False
        
        # 檢查 collection 是否存在
        collections = client.collections.list_all()
        found = collection_name in collections
        
        if not found:
            print(f"Collection '{collection_name}' 不存在")
            return False
        
        # 刪除 collection
        client.collections.delete(collection_name)
        print(f"Collection '{collection_name}' 已成功刪除")
        
        # 驗證刪除結果
        updated_collections = client.collections.list_all()
        if collection_name not in updated_collections:
            print("驗證成功：collection 已不存在")
            return True
        else:
            print("驗證失敗：collection 仍然存在")
            return False
    
    except Exception as e:
        print(f"刪除 collection 時發生錯誤: {e}")
        return False
    finally:
        # 關閉連接
        client.close()

def delete_all_collections() -> bool:
    """
    刪除所有 Weaviate 中的 collections
    
    回傳:
    bool: 所有 collections 都刪除成功返回 True，否則返回 False
    """
    try:
        # 連接到本地 Weaviate 資料庫
        client = weaviate.connect_to_local()
        
        # 檢查連接是否存活
        if not client.is_live():
            print("無法連接到 Weaviate 資料庫")
            return False
        
        # 獲取所有現有的 collections
        collections = client.collections.list_all()
        
        if not collections:
            print("Weaviate 中沒有任何 collections")
            return True
        
        print(f"找到 {len(collections)} 個 collections，開始刪除...")
        
        # 刪除所有 collections
        all_deleted = True
        for collection_name in collections:
            try:
                client.collections.delete(collection_name)
                print(f"Collection '{collection_name}' 已成功刪除")
            except Exception as e:
                print(f"刪除 Collection '{collection_name}' 時發生錯誤: {e}")
                all_deleted = False
        
        # 驗證所有 collections 都已刪除
        updated_collections = client.collections.list_all()
        if not updated_collections:
            print("驗證成功：所有 collections 已刪除")
            return True
        else:
            print(f"驗證失敗：仍有 {len(updated_collections)} 個 collections 存在")
            for collection in updated_collections:
                print(f"- {collection}")
            return False
    
    except Exception as e:
        print(f"刪除所有 collections 時發生錯誤: {e}")
        return False
    finally:
        # 關閉連接
        client.close()
        print("資料庫連接已關閉")

def main() -> None:
    """
    主函數，用於執行刪除所有 collections 的操作
    """
    print("Weaviate Collection 刪除工具")
    print("============================")
    
    # 連接到本地 Weaviate 資料庫
    client = weaviate.connect_to_local()
    
    try:
        # 獲取所有現有的 collections
        collections = client.collections.list_all()
        
        # 顯示所有現有的 collections
        print("\n現有的 collections:")
        if collections:
            for collection in collections:
                print(f"- {collection}")
            
            # 詢問用戶是刪除單一 collection 還是所有 collections
            print("\n選項:")
            print("1. 刪除所有 collections")
            print("2. 刪除特定 collection")
            
            choice = input("\n請選擇操作 (預設 1): ").strip() or "1"
            
            if choice == "1":
                print("\n準備刪除所有 collections...")
                success = delete_all_collections()
                
                if success:
                    print("\n所有 collections 已成功刪除！")
                else:
                    print("\n刪除所有 collections 時發生問題，請檢查錯誤訊息。")
            
            elif choice == "2":
                print("\n可刪除的 collections:")
                for i, collection in enumerate(collections, 1):
                    print(f"{i}. {collection}")
                
                try:
                    index = int(input("\n請輸入要刪除的 collection 編號: ").strip()) - 1
                    if 0 <= index < len(collections):
                        collection_to_delete = collections[index]
                        print(f"\n準備刪除 collection：{collection_to_delete}")
                        
                        # 執行刪除操作
                        success = delete_collection(collection_to_delete)
                        
                        if success:
                            print(f"\nCollection '{collection_to_delete}' 已成功刪除！")
                        else:
                            print(f"\n刪除 Collection '{collection_to_delete}' 失敗！")
                    else:
                        print("\n無效的選擇！")
                except ValueError:
                    print("\n請輸入有效的數字！")
            
            else:
                print("\n無效的選擇！")
                
        else:
            print("- 無現有 collections")
            print("\n沒有可刪除的 collections")
    
    except Exception as e:
        print(f"\n執行過程中發生錯誤: {e}")
    
    finally:
        # 關閉連接
        client.close()
        print("\n程式執行完畢")

if __name__ == "__main__":
    main()