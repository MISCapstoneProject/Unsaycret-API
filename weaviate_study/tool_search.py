import weaviate  # type: ignore
import os
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

def display_collections(client: weaviate.WeaviateClient) -> List[str]:
    """
    顯示 Weaviate 資料庫中所有的集合。
    
    Args:
        client: Weaviate 客戶端連接
        
    Returns:
        List[str]: 所有集合名稱的列表
    """
    collections = client.collections.list_all()
    print(f"\n資料庫中共有 {len(collections) if collections else 0} 個集合:")
    
    if not collections:
        print("- 資料庫中沒有任何集合")
        return []
        
    collections_list = list(collections)  # 確保轉換為列表，便於索引訪問
    for idx, collection in enumerate(collections_list, 1):
        print(f"{idx}. {collection}")
        
    return collections_list

def display_collection_details(client: weaviate.WeaviateClient, collection_name: str) -> None:
    """
    顯示特定集合的詳細資訊。
    
    Args:
        client: Weaviate 客戶端連接
        collection_name: 要顯示的集合名稱
    """
    try:
        collection = client.collections.get(collection_name)
        
        print(f"\n===== {collection_name} 集合詳細資訊 =====")
        
        # 獲取集合結構資訊
        try:
            schema = collection.config.get()
            
            # 顯示屬性
            print("\n屬性:")
            if hasattr(schema, 'properties'):
                for prop in schema.properties:
                    print(f"- {prop.name}: {prop.data_type}")
            else:
                print("- 無屬性或無法獲取屬性資訊")
                
            # 顯示參照
            print("\n參照:")
            if hasattr(schema, 'references'):
                for ref in schema.references:
                    # 使用 target_collections 而非 target_collection
                    target = ref.target_collections[0] if hasattr(ref, 'target_collections') and ref.target_collections else "未知目標"
                    print(f"- {ref.name} -> {target}")
            else:
                print("- 無參照或無法獲取參照資訊")
        except Exception as e:
            print(f"無法獲取集合結構: {str(e)}")
            print(f"詳細錯誤: {traceback.format_exc()}")
            
        # 顯示物件數量
        try:
            # 使用正確的 API 獲取物件數量
            count = len(collection.query.fetch_objects(limit=100).objects)
            print(f"\n物件總數: 至少 {count} 個")
        except Exception as e:
            print(f"\n物件總數: 無法計算 ({str(e)})")
            print(f"詳細錯誤: {traceback.format_exc()}")
            
    except Exception as e:
        print(f"\n無法獲取集合 '{collection_name}' 的詳細資訊: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")

def display_collection_objects(client: weaviate.WeaviateClient, collection_name: str, limit: int = 5, 
                              show_vectors: bool = False, vector_preview_size: int = 5,
                              show_vector_content: bool = False) -> None:
    """
    顯示特定集合中的物件。
    
    Args:
        client: Weaviate 客戶端連接
        collection_name: 集合名稱
        limit: 最多顯示的物件數量
        show_vectors: 是否顯示向量資訊
        vector_preview_size: 顯示向量的前幾個元素
        show_vector_content: 是否顯示向量的具體內容
    """
    try:
        collection = client.collections.get(collection_name)
        # 為了獲取向量需要設置 include_vector=True
        results = collection.query.fetch_objects(limit=limit, include_vector=show_vectors)
        
        print(f"\n===== {collection_name} 集合物件 (最多 {limit} 個) =====")
        
        if not results.objects:
            print("- 此集合中沒有物件")
            return
            
        for idx, obj in enumerate(results.objects, 1):
            print(f"\n物件 {idx} (UUID: {obj.uuid}):")
            
            # 顯示屬性
            print("  屬性:")
            for key, value in obj.properties.items():
                # 簡化長字串或複雜對象的顯示
                if isinstance(value, str) and len(value) > 50:
                    value_display = f"{value[:47]}..."
                elif isinstance(value, (list, dict)) and len(str(value)) > 50:
                    value_display = f"{str(value)[:47]}..."
                else:
                    value_display = value
                print(f"  - {key}: {value_display}")
            
            # 顯示向量資訊（如果請求顯示）
            if show_vectors and hasattr(obj, 'vector'):
                try:
                    # 獲取向量資料
                    vector_data = obj.vector
                    
                    # 檢查向量類型並進行適當處理
                    if vector_data is None:
                        print("  向量資訊: 無")
                    elif isinstance(vector_data, dict):
                        # 處理字典格式的向量資料
                        print(f"  向量類型: 字典格式")
                        
                        # 嘗試從字典中提取向量數據
                        vector_array = None
                        if 'vector' in vector_data:
                            vector_array = vector_data['vector']
                        elif 'values' in vector_data:
                            vector_array = vector_data['values']
                        else:
                            # 顯示字典中所有鍵，幫助調試
                            print(f"  向量字典鍵: {list(vector_data.keys())}")
                            
                            # 取第一個鍵值對查看
                            for key, value in vector_data.items():
                                if isinstance(value, (list, np.ndarray)):
                                    vector_array = value
                                    print(f"  找到向量數據在鍵 '{key}'")
                                    break
                        
                        if vector_array is not None:
                            # 如果成功提取向量數據
                            if isinstance(vector_array, (list, np.ndarray)):
                                vector_length = len(vector_array)
                                print(f"  向量維度: {vector_length}")
                                
                                if vector_length > 0:
                                    # 轉換為 NumPy 數組以進行統計計算
                                    if not isinstance(vector_array, np.ndarray):
                                        vector_array = np.array(vector_array)
                                    
                                    # 計算向量統計資訊
                                    print(f"  向量範圍: [{np.min(vector_array):.4f} 到 {np.max(vector_array):.4f}]")
                                    l2_norm = np.linalg.norm(vector_array)
                                    print(f"  向量 L2 範數: {l2_norm:.4f}")
                                    
                                    # 只有在特別要求顯示向量內容時才顯示
                                    if show_vector_content:
                                        # 安全地顯示向量前幾個元素
                                        preview_size = min(vector_preview_size, vector_length)
                                        preview = vector_array[:preview_size]
                                        print(f"  向量前 {preview_size} 個值: {preview}")
                            else:
                                print(f"  提取的向量格式不是列表或數組，而是 {type(vector_array)}")
                        else:
                            print("  無法從字典中提取向量數據")
                            # 顯示字典內容以協助調試
                            if show_vector_content:
                                print(f"  向量字典內容: {vector_data}")
                    elif isinstance(vector_data, (list, np.ndarray)):
                        # 如果是列表或 NumPy 陣列
                        vector_length = len(vector_data)
                        print(f"  向量維度: {vector_length}")
                        
                        if vector_length > 0:
                            # 計算向量統計資訊
                            vector_array = np.array(vector_data)
                            print(f"  向量範圍: [{np.min(vector_array):.4f} 到 {np.max(vector_array):.4f}]")
                            l2_norm = np.linalg.norm(vector_array)
                            print(f"  向量 L2 範數: {l2_norm:.4f}")
                            
                            # 只有在特別要求顯示向量內容時才顯示
                            if show_vector_content:
                                # 安全地顯示向量前幾個元素
                                preview_size = min(vector_preview_size, vector_length)
                                preview = vector_data[:preview_size]
                                print(f"  向量前 {preview_size} 個值: {preview}")
                    else:
                        # 如果是其他類型的向量資料
                        print(f"  向量類型: {type(vector_data)}")
                        if show_vector_content:
                            print(f"  向量內容: {vector_data}")
                except Exception as e:
                    print(f"  處理向量資料時發生錯誤: {str(e)}")
                    print(f"  向量類型: {type(getattr(obj, 'vector', None))}")
                    print(f"  錯誤詳情: {traceback.format_exc()}")
                
            # 顯示參照 (如果有)
            if hasattr(obj, 'references') and obj.references:
                print("  參照:")
                try:
                    for ref_name, ref_objects in obj.references.items():
                        # 根據參照對象的結構調整訪問方式
                        if ref_objects and hasattr(ref_objects[0], 'target') and hasattr(ref_objects[0].target, 'uuid'):
                            target_ids = [ref_obj.target.uuid for ref_obj in ref_objects]
                        else:
                            # 如果參照結構不符合預期，直接輸出
                            target_ids = [str(ref_obj) for ref_obj in ref_objects]
                        print(f"  - {ref_name}: {target_ids}")
                except Exception as e:
                    print(f"  無法處理參照資訊: {str(e)}")
                    print(f"  錯誤詳情: {traceback.format_exc()}")
                    
    except Exception as e:
        print(f"\n無法獲取集合 '{collection_name}' 的物件: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")

def display_all_collections_info(client: weaviate.WeaviateClient) -> None:
    """
    顯示所有集合的基本資訊。
    
    Args:
        client: Weaviate 客戶端連接
    """
    collections = client.collections.list_all()
    
    if not collections:
        print("- 資料庫中沒有任何集合")
        return
    
    for collection_name in collections:
        try:
            collection = client.collections.get(collection_name)
            
            print(f"\n===== {collection_name} =====")
            
            # 計算物件數量
            try:
                count = len(collection.query.fetch_objects(limit=100).objects)
                count_str = f"至少 {count} 個" if count >= 100 else str(count)
            except Exception as e:
                count_str = f"無法計算 ({str(e)})"
                
            print(f"物件數量: {count_str}")
            
            # 嘗試獲取配置並顯示屬性和參照
            try:
                schema = collection.config.get()
                
                # 顯示屬性名稱
                if hasattr(schema, 'properties'):
                    prop_names = [p.name for p in schema.properties]
                    print(f"屬性: {', '.join(prop_names)}")
                else:
                    print("屬性: 無法獲取屬性資訊")
                    
                # 顯示參照
                if hasattr(schema, 'references'):
                    # 使用 target_collections 而非 target_collection
                    ref_info = []
                    for r in schema.references:
                        target = r.target_collections[0] if hasattr(r, 'target_collections') and r.target_collections else "未知目標"
                        ref_info.append(f"{r.name} -> {target}")
                    
                    if ref_info:
                        print(f"參照: {', '.join(ref_info)}")
                    else:
                        print("參照: 無")
                else:
                    print("參照: 無法獲取參照資訊")
                    
            except Exception as e:
                print(f"無法獲取集合配置: {str(e)}")
                # 提供更詳細的錯誤訊息
                print(f"詳細錯誤: {traceback.format_exc()}")
                
        except Exception as e:
            print(f"\n無法獲取集合 '{collection_name}' 的資訊: {str(e)}")
            print(f"詳細錯誤: {traceback.format_exc()}")

def show_main_menu() -> str:
    """
    顯示主菜單並獲取用戶選擇。
    
    Returns:
        str: 用戶的選擇
    """
    print("\n==== Weaviate 資料庫查詢工具 - 主菜單 ====")
    print("1. 查看所有集合的概要資訊")
    print("2. 查看特定集合的詳細資訊")
    print("3. 查看集合中的物件（包含向量資訊）")
    print("0. 退出程式")
    
    return input("\n請輸入選項編號: ").strip()

def main() -> None:
    """
    主函數：連接到 Weaviate 並提供簡化的查詢功能。
    """
    print("Weaviate 資料庫查詢工具")
    print("=======================")
    
    # 連接到本地 Weaviate 資料庫
    try:
        client = weaviate.connect_to_local()
        print("成功連接到本地 Weaviate 資料庫")
    except Exception as e:
        print(f"連接 Weaviate 失敗: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        return
    
    try:
        # 測試連接是否存活
        is_live = client.is_live()
        print(f"連接狀態: {'存活' if is_live else '無法連接'}")
        
        if not is_live:
            print("無法連接到 Weaviate 資料庫，請確認服務是否運行。")
            return
            
        # 獲取資料庫版本資訊
        try:
            meta_info = client.get_meta()
            print(f"Weaviate 版本: {meta_info['version']}")
        except Exception as e:
            print(f"無法獲取 Weaviate 版本資訊: {str(e)}")
            print(f"詳細錯誤: {traceback.format_exc()}")
        
        # 主程式循環
        while True:
            # 顯示主菜單
            choice = show_main_menu()
            
            if choice == "0":
                print("\n感謝使用 Weaviate 資料庫查詢工具，程式執行結束。")
                break
                
            elif choice == "1":
                # 顯示所有集合的概要資訊
                display_all_collections_info(client)
                
            elif choice == "2":
                # 查看特定集合的詳細資訊
                collections = display_collections(client)
                if collections:
                    collection_idx = input("\n請選擇要查看的集合編號 (1-N 或 0 返回): ").strip()
                    if collection_idx == "0":
                        continue
                    
                    try:
                        idx = int(collection_idx) - 1
                        if 0 <= idx < len(collections):
                            collection_name = collections[idx]
                            display_collection_details(client, collection_name)
                        else:
                            print(f"無效的選擇！可用選項範圍是 1 到 {len(collections)}")
                    except ValueError:
                        print(f"請輸入有效的數字！您輸入的是 '{collection_idx}'")
                
            elif choice == "3":
                # 查看集合中的物件
                collections = display_collections(client)
                if collections:
                    collection_idx = input("\n請選擇要查看的集合編號 (1-N 或 0 返回): ").strip()
                    if collection_idx == "0":
                        continue
                    
                    try:
                        idx = int(collection_idx) - 1
                        if 0 <= idx < len(collections):
                            collection_name = collections[idx]
                            
                            # 設定顯示參數
                            limit_str = input("請輸入要顯示的物件數量 (預設 5): ").strip()
                            limit = int(limit_str) if limit_str else 5
                            
                            show_vectors_str = input("是否顯示向量資訊? (y/N): ").strip().lower()
                            show_vectors = show_vectors_str == 'y'
                            
                            show_vector_content = False
                            vector_preview_size = 5  # 預設顯示向量的前 5 個元素
                            
                            if show_vectors:
                                show_content_str = input("是否顯示向量內容? (y/N): ").strip().lower()
                                show_vector_content = show_content_str == 'y'
                                
                                if show_vector_content:
                                    preview_str = input("請輸入要顯示的向量元素數量 (預設 5): ").strip()
                                    if preview_str:
                                        vector_preview_size = int(preview_str)
                            
                            display_collection_objects(
                                client, 
                                collection_name, 
                                limit, 
                                show_vectors, 
                                vector_preview_size,
                                show_vector_content
                            )
                        else:
                            print(f"無效的選擇！可用選項範圍是 1 到 {len(collections)}")
                    except ValueError:
                        print(f"請輸入有效的數字！您輸入的是 '{collection_idx}'")
            
            else:
                print(f"無效的選項: '{choice}'！請重新選擇。")
            
            # 暫停讓用戶查看結果
            input("\n按 Enter 鍵繼續...")
                
    except Exception as e:
        print(f"\n執行過程中發生錯誤: {str(e)}")
        print(f"詳細錯誤: {traceback.format_exc()}")
        
    finally:
        try:
            # 關閉連接
            client.close()
            print("\n資料庫連接已關閉")
        except Exception as e:
            print(f"\n關閉資料庫連接時發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()