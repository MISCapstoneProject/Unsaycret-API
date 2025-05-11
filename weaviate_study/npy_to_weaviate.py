#!/usr/bin/env python3
"""
此程式將現有的 embeddingFiles 目錄下的所有聲紋嵌入向量（.npy 檔案）匯入到 Weaviate 資料庫。

每個說話者資料夾會成為 Speaker 集合中的一個實體，
而該資料夾內的每個 .npy 檔案會成為 VoicePrint 集合中的一個實體，並關聯到對應的說話者。

檔案命名格式: <用戶名>_<檔案獨立編號>_<更新次數>.npy
"""

import os
import re
import sys
import uuid
import weaviate # type: ignore
import numpy as np
from datetime import datetime, timezone, timedelta
import weaviate.classes.config as wc # type: ignore
from typing import Dict, List, Tuple, Optional, Any

# 全域常數
EMBEDDING_DIR = "../embeddingFiles"
# 修正：將日期格式設定為台北時區（UTC+8）
TAIPEI_TIMEZONE = timezone(timedelta(hours=8))
DEFAULT_DATE = datetime(2025, 1, 1, 0, 0, 0, tzinfo=TAIPEI_TIMEZONE)

def format_date_rfc3339(dt: datetime) -> str:
    """
    將日期格式化為符合 RFC3339 標準的字串，包含時區資訊
    
    Args:
        dt: 日期時間對象
        
    Returns:
        str: RFC3339 格式的日期字串
    """
    # 確保日期時間有時區資訊，若無則使用台北時區
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TAIPEI_TIMEZONE)
    
    # 格式化為 RFC3339，確保包含時區
    return dt.isoformat()

def connect_to_weaviate() -> Any:
    """
    連接到本地 Weaviate 實例
    
    Returns:
        weaviate.Client: Weaviate 客戶端實例
    """
    try:
        client = weaviate.connect_to_local()
        print("成功連接到 Weaviate 本地實例")
        return client
    except Exception as e:
        print(f"連接到 Weaviate 時發生錯誤: {str(e)}")
        sys.exit(1)

def check_collections_exist(client: Any) -> bool:
    """
    檢查 Weaviate 中是否存在所需的集合
    
    Args:
        client: Weaviate 客戶端實例
        
    Returns:
        bool: 如果兩個集合都存在則返回 True，否則返回 False
    """
    speaker_exists = client.collections.exists("Speaker")
    voiceprint_exists = client.collections.exists("VoicePrint")
    
    if not speaker_exists:
        print("錯誤: Speaker 集合不存在，請先執行 create_collections.py")
    
    if not voiceprint_exists:
        print("錯誤: VoicePrint 集合不存在，請先執行 create_collections.py")
    
    return speaker_exists and voiceprint_exists

def parse_file_info(file_name: str) -> Tuple[str, int, int]:
    """
    從檔案名稱中解析出用戶名、檔案編號與更新次數
    
    Args:
        file_name: 嵌入檔案名稱
        
    Returns:
        tuple: (用戶名, 檔案編號, 更新次數)
    """
    m = re.match(r"^(.*?)_(\d+)_(\d+)\.npy$", file_name)
    if m:
        speaker = m.group(1)
        file_id = int(m.group(2))
        update_count = int(m.group(3))
        return speaker, file_id, update_count
    else:
        # 如果解析失敗，採預設值
        speaker = file_name.split("_")[0]
        return speaker, 1, 1

def list_embedding_files() -> Dict[str, List[Tuple[str, str]]]:
    """
    遍歷 EMBEDDING_DIR 下所有子資料夾，
    返回一個字典，其中鍵是說話者資料夾，值是該資料夾內所有 .npy 檔案的列表及其完整路徑
    
    Returns:
        dict: {說話者資料夾: [(檔案名, 完整路徑), ...], ...}
    """
    result = {}
    
    if not os.path.exists(EMBEDDING_DIR):
        print(f"錯誤: {EMBEDDING_DIR} 目錄不存在")
        sys.exit(1)
    
    for folder in os.listdir(EMBEDDING_DIR):
        folder_path = os.path.join(EMBEDDING_DIR, folder)
        if os.path.isdir(folder_path):
            npy_files = []
            for file in os.listdir(folder_path):
                if file.endswith(".npy"):
                    full_path = os.path.join(folder_path, file)
                    npy_files.append((file, full_path))
            
            if npy_files:  # 只將有 .npy 檔案的資料夾加入結果
                result[folder] = npy_files
    
    return result

def create_speakers_in_weaviate(client: Any, embedding_files: Dict[str, List[Tuple[str, str]]]) -> Dict[str, str]:
    """
    在 Weaviate 中建立所有說話者實體
    
    Args:
        client: Weaviate 客戶端實例
        embedding_files: 從 list_embedding_files() 獲得的字典
        
    Returns:
        dict: {說話者資料夾: UUID, ...} - 說話者 ID 映射字典
    """
    speaker_collection = client.collections.get("Speaker")
    speaker_uuids = {}
    
    print(f"\n正在匯入 {len(embedding_files)} 個說話者資料...")
    
    for speaker_folder in embedding_files:
        # 為每個說話者生成一個 UUID
        speaker_uuid = str(uuid.uuid4())
        
        # 將說話者資料加入到 Speaker 集合
        # 注意：初始時 voiceprint_ids 是空列表，稍後會在匯入完成後更新
        # 修正：使用 format_date_rfc3339 函數格式化日期
        speaker_collection.data.insert(
            properties={
                "name": speaker_folder,
                "create_time": format_date_rfc3339(DEFAULT_DATE),
                "last_active_time": format_date_rfc3339(DEFAULT_DATE),
                "voiceprint_ids": []  # 最初是空列表，在匯入嵌入向量後再更新
            },
            uuid=speaker_uuid  # 將 UUID 作為參數傳遞
        )
        
        speaker_uuids[speaker_folder] = speaker_uuid
        print(f"已建立說話者實體: {speaker_folder} (UUID: {speaker_uuid})")
    
    return speaker_uuids

def import_embeddings_to_weaviate(client: Any, embedding_files: Dict[str, List[Tuple[str, str]]], speaker_uuids: Dict[str, str]) -> None:
    """
    將嵌入向量匯入到 Weaviate VoicePrint 集合，並建立正確的參照關係
    
    Args:
        client: Weaviate 客戶端實例
        embedding_files: 從 list_embedding_files() 獲得的字典
        speaker_uuids: 從 create_speakers_in_weaviate() 獲得的說話者 UUID 映射字典
    """
    voice_print_collection = client.collections.get("VoicePrint")
    speaker_collection = client.collections.get("Speaker")
    
    total_embeddings = sum(len(files) for files in embedding_files.values())
    print(f"\n正在匯入 {total_embeddings} 個嵌入向量檔案...")
    
    # 用於追蹤每個說話者關聯的聲紋 ID
    speaker_to_voiceprints = {speaker_id: [] for speaker_id in speaker_uuids.values()}
    
    for speaker_folder, npy_files in embedding_files.items():
        speaker_uuid = speaker_uuids[speaker_folder]
        
        for file_name, full_path in npy_files:
            # 解析檔案名稱
            _, file_id, update_count = parse_file_info(file_name)
            
            # 從 .npy 檔案中讀取嵌入向量
            embedding_vector = np.load(full_path)

            # 檢查維度
            print(f"向量維度為: {embedding_vector.shape}")  
            
            # 為嵌入向量生成一個 UUID
            voiceprint_uuid = str(uuid.uuid4())
            
            # 將此聲紋 ID 加入到對應說話者的追蹤列表中
            speaker_to_voiceprints[speaker_uuid].append(voiceprint_uuid)
            
            try:
                # 將嵌入向量加入到 VoicePrint 集合
                # 修正：參照格式必須為列表，且每個參照為一個單獨的字典
                voice_print_collection.data.insert(
                    properties={
                        "create_time": format_date_rfc3339(DEFAULT_DATE),
                        "updated_time": format_date_rfc3339(DEFAULT_DATE),
                        "update_count": update_count,
                        "speaker_name": speaker_folder,
                    },
                    references={
                        "speaker": [speaker_uuid]  # 修正：參照格式為列表，每個項目是 UUID
                    },
                    uuid=voiceprint_uuid,
                    vector=embedding_vector
                )
                print(f"已匯入嵌入向量: {speaker_folder}/{file_name} (更新次數: {update_count})")
            except Exception as e:
                print(f"匯入嵌入向量時發生錯誤 ({speaker_folder}/{file_name}): {str(e)}")
                # 繼續處理其他檔案，不中斷整個程序
                continue
    
    # 更新每個說話者的 voiceprint_ids 欄位
    print("\n正在更新說話者的關聯聲紋 ID 列表...")
    for speaker_uuid, voiceprint_ids in speaker_to_voiceprints.items():
        if voiceprint_ids:
            try:
                speaker_collection.data.update(
                    uuid=speaker_uuid,
                    properties={
                        "voiceprint_ids": voiceprint_ids
                    }
                )
                
                # 找出對應的說話者名稱以便輸出訊息
                speaker_name = next(name for name, uuid in speaker_uuids.items() if uuid == speaker_uuid)
                print(f"已更新說話者 {speaker_name} 的關聯聲紋 ID 列表 ({len(voiceprint_ids)} 個)")
            except Exception as e:
                print(f"更新說話者 UUID {speaker_uuid} 的關聯聲紋列表時發生錯誤: {str(e)}")

def main() -> None:
    """
    主函數，執行嵌入向量匯入流程
    """
    print("===== 開始將嵌入向量檔案匯入 Weaviate 資料庫 =====")
    
    # 連接到 Weaviate
    client = connect_to_weaviate()
    
    try:
        # 檢查集合是否存在
        if not check_collections_exist(client):
            print("請先執行 create_collections.py 建立必要的集合")
            sys.exit(1)
        
        # 列出所有嵌入檔案
        embedding_files = list_embedding_files()
        if not embedding_files:
            print(f"在 {EMBEDDING_DIR} 中沒有找到任何嵌入向量檔案")
            sys.exit(1)
        
        print(f"在 {EMBEDDING_DIR} 中找到 {len(embedding_files)} 個說話者資料夾")
        
        # 建立說話者實體
        speaker_uuids = create_speakers_in_weaviate(client, embedding_files)
        
        # 匯入嵌入向量並建立參照關係
        import_embeddings_to_weaviate(client, embedding_files, speaker_uuids)
        
        print("\n===== 完成嵌入向量匯入 =====")
        print("感謝使用 npy_to_weaviate 工具!")
    except Exception as e:
        print(f"執行過程中發生錯誤: {str(e)}")
    finally:
        # 確保在出現例外時也能關閉連接
        client.close()
        print("已關閉 Weaviate 連接")

if __name__ == "__main__":
    main()