"""預先建立語者並使用指定音檔建立聲紋。

此腳本支援兩種模式：
- 模式1（AVERAGE）：所有音檔平均合併成一個聲紋
- 模式2（INTELLIGENT）：使用智慧閾值邏輯，根據相似度決定更新或新增聲紋

模式2邏輯：
- 距離 >= THRESHOLD_NEW ：可能不屬於該語者或音檔品質不佳，跳過處理
- 距離 < THRESHOLD_LOW ：過於相似，跳過不處理
- 距離 < THRESHOLD_UPDATE ：適中相似，更新現有聲紋（加權平均）
- 距離 >= THRESHOLD_UPDATE 且 < THRESHOLD_NEW：較大差異，為該語者新增額外聲紋

使用方式：
1. 修改 CURRENT_MODE 變數選擇模式
2. 配置 SPEAKER_CONFIG 中的語者設定
3. 執行腳本即可批量建立語者
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

import numpy as np
from scipy.spatial.distance import cosine

# 將專案根目錄加入 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.database import DatabaseService
from modules.identification import SpeakerIdentifier
from utils.constants import THRESHOLD_LOW, THRESHOLD_UPDATE, THRESHOLD_NEW

# === 模式設定 ==============================================================
class BuildMode(Enum):
    """建立語者的模式"""
    AVERAGE = 1      # 模式1：所有音檔平均合併
    INTELLIGENT = 2  # 模式2：使用智慧閾值邏輯

# 設定使用的模式
CURRENT_MODE = BuildMode.INTELLIGENT  # 可以改為 BuildMode.AVERAGE

# === 語者設定 ==============================================================
# key: 新語者名稱
# value: {"folder": 資料夾路徑, "indices": [音檔索引列表]}
root_folder = "data_separated"  # 根目錄，所有語者資料夾都在這裡

SPEAKER_CONFIG: Dict[str, Dict[str, List[int]]] = {
    "spk1": {
        "folder": root_folder+"/speaker1",
        "indices": [1, 4, 5, 7, 8, 9, 12, 14, 15, 17, 18, 19, 20],
    },
    "spk2": {
        "folder": root_folder+"/speaker2",
        "indices": [1, 3, 7, 8, 10, 11, 12, 14, 15, 17, 18, 20],
    },
    "spk3": {
        "folder": root_folder+"/speaker3",
        "indices": [1, 2, 4, 5, 7, 8, 9, 14, 16, 17, 18],
    },
    "spk4": {
        "folder": root_folder+"/speaker4",
        "indices": [2, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 19, 20],
    },
    "spk5": {
        "folder": root_folder+"/speaker5",
        "indices": [1, 3, 4, 7, 9, 12, 14, 15, 16, 17, 18, 19, 20],
    },
    "spk6": {
        "folder": root_folder+"/speaker6",
        "indices": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20],
    },
    "spk7": {
        "folder": root_folder+"/speaker7",
        "indices": [1, 2, 8, 9, 10, 11, 13, 14, 16, 18, 19, 20],
    },
    "spk8": {
        "folder": root_folder+"/speaker8",
        "indices": [2, 4, 5, 6, 9, 10, 11, 12, 15, 17, 20],
    },
    "spk9": {
        "folder": root_folder+"/speaker9",
        "indices": [1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19],
    },
    "spk10": {
        "folder": root_folder+"/speaker10",
        "indices": [2, 4, 9, 10, 13, 19],
    },
    # 範例：若要建立更多語者，可在此處加入設定
    # "n4": {
    #     "folder": "data/clean/speaker4",
    #     "indices": [1, 3, 5, 7],
    # },
}

# 初始化識別器與資料庫
identifier = SpeakerIdentifier()
db = DatabaseService()


def build_speaker_average_mode(name: str, folder: str, indices: List[int]) -> None:
    """模式1：依索引平均音檔嵌入並建立語者。

    Args:
        name: 新語者名稱。
        folder: 音檔所在資料夾。
        indices: 需要使用的音檔索引（不含前置零）。
    """
    base = os.path.basename(folder)
    embeddings = []

    for idx in indices:
        file_path = os.path.join(folder, f"{base}_{idx:02d}.wav")
        if not os.path.exists(file_path):
            print(f"音檔 {file_path} 不存在，已跳過。")
            continue
        emb = identifier.audio_processor.extract_embedding(file_path)
        embeddings.append(emb)

    if not embeddings:
        print(f"語者 {name} 沒有有效音檔，跳過建立。")
        return

    avg_embedding = np.mean(np.stack(embeddings), axis=0)

    # 建立語者
    first_file = os.path.join(folder, f"{base}_{indices[0]:02d}.wav")
    speaker_uuid = db.create_speaker(full_name=name, first_audio=first_file)

    # 建立平均聲紋
    db.create_voiceprint(
        speaker_uuid,
        avg_embedding,
        audio_source="avg_of_indices",
    )
    print(f"已建立語者 {name} (UUID: {speaker_uuid})，使用 {len(embeddings)} 個音檔。")


def build_speaker_intelligent_mode(name: str, folder: str, indices: List[int]) -> None:
    """模式2：使用智慧閾值邏輯依序處理音檔。

    邏輯說明：
    1. 建立語者並用第一個音檔創建初始聲紋
    2. 對於後續音檔，**只與該語者現有聲紋**計算距離（確保同人比較）
    3. 根據最小距離決定操作：
       - >= THRESHOLD_NEW: 可能不屬於該語者，跳過並警告
       - < THRESHOLD_LOW: 過於相似，跳過
       - < THRESHOLD_UPDATE: 更新現有聲紋（加權平均）
       - >= THRESHOLD_UPDATE 且 < THRESHOLD_NEW: 新增聲紋

    Args:
        name: 新語者名稱。
        folder: 音檔所在資料夾。
        indices: 需要使用的音檔索引（不含前置零）。
    """
    base = os.path.basename(folder)
    valid_files = []
    
    # 先檢查所有音檔是否存在
    for idx in indices:
        file_path = os.path.join(folder, f"{base}_{idx:02d}.wav")
        if not os.path.exists(file_path):
            print(f"音檔 {file_path} 不存在，已跳過。")
            continue
        valid_files.append((idx, file_path))

    if not valid_files:
        print(f"語者 {name} 沒有有效音檔，跳過建立。")
        return

    print(f"開始處理語者 {name}，共 {len(valid_files)} 個有效音檔")
    
    # 統計變數
    stats = {
        'processed': 0,
        'skipped_too_similar': 0,
        'updated': 0,
        'added_new': 0,
        'skipped_not_same_person': 0
    }
    
    # 建立語者（使用第一個音檔）
    first_file = valid_files[0][1]
    speaker_uuid = db.create_speaker(full_name=name, first_audio=first_file)
    print(f"已建立語者 {name} (UUID: {speaker_uuid})")
    
    # 處理第一個音檔，直接建立第一個聲紋
    first_embedding = identifier.audio_processor.extract_embedding(first_file)
    first_voiceprint_uuid = db.create_voiceprint(
        speaker_uuid,
        first_embedding,
        audio_source=f"{base}_{valid_files[0][0]:02d}.wav",
    )
    print(f"已建立初始聲紋 (音檔: {os.path.basename(first_file)})")
    
    # 處理其餘音檔
    for idx, file_path in valid_files[1:]:
        print(f"\n處理音檔: {os.path.basename(file_path)}")
        stats['processed'] += 1
        current_embedding = identifier.audio_processor.extract_embedding(file_path)
        
        # 獲取該語者的所有現有聲紋（包含向量）
        speaker_voiceprints = db.get_speaker_voiceprints(speaker_uuid, include_vectors=True)
        
        if not speaker_voiceprints:
            print("無法獲取語者聲紋，跳過此音檔")
            continue
            
        # 計算與該語者所有現有聲紋的距離，找出最小距離
        # 注意：這裡只比較當前語者的聲紋，確保是同一個人
        min_distance = float('inf')
        closest_voiceprint_uuid = None
        closest_voiceprint_data = None
        
        print(f"與該語者現有的 {len(speaker_voiceprints)} 個聲紋進行比較...")
        
        for vp_data in speaker_voiceprints:
            vp_uuid = str(vp_data['uuid'])  # 確保轉換為字符串
            vp_embedding = vp_data.get('vector')
            
            if vp_embedding is not None:
                # 如果是字典格式，取出 default 向量
                if isinstance(vp_embedding, dict):
                    vp_embedding = vp_embedding.get('default', vp_embedding)
                    
                vp_embedding = np.array(vp_embedding, dtype=float)
                distance = cosine(current_embedding, vp_embedding)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_voiceprint_uuid = vp_uuid
                    closest_voiceprint_data = vp_data
        
        print(f"與該語者聲紋的最小距離: {min_distance:.4f}")
        
        # 檢查音檔是否確實屬於該語者（距離不能超過 THRESHOLD_NEW）
        if min_distance >= THRESHOLD_NEW:
            print(f"⚠️  距離過大 ({min_distance:.4f} >= {THRESHOLD_NEW})，該音檔可能：")
            print(f"    1. 不屬於語者 {name}")
            print(f"    2. 音檔品質不佳或有雜訊")
            print(f"    3. 標記錯誤")
            print(f"    音檔: {os.path.basename(file_path)}")
            stats['skipped_not_same_person'] += 1
            continue
        
        # 確認是同個人後，根據距離決定操作
        if min_distance < THRESHOLD_LOW:
            print(f"距離過小 ({min_distance:.4f} < {THRESHOLD_LOW})，跳過此音檔（過於相似）")
            stats['skipped_too_similar'] += 1
            
        elif min_distance < THRESHOLD_UPDATE:
            print(f"距離適中 ({min_distance:.4f} < {THRESHOLD_UPDATE})，更新現有聲紋")
            # 使用現有的更新方法，讓它自動計算新的 update_count
            new_update_count = db.update_voiceprint(closest_voiceprint_uuid, current_embedding)
            if new_update_count > 0:
                print(f"已更新聲紋，新的更新次數: {new_update_count}")
                stats['updated'] += 1
            else:
                print("更新聲紋失敗")
            
        else:  # THRESHOLD_UPDATE <= min_distance < THRESHOLD_NEW
            print(f"距離較大 ({min_distance:.4f} >= {THRESHOLD_UPDATE})，為同個語者新增聲紋")
            # 為現有語者新增聲紋，確保使用正確的語者名稱
            new_voiceprint_uuid = db.create_voiceprint(
                speaker_uuid,
                current_embedding,
                audio_source=f"{base}_{idx:02d}.wav",
            )
            if new_voiceprint_uuid:
                print(f"已新增聲紋 (UUID: {new_voiceprint_uuid})")
                stats['added_new'] += 1
            else:
                print("新增聲紋失敗")

    # 顯示統計結果
    print(f"\n完成處理語者 {name}")
    print(f"統計結果：")
    print(f"  處理音檔: {stats['processed']}")
    print(f"  更新聲紋: {stats['updated']}")
    print(f"  新增聲紋: {stats['added_new']}")
    print(f"  跳過(過於相似): {stats['skipped_too_similar']}")
    print(f"  跳過(可能不屬於該語者): {stats['skipped_not_same_person']}")
    
    # 獲取最終的聲紋數量
    final_voiceprints = db.get_speaker_voiceprints(speaker_uuid)
    print(f"  最終聲紋總數: {len(final_voiceprints)}")


def build_speaker(name: str, folder: str, indices: List[int], mode: Optional[BuildMode] = None) -> None:
    """根據指定模式建立語者。

    Args:
        name: 新語者名稱。
        folder: 音檔所在資料夾。
        indices: 需要使用的音檔索引（不含前置零）。
        mode: 建立模式，如果為 None 則使用全域設定。
    """
    if mode is None:
        mode = CURRENT_MODE
        
    if mode == BuildMode.AVERAGE:
        print(f"\n使用模式1（平均模式）處理語者 {name}")
        build_speaker_average_mode(name, folder, indices)
    elif mode == BuildMode.INTELLIGENT:
        print(f"\n使用模式2（智慧模式）處理語者 {name}")
        build_speaker_intelligent_mode(name, folder, indices)
    else:
        print(f"未知的模式: {mode}")


if __name__ == "__main__":
    print(f"當前模式: {CURRENT_MODE.name}")
    print(f"閾值設定: LOW={THRESHOLD_LOW}, UPDATE={THRESHOLD_UPDATE}, NEW={THRESHOLD_NEW}")
    print("=" * 60)
    
    for sp_name, cfg in SPEAKER_CONFIG.items():
        build_speaker(sp_name, cfg["folder"], cfg["indices"])
        print("=" * 60)