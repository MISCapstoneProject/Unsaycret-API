"""預先建立語者並使用指定音檔平均建立聲紋。

此腳本示範如何在程式碼中預先指定音檔索引，
依序將其平均成一個聲紋，並建立對應語者。
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from modules.database import DatabaseService      # ← 新增
from modules.identification import SpeakerIdentifier

# === 語者設定 ==============================================================
# key: 新語者名稱
# value: {"folder": 資料夾路徑, "indices": [音檔索引列表]}
SPEAKER_CONFIG: Dict[str, Dict[str, List[int]]] = {
    "spk1": {
        "folder": "data/clean/speaker1",
        "indices": [1, 4, 5, 7, 8, 9, 12, 14, 15, 17, 18, 19, 20],
    },
    "spk2": {
        "folder": "data/clean/speaker2",
        "indices": [1, 3, 7, 8, 10, 11, 12, 14, 15, 17, 18, 20],
    },
    "spk3": {
        "folder": "data/clean/speaker3",
        "indices": [1, 2, 4, 5, 7, 8, 9, 14, 16, 17, 18],
    },
    "spk4": {
        "folder": "data/clean/speaker4",
        "indices": [2, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 19, 20],
    },
    "spk5": {
        "folder": "data/clean/speaker5",
        "indices": [1, 3, 4, 7, 9, 12, 14, 15, 16, 17, 18, 19, 20],
    },
    "spk6": {
        "folder": "data/clean/speaker6",
        "indices": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20],
    },
    "spk7": {
        "folder": "data/clean/speaker7",
        "indices": [1, 2, 8, 9, 10, 11, 13, 14, 16, 18, 19, 20],
    },
    "spk8": {
        "folder": "data/clean/speaker8",
        "indices": [2, 4, 5, 6, 9, 10, 11, 12, 15, 17, 20],
    },
    "spk9": {
        "folder": "data/clean/speaker9",
        "indices": [1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19],
    },
    "spk10": {
        "folder": "data/clean/speaker10",
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
db = DatabaseService()                            # ← 用新的物件


def build_speaker(name: str, folder: str, indices: List[int]) -> None:
    """依索引平均音檔嵌入並建立語者。

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
    speaker_uuid = db.create_speaker(full_name=name,first_audio=first_file)

    # 建立平均聲紋
    db.create_voiceprint(
        speaker_uuid,
        avg_embedding,
        audio_source="avg_of_indices",
    )
    print(f"已建立語者 {name} (UUID: {speaker_uuid})，使用 {len(embeddings)} 個音檔。")


if __name__ == "__main__":
    for sp_name, cfg in SPEAKER_CONFIG.items():
        build_speaker(sp_name, cfg["folder"], cfg["indices"])