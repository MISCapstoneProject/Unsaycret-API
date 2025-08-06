"""手動語者辨識測試腳本

此腳本允許一次針對某個 speaker 的多個檔案進行辨識，
只要給定資料夾路徑和索引列表即可自動跑完。

使用方式:
    python -m examples.manual_add_speaker data/clean/speaker1 --indices 5 7 8
    # 會自動處理 speaker1_05.wav, speaker1_07.wav, speaker1_08.wav
"""

import argparse
from pathlib import Path
from modules.identification import SpeakerIdentifier


def main() -> None:
    """解析參數並依序執行語者辨識。"""
    parser = argparse.ArgumentParser(
        description="針對某個 speaker 資料夾內指定編號的檔案，執行語者辨識"
    )
    parser.add_argument(
        "speaker_dir",
        help="speaker 資料夾路徑（如 data/clean/speaker1）"
    )
    parser.add_argument(
        "--indices", "-i",
        required=True,
        nargs="+",
        type=int,
        metavar="N",
        help="要處理的檔案編號列表（不含前綴零），如 5 7 8"
    )
    args = parser.parse_args()

    speaker_path = Path(args.speaker_dir)
    if not speaker_path.is_dir():
        parser.error(f"{speaker_path!r} 不是一個有效的目錄")

    speaker_name = speaker_path.name  # e.g. "speaker1"
    identifier = SpeakerIdentifier()

    for idx in args.indices:
        # 兩位數補零
        filename = f"{speaker_name}_{idx:02d}.wav"
        audio_path = speaker_path / filename

        print(f"\n🔍 處理：{audio_path}")
        if not audio_path.is_file():
            print(f"⚠️ 檔案不存在，跳過：{audio_path}")
            continue

        result = identifier.process_audio_file(str(audio_path))
        if result:
            speaker_id, speaker_label, distance = result
            print(f"▶️ 語者: {speaker_label} (UUID: {speaker_id}) 相似度 {distance:.3f}")
        else:
            print("⚠️ 處理失敗或無法辨識")

if __name__ == "__main__":
    main()

#python -m examples.manual_add_speaker data/clean/speaker1 -i 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20