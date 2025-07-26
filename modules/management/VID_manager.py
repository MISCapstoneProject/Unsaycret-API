"""
===============================================================================
語者與聲紋管理系統 (Speaker and Voiceprint Management System)
===============================================================================

版本：v1.1.0 
作者：CYouuu
最後更新：2025-05-19

功能摘要：
-----------
本模組提供語者（Speaker）與聲紋（VoicePrint）在 Weaviate 資料庫的管理功能。
CLI 互動介面與資料操作分離，結構現代、易於維護。主要功能包括：

 1. 語者查詢：列出所有已註冊的語者及其基本資訊
 2. 語者編輯：修改語者名稱和其他相關屬性
 3. 聲紋管理：查看語者關聯的所有聲紋向量及其屬性
 4. 聲紋遷移：將聲紋從一個語者轉移到另一個語者
 5. 刪除功能：移除語者或特定聲紋向量

技術架構：
-----------
 - 資料庫：Weaviate 向量資料庫 (透過 VID_database 抽象層存取)
 - 介面：基於命令列的互動式介面
 - 架構：模組化設計，分離核心邏輯與介面層

使用方式：
-----------
1. 直接執行模組：
   ```
   python speaker_manager.py
   ```

2. 從整合系統選單進入：
   ```
   python speaker_system_v2.py
   # 選擇選項 2
   ```

前置需求：
-----------
 - Python 3.9+
 - Weaviate 向量資料庫 (需通過 Docker 啟動)
 - weaviate-client 套件

注意事項：
-----------
 - 使用前請確保 Weaviate 已啟動並初始化必要集合
 - 刪除操作不可復原，請謹慎操作
 - 部分功能可能需要資料庫管理權限

詳細資訊：
-----------
請參考專案文件: https://github.com/LCY000/ProjectStudy_SpeechRecognition

===============================================================================
"""
import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))
if root not in sys.path:
    sys.path.insert(0, root)
import re
import sys
import uuid
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Logging 設定
# ---------------------------------------------------------------------------
from utils.logger import get_logger

# 創建本模組的日誌器
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# UUID 工具
# ---------------------------------------------------------------------------
UUID_PATTERN = re.compile(r"^[0-9a-fA-F-]{36}$")

def valid_uuid(value: str) -> bool:
    """檢查字串是否為有效 UUID 格式。"""
    return bool(UUID_PATTERN.match(value))

# ---------------------------------------------------------------------------
# SpeakerManager (資料庫操作)
# ---------------------------------------------------------------------------

class SpeakerManager:
    """
    封裝所有與資料庫互動的語者管理操作。
    使用 VID_database 模組提供的 DatabaseService 作為統一接口。
    """

    def __init__(self):
        # 使用 DatabaseService 作為統一接口
        from modules.database.database import DatabaseService
        self._db = DatabaseService()
        
    def list_all_speakers(self) -> List[Dict[str, Any]]:
        """列出所有語者。"""
        try:
            speakers = self._db.list_all_speakers()
            # 使用 DatabaseService 的接口只需排序後返回即可
            speakers.sort(key=lambda s: s.get("last_active_time", ""), reverse=True)
            return speakers
        except Exception as exc:
            logger.error(f"列出語者時發生錯誤: {exc}")
            return []
            
    def get_speaker(self, speaker_uuid: str) -> Optional[Any]:
        """取得單一語者物件。"""
        try:
            return self._db.get_speaker(speaker_uuid)
        except Exception as exc:
            logger.error(f"獲取語者詳細資訊時發生錯誤: {exc}")
            return None

    def update_speaker_name(self, speaker_uuid: str, new_name: str) -> bool:
        """
        更改語者名稱（V2版本），並同步更新所有該語者底下聲紋的 speaker_name。
        """
        try:
            # V2: 默認更新 full_name
            return self._db.update_speaker_name(speaker_uuid, new_full_name=new_name)
        except Exception as exc:
            logger.error(f"更改語者名稱時發生錯誤: {exc}")
            return False

    def update_speaker(self, speaker_uuid: str, update_fields: dict) -> bool:
        """
        通用語者資料多欄位更新（V2，支援 PATCH）
        Args:
            speaker_uuid: 語者 UUID
            update_fields: 欲更新的欄位 dict
        Returns:
            bool: 是否成功
        """
        try:
            return self._db.update_speaker(speaker_uuid, update_fields)
        except Exception as exc:
            logger.error(f"更新語者欄位時發生錯誤: {exc}")
            return False

    def delete_speaker(self, speaker_uuid: str) -> bool:
        """刪除語者。"""
        try:
            return self._db.delete_speaker(speaker_uuid)
        except Exception as exc:
            logger.error(f"刪除語者時發生錯誤: {exc}")
            return False

    def transfer_voiceprints(
        self, source_uuid: str, dest_uuid: str, voiceprint_ids: Optional[List[str]] = None
    ) -> bool:
        """
        將來源語者的聲紋轉移到目標語者，並同步更新聲紋的 speaker_id 與 speaker_name。
        若來源語者已無聲紋，則自動刪除該語者。
        """
        try:
            return self._db.transfer_voiceprints(source_uuid, dest_uuid, voiceprint_ids)
        except Exception as exc:
            logger.error(f"轉移聲紋時發生錯誤: {exc}")
            return False

    def cleanup(self) -> None:
        """資料庫清理（示意、未實現）。"""
        try:
            self._db.database_cleanup()
            logger.info("資料庫清理完成")
        except Exception as exc:
            logger.error(f"資料庫清理時發生錯誤: {exc}") 
    
    def check_and_repair_database(self) -> None:
        """
        資料庫檢查與修復（Database check & repair）：
        1. 檢查每個 Speaker 的 voiceprint_ids 是否都存在並移除不存在的 ID
        2. 檢查 VoicePrint 的 speaker_name／ReferenceProperty 是否正確並修正
        3. 找出沒被任何 Speaker 參考的 VoicePrint
        4. (可選) 自動掛回孤兒 VoicePrint；結果在 step4_relinked_vp
        """
        try:
            report: Dict[str, Any] = self._db.check_and_repair_database()

            # ── 封裝輸出邏輯 ───────────────────────────────────────────
            messages = []

            # Step 1
            missing_vp = report.get("step1_missing_vp_count", 0)
            if missing_vp:
                messages.append(f"【步驟1】已移除 {missing_vp} 個不存在的 VoicePrint ID")
            else:
                messages.append("【步驟1】所有 Speaker 的 voiceprint_ids 均正常")

            # Step 2
            err_vps = report.get("step2_error_vp_ids", [])
            if err_vps:
                err_list = ", ".join(str(v)[:8] for v in err_vps)
                messages.append(f"【步驟2】修正 {len(err_vps)} 個 VoicePrint 屬性異常：{err_list}")
            else:
                messages.append("【步驟2】所有 VoicePrint 屬性均正確")

            # Step 3
            orphan_vps = report.get("step3_unreferenced_vp_ids", [])
            if orphan_vps:
                orphan_list = ", ".join(str(v)[:8] for v in orphan_vps)
                messages.append(f"【步驟3】{len(orphan_vps)} 個 VoicePrint 未被 Speaker 參考：{orphan_list}")
            else:
                messages.append("【步驟3】所有 VoicePrint 均有被 Speaker 參考")            # Step 4
            relinked = report.get("step4_relinked_vp")
            if relinked:
                total = sum(len(v) for v in relinked.values())
                sp_cnt = len(relinked)
                messages.append(f"【步驟4】自動掛回 {total} 個 VoicePrint 至 {sp_cnt} 位 Speaker")
            
            # 刪除的孤兒聲紋
            deleted_orphans = report.get("step4_deleted_orphans", [])
            if deleted_orphans:
                messages.append(f"【步驟4】已刪除 {len(deleted_orphans)} 個無法確定歸屬的孤兒聲紋")
                if len(deleted_orphans) <= 10:
                    orphan_list = ", ".join(str(v)[:8] for v in deleted_orphans)
                    messages.append(f"刪除的聲紋 ID: {orphan_list}")
                else:
                    messages.append(f"(聲紋數量過多，僅顯示前10個)")
                    orphan_list = ", ".join(str(v)[:8] for v in deleted_orphans[:10])
                    messages.append(f"刪除的聲紋 ID: {orphan_list}...")

            # 輸出
            for msg in messages:
                logger.info(msg)            # ✅ 對外採用 logger
                print(msg)                  #    如果還想在 CLI 顯示，可保留

            if report.get("success"):
                logger.info("✅ 資料庫檢查與修復已完成")
                print("\n✅ 資料庫檢查與修復已完成\n")
            else:
                logger.warning("⚠️ 資料庫檢查完成，但部分步驟失敗")
                print("\n⚠️ 資料庫檢查完成，但部分步驟失敗\n")

        except Exception as exc:
            logger.exception("❌ 執行資料庫檢查與修復時發生錯誤")
            print(f"❌ 執行資料庫檢查與修復時發生錯誤：{exc}")


# ---------------------------------------------------------------------------
# CLI (Command‑line Interface)
# ---------------------------------------------------------------------------

class SpeakerManagerCLI:
    """命令列互動介面 (CLI)。"""
    MENU = (
        """
============================================================
                       語者與聲紋管理系統
============================================================
1. 列出所有語者
2. 檢視語者詳細資訊
3. 更改語者名稱
4. 轉移聲紋到其他語者
5. 刪除語者
6. 資料庫清理與修復
0. 離開
------------------------------------------------------------
"""
    )

    def __init__(self, manager: SpeakerManager) -> None:
        self.manager = manager
        self.index2uuid: Dict[str, str] = {}

    @staticmethod
    def _print_speakers_table(speakers: List[Dict[str, Any]]) -> None:
        header = f"{'No.':<4} | {'ID':<36} | {'名稱':<20} | {'聲紋數量':<10} | {'最後活動時間'}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for idx, sp in enumerate(speakers, start=1):
            print(
                f"{idx:<4} | {sp['uuid']:<36} | {sp['name']:<20} | "
                f"{sp['voiceprint_count']:<10} | {sp['last_active_time']}"
            )
        print("-" * len(header))

    def _resolve_id(self, raw: str) -> Optional[str]:
        raw = raw.strip()
        return self.index2uuid.get(raw) if raw.isdigit() else raw

    def _action_list(self) -> None:
        speakers = self.manager.list_all_speakers()
        if not speakers:
            print("目前沒有語者紀錄。")
            return
        self.index2uuid = {str(i): sp["uuid"] for i, sp in enumerate(speakers, start=1)}
        self._print_speakers_table(speakers)

    def _action_view(self) -> None:
        raw = input("請輸入語者序號或 ID: ")
        sp_id = self._resolve_id(raw)
        if not sp_id or not valid_uuid(sp_id):
            print("❌ 無效的語者 ID，請重新嘗試。")
            return
        obj = self.manager.get_speaker(sp_id)
        if obj is None:
            print("❌ 查無此語者。")
            return
        props = obj.properties
        print("\n語者詳細資訊:")
        print(f"UUID            : {obj.uuid}")
        print(f"名稱            : {props.get('name', '未命名')}")
        print(f"建立時間        : {props.get('create_time', '未知')}")
        print(f"最後活動時間    : {props.get('last_active_time', '未知')}")
        print(f"聲紋數量        : {len(props.get('voiceprint_ids', []))}\n")

    def _action_rename(self) -> None:
        raw = input("請輸入語者序號或 ID: ")
        sp_id = self._resolve_id(raw)
        if not sp_id or not valid_uuid(sp_id):
            print("❌ 無效的語者 ID。")
            return
        new_name = input("請輸入新名稱: ").strip()
        if not new_name:
            print("❌ 名稱不可為空。")
            return
        if self.manager.update_speaker_name(sp_id, new_name):
            print(f"✅ 已更新語者名稱為：{new_name}")
        else:
            print("❌ 更新失敗。")

    def _action_delete(self) -> None:
        raw = input("請輸入要刪除的語者序號或 ID: ")
        sp_id = self._resolve_id(raw)
        if not sp_id or not valid_uuid(sp_id):
            print("❌ 無效的語者 ID。")
            return
        confirm = input(f"⚠️ 警告：刪除操作不可逆，確定要刪除語者 {sp_id}？(Y/n): ")
        if confirm.lower() not in ["y", "yes"]:
            print("已取消刪除操作。")
            return
        if self.manager.delete_speaker(sp_id):
            print("✅ 語者已刪除。")
        else:
            print("❌ 刪除失敗。")

    def _action_transfer(self) -> None:
        src_raw = input("請輸入來源語者序號或 ID: ")
        src_id = self._resolve_id(src_raw)
        if not src_id or not valid_uuid(src_id):
            print("❌ 無效的來源語者 ID。")
            return
        dest_raw = input("請輸入目標語者序號或 ID: ")
        dest_id = self._resolve_id(dest_raw)
        if not dest_id or not valid_uuid(dest_id):
            print("❌ 無效的目標語者 ID。")
            return
        if src_id == dest_id:
            print("❌ 來源和目標語者不能相同。")
            return
        confirm = input(
            f"⚠️ 確定要將來源語者 {src_id} 的所有聲紋轉移到目標語者 {dest_id}？(Y/n): "
        )
        if confirm.lower() not in ["y", "yes"]:
            print("已取消轉移操作。")
            return
        if self.manager.transfer_voiceprints(src_id, dest_id):
            print("✅ 聲紋成功轉移。")
        else:
            print("❌ 轉移失敗。")

    def _action_cleanup(self) -> None:
        confirm = input(f"⚠️ 是否確定執行資料庫檢查與修復？部分操作可能不可逆。(Y/n): ")
        if confirm.lower() not in ["y", "yes"]:
            print("已取消檢查與修復操作。")
            return
        self.manager.check_and_repair_database()
        print("✅ 資料庫檢查與修復完成。\n")

    def run(self) -> None:
        """開始運行命令列界面。"""
        actions = {
            "1": self._action_list,
            "2": self._action_view,
            "3": self._action_rename,
            "4": self._action_transfer,
            "5": self._action_delete,
            "6": self._action_cleanup,
            "0": lambda: sys.exit(0),
        }
        while True:
            print(self.MENU)
            choice = input("請選擇操作 (0-6): ").strip()
            action = actions.get(choice)
            if action:
                action()
            else:
                print("❌ 無效選項，請重新輸入 0-6。\n")

# ---------------------------------------------------------------------------
# 入口點
# ---------------------------------------------------------------------------

def main() -> None:
    """程序主入口點"""
    manager = SpeakerManager()
    cli = SpeakerManagerCLI(manager)
    cli.run()

if __name__ == "__main__":
    main()
