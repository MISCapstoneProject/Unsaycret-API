"""
Weaviate Collections V2 測試模組

此模組負責測試新版本的 Weaviate 資料庫結構，包含：
- 測試資料插入功能
- 複雜查詢驗證
- 關聯性測試
- 語義搜尋測試

使用方法：
    python -m utils.test_init_v2_collections
    python -m utils.test_init_v2_collections --host localhost --port 8080
    python -m utils.test_init_v2_collections --no-test-data  # 只建立集合，不插入測試資料
"""

import weaviate  # type: ignore
import weaviate.classes.config as wc  # type: ignore
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timezone
import sys
import time
from tqdm import tqdm
from dataclasses import dataclass

from utils.logger import get_logger
from utils.init_v2_collections import (
    WeaviateV2CollectionManager, 
    SpeakerData, 
    SessionData, 
    SpeechLogData, 
    VoicePrintData,
    ensure_weaviate_v2_collections
)

# 創建本模組的日誌器
logger = get_logger(__name__)


class WeaviateV2TestManager:
    """Weaviate V2 測試管理器"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        """
        初始化測試管理器
        
        Args:
            host: Weaviate 服務器主機地址
            port: Weaviate 服務器端口
        """
        self.host = host
        self.port = port
        self.manager = WeaviateV2CollectionManager(host, port)
    
    def __enter__(self):
        """進入 context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """離開 context manager"""
        if self.manager.client:
            self.manager.disconnect()
    
    def insert_comprehensive_test_data(self) -> bool:
        """
        插入完整的測試資料（包含複雜關聯）
        
        Returns:
            bool: 是否成功插入測試資料
        """
        # 確保連接
        if not self.manager.client:
            self.manager.connect()
        
        try:
            logger.info("🧪 開始插入綜合測試資料...")
            
            # ========== 1. 插入 3 個 Speaker ==========
            speaker_collection = self.manager.client.collections.get("Speaker")
            speakers_data = [
                SpeakerData(
                    speaker_id=1,
                    full_name="測試用戶一號",
                    nickname="Alice",
                    gender="female",
                    meet_count=5,
                    meet_days=3,
                    first_audio="alice_sample_001.wav"
                ),
                SpeakerData(
                    speaker_id=2,
                    full_name="測試用戶二號", 
                    nickname="Bob",
                    gender="male",
                    meet_count=3,
                    meet_days=2,
                    first_audio="bob_sample_001.wav"
                ),
                SpeakerData(
                    speaker_id=3,
                    full_name="測試用戶三號",
                    nickname="Carol",
                    gender="female",
                    meet_count=1,
                    meet_days=1,
                    first_audio="carol_sample_001.wav"
                )
            ]
            
            speaker_uuids = []
            with speaker_collection.batch as batch:
                for speaker in speakers_data:
                    speaker_uuid = batch.add_object({
                        "speaker_id": speaker.speaker_id,
                        "full_name": speaker.full_name,
                        "nickname": speaker.nickname,
                        "gender": speaker.gender,
                        "created_at": speaker.created_at,
                        "last_active_at": speaker.last_active_at,
                        "meet_count": speaker.meet_count,
                        "meet_days": speaker.meet_days,
                        "voiceprint_ids": speaker.voiceprint_ids,
                        "first_audio": speaker.first_audio
                    })
                    speaker_uuids.append(speaker_uuid)
            
            logger.info(f"✅ 成功插入 {len(speakers_data)} 個 Speaker")
            
            # ========== 2. 插入 3 個 Session ==========
            session_collection = self.manager.client.collections.get("Session")
            sessions_data = [
                SessionData(
                    session_id="meeting_001",
                    session_type="meeting",
                    title="週會討論：項目進度報告",
                    start_time=datetime(2025, 7, 19, 10, 0, 0, tzinfo=timezone.utc),
                    end_time=datetime(2025, 7, 19, 11, 30, 0, tzinfo=timezone.utc),
                    summary="討論本週工作進度和下週計劃，重點關注API開發和資料庫重構"
                ),
                SessionData(
                    session_id="call_002",
                    session_type="call",
                    title="客戶諮詢電話：產品功能介紹",
                    start_time=datetime(2025, 7, 19, 14, 0, 0, tzinfo=timezone.utc),
                    end_time=datetime(2025, 7, 19, 14, 45, 0, tzinfo=timezone.utc),
                    summary="客戶詢問產品功能和價格，重點介紹語音識別和聲紋技術"
                ),
                SessionData(
                    session_id="demo_003",
                    session_type="demo",
                    title="系統演示：語音識別功能測試",
                    start_time=datetime(2025, 7, 19, 16, 0, 0, tzinfo=timezone.utc),
                    summary="展示語音識別系統的各項功能和準確度表現"
                )
            ]
            
            session_uuids = []
            with session_collection.batch as batch:
                for i, session in enumerate(sessions_data):
                    # 建立參與者關聯
                    if i == 0:  # meeting: Alice + Bob
                        participants = [speaker_uuids[0], speaker_uuids[1]]
                    elif i == 1:  # call: Bob + Carol  
                        participants = [speaker_uuids[1], speaker_uuids[2]]
                    else:  # demo: 所有人
                        participants = speaker_uuids
                    
                    session_uuid = batch.add_object(
                        properties={
                            "session_id": session.session_id,
                            "session_type": session.session_type,
                            "title": session.title,
                            "start_time": session.start_time,
                            "end_time": session.end_time,
                            "summary": session.summary
                        },
                        references={
                            "participants": participants
                        }
                    )
                    session_uuids.append(session_uuid)
            
            logger.info(f"✅ 成功插入 {len(sessions_data)} 個 Session")
            
            # ========== 3. 插入 12 條 SpeechLog ==========
            speechlog_collection = self.manager.client.collections.get("SpeechLog")
            speechlogs_data = [
                # Meeting session 語音記錄
                SpeechLogData("歡迎大家參加本週的週會", datetime(2025, 7, 19, 10, 5, 0, tzinfo=timezone.utc), 0.95, 2.3),
                SpeechLogData("我們先來回顧一下上週的工作成果", datetime(2025, 7, 19, 10, 7, 0, tzinfo=timezone.utc), 0.92, 3.1),
                SpeechLogData("我負責的API開發已經完成80%", datetime(2025, 7, 19, 10, 10, 0, tzinfo=timezone.utc), 0.88, 2.8),
                SpeechLogData("資料庫結構重構的部分需要再討論", datetime(2025, 7, 19, 10, 12, 0, tzinfo=timezone.utc), 0.90, 3.5),
                
                # Call session 語音記錄
                SpeechLogData("您好，請問有什麼可以幫助您的嗎？", datetime(2025, 7, 19, 14, 2, 0, tzinfo=timezone.utc), 0.97, 2.1),
                SpeechLogData("我想了解貴公司的語音識別產品", datetime(2025, 7, 19, 14, 5, 0, tzinfo=timezone.utc), 0.89, 2.7),
                SpeechLogData("這個系統的準確度如何？", datetime(2025, 7, 19, 14, 8, 0, tzinfo=timezone.utc), 0.94, 2.2),
                SpeechLogData("我們的語音識別準確度達到95%以上", datetime(2025, 7, 19, 14, 10, 0, tzinfo=timezone.utc), 0.96, 3.4),
                
                # Demo session 語音記錄
                SpeechLogData("現在開始演示語音識別功能", datetime(2025, 7, 19, 16, 2, 0, tzinfo=timezone.utc), 0.93, 2.8),
                SpeechLogData("這個系統可以識別多種語言", datetime(2025, 7, 19, 16, 5, 0, tzinfo=timezone.utc), 0.91, 3.2),
                
                # 無 session 的語音記錄
                SpeechLogData("今天天氣真好，適合測試語音功能", datetime(2025, 7, 19, 17, 0, 0, tzinfo=timezone.utc), 0.92, 3.1),
                SpeechLogData("希望未來能支援更多語言和方言", datetime(2025, 7, 19, 17, 5, 0, tzinfo=timezone.utc), 0.89, 3.8)
            ]
            
            with speechlog_collection.batch as batch:
                for i, speechlog in enumerate(speechlogs_data):
                    # 分配 speaker 和 session
                    speaker_ref = speaker_uuids[i % len(speaker_uuids)]
                    
                    # 前 4 條 -> meeting session, 中間 4 條 -> call session, 再 2 條 -> demo session, 最後 2 條無 session
                    if i < 4:
                        session_ref = session_uuids[0]  # meeting
                    elif i < 8:
                        session_ref = session_uuids[1]  # call
                    elif i < 10:
                        session_ref = session_uuids[2]  # demo
                    else:
                        session_ref = None  # 無 session
                    
                    references = {"speaker": speaker_ref}
                    if session_ref:
                        references["session"] = session_ref
                    
                    batch.add_object(
                        properties={
                            "content": speechlog.content,
                            "timestamp": speechlog.timestamp,
                            "confidence": speechlog.confidence,
                            "duration": speechlog.duration,
                            "language": speechlog.language
                        },
                        references=references
                    )
            
            logger.info(f"✅ 成功插入 {len(speechlogs_data)} 條 SpeechLog")
            
            # ========== 4. 插入對應的 VoicePrint 記錄 ==========
            voiceprint_collection = self.manager.client.collections.get("VoicePrint")
            voiceprints_data = [
                VoicePrintData(
                    voiceprint_id="vp_001_alice",
                    update_count=5,
                    sample_count=25,
                    quality_score=0.94,
                    speaker_name="測試用戶一號"
                ),
                VoicePrintData(
                    voiceprint_id="vp_002_bob",
                    update_count=3,
                    sample_count=18,
                    quality_score=0.91,
                    speaker_name="測試用戶二號"
                ),
                VoicePrintData(
                    voiceprint_id="vp_003_carol",
                    update_count=2,
                    sample_count=12,
                    quality_score=0.87,
                    speaker_name="測試用戶三號"
                )
            ]
            
            with voiceprint_collection.batch as batch:
                for i, voiceprint in enumerate(voiceprints_data):
                    batch.add_object(
                        properties={
                            "voiceprint_id": voiceprint.voiceprint_id,
                            "created_at": voiceprint.created_at,
                            "updated_at": voiceprint.updated_at,
                            "update_count": voiceprint.update_count,
                            "sample_count": voiceprint.sample_count,
                            "quality_score": voiceprint.quality_score,
                            "speaker_name": voiceprint.speaker_name
                        },
                        references={
                            "speaker": speaker_uuids[i]
                        }
                    )
            
            logger.info(f"✅ 成功插入 {len(voiceprints_data)} 個 VoicePrint")
            
            logger.info("🎉 所有測試資料插入完成！")
            return True
            
        except Exception as e:
            logger.error(f"插入測試資料時發生錯誤: {str(e)}")
            return False
    
    def run_comprehensive_tests(self) -> bool:
        """
        執行綜合測試驗證
        
        Returns:
            bool: 是否所有測試都通過
        """
        if not self.manager.client:
            self.manager.connect()
        
        try:
            logger.info("🧪 開始執行綜合測試驗證...")
            
            # 測試 1: 基本集合查詢
            logger.info("📋 測試 1: 基本集合查詢")
            collections = ["Speaker", "Session", "SpeechLog", "VoicePrint"]
            for collection_name in collections:
                collection = self.manager.client.collections.get(collection_name)
                count = len(collection.query.fetch_objects().objects)
                logger.info(f"   {collection_name}: {count} 條記錄")
            
            # 測試 2: Session → participants 關聯查詢
            logger.info("🔗 測試 2: Session → participants 關聯查詢")
            session_collection = self.manager.client.collections.get("Session")
            sessions = session_collection.query.fetch_objects(
                include_vector=False,
                return_references=[
                    wc.QueryReference(
                        link_on="participants",
                        return_properties=["speaker_id", "full_name", "nickname"]
                    )
                ]
            )
            logger.info(f"   找到 {len(sessions.objects)} 個 session 及其參與者")
            for session in sessions.objects:
                participants_count = len(session.references.get("participants", []))
                logger.info(f"   Session '{session.properties['title']}': {participants_count} 個參與者")
            
            # 測試 3: 語義搜尋測試
            logger.info("🔍 測試 3: 語義搜尋測試")
            speechlog_collection = self.manager.client.collections.get("SpeechLog")
            
            # 搜尋「會議討論工作」
            speech_results = speechlog_collection.query.near_text(
                query="會議討論工作",
                limit=3,
                return_metadata=wc.MetadataQuery(score=True)
            )
            logger.info(f"   搜尋 '會議討論工作': 找到 {len(speech_results.objects)} 條相關記錄")
            
            # 搜尋「語音識別產品」
            product_results = speechlog_collection.query.near_text(
                query="語音識別產品",
                limit=3,
                return_metadata=wc.MetadataQuery(score=True)
            )
            logger.info(f"   搜尋 '語音識別產品': 找到 {len(product_results.objects)} 條相關記錄")
            
            # 測試 4: Session 語義搜尋
            logger.info("📝 測試 4: Session 語義搜尋")
            session_results = session_collection.query.near_text(
                query="項目進度報告",
                limit=2,
                return_metadata=wc.MetadataQuery(score=True)
            )
            logger.info(f"   搜尋 '項目進度報告': 找到 {len(session_results.objects)} 個相關 session")
            
            # 測試 5: Speaker ID 精確查詢
            logger.info("🎯 測試 5: Speaker ID 精確查詢")
            speaker_collection = self.manager.client.collections.get("Speaker")
            speaker_results = speaker_collection.query.fetch_objects(
                where=wc.Filter.by_property("speaker_id").equal(1),
                include_vector=False
            )
            logger.info(f"   查詢 speaker_id=1: 找到 {len(speaker_results.objects)} 個 speaker")
            
            # 測試 6: 時間範圍查詢
            logger.info("⏰ 測試 6: 時間範圍查詢")
            time_results = speechlog_collection.query.fetch_objects(
                where=wc.Filter.by_property("timestamp").greater_than(
                    datetime(2025, 7, 19, 14, 0, 0, tzinfo=timezone.utc)
                ),
                include_vector=False
            )
            logger.info(f"   查詢 14:00 後的記錄: 找到 {len(time_results.objects)} 條記錄")
            
            # 測試 7: 複雜組合查詢
            logger.info("🔀 測試 7: 複雜組合查詢")
            complex_results = speechlog_collection.query.fetch_objects(
                where=wc.Filter.by_property("confidence").greater_than(0.9) &
                      wc.Filter.by_property("language").equal("zh-TW"),
                include_vector=False
            )
            logger.info(f"   查詢高信心度中文記錄: 找到 {len(complex_results.objects)} 條記錄")
            
            logger.info("✅ 所有測試驗證完成！")
            return True
            
        except Exception as e:
            logger.error(f"執行測試時發生錯誤: {str(e)}")
            return False


def run_full_test_suite(host: str = "localhost", port: int = 8080, 
                       skip_collection_setup: bool = False) -> bool:
    """
    執行完整的測試套件
    
    Args:
        host: Weaviate 服務器主機地址
        port: Weaviate 服務器端口
        skip_collection_setup: 是否跳過集合建立（假設已存在）
        
    Returns:
        bool: 是否所有測試都通過
    """
    logger.info("🚀 開始執行 Weaviate V2 完整測試套件...")
    
    try:
        # 步驟 1: 確保集合存在（除非跳過）
        if not skip_collection_setup:
            logger.info("📋 步驟 1: 確保 V2 集合存在...")
            if not ensure_weaviate_v2_collections(host, port, insert_test_data=False):
                logger.error("❌ 集合建立失敗")
                return False
        else:
            logger.info("⏭️  跳過集合建立步驟")
        
        # 步驟 2: 插入測試資料和執行驗證
        with WeaviateV2TestManager(host, port) as test_manager:
            logger.info("📊 步驟 2: 插入綜合測試資料...")
            if not test_manager.insert_comprehensive_test_data():
                logger.error("❌ 測試資料插入失敗")
                return False
            
            logger.info("🧪 步驟 3: 執行綜合測試驗證...")
            if not test_manager.run_comprehensive_tests():
                logger.error("❌ 測試驗證失敗")
                return False
        
        logger.info("🎉 所有測試通過！Weaviate V2 系統運行正常")
        return True
        
    except Exception as e:
        logger.error(f"❌ 測試套件執行失敗: {str(e)}")
        return False


def main() -> None:
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Weaviate V2 集合測試工具")
    parser.add_argument("--host", default="localhost", help="Weaviate 服務器主機地址")
    parser.add_argument("--port", type=int, default=8080, help="Weaviate 服務器端口")
    parser.add_argument("--skip-setup", action="store_true", help="跳過集合建立（假設已存在）")
    parser.add_argument("--test-only", action="store_true", help="只執行測試，不插入資料")
    
    args = parser.parse_args()
    
    if args.test_only:
        # 只執行測試驗證
        with WeaviateV2TestManager(args.host, args.port) as test_manager:
            success = test_manager.run_comprehensive_tests()
    else:
        # 執行完整測試套件
        success = run_full_test_suite(
            host=args.host,
            port=args.port,
            skip_collection_setup=args.skip_setup
        )
    
    if success:
        print("🎉 測試通過！Weaviate V2 系統運行正常")
        print("📊 測試內容包含:")
        print("   ✅ 基本集合查詢")
        print("   ✅ 關聯查詢驗證")
        print("   ✅ 語義搜尋功能")
        print("   ✅ 精確查詢功能")
        print("   ✅ 時間範圍查詢")
        print("   ✅ 複雜組合查詢")
        sys.exit(0)
    else:
        print("❌ 測試失敗！請檢查錯誤日誌")
        sys.exit(1)


if __name__ == "__main__":
    main()
