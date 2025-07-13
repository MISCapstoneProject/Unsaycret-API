# services/handlers/speaker_handler.py
"""
語者管理業務邏輯處理器

此模組包含所有與語者管理相關的業務邏輯，
將 HTTP 層與業務邏輯分離，提高程式碼的可維護性和可測試性。
"""
from typing import Dict, Any, Optional, List
from fastapi import HTTPException
from modules.management.VID_manager import SpeakerManager
from modules.identification.VID_identify_v5 import AudioProcessor
from modules.database.database import DatabaseService
from utils.logger import get_logger

logger = get_logger(__name__)

class SpeakerHandler:
    """語者管理業務邏輯處理器"""
    
    def __init__(self) -> None:
        """初始化語者處理器"""
        self.speaker_manager = SpeakerManager()
        # 初始化語音處理器和資料庫服務（用於純讀取的語音驗證）
        self.audio_processor = AudioProcessor()
        self.database = DatabaseService()
    
    def rename_speaker(
        self, 
        speaker_id: str, 
        current_name: str, 
        new_name: str
    ) -> Dict[str, Any]:
        """
        更改語者名稱的業務邏輯
        
        Args:
            speaker_id: 語者的唯一識別碼
            current_name: 當前語者名稱
            new_name: 新的語者名稱
            
        Returns:
            Dict[str, Any]: 包含操作結果的字典
            
        Raises:
            HTTPException: 當操作失敗時拋出相應的 HTTP 異常
        """
        try:
            # 1. 驗證輸入參數
            if not speaker_id.strip():
                raise HTTPException(status_code=400, detail="語者ID不能為空")
            
            if not new_name.strip():
                raise HTTPException(status_code=400, detail="新名稱不能為空")
            
            # 2. 檢查語者是否存在
            obj = self.speaker_manager.get_speaker(speaker_id)
            if not obj:
                raise HTTPException(
                    status_code=404, 
                    detail=f"找不到ID為 {speaker_id} 的語者"
                )
            
            # 3. 驗證當前名稱是否匹配（安全檢查）
            current_speaker_name = obj.properties.get('name', '')
            if current_name.strip() != current_speaker_name:
                raise HTTPException(
                    status_code=400, 
                    detail=f"提供的當前名稱 '{current_name}' 與資料庫中的名稱 '{current_speaker_name}' 不符"
                )
            
            # 4. 執行改名操作
            success = self.speaker_manager.update_speaker_name(
                speaker_id, 
                new_name.strip()
            )
            
            if success:
                logger.info(f"成功將語者 {speaker_id} 從 '{current_name}' 更名為 '{new_name}'")
                return {
                    "success": True,
                    "message": f"成功將語者 '{current_name}' 更名為 '{new_name}'",
                    "data": {
                        "speaker_id": speaker_id,
                        "old_name": current_name,
                        "new_name": new_name.strip()
                    }
                }
            else:
                logger.error(f"更名操作失敗：speaker_id={speaker_id}")
                raise HTTPException(
                    status_code=500, 
                    detail="更名操作失敗，請檢查日誌以獲取詳細資訊"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"更名操作發生未預期錯誤：{str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"伺服器內部錯誤：{str(e)}"
            )
    
    def transfer_voiceprints(
        self,
        source_speaker_id: str,
        source_speaker_name: str,
        target_speaker_id: str,
        target_speaker_name: str
    ) -> Dict[str, Any]:
        """
        將聲紋從來源語者轉移到目標語者的業務邏輯
        
        Args:
            source_speaker_id: 來源語者ID
            source_speaker_name: 來源語者名稱
            target_speaker_id: 目標語者ID
            target_speaker_name: 目標語者名稱
            
        Returns:
            Dict[str, Any]: 包含操作結果的字典
            
        Raises:
            HTTPException: 當操作失敗時拋出相應的 HTTP 異常
        """
        try:
            # 1. 驗證輸入參數
            if not source_speaker_id.strip():
                raise HTTPException(status_code=400, detail="來源語者ID不能為空")
            
            if not target_speaker_id.strip():
                raise HTTPException(status_code=400, detail="目標語者ID不能為空")
            
            if source_speaker_id == target_speaker_id:
                raise HTTPException(status_code=400, detail="來源語者和目標語者不能是同一人")
            
            # 2. 檢查來源語者是否存在
            source_speaker = self.speaker_manager.get_speaker(source_speaker_id)
            if not source_speaker:
                raise HTTPException(
                    status_code=404, 
                    detail=f"找不到ID為 {source_speaker_id} 的來源語者"
                )
            
            # 3. 檢查目標語者是否存在
            target_speaker = self.speaker_manager.get_speaker(target_speaker_id)
            if not target_speaker:
                raise HTTPException(
                    status_code=404, 
                    detail=f"找不到ID為 {target_speaker_id} 的目標語者"
                )
            
            # 4. 驗證語者名稱是否匹配（安全檢查）
            source_name = source_speaker.properties.get('name', '')
            target_name = target_speaker.properties.get('name', '')
            
            if source_speaker_name.strip() != source_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"提供的來源語者名稱 '{source_speaker_name}' 與資料庫中的名稱 '{source_name}' 不符"
                )
            
            if target_speaker_name.strip() != target_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"提供的目標語者名稱 '{target_speaker_name}' 與資料庫中的名稱 '{target_name}' 不符"
                )
            
            # 5. 執行聲紋轉移操作（轉移所有聲紋）
            success = self.speaker_manager.transfer_voiceprints(
                source_uuid=source_speaker_id,
                dest_uuid=target_speaker_id,
                voiceprint_ids=None  # None 表示轉移所有聲紋
            )
            
            if success:
                logger.info(f"成功轉移聲紋：{source_speaker_id} -> {target_speaker_id}")
                return {
                    "success": True,
                    "message": f"成功將語者 '{source_speaker_name}' 的所有聲紋轉移到 '{target_speaker_name}' 並刪除來源語者",
                    "data": {
                        "source_speaker_id": source_speaker_id,
                        "source_speaker_name": source_speaker_name,
                        "target_speaker_id": target_speaker_id,
                        "target_speaker_name": target_speaker_name
                    }
                }
            else:
                logger.error(f"聲紋轉移失敗：{source_speaker_id} -> {target_speaker_id}")
                raise HTTPException(
                    status_code=500, 
                    detail="聲紋轉移操作失敗，請檢查日誌以獲取詳細資訊"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"聲紋轉移發生未預期錯誤：{str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"伺服器內部錯誤：{str(e)}"
            )
    
    def get_speaker_info(self, speaker_id: str) -> Dict[str, Any]:
        """
        獲取語者資訊
        
        Args:
            speaker_id: 語者ID
            
        Returns:
            Dict[str, Any]: 語者資訊
            
        Raises:
            HTTPException: 當語者不存在時
        """
        try:
            obj = self.speaker_manager.get_speaker(speaker_id)
            if not obj:
                raise HTTPException(
                    status_code=404, 
                    detail=f"找不到ID為 {speaker_id} 的語者"
                )
            
            props = obj.properties

            return {
                "speaker_id": obj.uuid,
                "speaker_name": props.get('name', ''),
                "created_time": props.get('create_time', ''),
                "last_active_time": props.get('last_active_time', ''),
                "first_audio_id": props.get('first_audio_id'),
                "voiceprint_count": len(props.get('voiceprint_ids', []))
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"獲取語者資訊發生錯誤：{str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"伺服器內部錯誤：{str(e)}"
            )

    def list_all_speakers(self) -> List[Dict[str, Any]]:
        """
        列出所有語者的業務邏輯
        
        Returns:
            List[Dict[str, Any]]: 語者列表，包含完整資訊（含 first_audio_id）
            
        Raises:
            HTTPException: 當操作失敗時拋出相應的 HTTP 異常
        """
        try:
            # 1. 從 SpeakerManager 獲取所有語者（現在已包含 voiceprint_ids）
            speakers = self.speaker_manager.list_all_speakers()
            
            # 2. 轉換為 API 回應格式
            api_speakers = []
            for speaker in speakers:
                voiceprint_ids = speaker.get("voiceprint_ids", [])
                
                # 轉換 datetime 物件為字串
                created_at = speaker.get("create_time")
                if created_at and hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                elif created_at == "未知":
                    created_at = None
                
                updated_at = speaker.get("last_active_time") 
                if updated_at and hasattr(updated_at, 'isoformat'):
                    updated_at = updated_at.isoformat()
                elif updated_at == "未知":
                    updated_at = None
                
                # 轉換 UUID 物件為字串
                speaker_id = str(speaker["uuid"])
                voiceprint_ids_str = [str(vp_id) for vp_id in voiceprint_ids]
                
                api_speaker = {
                    "speaker_id": speaker_id,
                    "name": speaker["name"],
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "voiceprint_ids": voiceprint_ids_str,
                    # 包含 first_audio_id
                    "first_audio_id": str(speaker.get("first_audio_id"))
                }
                
                api_speakers.append(api_speaker)
            
            logger.info(f"成功列出 {len(api_speakers)} 位語者")
            return api_speakers
            
        except Exception as e:
            logger.error(f"列出語者發生未預期錯誤：{str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"伺服器內部錯誤：{str(e)}"
            )
    
    def delete_speaker(self, speaker_id: str) -> Dict[str, Any]:
        """
        刪除語者及其所有聲紋的業務邏輯
        
        Args:
            speaker_id: 語者的唯一識別碼
            
        Returns:
            Dict[str, Any]: 包含操作結果的字典
            
        Raises:
            HTTPException: 當操作失敗時拋出相應的 HTTP 異常
        """
        try:
            # 1. 驗證輸入參數
            if not speaker_id.strip():
                raise HTTPException(status_code=400, detail="語者ID不能為空")

            # 2. 檢查語者是否存在
            obj = self.speaker_manager.get_speaker(speaker_id)
            if not obj:
                raise HTTPException(
                    status_code=404, 
                    detail=f"找不到ID為 {speaker_id} 的語者"
                )
            
            # 3. 獲取語者資訊用於回傳
            speaker_name = obj.properties.get('name', '未命名')
            voiceprint_count = len(obj.properties.get('voiceprint_ids', []))
            
            # 4. 執行刪除操作
            success = self.speaker_manager.delete_speaker(speaker_id)
            
            if success:
                logger.info(f"成功刪除語者 {speaker_id} (名稱: {speaker_name}) 及其 {voiceprint_count} 個聲紋")
                return {
                    "success": True,
                    "message": f"成功刪除語者 '{speaker_name}' 及其 {voiceprint_count} 個聲紋",
                    "data": {
                        "speaker_id": speaker_id,
                        "speaker_name": speaker_name,
                        "deleted_voiceprint_count": voiceprint_count
                    }
                }
            else:
                logger.error(f"刪除操作失敗：speaker_id={speaker_id}")
                raise HTTPException(
                    status_code=500, 
                    detail="刪除操作失敗，請檢查日誌以獲取詳細資訊"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"刪除語者發生未預期錯誤：{str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"伺服器內部錯誤：{str(e)}"
            )
    
    def verify_speaker_voice(
        self, 
        audio_file_path: str,
        threshold: float = 0.39,
        max_results: int = 3
    ) -> Dict[str, Any]:
        """
        驗證音檔中的語者身份，純讀取操作，不會修改任何資料
        
        Args:
            audio_file_path: 音檔的暫存路徑
            threshold: 比對閾值，距離小於此值才認為是匹配
            max_results: 返回最相似的結果數量
            
        Returns:
            Dict[str, Any]: 包含識別結果的字典
            
        Raises:
            HTTPException: 當操作失敗時拋出相應的 HTTP 異常
        """
        try:
            logger.info(f"開始驗證音檔: {audio_file_path}")
            
            # 1. 從音檔提取語音特徵向量
            try:
                embedding = self.audio_processor.extract_embedding(audio_file_path)
                logger.debug(f"成功提取語音特徵向量，維度: {embedding.shape}")
            except Exception as e:
                logger.error(f"提取語音特徵失敗: {str(e)}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"音檔處理失敗：{str(e)}"
                )
            
            # 2. 與資料庫中的聲紋進行比對（純讀取操作）
            try:
                best_id, best_name, best_distance, all_distances = self.database.find_similar_voiceprints(
                    embedding=embedding,
                    limit=max_results
                )
                logger.debug(f"比對完成，找到 {len(all_distances)} 個候選結果")
            except Exception as e:
                logger.error(f"聲紋比對失敗: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"聲紋比對失敗：{str(e)}"
                )
            
            # 3. 處理比對結果
            if not all_distances:
                logger.info("資料庫中沒有任何聲紋資料")
                return {
                    "success": True,
                    "message": "資料庫中沒有任何聲紋資料",
                    "is_known_speaker": False,
                    "best_match": None,
                    "all_candidates": [],
                    "threshold": threshold,
                    "total_candidates": 0
                }
            
            # 4. 判斷是否為已知語者
            is_known_speaker = best_distance < threshold
            
            # 5. 準備返回的候選結果
            candidates = []
            for voice_id, speaker_name, distance, update_count in all_distances:
                candidates.append({
                    "voiceprint_id": str(voice_id),
                    "speaker_name": speaker_name,
                    "distance": float(distance),
                    "update_count": update_count,
                    "is_match": distance < threshold
                })
            
            # 6. 準備最佳匹配結果
            best_match = None
            if best_id and best_name:
                best_match = {
                    "voiceprint_id": str(best_id),
                    "speaker_name": best_name,
                    "distance": float(best_distance),
                    "is_match": is_known_speaker
                }
            
            # 7. 記錄驗證結果
            if is_known_speaker:
                logger.info(f"驗證成功 - 識別為語者: {best_name}, 距離: {best_distance:.4f}")
            else:
                logger.info(f"驗證結果 - 未知語者, 最相似語者: {best_name if best_name else '無'}, 距離: {best_distance:.4f}")
            
            return {
                "success": True,
                "message": "語音驗證完成" if is_known_speaker else "未找到匹配的語者",
                "is_known_speaker": is_known_speaker,
                "best_match": best_match,
                "all_candidates": candidates,
                "threshold": threshold,
                "total_candidates": len(candidates)
            }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"語音驗證過程發生未預期錯誤：{str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"伺服器內部錯誤：{str(e)}"
            )
