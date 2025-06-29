# services/handlers/speaker_handler.py
"""
說話者管理業務邏輯處理器

此模組包含所有與說話者管理相關的業務邏輯，
將 HTTP 層與業務邏輯分離，提高程式碼的可維護性和可測試性。
"""
from typing import Dict, Any, Optional
from fastapi import HTTPException
from modules.management.VID_manager import SpeakerManager
from utils.logger import get_logger

logger = get_logger(__name__)

class SpeakerHandler:
    """說話者管理業務邏輯處理器"""
    
    def __init__(self) -> None:
        """初始化說話者處理器"""
        self.speaker_manager = SpeakerManager()
    
    def rename_speaker(
        self, 
        speaker_id: str, 
        current_name: str, 
        new_name: str
    ) -> Dict[str, Any]:
        """
        更改說話者名稱的業務邏輯
        
        Args:
            speaker_id: 說話者的唯一識別碼
            current_name: 當前說話者名稱
            new_name: 新的說話者名稱
            
        Returns:
            Dict[str, Any]: 包含操作結果的字典
            
        Raises:
            HTTPException: 當操作失敗時拋出相應的 HTTP 異常
        """
        try:
            # 1. 驗證輸入參數
            if not speaker_id.strip():
                raise HTTPException(status_code=400, detail="說話者ID不能為空")
            
            if not new_name.strip():
                raise HTTPException(status_code=400, detail="新名稱不能為空")
            
            # 2. 檢查說話者是否存在
            obj = self.speaker_manager.get_speaker(speaker_id)
            if not obj:
                raise HTTPException(
                    status_code=404, 
                    detail=f"找不到ID為 {speaker_id} 的說話者"
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
                logger.info(f"成功將說話者 {speaker_id} 從 '{current_name}' 更名為 '{new_name}'")
                return {
                    "success": True,
                    "message": f"成功將說話者 '{current_name}' 更名為 '{new_name}'",
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
        將聲紋從來源說話者轉移到目標說話者的業務邏輯
        
        Args:
            source_speaker_id: 來源說話者ID
            source_speaker_name: 來源說話者名稱
            target_speaker_id: 目標說話者ID
            target_speaker_name: 目標說話者名稱
            
        Returns:
            Dict[str, Any]: 包含操作結果的字典
            
        Raises:
            HTTPException: 當操作失敗時拋出相應的 HTTP 異常
        """
        try:
            # 1. 驗證輸入參數
            if not source_speaker_id.strip():
                raise HTTPException(status_code=400, detail="來源說話者ID不能為空")
            
            if not target_speaker_id.strip():
                raise HTTPException(status_code=400, detail="目標說話者ID不能為空")
            
            if source_speaker_id == target_speaker_id:
                raise HTTPException(status_code=400, detail="來源說話者和目標說話者不能是同一人")
            
            # 2. 檢查來源說話者是否存在
            source_speaker = self.speaker_manager.get_speaker(source_speaker_id)
            if not source_speaker:
                raise HTTPException(
                    status_code=404, 
                    detail=f"找不到ID為 {source_speaker_id} 的來源說話者"
                )
            
            # 3. 檢查目標說話者是否存在
            target_speaker = self.speaker_manager.get_speaker(target_speaker_id)
            if not target_speaker:
                raise HTTPException(
                    status_code=404, 
                    detail=f"找不到ID為 {target_speaker_id} 的目標說話者"
                )
            
            # 4. 驗證說話者名稱是否匹配（安全檢查）
            source_name = source_speaker.properties.get('name', '')
            target_name = target_speaker.properties.get('name', '')
            
            if source_speaker_name.strip() != source_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"提供的來源說話者名稱 '{source_speaker_name}' 與資料庫中的名稱 '{source_name}' 不符"
                )
            
            if target_speaker_name.strip() != target_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"提供的目標說話者名稱 '{target_speaker_name}' 與資料庫中的名稱 '{target_name}' 不符"
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
                    "message": f"成功將說話者 '{source_speaker_name}' 的所有聲紋轉移到 '{target_speaker_name}' 並刪除來源說話者",
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
        獲取說話者資訊
        
        Args:
            speaker_id: 說話者ID
            
        Returns:
            Dict[str, Any]: 說話者資訊
            
        Raises:
            HTTPException: 當說話者不存在時
        """
        try:
            obj = self.speaker_manager.get_speaker(speaker_id)
            if not obj:
                raise HTTPException(
                    status_code=404, 
                    detail=f"找不到ID為 {speaker_id} 的說話者"
                )
            
            props = obj.properties

            return {
                "speaker_id": obj.uuid,
                "speaker_name": props.get('name', ''),
                "created_time": props.get('create_time', ''),
                "last_active_time": props.get('last_active_time', ''),
                "voiceprint_count": len(props.get('voiceprint_ids', []))
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"獲取說話者資訊發生錯誤：{str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"伺服器內部錯誤：{str(e)}"
            )
