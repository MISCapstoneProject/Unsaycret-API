from modules.identification import VID_identify_v5 as speaker_id
from modules.management import VID_manager
from modules.separation import separator as VID_system
from utils.logger import get_logger

# 創建本模組的日誌器
logger = get_logger(
    name="VoiceID.main", 
    log_file="system_output.log", 
    append_mode=True
)

def check_weaviate_connection() -> bool:
    """
    檢查 Weaviate 資料庫連線狀態。

    Returns:
        bool: 若連線成功回傳 True，否則回傳 False。
    """
    return VID_system.check_weaviate_connection()

def main():
    """
    先分離再識別的主流程 - 即時版本
    
    這個版本下，語音分離和識別是並行處理的：
    1. 語音分離後立即進行識別
    2. 不需要等待所有音檔全部錄製完畢才開始識別
    3. 可以實時顯示識別結果
    """
    try:
        # 顯示主選單
        print("\n" + "="*60)
        print("語者分離與識別系統".center(56))
        print("="*60)
        print("1. 啟動即時語者分離與識別")
        print("2. 管理說話者和聲紋")
        print("0. 退出")
        print("-"*60)
        choice = input("請選擇操作 (0-2): ").strip()
        
        if choice == "0":
            print("程式結束")
            return
        
        elif choice == "1":
            logger.info("語者分離與識別系統啟動 (即時識別模式)")
            
            # 在啟動時檢查 Weaviate 連線狀態
            if not check_weaviate_connection():
                logger.critical("由於無法連線至 Weaviate 資料庫，程式將終止執行。")
                return  # 直接結束程式
            
            # 步驟2: 初始化語者分離器
            separator = VID_system.AudioSeparator()
            
            # 步驟3: 錄音並進行語者分離 (分離的同時進行識別)
            # 注意：識別功能已整合到 separate_and_identify 方法中
            logger.info("開始錄音並執行即時分離與識別...")
            mixed_audio_file = separator.record_and_process(VID_system.OUTPUT_DIR)
            
            # 步驟4: 錄音結束後的摘要
            separated_files = separator.get_output_files()
            logger.info(f"處理完成，共產生 {len(separated_files)} 個分離後的音檔")
            
            # 步驟5: 顯示錄音結果
            print(f"\n原始混合音檔: {mixed_audio_file}")
            print(f"分離後的音檔已保存至: {VID_system.OUTPUT_DIR}")
            
        elif choice == "2":
            # 使用從 speaker_manager 模組導入的函數
            # 啟動說話者管理系統
            VID_manager.main()
        
        else:
            print("無效的選擇")
            
    except KeyboardInterrupt:
        logger.info("\n接收到停止信號")
        if 'separator' in locals():
            separator.stop_recording()
    except Exception as e:
        logger.error(f"程式執行時發生錯誤：{e}")

# 主程式進入點
if __name__ == "__main__":
    main()

