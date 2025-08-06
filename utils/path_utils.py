"""
路徑處理工具模組
提供更好的檔案路徑處理和日誌前綴生成功能
"""

import os
from pathlib import Path
from typing import Optional

def generate_smart_prefix(file_path: str, max_depth: int = 2) -> str:
    """
    生成智慧化的檔案前綴，保留重要的資料夾資訊
    
    Args:
        file_path: 完整檔案路徑
        max_depth: 最大保留的資料夾深度
        
    Returns:
        str: 智慧化前綴字串
        
    Examples:
        >>> generate_smart_prefix("stream_output/20250728_210015/segment_000/speaker1.wav")
        "segment_000/speaker1.wav"
        >>> generate_smart_prefix("/long/path/to/data/speaker_A/audio1.wav") 
        "speaker_A/audio1.wav"
        >>> generate_smart_prefix("test.wav")
        "test.wav"
    """
    if not file_path:
        return "unknown"
    
    # 標準化路徑分隔符
    normalized_path = file_path.replace('\\', '/')
    path_parts = normalized_path.split('/')
    
    # 移除空字串（避免開頭或結尾的斜線造成問題）
    path_parts = [part for part in path_parts if part]
    
    if len(path_parts) <= max_depth:
        return '/'.join(path_parts)
    
    # 保留最後 max_depth 個部分（包含檔案名）
    return '/'.join(path_parts[-max_depth:])

def extract_segment_info(file_path: str) -> Optional[str]:
    """
    從路徑中提取分段資訊
    
    Args:
        file_path: 檔案路徑
        
    Returns:
        Optional[str]: 分段資訊，如 "segment_000" 或 None
    """
    path_parts = file_path.replace('\\', '/').split('/')
    
    for part in path_parts:
        if part.startswith('segment_'):
            return part
    
    return None

def get_relative_path(file_path: str, base_dirs: list = None) -> str:
    """
    獲取相對於常見基礎目錄的路徑
    
    Args:
        file_path: 完整檔案路徑
        base_dirs: 基礎目錄列表，預設為常見的專案目錄
        
    Returns:
        str: 相對路徑或簡化後的路徑
    """
    if base_dirs is None:
        base_dirs = [
            'stream_output',
            'data',
            'test',
            'examples',
            'temp',
            'tmp'
        ]
    
    normalized_path = file_path.replace('\\', '/')
    
    for base_dir in base_dirs:
        if f'/{base_dir}/' in normalized_path:
            # 找到基礎目錄，返回從該目錄開始的相對路徑
            base_index = normalized_path.find(f'/{base_dir}/')
            return normalized_path[base_index + 1:]  # +1 移除開頭的斜線
    
    # 如果沒找到匹配的基礎目錄，使用智慧前綴
    return generate_smart_prefix(file_path, max_depth=3)

def format_process_prefix(process_id: int, file_path: str, 
                         include_segment: bool = True,
                         max_depth: int = 2) -> str:
    """
    格式化處理流程前綴
    
    Args:
        process_id: 處理流程ID
        file_path: 檔案路徑
        include_segment: 是否特別標示分段資訊
        max_depth: 路徑最大深度
        
    Returns:
        str: 格式化的前綴，如 "[#1 segment_000/speaker1.wav]"
    """
    # 標準化路徑並取得檔案名
    normalized_path = file_path.replace('\\', '/')
    filename = os.path.basename(normalized_path)
    
    # 如果要包含分段資訊，嘗試提取
    if include_segment:
        segment_info = extract_segment_info(file_path)
        if segment_info:
            # 只返回 segment_xxx/filename 格式
            smart_path = f"{segment_info}/{filename}"
        else:
            # 沒有分段資訊，只返回檔案名
            smart_path = filename
    else:
        smart_path = filename
    
    return f"[#{process_id} {smart_path}]"

# 測試函數
if __name__ == "__main__":
    test_paths = [
        "stream_output/20250728_210015/segment_000/speaker1.wav",
        "/Users/cyouuu/Desktop/Unsaycret-API/data/clean/speaker1/audio1.wav",
        "test.wav",
        "C:\\Windows\\temp\\segment_005\\speaker2.wav",
        "examples/test_data/speaker_A/sample.wav"
    ]
    
    print("=== 路徑處理測試 ===")
    for i, path in enumerate(test_paths, 1):
        prefix = format_process_prefix(i, path)
        smart_path = get_relative_path(path)
        segment = extract_segment_info(path)
        
        print(f"原始路徑: {path}")
        print(f"前綴: {prefix}")
        print(f"智慧路徑: {smart_path}")
        print(f"分段資訊: {segment}")
        print("-" * 50)
