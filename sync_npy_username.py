#!/usr/bin/env python3
"""
此腳本會自動掃描 "embeddingFiles" 資料夾中所有子資料夾，
檢查每個 npy 檔案的檔名是否符合格式：<用戶名>_<檔案獨立編號>_<更新次數>.npy，
並且檔案中用戶名部分是否與所在子資料夾名稱一致。
如果不一致，則自動重新命名該檔案，使其用戶名部分與子資料夾名稱同步。

例如：
    若子資料夾名稱改為 "David"，而檔案名稱為 "n1_1_2.npy"，則會更新為 "David_1_2.npy"。
"""

import os
import re

# 母資料夾路徑，請根據實際情況修改
EMBEDDING_DIR = "embeddingFiles"

def sync_npy_filenames_in_folder(folder_path):
    """
    處理單一子資料夾：
      1. 以該子資料夾名稱作為最新的用戶名。
      2. 遍歷該資料夾內所有 npy 檔案。
      3. 用正則表達式解析檔案名稱，預期格式為：<用戶名>_<檔案獨立編號>_<更新次數>.npy
      4. 如果解析到的用戶名與子資料夾名稱不符，則重新命名檔案，將用戶名部分更新為子資料夾名稱。
    """
    # 取得子資料夾名稱，作為新的用戶名
    new_username = os.path.basename(os.path.normpath(folder_path))
    
    # 遍歷子資料夾內所有檔案
    for filename in os.listdir(folder_path):
        # 只處理 npy 檔案
        if not filename.endswith(".npy"):
            continue
        
        # 用正則解析檔案名稱，預期格式： <用戶名>_<檔案獨立編號>_<更新次數>.npy
        m = re.match(r"^(.*?)_(\d+)_(\d+)\.npy$", filename)
        if not m:
            print(f"檔案 '{filename}' 不符合預期格式，跳過。")
            continue
        
        old_username, file_id, update_count = m.groups()
        
        # 若檔案用戶名與子資料夾名稱不同，就進行重新命名
        if old_username != new_username:
            new_filename = f"{new_username}_{file_id}_{update_count}.npy"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            print(f"將檔案 '{filename}' 重新命名為 '{new_filename}'")
            os.rename(old_path, new_path)

def sync_all_folders():
    """
    掃描母資料夾 EMBEDDING_DIR 中所有子資料夾，並對每個子資料夾內的 npy 檔案進行同步更新。
    """
    print("\n* * * * * 檢測檔案用戶名 * * * * *")

    if not os.path.exists(EMBEDDING_DIR):
        print(f"母資料夾 '{EMBEDDING_DIR}' 不存在。")
        return
    
    # 列出母資料夾內所有項目，並檢查哪些是子資料夾
    for entry in os.listdir(EMBEDDING_DIR):
        subfolder = os.path.join(EMBEDDING_DIR, entry)
        if os.path.isdir(subfolder):
            # print(f"處理子資料夾：{subfolder}")
            sync_npy_filenames_in_folder(subfolder)

    print("* * * * * * * * * * * * * * * * * \n")

def main():
    """
    主函式：執行同步處理，並在最後提示完成訊息。
    """
    sync_all_folders()
    print("所有子資料夾內 npy 檔案的用戶名同步更新完成。")

if __name__ == "__main__":
    main()
