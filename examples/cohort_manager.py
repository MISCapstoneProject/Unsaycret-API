#!/usr/bin/env python3
"""
AS-Norm Cohort 資料庫管理工具

⚠️ 此檔案已廢棄，請使用新的模組：

新的位置：
- 主要模組: modules/database/cohort_manager.py
- 快速工具: examples/cohort_cli.py

遷移指令：
- 舊: python examples/cohort_manager.py --action init
- 新: python examples/cohort_cli.py init

- 舊: python examples/cohort_manager.py --action generate --count 100
- 新: python examples/cohort_cli.py import /path/to/audio/folder

- 舊: python examples/cohort_manager.py --action stats  
- 新: python examples/cohort_cli.py stats
"""

import sys
import os

# 添加模組路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("⚠️ 此工具已廢棄！")
print("📍 請使用新的 cohort 管理工具:")
print("   快速工具: python examples/cohort_cli.py help")
print("   完整模組: from modules.database.cohort_manager import CohortDatabaseManager")
print("\n🔄 自動重導向到新工具...")

# 自動重導向到新工具
try:
    from examples.cohort_cli import main as new_main
    new_main()
except ImportError:
    print("❌ 無法載入新工具，請檢查安裝")
