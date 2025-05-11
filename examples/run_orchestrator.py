import sys
from pipelines.orchestrator import run_pipeline

# 如果你有傳參數，就用第一個；沒有就 fallback demo.wav
input_path = sys.argv[1] if len(sys.argv) > 1 else "examples/demo.wav"
run_pipeline(input_path)
