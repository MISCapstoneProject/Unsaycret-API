# pipelines

整合語者分離、辨識與 ASR 的處理流程。
提供兩個函式：`run_pipeline_file` 處理單一音檔，`run_pipeline_dir` 批次處理資料夾。

```python
from pipelines import run_pipeline_file
segments, pretty, stats = run_pipeline_file('input.wav')
```
