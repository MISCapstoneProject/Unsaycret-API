## Chinese Punctuator

Immediate cleanup module for Chinese ASR text: restores punctuation, optionally fixes obvious typos, and normalizes output to Taiwanese Traditional script.

### Models
- Punctuation restoration: [p208p2002/zh-wiki-punctuation-restore](https://huggingface.co/p208p2002/zh-wiki-punctuation-restore) via the `zhpr` sliding-window helpers.
- Optional spelling correction: [shibing624/macbert4csc-base-chinese](https://huggingface.co/shibing624/macbert4csc-base-chinese) through `pycorrector`.
- Script conversion and punctuation style: [`opencc` s2twp](https://github.com/BYVoid/OpenCC).

### Installation
```bash
pip install zhpr transformers torch opencc-python-reimplemented pycorrector
```

### Usage
```python
from modules.text.punctuator import ChinesePunctuator

punctuator = ChinesePunctuator(device="cuda")
clean_text = punctuator.apply(raw_text, use_macbert=True)
```

Disable the MacBERT pass by calling `apply(text, use_macbert=False)` if you want punctuation-only cleanup.

### Performance Notes
- Runs inline with low latency; all workloads stay on the configured device (GPU if available, CPU otherwise).
- Any import or inference failure falls back to the original ASR text (fail-open), so subtitles are never blocked.
- Existing downstream repair or LLM post-processing pipelines remain untouched.

### Processing Order
1. Detect Traditional-heavy text and convert to Simplified (t2s) to maximise downstream model accuracy.
2. Restore punctuation with `zhpr`.
3. (Optional) Run MacBERT4CSC for short snippets to catch easy spelling slips.
4. Convert to Taiwanese Traditional with `opencc` s2twp for final script and punctuation style.
