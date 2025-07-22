# text_utils.py
try:
    import jieba_fast as jieba  # 安裝了 jieba_fast 就用它
except ModuleNotFoundError:
    import jieba
from typing import List, Dict
import re

# jieba.add_word("公務員")
# jieba.add_word("畢業後")
# jieba.load_userdict("user_dict.txt")

def merge_char_to_word(full_txt: str,
                       char_words: List[Dict]) -> List[Dict]:
    """
    把 char-level timestamps (Whisper) 合併成 jieba 詞級 timestamps
    char_words 必須依時間排序，每筆含 start/end/word/probability
    """
    
    merged, cw_idx = [], 0
    for tok in jieba.cut(full_txt):
        buf, buf_words = "", []
        while len(buf) < len(tok) and cw_idx < len(char_words):
            w = char_words[cw_idx]
            buf += w["word"]
            buf_words.append(w)
            cw_idx += 1
        if buf != tok or not buf_words:      # 對不到就略過
            continue
        merged.append({
            "word": tok,
            "start": min(x["start"] for x in buf_words),
            "end":   max(x["end"]   for x in buf_words),
            "probability": (sum(x["probability"] for x in buf_words)
                            / len(buf_words)),
        })
    return merged


def _levenshtein(a: List[str], b: List[str]) -> int:
    """Simple Levenshtein distance."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_tokens = _normalize(reference).split()
    hyp_tokens = _normalize(hypothesis).split()
    if not ref_tokens:
        return 0.0
    dist = _levenshtein(ref_tokens, hyp_tokens)
    return dist / len(ref_tokens)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    ref_chars = list(_normalize(reference).replace(" ", ""))
    hyp_chars = list(_normalize(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0
    dist = _levenshtein(ref_chars, hyp_chars)
    return dist / len(ref_chars)