# text_utils.py
try:
    import jieba_fast as jieba  # 安裝了 jieba_fast 就用它
except ModuleNotFoundError:
    import jieba
from typing import List, Dict
import re
try:
    from opencc import OpenCC
    cc = OpenCC('s2twp')  # 簡→繁（台灣用字）
except ImportError:
    cc = None

_normalize_pattern = re.compile(r'[^\u4e00-\u9fffA-Za-z0-9]')

def normalize_zh(text: str) -> str:
    """
    先(可選)把簡體轉繁體，再移除標點與空白，只留中英數。
    """
    if cc:
        text = cc.convert(text)
    return _normalize_pattern.sub('', text)


def cer_zh(ref_norm: str, hyp_norm: str) -> float:
    """
    ref_norm / hyp_norm 已經是 normalize_zh 後的字串
    直接用字級 Levenshtein 距離算 CER
    """
    if not ref_norm:
        return 0.0
    dist = _levenshtein(list(ref_norm), list(hyp_norm))
    return dist / len(ref_norm)


def wer_zh(ref_norm: str, hyp_norm: str) -> float:
    """
    中文 WER：用 jieba 斷詞後計算。若只想看 CER，可不用此函式。
    """
    if not ref_norm:
        return 0.0
    ref_tok = list(jieba.cut(ref_norm))
    hyp_tok = list(jieba.cut(hyp_norm))
    dist = _levenshtein(ref_tok, hyp_tok)
    return dist / len(ref_tok)

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


def clip_words_to_window(words, win_start, win_end):
    """Keep words overlapping [win_start, win_end) and clamp their spans."""
    out = []
    for w in words or []:
        s = float(w.get("start", 0.0))
        e = float(w.get("end", 0.0))
        if e <= win_start or s >= win_end:
            continue
        w2 = dict(w)
        w2["start"] = max(win_start, s)
        w2["end"] = min(win_end, e)
        out.append(w2)
    return out


def edge_sanitize(words, win_start, win_end, edge_ms, edge_conf, edge_min_dur):
    """Apply stricter confidence/duration near window edges."""
    if not words:
        return []
    head_edge = win_start + edge_ms
    tail_edge = win_end - edge_ms
    keep = []
    for w in words:
        s = float(w.get("start", 0.0))
        e = float(w.get("end", 0.0))
        p = float(w.get("probability", 1.0))
        dur = max(0.0, e - s)
        in_edge = (s < head_edge) or (e > tail_edge)
        if in_edge:
            if (p >= edge_conf) and (dur >= edge_min_dur):
                keep.append(w)
        else:
            keep.append(w)
    return keep


def suppress_tail_punct(text, last_word_end, win_end, tail_gap):
    """Strip trailing punctuation if last word ends too close to window end."""
    if not text:
        return text
    if (win_end - float(last_word_end or 0.0)) < tail_gap:
        stripped = text.rstrip()
        for tail in ("。", "！", "？", ".", "!", "?", "…", "..."):
            if stripped.endswith(tail):
                return stripped[: -len(tail)]
    return text


def rebuild_text_from_words(words):
    """Rebuild utterance text using spacing rules for CJK tokens."""
    if not words:
        return ""
    buf = []
    for i, w in enumerate(words):
        tok = str(w.get("word", "") or "")
        if not tok:
            continue
        if i > 0 and tok and ord(tok[0]) < 128:
            buf.append(" ")
        buf.append(tok)
    return "".join(buf).strip()
