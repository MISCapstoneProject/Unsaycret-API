# text_utils.py
try:
    import jieba_fast as jieba  # 安裝了 jieba_fast 就用它
except ModuleNotFoundError:
    import jieba
from typing import List, Dict

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
