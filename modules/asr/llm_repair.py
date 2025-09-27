# =============================================
# File: llm_repair.py
# Purpose: Pre/Rule/LLM repair for zh-TW ASR text + SRT reseg
# Author: Airi
# =============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os, json, re, time
import requests

try:
    import opencc  # pip install opencc (plus native OpenCC library)
    _HAS_OPENCC = True
except Exception:
    _HAS_OPENCC = False

# ---------------------- helpers ----------------------
_CJK_RNG = ('\u4e00', '\u9fff')
_DEF_WHITELIST = {
    # 常見混淆（僅示例；可在外部 JSON 擴充）
    "戰機": "戰績",
    "高興挖走": "高薪挖走",
    "兼韌": "兼任",
    "市大運": "世大運",
}

_ZH_SENT_SPLIT = re.compile(r"([。！？!?])")  # 句末標點保留
_RE_MULTISPACE = re.compile(r"\s+")
_RE_DUPE_CHAR = re.compile(r"(\S)\1{1,}")  # 連續同字 >=2 → 壓成1
_RE_DUPE_ABAB = re.compile(r"(\S{1,2})\1")


def _is_cjk(ch: str) -> bool:
    return len(ch) == 1 and _CJK_RNG[0] <= ch <= _CJK_RNG[1]


def _normalize_spaces(s: str) -> str:
    # 壓縮空白；避免把 CJK 與字母數字之間的必要空白吃掉
    return _RE_MULTISPACE.sub(" ", s.strip())


@dataclass
class Edit:
    type: str
    old: str
    new: str
    reason: str
    start: int
    end: int


@dataclass
class RepairResult:
    clean_text: str
    edits: List[Edit]


# ---------------------- pre/rule repairs ----------------------

def hans_to_hant(text: str, config: str = "s2twp.json") -> str:
    """簡→繁（台灣用詞）；若缺 OpenCC 則原文返回。"""
    if not _HAS_OPENCC:
        return text
    try:
        cfg = config if config.endswith('.json') else f"{config}"
        cc = opencc.OpenCC(cfg)
        return cc.convert(text)
    except Exception:
        return text


def dedup_text(s: str) -> str:
    # AA → A
    s = _RE_DUPE_CHAR.sub(lambda m: m.group(1), s)
    # ABAB → AB
    s = _RE_DUPE_ABAB.sub(lambda m: m.group(1), s)
    return s


def apply_whitelist(s: str, mapping: Dict[str, str]) -> Tuple[str, List[Edit]]:
    edits: List[Edit] = []
    for k, v in mapping.items():
        if k in s:
            start = s.find(k)
            s = s.replace(k, v)
            edits.append(Edit(type="whitelist", old=k, new=v, reason="known_confusion", start=start, end=start+len(v)))
    return s, edits


def seed_punct_by_gaps(tokens: List[Dict[str, Any]], period_gap: float = 0.5, comma_gap: float = 0.25) -> str:
    """
    依 token 時間差先打一版『粗標點』：gap>=period_gap→句號，>=comma_gap→逗點。
    tokens: [{"text": str, "start": float, "end": float}]
    """
    if not tokens:
        return ""
    out: List[str] = []
    prev_end = None
    for t in tokens:
        txt = t.get("text", "").strip()
        if not txt:
            continue
        if prev_end is None:
            out.append(txt)
        else:
            gap = float(t.get("start", 0.0)) - float(prev_end)
            if gap >= period_gap:
                out.append("。")
                out.append(txt)
            elif gap >= comma_gap:
                out.append("，")
                out.append(txt)
            else:
                # CJK 不加空白，英數與英數間補空白
                if (out and out[-1] and out[-1][-1].isalnum() and txt[:1].isalnum()):
                    out.append(" ")
                out.append(txt)
        prev_end = float(t.get("end", 0.0))
    return "".join(out)


# ---------------------- LLM providers ----------------------

class BaseLLM:
    def repair(self, raw_text: str, glossary: List[str], temperature: float = 0.2, max_tokens: int = 2048) -> RepairResult:
        raise NotImplementedError

    @staticmethod
    def _prompt(raw_text: str, glossary: List[str]) -> Tuple[str, Dict[str, Any]]:
        system = (
            "你是正體中文的文字校對器。只做：1) 近音/語意錯詞修正，2) 專有名詞依詞庫，3) 簡→繁與台灣標點，"
            "4) 去除殘留重複(AA/ABAB/ABA)，5) 合理補逗點句號。不能新增事實；不要改動數字/日期(除非明顯錯)。"
        )
        task = {
            "instructions": (
                "請只回傳 JSON：{\"clean_text\": str, \"edits\":[{\"type\":str,\"old\":str,\"new\":str,\"reason\":str,\"start\":int,\"end\":int}]}。"
                "\n- 用全形標點（，。！？）；\n- 名詞以 glossary 為準；\n- 不確定就保留原文。"
            ),
            "glossary": glossary,
            "raw_text": raw_text,
        }
        return system, task


class OllamaLLM(BaseLLM):
    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5:7b-instruct"):
        self.host = host.rstrip('/')
        self.model = model

    def repair(self, raw_text: str, glossary: List[str], temperature: float = 0.2, max_tokens: int = 2048) -> RepairResult:
        sys_prompt, payload = self._prompt(raw_text, glossary)
        body = {
            "model": self.model,
            "system": sys_prompt,
            "prompt": json.dumps(payload, ensure_ascii=False),
            "stream": False,
            "options": {"temperature": temperature},
            "format": "json",
        }
        url = f"{self.host}/api/generate"
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        content = data.get("response", "{}")
        try:
            obj = json.loads(content)
            edits = [Edit(**e) for e in obj.get("edits", []) if isinstance(e, dict)]
            return RepairResult(clean_text=_normalize_spaces(obj.get("clean_text", "")), edits=edits)
        except Exception:
            # 非 JSON：當作原文返回
            return RepairResult(clean_text=_normalize_spaces(content), edits=[])


class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile", endpoint: str = "https://api.groq.com/openai/v1/chat/completions"):
        self.key = api_key
        self.model = model
        self.endpoint = endpoint

    def repair(self, raw_text: str, glossary: List[str], temperature: float = 0.2, max_tokens: int = 2048) -> RepairResult:
        sys_prompt, payload = self._prompt(raw_text, glossary)
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": max_tokens,
        }
        r = requests.post(self.endpoint, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        out = r.json()
        content = out.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        try:
            obj = json.loads(content)
            edits = [Edit(**e) for e in obj.get("edits", []) if isinstance(e, dict)]
            return RepairResult(clean_text=_normalize_spaces(obj.get("clean_text", "")), edits=edits)
        except Exception:
            return RepairResult(clean_text=_normalize_spaces(content), edits=[])


# ---------------------- main repair pipeline ----------------------

def llm_repair_pipeline(tokens: List[Dict[str, Any]],
                        provider: str = "ollama",
                        model: str = "qwen2.5:7b-instruct",
                        host: str = "http://localhost:11434",
                        api_key: Optional[str] = None,
                        opencc_config: str = "s2twp.json",
                        whitelist_path: Optional[str] = None,
                        glossary_path: Optional[str] = None,
                        temperature: float = 0.2,
                        gap_period: float = 0.5,
                        gap_comma: float = 0.25) -> RepairResult:
    """
    1) Hans→Hant  2) 初步去重  3) 白名單替換  4) 依 gap 打粗標點  5) LLM 精修
    返回：clean_text + edits（可落地審計）
    """
    # a) hans→hant on the plain join text
    seed_text = seed_punct_by_gaps(tokens, period_gap=gap_period, comma_gap=gap_comma)
    s = hans_to_hant(seed_text, config=opencc_config)
    s = dedup_text(s)

    # b) whitelist
    mapping = _DEF_WHITELIST.copy()
    if whitelist_path and os.path.exists(whitelist_path):
        try:
            mapping.update(json.loads(open(whitelist_path, 'r', encoding='utf-8').read()))
        except Exception:
            pass
    s, wl_edits = apply_whitelist(s, mapping)

    # c) glossary
    glossary: List[str] = []
    if glossary_path and os.path.exists(glossary_path):
        try:
            glossary = json.loads(open(glossary_path, 'r', encoding='utf-8').read())
        except Exception:
            # 也允許一行一詞的純文字
            try:
                glossary = [line.strip() for line in open(glossary_path, 'r', encoding='utf-8') if line.strip()]
            except Exception:
                glossary = []

    # d) call LLM
    if provider == "ollama":
        llm = OllamaLLM(host=host, model=model)
    elif provider == "groq":
        if not api_key:
            raise RuntimeError("GROQ_API_KEY missing")
        llm = GroqLLM(api_key=api_key, model=model)
    else:
        raise RuntimeError(f"Unknown provider: {provider}")

    res = llm.repair(s, glossary)
    # 合併白名單 edits
    res.edits = wl_edits + res.edits
    return res


# ---------------------- SRT reseggestion ----------------------

def srt_segments_from_tokens(tokens: List[Dict[str, Any]], gap_break: float = 0.5) -> List[Tuple[float, float]]:
    """只用時間 gap 產生句段 (start,end)。"""
    if not tokens:
        return []
    segs: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for t in tokens:
        if not cur:
            cur = [t]
            continue
        gap = float(t.get("start", 0.0)) - float(cur[-1].get("end", 0.0))
        if gap >= gap_break:
            segs.append(cur)
            cur = [t]
        else:
            cur.append(t)
    if cur:
        segs.append(cur)
    return [(seg[0]["start"], seg[-1]["end"]) for seg in segs]


def split_text_by_sentences(clean_text: str) -> List[str]:
    parts: List[str] = []
    if not clean_text.strip():
        return parts
    chunks = _ZH_SENT_SPLIT.split(clean_text)
    # 拼回：content + end-punct
    buf = ""
    for c in chunks:
        if not c:
            continue
        if _ZH_SENT_SPLIT.fullmatch(c):
            buf += c
            parts.append(buf.strip())
            buf = ""
        else:
            buf += c
    if buf.strip():
        parts.append(buf.strip())
    return parts


def assign_text_to_segments(sentences: List[str], timings: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], str]]:
    """將句子串對齊到時間段：數量不同時做保守映射。"""
    if not timings:
        return []
    if not sentences:
        # 沒文字：產生空白
        return [(timings[0], "")] if timings else []

    n_t, n_s = len(timings), len(sentences)
    if n_s == n_t:
        return list(zip(timings, sentences))

    # 若句子較少：按比例把相鄰段落合併文字
    out: List[Tuple[Tuple[float, float], str]] = []
    i = 0
    for t in timings:
        if i < n_s:
            out.append((t, sentences[i]))
            i += 1
        else:
            out.append((t, ""))
    # 若句子較多：餘下句子全部附加到最後一段
    if n_s > n_t:
        extra = "".join(sentences[n_t:])
        last_t, last_txt = out[-1]
        out[-1] = (last_t, (last_txt + ("" if last_txt == "" else " ") + extra).strip())
    return out


def format_ts(x: float) -> str:
    h = int(x // 3600); x -= h*3600
    m = int(x // 60);   x -= m*60
    s = int(x);         ms = int((x - s) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt_with_repair(path: str,
                          tokens: List[Dict[str, Any]],
                          clean_text: str,
                          gap_break: float = 0.5,
                          max_line: int = 18) -> None:
    timings = srt_segments_from_tokens(tokens, gap_break=gap_break)
    sents = split_text_by_sentences(clean_text)
    aligned = assign_text_to_segments(sents, timings)

    with open(path, 'w', encoding='utf-8') as f:
        for i, (tspan, text) in enumerate(aligned, start=1):
            s, e = tspan
            # 簡單依長度斷行
            lines: List[str] = []
            buf = text
            while len(buf) > max_line:
                lines.append(buf[:max_line])
                buf = buf[max_line:]
            if buf:
                lines.append(buf)
            f.write(f"{i}\n{format_ts(s)} --> {format_ts(e)}\n")
            f.write("\n".join(lines) + "\n\n")


# ======================== end of llm_repair.py ========================


# =============================================
# PATCH: fast_test.py (integration points)
# - add CLI flags for LLM repair & pre/rule steps
# - call pipeline after agg.finalize()
# - write SRT using write_srt_with_repair
# =============================================
# --- 在你的 fast_test.py 中：匯入 ---
# from llm_repair import llm_repair_pipeline, write_srt_with_repair
#
# --- 在 parse_args() 追加參數（示例） ---
# p.add_argument('--repair-enable', action='store_true', help='Enable LLM repair (file-mode only)')
# p.add_argument('--repair-provider', type=str, default='ollama', choices=['ollama','groq'], help='LLM provider')
# p.add_argument('--repair-host', type=str, default='http://localhost:11434', help='Ollama host if provider=ollama')
# p.add_argument('--repair-model', type=str, default='qwen2.5:7b-instruct', help='Model name for LLM provider')
# p.add_argument('--repair-api-key', type=str, default=None, help='API key (e.g., GROQ_API_KEY)')
# p.add_argument('--opencc-config', type=str, default='s2twp.json', help='OpenCC config (e.g., s2twp.json)')
# p.add_argument('--whitelist', type=str, default=None, help='JSON mapping for known confusions')
# p.add_argument('--glossary', type=str, default=None, help='JSON/lines glossary for NER')
# p.add_argument('--seed-period-gap', type=float, default=0.5, help='gap≥x→句號')
# p.add_argument('--seed-comma-gap', type=float, default=0.25, help='gap≥x→逗點')
# p.add_argument('--srt-gap-break', type=float, default=0.5, help='SRT 分段 gap 門檻')
# p.add_argument('--srt-max-line', type=int, default=18, help='每行最大字數 (CJK 約 2~3 秒可讀)')
#
# --- 在 run_one_file() finalize 後改成： ---
# agg.finalize()
# committed = agg.state.committed
# 
# # 轉成 dict tokens for repair
# toks = [{"text": t.text, "start": t.start, "end": t.end, "prob": t.prob} for t in committed]
# hyp_text = tokens_to_text(committed)
# 
# if args.repair-enable:
#     res = llm_repair_pipeline(
#         tokens=toks,
#         provider=args.repair_provider,
#         model=args.repair_model,
#         host=args.repair_host,
#         api_key=args.repair_api_key or os.environ.get('GROQ_API_KEY'),
#         opencc_config=args.opencc_config,
#         whitelist_path=args.whitelist,
#         glossary_path=args.glossary,
#         gap_period=args.seed_period_gap,
#         gap_comma=args.seed_comma_gap,
#     )
#     clean_text = res.clean_text
# else:
#     # 不修：直接用 hyp_text
#     clean_text = hyp_text
# 
# # 輸出字幕
# os.makedirs(args.out_dir, exist_ok=True)
# stem = audio_path.stem
# if args.subtitle_format == 'srt':
#     if args.repair_enable:
#         write_srt_with_repair(str(Path(args.out_dir) / f"{stem}.srt"), toks, clean_text, gap_break=args.srt_gap_break, max_line=args.srt_max_line)
#     else:
#         write_srt(Path(args.out_dir) / f"{stem}.srt", committed)
# else:
#     with open(Path(args.out_dir) / f"{stem}.txt", 'w', encoding='utf-8') as f:
#         f.write(clean_text + "\n")
# 
# # 另存 edits 日誌（若有啟用）
# if args.repair_enable:
#     with open(Path(args.out_dir) / f"{stem}.edits.jsonl", 'w', encoding='utf-8') as ef:
#         try:
#             for e in res.edits:
#                 ef.write(json.dumps(e.__dict__, ensure_ascii=False) + "\n")
#         except Exception:
#             pass
# 
# # JSONL (committed token dump) 如你原有邏輯
# if args.keep_jsonl and (not args.no_jsonl):
#     write_jsonl(Path(args.out_dir) / f"{stem}.jsonl", committed)
#
# --- 完成 ---
