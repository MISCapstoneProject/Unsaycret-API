import logging
import re,os, json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, List, Dict, Any

class ChinesePunctuator:
    """
    Immediate cleanup for Chinese ASR text.

    Steps:
      1) (optional) t2s with OpenCC if text appears mostly Traditional.
      2) Punctuation restoration using zhpr + p208p2002/zh-wiki-punctuation-restore.
      3) (optional) very-light spelling correction with pycorrector's MacBERT4CSC (short inputs only).
      4) s2twp normalization to Taiwanese Traditional.
      5) tiny rules: collapse duplicated punctuation/characters; ensure final sentence punctuation.

    All model failures fall back to returning the original text unchanged.
    """

    _WINDOW_SIZE = 128
    _WINDOW_STEP = 64
    _BATCH_SIZE = 8
    _FINAL_PUNCTUATION = "。！？?!"

    def __init__(
        self,
        device: str = "cpu",
        macbert_max_len: int = 64,
        macbert_timeout_ms: int = 80,
        corrections_path: str = None,
        asr_corrections_path: str = None
    ):
        self.logger = logging.getLogger(__name__)
        self.macbert_max_len = max(0, macbert_max_len)
        self.macbert_timeout_s = max(0, macbert_timeout_ms) / 1000.0

        try:
            import torch  # type: ignore
            from torch.utils.data import DataLoader  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ChinesePunctuator requires `torch`. Install via `pip install torch`."
            ) from exc

        try:
            from transformers import (  # type: ignore
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "ChinesePunctuator requires `transformers`. Install via `pip install transformers`."
            ) from exc

        try:
            from zhpr.predict import (  # type: ignore
                DocumentDataset,
                decode_pred,
                merge_stride,
            )
        except ImportError as exc:
            raise ImportError(
                "ChinesePunctuator requires `zhpr`. Install via `pip install zhpr`."
            ) from exc

        try:
            from opencc import OpenCC  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ChinesePunctuator requires `opencc-python-reimplemented`. "
                "Install via `pip install opencc-python-reimplemented`."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning(
                "ChinesePunctuator requested CUDA but it is unavailable; using CPU instead."
            )
            device = "cpu"

        self.device = device
        self._torch = torch
        self._DataLoader = DataLoader
        self._AutoTokenizer = AutoTokenizer
        self._AutoModelForTokenClassification = AutoModelForTokenClassification
        self._DocumentDataset = DocumentDataset
        self._merge_stride = merge_stride
        self._decode_pred = decode_pred
        self._opencc_t2s = OpenCC("t2s")
        self._opencc_s2twp = OpenCC("s2twp")

        self._tokenizer = self._AutoTokenizer.from_pretrained(
            "bert-base-chinese", use_fast=True
        )
        self._pad_token_id = self._tokenizer.pad_token_id
        if self._pad_token_id is None:
            pad_token = self._tokenizer.pad_token or "[PAD]"
            self._pad_token_id = self._tokenizer.convert_tokens_to_ids(pad_token)

        self._punct_model = self._AutoModelForTokenClassification.from_pretrained(
            "p208p2002/zh-wiki-punctuation-restore"
        )
        self._punct_model.to(self._torch.device(self.device))
        self._punct_model.eval()
        self._id2label = {
            int(idx): label for idx, label in self._punct_model.config.id2label.items()
        }

        self._macbert_corrector = None
        self._macbert_available: Optional[bool] = None
        self._macbert_lock = threading.Lock()
        # === Load corrections (fail-open) ===
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 預設路徑（可依你的專案實際位置調整）
        default_general = os.path.join(base_dir, "dictionary_corrections.json")
        default_asr     = os.path.join(base_dir, "dictionary_asr_corrections.json")

        self._corrections_path = corrections_path or default_general
        self._asr_corrections_path = asr_corrections_path or default_asr

        self._corr_rules = self._load_correction_rules(
            [self._corrections_path, self._asr_corrections_path]
        )

    def apply(self, text: str, use_macbert: bool = True) -> str:
        if not isinstance(text, str) or not text:
            return text

        original_text = text

        try:
            working = text

            if self._looks_traditional(working):
                working = self._opencc_t2s.convert(working)
            # C) 在送入標點模型前，先把 ASCII 標點統一成中文，降低相鄰插標點機率
            working = self._normalize_ascii_punct(working)
            
            working = self._restore_punctuation(working)

            if use_macbert and len(working) <= self.macbert_max_len:
                working = self._macbert_cleanup(working)

            working = self._opencc_s2twp.convert(working)
            # E) s2twp 之後再做一次去重與清理（含移除 [UNK]）
            working = self._post_process(working)
            # 新增：合併保護詞（避免「不。過」等）
            working = self._protected_merge(working)
            # 新增：功能詞前不收句（把誤切句號換成逗號）
            working = self._no_end_before_function_word(working)
            # 新增：常見 ASR 錯字輕量修正
            working = self._fix_common_asr_errors(working)

            return working
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(
                "ChinesePunctuator failed open with error: %s", exc, exc_info=True
            )
            return original_text

    def _restore_punctuation(self, text: str) -> str:
        dataset = self._DocumentDataset(
            text, window_size=self._WINDOW_SIZE, step=self._WINDOW_STEP
        )
        dataloader = self._DataLoader(dataset, batch_size=self._BATCH_SIZE)

        stride_outputs = []
        inference_ctx = (
            self._torch.inference_mode()
            if hasattr(self._torch, "inference_mode")
            else self._torch.no_grad()
        )
        with inference_ctx:
            for batch in dataloader:
                input_ids = batch.to(self._punct_model.device)
                logits = self._punct_model(input_ids).logits
                predictions = logits.argmax(dim=-1).cpu().tolist()
                batch_ids = batch.cpu().tolist()

                for token_ids, label_ids in zip(batch_ids, predictions):
                    stride = []
                    for token_id, label_id in zip(token_ids, label_ids):
                        if token_id == self._pad_token_id:
                            break
                        token = self._tokenizer.convert_ids_to_tokens([token_id])[0]
                        token = token.replace("##", "")
                        label = self._id2label.get(int(label_id), "O")
                        stride.append((token, label))
                    if stride:
                        stride_outputs.append(stride)

        if not stride_outputs:
            return text

        merged = self._merge_stride(stride_outputs, dataset.step)
        decoded = self._decode_pred(merged)
        return "".join(decoded)

    def _macbert_cleanup(self, text: str) -> str:
        if not text:
            return text

        if not self._ensure_macbert():
            return text

        def _run_correction():
            result = self._macbert_corrector.correct(text, silent=True)
            if isinstance(result, dict):
                return result.get("target") or text
            if isinstance(result, list) and result:
                entry = result[0]
                if isinstance(entry, dict):
                    return entry.get("target") or text
            return text

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_correction)
            try:
                corrected = future.result(timeout=self.macbert_timeout_s)
            except TimeoutError:
                self.logger.debug("MacBERT cleanup timed out after %.3fs", self.macbert_timeout_s)
                future.cancel()
                return text
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug("MacBERT cleanup failed: %s", exc)
                return text

        return corrected or text

    def _ensure_macbert(self) -> bool:
        if self._macbert_available is False:
            return False
        if self._macbert_corrector is not None:
            return True

        with self._macbert_lock:
            if self._macbert_corrector is not None:
                return True
            try:
                from pycorrector.macbert.macbert_corrector import (  # type: ignore
                    MacBertCorrector,
                )
            except ImportError:
                self.logger.debug(
                    "pycorrector not installed; skipping MacBERT cleanup step."
                )
                self._macbert_available = False
                return False

            try:
                self._macbert_corrector = MacBertCorrector(
                    "shibing624/macbert4csc-base-chinese"
                )
                self._macbert_available = True
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug("Failed to initialize MacBERT: %s", exc)
                self._macbert_available = False
                self._macbert_corrector = None
                return False

        return self._macbert_available is True

    def _looks_traditional(self, text: str) -> bool:
        sample_chars = [ch for ch in text if self._is_cjk(ch)]
        if not sample_chars:
            return False

        sample = "".join(sample_chars)
        converted = self._opencc_t2s.convert(sample)
        limit = min(len(sample), len(converted))
        if limit == 0:
            return False

        diffs = sum(
            1 for idx in range(limit) if sample[idx] != converted[idx]
        )
        ratio = diffs / limit
        return ratio >= 0.3

    def _post_process(self, text: str) -> str:
        if not text:
            return text
        
        # B) 相同標點連發去重
        collapsed = re.sub(r"([，、。？！；])\1+", r"\1", text)
        # B) 混合標點去重（中文優先）
        collapsed = re.sub(r"[，,]{2,}", "，", collapsed)
        collapsed = re.sub(r"[。.\u3002]{2,}", "。", collapsed)
        collapsed = re.sub(r"[！!]{2,}", "！", collapsed)
        collapsed = re.sub(r"[？?]{2,}", "？", collapsed)
        # B) 中西標點相鄰 → 收斂為中文一個
        collapsed = re.sub(r"，\s*[,，]", "，", collapsed)
        collapsed = re.sub(r"。\s*[。\.]", "。", collapsed)
        collapsed = re.sub(r"！\s*[!！]", "！", collapsed)
        collapsed = re.sub(r"？\s*[?\?]", "？", collapsed)
        # E) 任意字連發 ≥4 壓到 2（如 哈哈哈哈 → 哈哈）
        collapsed = re.sub(r"(.)\1{3,}", lambda m: m.group(1) * 2, collapsed)
        # E) 清掉解碼副產品（UNK 類）
        collapsed = re.sub(r"\[UNK\]+", "", collapsed)

        stripped = collapsed.rstrip()
        trail_len = len(collapsed) - len(stripped)
        trail = collapsed[-trail_len:] if trail_len else ""

        if stripped and stripped[-1] not in self._FINAL_PUNCTUATION:
            stripped += "。"

        return stripped + trail
    
    # 在 ChinesePunctuator 類別內，補這三個 helper，並在 apply() 結尾串起來

    def _protected_merge(self, text: str) -> str:
        """
        合併容易被錯切的連詞/轉折詞，避免出現「不。過」「因,此」之類。
        """
        protected = [
            "不過","因此","此外","至於","而且","另外","接著","總之","最後","首先","然後","然而"
        ]
        t = text
        for w in protected:
            if len(w) == 2:
                a, b = w[0], w[1]
                t = re.sub(fr"{a}[。．\.,，、]{b}", w, t)
        return t

    def _no_end_before_function_word(self, text: str) -> str:
        """
        若句號後面是功能詞/介詞開頭，且前一句過短，將該句號改為逗號。
        例: 「…表示。為防止…」→「…表示，為防止…」
        """
        func_heads = "為並且及若如而與或等再能又則及其"
        # 找到「。 + （可有空白） + 功能詞」
        pat = re.compile(rf"(.{{1,10}})。(?:\s*)([{func_heads}])")
        def _rep(m):
            prev = m.group(1)
            head = m.group(2)
            # 前半句過短才視為誤切，改成逗號
            return f"{prev}，{head}"
        return pat.sub(_rep, text)


    def _load_correction_rules(self, paths: List[str]) -> List[Dict[str, Any]]:
        """
        從多個 JSON 詞庫載入並合併成可執行規則：
        每筆 item 形如:
          { "wrong": "戰機", "correct": "戰績",
            "whitelist": ["戰機模型"], "notes": "..."}
        回傳格式:
          [{ "pat": compiled_regex, "replace": str, "whitelist": set([...]) }, ...]
        缺檔 or 格式錯誤 → 忽略（fail-open）
        """
        rules: List[Dict[str, Any]] = []
        seen_wrong = set()

        for p in paths:
            try:
                if not p or not os.path.exists(p):
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    # 若是 dict 形式 {"錯字":"正字"} 也支援
                    if isinstance(data, dict):
                        data = [{"wrong": k, "correct": v} for k, v in data.items()]
                    else:
                        continue

                for item in data:
                    wrong = (item.get("wrong") or "").strip()
                    correct = (item.get("correct") or "").strip()
                    if not wrong or not correct:
                        continue
                    if wrong in seen_wrong:
                        # 先載入者優先；ASR 詞庫放在後面路徑可覆蓋一般詞庫
                        continue
                    seen_wrong.add(wrong)

                    wl = item.get("whitelist") or []
                    if isinstance(wl, str):
                        wl = [wl]
                    wl = [s for s in wl if isinstance(s, str) and s.strip()]
                    pat = re.compile(re.escape(wrong))
                    rules.append({
                        "pat": pat,
                        "replace": correct,
                        "whitelist": set(wl)
                    })
            except Exception:
                # 任一檔案讀取失敗就忽略，不影響主流程
                continue
        return rules

    def _fix_common_asr_errors(self, text: str) -> str:
        """
        常見同音/形近字修正 — 讀取 JSON 詞庫（一般 + ASR 專用）並套用。
        - 命中白名單片語就跳過該條替換，避免誤殺；
        - 任一錯誤/缺檔一律 fail-open（回傳原文）。
        """
        t = text
        if not t or not self._corr_rules:
            return t

        for rule in self._corr_rules:
            wl = rule.get("whitelist", set())
            # 任一白名單片語出現於文本 → 跳過這條規則
            if wl and any(phrase in t for phrase in wl):
                continue
            t = rule["pat"].sub(rule["replace"], t)
        return t


    def _normalize_ascii_punct(self, text: str) -> str:
        """C) 將常見 ASCII 標點先轉為中文全形，降低模型重複插入機率。"""
        if not text:
            return text
        t = text
        t = t.replace(",", "，").replace(".", "。").replace("!", "！").replace("?", "？").replace(";", "；")
        # 避免把英文數字的千分位/小數點誤改，可再視需求加語境判斷
        return t


    @staticmethod
    def _is_cjk(char: str) -> bool:
        if not char:
            return False
        code = ord(char)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
        )
