"""
fast_aggregator.py
最終版：配合 orchestrator_v2 使用的 FAST 聚合器。

目標：
- 行為貼近 sample / Fast_Test：
  - committed / tail 雙區域
  - guard_sec / protect_head_sec / final_protect_sec 等窗際防重疊策略
  - fuse_back / commit_tail_sec 等尾巴延展與「成熟句」提交邏輯
- 介面貼近 v2 現在呼叫方式：
  FastAggregator().append_window_tokens(...)
  FastAggregator().finalize()
  FastAggregator().export_txt(...)
  FastAggregator().broadcast_snapshot(...)

同時支援 SSE /snapshot /stream 用於即時監看。
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("fast_aggregator")


# -------------------------------------------------
# 低階資料型別
# -------------------------------------------------

@dataclass
class Tok:
    """
    一小段文字 + 時間範圍
    """
    text: str
    start: float
    end: float
    prob: float = 0.0


@dataclass
class AggregatorState:
    """
    Aggregator 內部狀態
    - committed: 已經「鎖定」的 token，不再修改
    - tail: 仍可被更新/覆蓋/延長的 token
    - last_committed_end: committed 最後時間，當成地板
    """
    committed: List[Tok] = field(default_factory=list)
    tail: List[Tok] = field(default_factory=list)
    last_committed_end: float = 0.0


# -------------------------------------------------
# 小工具
# -------------------------------------------------

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """
    傳回兩段時間 [a0,a1] 與 [b0,b1] 的重疊秒數
    """
    left = max(a0, b0)
    right = min(a1, b1)
    if right <= left:
        return 0.0
    return right - left


def _len(tok: Tok) -> float:
    return max(0.0, float(tok.end) - float(tok.start))


def _bigram_set(text: str) -> List[str]:
    """
    用來做重複偵測/相似偵測的粗略方法
    """
    s = text.strip()
    if len(s) < 2:
        return [s] if s else []
    return [s[i:i+2] for i in range(len(s)-1)]


# 英數相鄰才放空白，中文直接黏
import re
_RE_ALNUM_END = re.compile(r"[A-Za-z0-9]$")
_RE_ALNUM_START = re.compile(r"^[A-Za-z0-9]")


def _needs_space(a: str, b: str) -> bool:
    return bool(_RE_ALNUM_END.search(a)) and bool(_RE_ALNUM_START.search(b))


def tokens_to_text(tokens: List[Tok]) -> str:
    """
    把一串 Tok 轉成人類可讀文字。
    - 中文不亂塞空格
    - 英數之間保留空格
    """
    out: List[str] = []
    prev = None
    for t in tokens:
        seg = t.text.strip()
        if not seg:
            continue
        if prev is None:
            out.append(seg)
        else:
            if _needs_space(prev, seg):
                out.append(" ")
            out.append(seg)
        prev = seg
    return "".join(out)


# -------------------------------------------------
# Aggregator 本體：合併 & 去重 & commit tail
# -------------------------------------------------

class Aggregator:
    """
    每一個說話人一個 Aggregator。

    行為重點：
    - append_fast()：把新窗口的 tokens 合進來，
      會做去重、融合 (fuse_back)、以及「成熟句提交」commit_tail_sec。
    - state.committed / state.tail：
        committed 是已經確定的歷史，tail 是最近幾秒內還在長/可能覆寫的句尾。
    """

    def __init__(
        self,
        commit_tail_sec: float = 1.20,
        epsilon: float = 1e-3,
        san_near_dup_gap: float = 0.25,
        san_back_overlap_tol: float = 0.12,
        san_overlap_ratio: float = 0.60,
        cov_min_overlap_sec: float = 0.10,
        cov_min_cover_ratio: float = 0.35,
    ) -> None:
        # 狀態
        self.state = AggregatorState()

        # 時間/融合參數
        self.commit_tail_sec = float(commit_tail_sec)
        self.eps = float(epsilon)

        # 影響尾巴延展的寬度
        # fuse_back: 允許把新 token 往前「吃進去」舊 token 多少秒，達到縫合
        self.fuse_back = 0.35  # 後續會由上層 Manager 用 cfg.fuse_back 覆蓋

        # 供 sanitize/back_overlap 用
        self.san_near_dup_gap = float(san_near_dup_gap)
        self.san_back_overlap_tol = float(san_back_overlap_tol)
        self.san_overlap_ratio = float(san_overlap_ratio)
        self.cov_min_overlap_sec = float(cov_min_overlap_sec)
        self.cov_min_cover_ratio = float(cov_min_cover_ratio)

        # 其他重複移除策略 (Manager 建好 aggregator 後再覆蓋)
        self.dedup_near_gap = self.san_near_dup_gap
        self.dedup_overlap_ratio = self.san_overlap_ratio
        self.dedup_repeat_gap = 1.20
        self.dedup_bigram_gap = 1.20

    # ---- public-ish helpers ----

    def tail_duration(self) -> float:
        if not self.state.tail:
            return 0.0
        return max(0.0, self.state.tail[-1].end - self.state.tail[0].start)

    def all_tokens(self) -> List[Tok]:
        """
        committed + tail
        """
        return list(self.state.committed) + list(self.state.tail)

    # ---- main entry ----

    def append_fast(self, toks: List[Tok]) -> None:
        """
        合併一個 window 的 tokens。
        """
        if not toks:
            return

        # 時間排序
        toks = sorted(toks, key=lambda t: (t.start, t.end))

        # 先做清理 / 去重
        toks = self._sanitize_tokens(toks)
        toks = self._back_overlap_sanitize(toks)

        if not toks:
            return

        # 接到 tail 裡，並嘗試 fuse_back
        for t in toks:
            self._append_one(t)

        # 把已經「成熟」(太舊) 的 tail 送進 committed
        self._commit_aged_tail()

    # ---- inner logic ----

    def _append_one(self, t: Tok) -> None:
        """
        把一顆 Tok 串到 tail。
        嘗試跟最後一顆 tail fuse，如果 gap 很小 (<= fuse_back) 就合併成較長的那個。
        """
        st = self.state

        # 我們只會動 tail，不動 committed
        if st.tail:
            last = st.tail[-1]
        else:
            last = None

        if last:
            # gap <= fuse_back -> merge
            gap = t.start - last.end
            if gap <= self.fuse_back + self.eps and gap >= -self.fuse_back - self.eps:
                # 合併文字：偏向選比較長 / 更新到最新
                merged_text = self._merge_text(last.text, t.text)
                merged_end = max(last.end, t.end)
                merged_prob = max(last.prob, t.prob)
                st.tail[-1] = Tok(
                    text=merged_text,
                    start=last.start,
                    end=merged_end,
                    prob=merged_prob,
                )
                return

        # 否則直接 push 進 tail
        st.tail.append(t)

    def _merge_text(self, a: str, b: str) -> str:
        """
        merge 兩段相近文本，盡量保留 b 的更新，同時避免重複。
        策略：
        - 如果 b 是 a 的延伸 (a 為 prefix) -> 用 b
        - 如果 a 是 b 的延伸 -> 用 a
        - 否則就 a + " " + b（用 tokens_to_text 規則其實更好，不過這裡簡化）
        """
        a_strip = a.strip()
        b_strip = b.strip()
        if not a_strip:
            return b_strip
        if not b_strip:
            return a_strip
        if b_strip.startswith(a_strip):
            return b_strip
        if a_strip.startswith(b_strip):
            return a_strip
        # fallback: 串起來（中文不需要空白，但混中英時 tokens_to_text 處理 spacing）
        return a_strip + b_strip

    def _commit_aged_tail(self) -> None:
        """
        將 tail 中「已經超過 commit_tail_sec 的前半段」搬進 committed。
        commit_tail_sec 表示 tail 可以被編輯的「最近幾秒」。
        比如 commit_tail_sec=1.2s => 只保留最後1.2秒在 tail，可以被覆寫。
        """
        st = self.state
        if not st.tail:
            return

        horizon = st.tail[-1].end - self.commit_tail_sec
        push: List[Tok] = []
        keep: List[Tok] = []
        for tk in st.tail:
            if tk.end <= horizon + self.eps:
                push.append(tk)
            else:
                keep.append(tk)

        if push:
            st.committed.extend(push)
            st.last_committed_end = max(st.last_committed_end, push[-1].end)

        st.tail = keep

    # ---- sanitization / dedup / overlap fix ----

    def _sanitize_tokens(self, toks: List[Tok]) -> List[Tok]:
        """
        去掉明顯重複 or 極度相似的片段。
        規則是依照 Fast_Test 的精神：
        - 近距離重複：時間幾乎一樣、內容大同小異 -> 只留一個（通常長的那個）
        - 連續 bigram 幾乎一樣的重覆（"整體流程沒有誤差" vs "流程沒有誤差"）-> 留較新的那個並覆蓋舊的
        """
        out: List[Tok] = []
        last_time_by_text: Dict[str, float] = {}

        for t in toks:
            txt = t.text.strip()
            if not txt:
                continue

            # 1) 最近是否出現過一模一樣的字串？
            lt = last_time_by_text.get(txt)
            if lt is not None and (t.start - lt) <= (self.dedup_repeat_gap + self.eps):
                # 重複太近 => 跟 out[-1] 比長度，留長的
                if out:
                    prev = out[-1]
                    if prev.text.strip() == txt:
                        # 選比較長的時間範圍
                        if _len(t) > _len(prev):
                            out[-1] = t
                        # 否則 skip
                        continue
                # 若 out 空或 out[-1] 不是同字，還是可以加
            last_time_by_text[txt] = t.start

            # 2) 檢查跟上一顆 out[-1] 是否高度重疊，時間又很近
            if out:
                p = out[-1]
                gap = t.start - p.end
                if gap <= self.dedup_near_gap + self.eps:
                    ov = _overlap(p.start, p.end, t.start, t.end)
                    total = max(_len(p), _len(t))
                    if total > 0:
                        ratio = ov / total
                    else:
                        ratio = 0.0

                    # 若文字相似且重疊比率高，保留較長/較新的
                    if ratio >= self.dedup_overlap_ratio - self.eps:
                        # 假如 t.text 包含 p.text，就用 t；反之亦然
                        if t.text.strip().startswith(p.text.strip()):
                            out[-1] = Tok(
                                text=t.text,
                                start=p.start,
                                end=max(p.end, t.end),
                                prob=max(p.prob, t.prob),
                            )
                            continue
                        if p.text.strip().startswith(t.text.strip()):
                            # keep p, maybe just extend end
                            out[-1] = Tok(
                                text=p.text,
                                start=p.start,
                                end=max(p.end, t.end),
                                prob=max(p.prob, t.prob),
                            )
                            continue

                # 3) bigram 類似詞，時間接近 -> 偏向較新那個
                gap2 = t.start - p.end
                if gap2 <= self.dedup_bigram_gap + self.eps:
                    bg_a = set(_bigram_set(p.text))
                    bg_b = set(_bigram_set(t.text))
                    inter = bg_a.intersection(bg_b)
                    if bg_a and bg_b:
                        sim = len(inter) / float(min(len(bg_a), len(bg_b)))
                    else:
                        sim = 0.0
                    if sim >= 0.8:
                        # 兩段其實在說同一件事，只是後者更新了
                        out[-1] = Tok(
                            text=t.text,
                            start=min(p.start, t.start),
                            end=max(p.end, t.end),
                            prob=max(p.prob, t.prob),
                        )
                        continue

            out.append(t)

        return out

    def _back_overlap_sanitize(self, toks: List[Tok]) -> List[Tok]:
        """
        修正「後一個 token 倒貼進上一個 token」的場景。
        如果太嚴重重疊，就把前一個 token 的終點往前剪，避免兩段時間互相覆蓋過頭。
        """
        if not toks:
            return []

        out = [toks[0]]
        for t in toks[1:]:
            prev = out[-1]
            ov = _overlap(prev.start, prev.end, t.start, t.end)
            if ov > self.san_back_overlap_tol + self.eps:
                # 把 prev 的 end 往前拉，盡量不超過 t.start
                new_end = min(prev.end, t.start - self.eps)
                if new_end < prev.start:
                    new_end = prev.start
                out[-1] = Tok(
                    text=prev.text,
                    start=prev.start,
                    end=new_end,
                    prob=prev.prob,
                )
            out.append(t)
        return out


# -------------------------------------------------
# Track 狀態
# -------------------------------------------------

@dataclass
class TrackAggregatorState:
    track_id: int
    aggregator: Aggregator
    windows_seen: int = 0
    last_window_end: float = 0.0


# -------------------------------------------------
# SSE Broker + HTTP Server
# -------------------------------------------------

class SSEBroker:
    """
    管理 SSE 訂閱者 (一個訂閱者 -> 一個 Queue)
    """
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.subscribers: List[queue.Queue] = []
        self.running = True

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass

    def stop(self) -> None:
        self.running = False
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(None)
            except Exception:
                pass

    def publish(self, payload: str) -> None:
        """
        payload: 預期是一個 JSON 字串 (event)
        """
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                # 丟掉最舊那個
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except Exception:
                    pass


class AggregatorHTTPServer(ThreadingMixIn, HTTPServer):
    """
    帶著 broker 與 manager 的 HTTPServer，ThreadingMixIn 讓每個 request 獨立 thread。
    """
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, broker: SSEBroker, manager: "FastAggregatorManager"):
        self.broker = broker
        self.manager = manager
        super().__init__(server_address, RequestHandlerClass)


class AggregatorRequestHandler(BaseHTTPRequestHandler):
    """
    /healthz   -> 200 OK
    /snapshot  -> 回傳目前 snapshot JSON
    /stream    -> SSE，不斷送出 view_update/snapshot
    """
    server: AggregatorHTTPServer  # type hint

    def log_message(self, format, *args):
        # 靜音，避免 HTTPServer 把每次 poll 都 print 出來
        return

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path == "/healthz":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if path == "/snapshot":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            snap = self.server.manager.build_snapshot()
            payload = json.dumps({"snapshot": snap}, ensure_ascii=False)
            self.wfile.write(payload.encode("utf-8"))
            return

        if path == "/stream":
            # SSE
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = self.server.broker.subscribe()
            try:
                # 先送一包 init
                init_snap = self.server.manager.build_snapshot()
                init_event = json.dumps({"type": "init", "snapshot": init_snap}, ensure_ascii=False)
                self.wfile.write(f"data: {init_event}\n\n".encode("utf-8"))
                self.wfile.flush()

                while self.server.broker.running:
                    try:
                        msg = q.get(timeout=1.0)
                    except queue.Empty:
                        # keep-alive ping
                        keepalive = "data: {\"type\":\"ping\"}\n\n".encode("utf-8")
                        self.wfile.write(keepalive)
                        self.wfile.flush()
                        continue
                    if msg is None:
                        break
                    # SSE 要求每則訊息用 data:
                    buf = f"data: {msg}\n\n".encode("utf-8")
                    self.wfile.write(buf)
                    self.wfile.flush()
            except BrokenPipeError:
                pass
            except ConnectionResetError:
                pass
            finally:
                self.server.broker.unsubscribe(q)
            return

        # 其他路徑
        self.send_response(404)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"not found")


class ViewUpdateServer:
    """
    包裝 HTTP server + SSEBroker
    """
    def __init__(self, manager: "FastAggregatorManager", host: str, port: int, enable: bool = False) -> None:
        self.manager = manager
        self.httpd: Optional[AggregatorHTTPServer] = None
        self.broker: Optional[SSEBroker] = None
        self.thread: Optional[threading.Thread] = None
        self.enabled = False
        if enable:
            self.start(host, port)

    def start(self, host: str, port: int) -> None:
        self.broker = SSEBroker()
        self.httpd = AggregatorHTTPServer((host, port), AggregatorRequestHandler, self.broker, self.manager)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.enabled = True
        logger.info("FAST SSE server listening on http://%s:%d", host, port)

    def publish(self, payload: str) -> None:
        if not self.enabled or not self.broker:
            return
        self.broker.publish(payload)

    def shutdown(self) -> None:
        if self.httpd:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
        if self.broker:
            try:
                self.broker.stop()
            except Exception:
                pass
        self.enabled = False


# -------------------------------------------------
# Aggregator 設定 (對應 sample)
# -------------------------------------------------

@dataclass
class AggregatorCLIConfig:
    """
    這些是 sample / fast_test 中調過的閾值。
    小心：這些數值會直接影響字幕是否重疊或截尾。
    """
    enabled: bool = True
    track_limit: int = 32

    # 窗口邏輯
    guard_sec: float = 0.40            # 視窗左右邊的保護帶，避免太早／太晚截取
    last_slack_sec: float = 0.35       # 最後一窗可多吃一點到右邊，幫句尾收乾淨
    protect_head_sec: float = 0.30     # tail 前端保護，不要立刻重複送
    final_protect_sec: float = 0.20    # finalize 時保護最後這段不被過早 commit

    # tail / commit
    commit_tail_sec: float = 1.20      # tail 可編輯長度(秒)，越大=越晚鎖定句尾
    epsilon: float = 1e-3              # 時間上的容忍度(秒)

    # fuse / dedup
    fuse_back: float = 0.35            # 新 token 可以往前黏的寬度
    dedup_near_gap: float = 0.25       # 兩個 token 時間超接近 -> 視為同一段候選
    dedup_overlap_ratio: float = 0.60  # 如果兩段重疊比例 >= 這個值，就偏向只留更新的
    dedup_repeat_gap: float = 1.20     # 一模一樣文字在這秒數內重出 -> 視為重複
    dedup_bigram_gap: float = 1.20     # bigram 相似度極高又很近 -> 視為同一句更新版
    back_overlap_tol: float = 0.12     # 容許上一段往後蓋住下一段多少秒

    # coverage-based剪尾 (commit判定相關, 讓很長重疊的東西可以取代舊尾)
    cov_min_overlap_sec: float = 0.10
    cov_min_cover_ratio: float = 0.35  # "新token覆蓋舊token" 的最小佔比


# -------------------------------------------------
# FastAggregatorManager
# -------------------------------------------------

class FastAggregatorManager:
    """
    1. 每個說話人一個 Aggregator。
    2. 對外提供：
        - append_window_tokens()
        - finalize_all()
        - export_txt()
        - build_snapshot() / broadcast_snapshot()
        - enable_sse_server()
    """

    def __init__(
        self,
        config: AggregatorCLIConfig,
        edits_log_path: Path,
        server_host: str = "127.0.0.1",
        server_port: int = 9877,
        server_enable: bool = False,
    ) -> None:
        self.config = config
        self.tracks: Dict[int, TrackAggregatorState] = {}
        self.track_limit = int(config.track_limit)
        self.enabled = bool(config.enabled and self.track_limit > 0)
        self.lock = threading.Lock()

        self.export_path: Optional[str] = None
        self.session_stem: Optional[str] = None

        # SSE server
        self.server = ViewUpdateServer(self, server_host, server_port, enable=(self.enabled and server_enable))

        # 編輯 log 目前先保留路徑 (未實作編輯 API 但避免未來 crash)
        self.edits_log_path = edits_log_path
        if self.enabled:
            self.edits_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------
    # lifecycle
    # ---------------------------------------------

    def shutdown(self) -> None:
        if self.server:
            self.server.shutdown()

    def enable_sse_server(self, host: str, port: int) -> None:
        """
        動態開 SSE server
        """
        # 先關舊的
        if self.server:
            try:
                self.server.shutdown()
            except Exception:
                pass
        self.server = ViewUpdateServer(self, host, port, enable=True)

    def set_export_path(self, path: str) -> None:
        self.export_path = path

    def set_session_stem(self, stem: str) -> None:
        self.session_stem = stem

    # ---------------------------------------------
    # state helpers
    # ---------------------------------------------

    def _build_aggregator(self) -> Aggregator:
        """
        建立一個新的 Aggregator，並套用 config 的參數。
        """
        cfg = self.config
        agg = Aggregator(
            commit_tail_sec=cfg.commit_tail_sec,
            epsilon=cfg.epsilon,
            san_near_dup_gap=cfg.dedup_near_gap,
            san_back_overlap_tol=cfg.back_overlap_tol,
            san_overlap_ratio=cfg.dedup_overlap_ratio,
            cov_min_overlap_sec=cfg.cov_min_overlap_sec,
            cov_min_cover_ratio=cfg.cov_min_cover_ratio,
        )
        # 對齊 sample：把 config 的閾值塞回去 aggregator
        agg.fuse_back = cfg.fuse_back
        agg.dedup_near_gap = cfg.dedup_near_gap
        agg.dedup_overlap_ratio = cfg.dedup_overlap_ratio
        agg.dedup_repeat_gap = cfg.dedup_repeat_gap
        agg.dedup_bigram_gap = cfg.dedup_bigram_gap
        return agg

    def _ensure_state(self, track_id: int) -> TrackAggregatorState:
        st = self.tracks.get(track_id)
        if st is None:
            agg = self._build_aggregator()
            st = TrackAggregatorState(track_id=track_id, aggregator=agg)
            self.tracks[track_id] = st
        return st

    # ---------------------------------------------
    # main external APIs
    # ---------------------------------------------

    def append_window_tokens(
        self,
        track_id: int,
        tokens: List[Dict[str, Any]],
        t_start: float,
        t_end: float,
        is_last_window: bool = False,
        window_index: int = -1,
        rtf: float = 0.0,
    ) -> None:
        """
        把單一說話人的一個 window 的句段 tokens 丟進聚合器。
        orchestrator_v2.handle_window_result() 會對每個 speaker 呼叫一次。
        """
        if not self.enabled:
            return
        if track_id >= self.track_limit:
            return

        # 準備資料
        incoming_toks: List[Tok] = []
        for tk in tokens:
            txt = str(tk.get("text", "")).strip()
            if not txt:
                continue
            start = float(tk.get("start", 0.0))
            end = float(tk.get("end", start))
            prob = float(tk.get("prob", 0.0) or 0.0)
            incoming_toks.append(Tok(text=txt, start=start, end=end, prob=prob))

        if not incoming_toks:
            return

        with self.lock:
            st = self._ensure_state(track_id)
            kept = self._filter_tokens(
                state=st,
                tokens=incoming_toks,
                win_start=t_start,
                win_end=t_end,
                is_last_window=is_last_window,
            )
            if kept:
                st.aggregator.append_fast(kept)
            st.windows_seen += 1
            st.last_window_end = max(st.last_window_end, t_end)

    def finalize_all(self) -> None:
        """
        Window flush 時呼叫。概念上：
        1) 讓各 track 的 aggregator 先做「成熟 tail commit」
        2) 把除了最後 final_protect_sec 之內的 tail 也 push 到 committed
        如此一來，export_txt() 看到的 committed 幾乎就是「你能接受的完整字幕」，
        只有最後不到 final_protect_sec 秒還在 tail 裡持續長。
        """
        if not self.enabled:
            return

        cfg = self.config
        with self.lock:
            for st in self.tracks.values():
                agg = st.aggregator
                # 先做基本 aged commit
                agg._commit_aged_tail()

                # 再做 final_protect，把 tail 中「比較舊」的部分也推進 committed
                if agg.state.tail:
                    cutoff = agg.state.tail[-1].end - cfg.final_protect_sec
                else:
                    cutoff = 0.0

                push: List[Tok] = []
                keep: List[Tok] = []
                for tk in agg.state.tail:
                    if tk.end <= cutoff + agg.eps:
                        push.append(tk)
                    else:
                        keep.append(tk)
                if push:
                    agg.state.committed.extend(push)
                    agg.state.last_committed_end = max(
                        agg.state.last_committed_end,
                        push[-1].end,
                    )
                agg.state.tail = keep

    def export_txt(self, path: str) -> None:
        """
        寫出「每個說話者各自乾淨的 transcript」。
        檔名：<stem>_01.txt, <stem>_02.txt, ...
        committed -> tokens_to_text()
        """
        if not self.enabled:
            return
        base_path = Path(path)
        stem = base_path.stem
        out_dir = base_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        with self.lock:
            for tid, st in sorted(self.tracks.items(), key=lambda kv: kv[0]):
                toks = list(st.aggregator.state.committed)
                if not toks:
                    continue
                text = tokens_to_text(toks).strip()
                fname = f"{stem}_{tid+1:02d}.txt"
                out_file = out_dir / fname
                try:
                    out_file.write_text(text + "\n", encoding="utf-8")
                except Exception as exc:
                    logger.warning("Failed to write %s: %s", out_file, exc)

    # ---------------------------------------------
    # snapshot / SSE
    # ---------------------------------------------

    def _merge_contiguous_tokens(self, toks: List[Tok], gap_tol: float = 0.5) -> List[Tuple[float, float, List[Tok]]]:
        """
        把同一個 track 的 token 合併成較長片段，避免 timeline 太碎。
        回傳 [(start,end,[Tok,...]), ...]
        """
        if not toks:
            return []
        toks = sorted(toks, key=lambda t: (t.start, t.end))
        groups: List[Tuple[float, float, List[Tok]]] = []
        cur_start = toks[0].start
        cur_end = toks[0].end
        cur_list: List[Tok] = [toks[0]]
        for tk in toks[1:]:
            gap = tk.start - cur_end
            if gap <= gap_tol:
                cur_list.append(tk)
                cur_end = max(cur_end, tk.end)
            else:
                groups.append((cur_start, cur_end, cur_list))
                cur_start = tk.start
                cur_end = tk.end
                cur_list = [tk]
        groups.append((cur_start, cur_end, cur_list))
        return groups

    def build_snapshot(self) -> Dict[str, Any]:
        """
        產生一個 snapshot dict，給 /snapshot 以及 SSE 初始包。
        結構跟我們前端 viewer 對齊：
        {
          "session": "...",
          "tracks": {
             "0": {"track_id":0,"text":"...","tail_duration":1.2,...},
             ...
          },
          "timeline":[
             {"track_id":0,"start":12.3,"end":15.2,"text":"..."},
             ...
          ]
        }
        """
        snap_tracks: Dict[str, Dict[str, Any]] = {}
        timeline_segments: List[Dict[str, Any]] = []

        with self.lock:
            for tid, st in sorted(self.tracks.items(), key=lambda kv: kv[0]):
                agg = st.aggregator
                committed_text = tokens_to_text(agg.state.committed)
                tail_text = tokens_to_text(agg.state.tail)
                full_text = committed_text + tail_text
                snap_tracks[str(tid)] = {
                    "track_id": tid,
                    "text": full_text,
                    "committed_text": committed_text,
                    "tail_text": tail_text,
                    "tail_duration": agg.tail_duration(),
                    "last_committed_end": agg.state.last_committed_end,
                }

                # timeline: 把 committed+tail 合併
                toks_all = agg.all_tokens()
                for (seg_start, seg_end, group_tokens) in self._merge_contiguous_tokens(toks_all):
                    timeline_segments.append({
                        "track_id": tid,
                        "start": seg_start,
                        "end": seg_end,
                        "text": tokens_to_text(group_tokens),
                    })

        timeline_segments.sort(key=lambda s: (s["start"], s["end"]))

        return {
            "session": self.session_stem or "session",
            "tracks": snap_tracks,
            "timeline": timeline_segments,
        }

    def broadcast_snapshot(self, session_stem: Optional[str] = None) -> None:
        """
        將完整 snapshot 丟到 SSE。v2 在每個 window flush 後會叫我。
        """
        if not self.enabled or not self.server or not self.server.enabled:
            return
        if session_stem:
            self.session_stem = session_stem
        snap = self.build_snapshot()
        event = {
            "type": "snapshot",
            "snapshot": snap,
        }
        payload = json.dumps(event, ensure_ascii=False)
        self.server.publish(payload)

    # ---------------------------------------------
    # token 篩選邏輯 (跨窗 dedup / 保護尾巴)
    # ---------------------------------------------

    def _filter_tokens(
        self,
        state: TrackAggregatorState,
        tokens: List[Tok],
        win_start: float,
        win_end: float,
        is_last_window: bool,
    ) -> List[Tok]:
        """
        窗口級過濾邏輯（近似 sample / fast_test）：
        - 不要重送太早的內容
        - 右邊留 guard，最後一窗可以 last_slack_sec
        - 保護上一窗 tail 的頭 (protect_head_sec)；允許「接縫」(seam_free)
        - 在最後一窗留下 final_protect_sec 的尾巴做為未定稿
        """
        agg = state.aggregator
        cfg = self.config
        eps = cfg.epsilon

        # --- 1) 基本左邊界 accept_s ---
        # 第一窗 / 很早的窗：允許從0開始
        # 之後的窗：至少要跳過 guard_sec
        if state.windows_seen == 0 or win_start < cfg.guard_sec:
            accept_s = 0.0
        else:
            accept_s = win_start + cfg.guard_sec

        # 不能回到已提交的 committed 前面
        accept_s = max(accept_s, agg.state.last_committed_end - eps)

        # --- 2) 右邊界：避免吃太寬 ---
        def _within_right_bound(tk: Tok) -> bool:
            if is_last_window:
                # 最後一窗多吃一點，幫忙收尾
                return tk.end <= (win_end + cfg.last_slack_sec) + eps
            else:
                # 中間窗留 guard_sec，不要黏太右邊
                return tk.end <= (win_end - cfg.guard_sec) + eps

        # 先只用右邊界篩
        cand: List[Tok] = [tk for tk in tokens if _within_right_bound(tk)]

        # --- 3) protect_head_sec 與 seam_free ---
        #    如果這不是第一個窗，tail 裡其實已經有上一窗的尾巴。
        #    我們不想把 tail 最前面那一小段又當成"新句子的開頭"重送，
        #    但允許靠近 tail 尾巴的「接縫」(seam_free)。
        if agg.state.tail and cfg.protect_head_sec > 0.0 and cand:
            tail_head = agg.state.tail[0].start
            seam_free = agg.state.tail[-1].end - agg.fuse_back

            cutoff = max(
                tail_head + cfg.protect_head_sec,
                agg.state.last_committed_end + 0.005,
                accept_s,
            )

            tmp: List[Tok] = []
            for tk in cand:
                if tk.start < accept_s - agg.eps:
                    # token 開始時間甚至比 accept_s 還早，
                    # 只有在它正好接在舊 tail 尾巴(>= seam_free)時才放行，
                    # 讓句子可以順著往下長。
                    if tk.start >= seam_free - agg.eps:
                        tmp.append(tk)
                    continue

                # tk.start >= accept_s
                # 仍須檢查 cutoff（避免把舊 tail 的頭又重送）
                if tk.start >= cutoff - agg.eps:
                    tmp.append(tk)
                elif tk.start >= seam_free - agg.eps:
                    # 接縫例外
                    tmp.append(tk)
            kept = tmp
        else:
            # 這是第一個窗 or tail 還沒東西 => 直接用 accept_s 來擋太舊的
            kept = [tk for tk in cand if tk.start >= accept_s - agg.eps]

        # --- 4) final_protect_sec ---
        # 最後一窗時，我們想留下一小段尾巴在 tail，
        # 讓它還可以在下一個窗被修正，而不是馬上被當最終稿。
        if is_last_window and agg.state.tail and cfg.final_protect_sec > 0.0 and kept:
            tail_end = agg.state.tail[-1].end
            protect_start = max(accept_s, tail_end - cfg.final_protect_sec)
            kept = [tk for tk in kept if tk.start >= protect_start - agg.eps]

        return kept


# -------------------------------------------------
# v2 對接外層
# -------------------------------------------------

class FastAggregator:
    """
    orchestrator_v2 會使用的對接層。
    外面不用直接碰 FastAggregatorManager / Aggregator。
    """

    def __init__(self) -> None:
        cfg = AggregatorCLIConfig()
        edits_log_path = Path("fast_edits.log")
        self.manager = FastAggregatorManager(
            config=cfg,
            edits_log_path=edits_log_path,
            server_host="127.0.0.1",
            server_port=9877,
            server_enable=False,
        )
        # 讓 v2 可以存這些資訊，不用每次重傳
        self._export_path: Optional[str] = None
        self._session_stem: Optional[str] = None

    def set_export_path(self, path: str) -> None:
        self._export_path = path
        self.manager.set_export_path(path)

    def set_session_stem(self, stem: str) -> None:
        self._session_stem = stem
        self.manager.set_session_stem(stem)

    def enable_sse_server(self, host: str = "127.0.0.1", port: int = 9877) -> None:
        """
        啟動 SSE server (/snapshot, /stream) 讓你即時在瀏覽器看到字幕。
        """
        self.manager.enable_sse_server(host, port)

    def append_window_tokens(
        self,
        track_id: int,
        tokens: List[Dict[str, Any]],
        t_start: float,
        t_end: float,
        is_last_window: bool = False,
        window_index: int = -1,
        rtf: float = 0.0,
    ) -> None:
        self.manager.append_window_tokens(
            track_id=track_id,
            tokens=tokens,
            t_start=t_start,
            t_end=t_end,
            is_last_window=is_last_window,
            window_index=window_index,
            rtf=rtf,
        )

    def finalize(self) -> None:
        """
        在每個 window flush 後呼叫，讓字幕更完整（但仍保留最後final_protect_sec秒在 tail）。
        """
        self.manager.finalize_all()

    def export_txt(self, path: str) -> None:
        """
        依目前 committed 內容寫出每位說話者各自的稿。
        orchestrator_v2 會在 finalize() 之後呼叫這個。
        """
        self.manager.export_txt(path)

    def broadcast_snapshot(self, session_stem: Optional[str] = None) -> None:
        """
        把目前 snapshot 丟到 SSE (type="snapshot")
        """
        self.manager.broadcast_snapshot(session_stem=session_stem)

    # 可能之後要用
    def shutdown(self) -> None:
        self.manager.shutdown()
