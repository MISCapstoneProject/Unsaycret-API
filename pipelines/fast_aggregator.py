# fast_aggregator.py
#
# 這個模組是從原本 orchestrator 大檔案裡，把 FAST 聚合器相關的
# 各種 class / dataclass / helper function / SSE server 包成獨立檔案。
#
# 目的：主 orchestrator 之後只需要 `from fast_aggregator import *`
# 就能用 AggregatorCLIConfig / FastAggregatorManager / FAST_COMPONENTS 等功能，
# 而不用在同一支檔案裡塞滿上千行。
#
# 注意：邏輯基本維持原狀，沒有主動簡化演算法或砍功能。
# 只有 import / module 邊界做了重組。


import http.server
import importlib
import json
import queue
import socketserver
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


# --- FastComponents / _load_fast_components -----------------
# 用來動態載入 FAST 模組 (Aggregator, Tok, tokens_to_text, write_srt...)
# 如果找不到，FAST_COMPONENTS 就會是 None，聚合器就會被標成 disabled

@dataclass
class FastComponents:
    Aggregator: Any
    Tok: Any
    tokens_to_text: Any
    write_srt: Any
    write_txt: Any
    write_srt_with_repair: Optional[Any]


def _load_fast_components() -> Optional[FastComponents]:
    module_candidates = ["modules.asr.Fast_Test", "Fast_Test"]
    for name in module_candidates:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        required = ("Aggregator", "Tok", "tokens_to_text", "write_srt")
        if not all(hasattr(mod, attr) for attr in required):
            continue

        write_txt_func = getattr(mod, "write_txt", None)
        if write_txt_func is None:
            # 如果 FAST_Test 裡沒提供 write_txt，我們 fallback 成一個基本版
            def _fallback_write_txt(path: Path, toks: List[Any]) -> None:
                text = getattr(mod, "tokens_to_text")(toks)
                Path(path).write_text(text + "\n", encoding="utf-8")

            write_txt_func = _fallback_write_txt

        # 可選：有沒有 LLM repair 版本字幕輸出
        write_srt_with_repair = None
        for repair_mod in ("modules.asr.llm_repair", "llm_repair"):
            try:
                rep = importlib.import_module(repair_mod)
                if hasattr(rep, "write_srt_with_repair"):
                    write_srt_with_repair = getattr(rep, "write_srt_with_repair")
                    break
            except Exception:
                continue

        return FastComponents(
            Aggregator=getattr(mod, "Aggregator"),
            Tok=getattr(mod, "Tok"),
            tokens_to_text=getattr(mod, "tokens_to_text"),
            write_srt=getattr(mod, "write_srt"),
            write_txt=write_txt_func,
            write_srt_with_repair=write_srt_with_repair,
        )

    logger.warning("FAST aggregator modules not found; aggregation disabled.")
    return None


FAST_COMPONENTS = _load_fast_components()


# --- AggregatorCLIConfig / TrackAggregatorState --------------
# 設定聚合器行為 & 追蹤每個 track 的狀態

@dataclass
class AggregatorCLIConfig:
    enabled: bool
    track_limit: int
    guard_sec: float
    last_slack_sec: float
    protect_head_sec: float
    final_protect_sec: float
    commit_tail_sec: float
    epsilon: float
    fuse_back: float
    dedup_near_gap: float
    dedup_overlap_ratio: float
    dedup_repeat_gap: float
    dedup_bigram_gap: float
    back_overlap_tol: float
    cov_min_overlap_sec: float
    cov_min_cover_ratio: float
    srt_gap_break: float
    srt_max_line: int
    repair_enable: bool


@dataclass
class TrackAggregatorState:
    track_id: int
    aggregator: Any
    windows_seen: int = 0
    last_window_end: float = 0.0


# --- SSE / HTTP streaming infra ------------------------------
# 讓外部 (例如即時字幕前端) 可以訂閱目前聚合器的狀態
# 這些東西在 FastAggregatorManager 裡的 self.server 會被使用
# 你說 SSE server 沒差，我保留原邏輯，未來要砍再一起砍

class SSEBroker:
    def __init__(self) -> None:
        self.subscribers: List[queue.Queue] = []
        self.lock = threading.Lock()
        self.running = True

    def register(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unregister(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def publish(self, payload: str) -> None:
        with self.lock:
            targets = list(self.subscribers)
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                # queue滿了就丟掉最舊的，嘗試再塞
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

    def stop(self) -> None:
        self.running = False
        with self.lock:
            targets = list(self.subscribers)
        for q in targets:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass


class AggregatorHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class AggregatorRequestHandler(http.server.BaseHTTPRequestHandler):
    server_version = "AggStream/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - debug noise
        logger.debug("agg_server: " + format, *args)

    def do_GET(self) -> None:  # pragma: no cover - I/O heavy
        if self.path.startswith("/stream"):
            broker: Optional[SSEBroker] = getattr(self.server, "broker", None)
            if broker is None or not broker.running:
                self.send_error(http.HTTPStatus.SERVICE_UNAVAILABLE, "stream disabled")
                return

            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = broker.register()
            try:
                while broker.running:
                    try:
                        payload = q.get(timeout=1.0)
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    if payload is None:
                        break
                    msg = f"data: {payload}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                broker.unregister(q)

        elif self.path.startswith("/healthz"):
            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{\"status\":\"ok\"}")
        else:
            self.send_error(http.HTTPStatus.NOT_FOUND, "unknown endpoint")

    def do_PATCH(self) -> None:  # pragma: no cover - network I/O
        if not self.path.startswith("/tracks/") or not self.path.endswith("/edits"):
            self.send_error(http.HTTPStatus.NOT_FOUND, "unknown endpoint")
            return

        manager = getattr(self.server, "manager", None)
        if manager is None or not getattr(manager, "enabled", False):
            self.send_error(http.HTTPStatus.SERVICE_UNAVAILABLE, "aggregator disabled")
            return

        parts = [p for p in self.path.strip("/").split("/") if p]
        if len(parts) < 3:
            self.send_error(http.HTTPStatus.BAD_REQUEST, "invalid track path")
            return

        try:
            track_id = int(parts[1])
        except ValueError:
            self.send_error(http.HTTPStatus.BAD_REQUEST, "invalid track id")
            return

        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0

        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(http.HTTPStatus.BAD_REQUEST, "invalid json")
            return

        ok, resp = manager.apply_edit(track_id, payload)
        if not ok:
            self.send_error(http.HTTPStatus.BAD_REQUEST, resp.get("error", "edit failed"))
            return

        self.send_response(http.HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))


class ViewUpdateServer:
    def __init__(
        self,
        manager: "FastAggregatorManager",
        host: str,
        port: int,
        enable: bool,
    ) -> None:
        self.manager = manager
        self.host = host
        self.port = port
        self.enable = enable

        self.httpd: Optional[AggregatorHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.broker: Optional[SSEBroker] = None

        if not enable:
            return

        try:
            self.broker = SSEBroker()
            self.httpd = AggregatorHTTPServer((host, port), AggregatorRequestHandler)
            self.httpd.manager = manager
            self.httpd.broker = self.broker

            self.thread = threading.Thread(
                target=self.httpd.serve_forever,
                daemon=True,
            )
            self.thread.start()
            logger.info(
                "Aggregator stream server listening on http://%s:%s/stream",
                host,
                port,
            )
        except Exception as exc:
            logger.warning(
                "Failed to start aggregator stream server (%s:%s): %s",
                host,
                port,
                exc,
            )
            self.enable = False
            self.httpd = None
            self.broker = None

    def publish(self, event: Dict[str, Any]) -> None:
        if not self.enable or not self.broker:
            return
        payload = json.dumps(event, ensure_ascii=False)
        self.broker.publish(payload)

    def shutdown(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.broker:
            self.broker.stop()


# --- FastAggregatorManager -----------------------------------
# 這是整個 FAST 聚合核心：收每個 window 的 word-level token，
# 做去重、拼接、保護尾端可編輯區段、最後輸出 txt/srt，
# 還可以被 HTTP PATCH 修改 (人工修正字幕)，並 push SSE 給前端。

class FastAggregatorManager:
    def __init__(
        self,
        components: Optional[FastComponents],
        config: AggregatorCLIConfig,
        track_limit: int,
        server_host: str,
        server_port: int,
        server_enable: bool,
        edits_log_path: Path,
    ) -> None:
        self.components = components
        self.config = config
        self.track_limit = max(0, track_limit)
        self.enabled = bool(components and config.enabled and self.track_limit > 0)

        self.lock = threading.Lock()
        self.tracks: Dict[int, TrackAggregatorState] = {}

        self.edits_log_path = edits_log_path
        if self.enabled:
            self.edits_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.server = ViewUpdateServer(
            self,
            server_host,
            server_port,
            enable=self.enabled and server_enable,
        )

    def shutdown(self) -> None:
        if self.server:
            self.server.shutdown()

    def _ensure_state(self, track_id: int) -> TrackAggregatorState:
        state = self.tracks.get(track_id)
        if state is None:
            agg = self._build_aggregator()
            state = TrackAggregatorState(track_id=track_id, aggregator=agg)
            self.tracks[track_id] = state
        return state

    def _build_aggregator(self) -> Any:
        assert self.components is not None
        cfg = self.config
        agg = self.components.Aggregator(
            commit_tail_sec=cfg.commit_tail_sec,
            epsilon=cfg.epsilon,
            san_near_dup_gap=cfg.dedup_near_gap,
            san_back_overlap_tol=cfg.back_overlap_tol,
            san_overlap_ratio=cfg.dedup_overlap_ratio,
            cov_min_overlap_sec=cfg.cov_min_overlap_sec,
            cov_min_cover_ratio=cfg.cov_min_cover_ratio,
        )
        agg.fuse_back = cfg.fuse_back
        agg.dedup_near_gap = cfg.dedup_near_gap
        agg.dedup_overlap_ratio = cfg.dedup_overlap_ratio
        agg.dedup_repeat_gap = cfg.dedup_repeat_gap
        agg.dedup_bigram_gap = cfg.dedup_bigram_gap
        agg.front_grace_sec = cfg.guard_sec
        return agg

    def append_window_tokens(
        self,
        track_id: int,
        words: List[Dict[str, Any]],
        window_start: float,
        window_end: float,
        is_last_window: bool,
        window_index: int,
        rtf: float,
    ) -> None:
        if not self.enabled or self.components is None:
            return
        if track_id >= self.track_limit:
            return

        tokens = self._words_to_tokens(words)
        if not tokens:
            return

        with self.lock:
            state = self._ensure_state(track_id)

            filtered = self._filter_tokens(
                state,
                tokens,
                window_start,
                window_end,
                is_last_window,
            )

            if not filtered and not is_last_window:
                state.windows_seen += 1
                state.last_window_end = window_end
                return

            if filtered:
                state.aggregator.append_fast(filtered)

            state.windows_seen += 1
            state.last_window_end = window_end

        self._publish_view_update(track_id, window_end, window_index, rtf)

    def finalize_all(self) -> None:
        if not self.enabled or self.components is None:
            return

        with self.lock:
            for state in self.tracks.values():
                state.aggregator.finalize()

        for track_id, state in self.tracks.items():
            self._publish_view_update(track_id, state.last_window_end, -1, 0.0)

    def export_transcripts(
        self,
        base_dir: Path,
        stem: str,
        repair_enable: bool,
        gap_break: float,
        max_line: int,
    ) -> None:
        if not self.enabled or self.components is None:
            return

        base_dir.mkdir(parents=True, exist_ok=True)
        for track_id, state in sorted(self.tracks.items()):
            toks = list(state.aggregator.state.committed)
            if not toks:
                continue

            txt_path = base_dir / f"{stem}_track{track_id}.txt"
            self.components.write_txt(txt_path, toks)

            srt_path = base_dir / f"{stem}_track{track_id}.srt"
            if repair_enable and self.components.write_srt_with_repair:
                dict_tokens = [
                    {
                        "text": t.text,
                        "start": t.start,
                        "end": t.end,
                        "prob": getattr(t, "prob", 0.0),
                    }
                    for t in toks
                ]
                clean_text = self.components.tokens_to_text(toks)
                self.components.write_srt_with_repair(
                    str(srt_path),
                    dict_tokens,
                    clean_text,
                    gap_break=gap_break,
                    max_line=max_line,
                )
            else:
                self.components.write_srt(
                    srt_path,
                    toks,
                    max_len=max_line,
                    gap_break=gap_break,
                )

    def apply_edit(
        self,
        track_id: int,
        payload: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        提供給 HTTP PATCH：讓使用者修正已經 commit 或 tail 的區段
        """
        if not self.enabled or self.components is None:
            return False, {"error": "aggregator disabled"}

        mode = str(payload.get("mode", "commit")).lower()
        if mode not in {"commit", "tail"}:
            return False, {"error": "mode must be 'commit' or 'tail'"}

        range_info = payload.get("range") or {}
        try:
            start = float(range_info.get("start"))
            end = float(range_info.get("end"))
        except (TypeError, ValueError):
            return False, {"error": "range.start/end required"}

        if end < start:
            start, end = end, start

        replace_text = str(payload.get("replace_text", "")).strip()
        user = str(payload.get("user", "anonymous"))

        with self.lock:
            state = self.tracks.get(track_id)
            if state is None:
                return False, {"error": "track not found"}

            tokens = (
                state.aggregator.state.committed
                if mode == "commit"
                else state.aggregator.state.tail
            )
            if not tokens:
                return False, {"error": "no tokens to edit"}

            replaced, old_text = self._replace_tokens(
                tokens,
                start,
                end,
                replace_text,
            )
            if not replaced:
                return False, {"error": "no overlapping tokens"}

            self._log_edit(
                track_id,
                user,
                start,
                end,
                old_text,
                replace_text,
                mode,
            )

        self._publish_view_update(
            track_id,
            self.tracks[track_id].last_window_end,
            -1,
            0.0,
        )
        return True, {"status": "ok"}

    # -------- internal helpers below -------------------------

    def _replace_tokens(
        self,
        tokens: List[Any],
        start: float,
        end: float,
        new_text: str,
    ) -> Tuple[Optional[Any], str]:
        # 找到重疊 token，刪除後塞一個新的 token 取代
        overlapping = [t for t in tokens if not (t.end <= start or t.start >= end)]
        if not overlapping:
            return None, ""

        span_start = min(start, min(t.start for t in overlapping))
        span_end = max(end, max(t.end for t in overlapping))
        old_text = self.components.tokens_to_text(overlapping)

        remaining = [t for t in tokens if t not in overlapping]
        new_tok = self.components.Tok(
            text=new_text or "",
            start=span_start,
            end=span_end,
            prob=1.0,
        )
        remaining.append(new_tok)
        remaining.sort(key=lambda t: (t.start, t.end))
        tokens[:] = remaining
        return new_tok, old_text

    def _words_to_tokens(self, words: List[Dict[str, Any]]) -> List[Any]:
        if not words:
            return []
        tokens = []
        for w in words:
            text = str(w.get("word") or w.get("text") or "").strip()
            if not text:
                continue
            start = float(w.get("start") or 0.0)
            end = float(w.get("end") or start)
            if end < start:
                end = start
            prob = float(w.get("probability") or w.get("prob") or 0.0)
            tokens.append(
                self.components.Tok(
                    text=text,
                    start=start,
                    end=end,
                    prob=prob,
                )
            )
        return tokens

    def _filter_tokens(
        self,
        state: TrackAggregatorState,
        tokens: List[Any],
        win_start: float,
        win_end: float,
        is_last_window: bool,
    ) -> List[Any]:
        """
        這段是 FAST 聚合器最複雜的地方：
        - guard_sec / last_slack_sec / protect_head_sec / final_protect_sec
          這些參數會控制「哪些 token 允許立刻 commit」，
          避免前面時間線又被後面新的辨識修正，導致字幕抖動。
        """
        agg = state.aggregator
        cfg = self.config
        eps = cfg.epsilon

        if state.windows_seen == 0 or win_start < cfg.guard_sec:
            accept_s = 0.0
        else:
            accept_s = win_start + cfg.guard_sec
        accept_s = max(accept_s, agg.state.last_committed_end - eps)

        def keep_token(tok: Any) -> bool:
            # 在最後一個 window，允許一些額外 slack，讓尾巴也能被提交
            if tok.start < accept_s - eps:
                return False
            if is_last_window:
                return tok.end <= (win_end + cfg.last_slack_sec) + eps
            return tok.end <= (win_end - cfg.guard_sec) + eps

        kept = [t for t in tokens if keep_token(t)]

        # 以下這一坨是 tail 保護邏輯，避免還在「可編輯區」的尾巴被提早鎖死
        if agg.state.tail and cfg.protect_head_sec > 0:
            tail_head = agg.state.tail[0].start
            cutoff = max(
                tail_head + cfg.protect_head_sec,
                agg.state.last_committed_end + 0.005,
                accept_s,
            )
            seam_free = agg.state.tail[-1].end - agg.fuse_back
            filtered = []
            for t in kept:
                if t.start >= cutoff - agg.eps:
                    filtered.append(t)
                elif t.start >= seam_free - agg.eps:
                    filtered.append(t)
            kept = filtered

        # final_protect_sec: 在最後一個 window，上一小段尾巴不要馬上 commit，
        # 以免人還在講話就被鎖死
        if is_last_window and agg.state.tail and cfg.final_protect_sec > 0:
            tail_end = agg.state.tail[-1].end
            protect_start = max(accept_s, tail_end - cfg.final_protect_sec)
            kept = [t for t in kept if t.start >= protect_start - agg.eps]

        return kept

    def _publish_view_update(
        self,
        track_id: int,
        timestamp: float,
        window_index: int,
        rtf: float,
    ) -> None:
        if not self.enabled or self.components is None:
            return

        state = self.tracks.get(track_id)
        if state is None:
            return

        agg = state.aggregator
        committed_text = self.components.tokens_to_text(agg.state.committed)
        tail_text = self.components.tokens_to_text(agg.state.tail)
        tail_duration = agg.tail_duration()
        tail_probs = [getattr(t, "prob", 0.0) for t in agg.state.tail]
        avg_tail_prob = float(sum(tail_probs) / len(tail_probs)) if tail_probs else 0.0

        event = {
            "type": "view_update",
            "t": round(float(timestamp), 3),
            "track_id": track_id,
            "window_index": window_index,
            "committed_text": committed_text,
            "tail_text": tail_text,
            "tail_duration": tail_duration,
            "last_committed_end": agg.state.last_committed_end,
            "stats": {
                "rtf": float(rtf),
                "avg_tail_prob": avg_tail_prob,
            },
        }

        if self.server:
            self.server.publish(event)
            if agg.state.tail:
                tail_event = {
                    "type": "tail_tokens",
                    "track_id": track_id,
                    "tokens": [
                        {
                            "text": t.text,
                            "start": t.start,
                            "end": t.end,
                            "prob": getattr(t, "prob", 0.0),
                        }
                        for t in agg.state.tail
                    ],
                }
                self.server.publish(tail_event)

    def _log_edit(
        self,
        track_id: int,
        user: str,
        start: float,
        end: float,
        old: str,
        new: str,
        mode: str,
    ) -> None:
        if not self.enabled:
            return

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "track_id": track_id,
            "user": user,
            "mode": mode,
            "range": {"start": start, "end": end},
            "old_text": old,
            "new_text": new,
        }

        try:
            with self.edits_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Failed to write edit log entry", exc_info=True)
