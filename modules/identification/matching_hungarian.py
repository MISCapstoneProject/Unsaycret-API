from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianAB:
    """
    Lightweight two-track matcher with hysteresis and inertia.

    The matcher keeps short rolling prototypes per track, solves a
    Hungarian assignment on cosine distance, and applies a double
    threshold to stabilise updates. A small inertia penalty is added
    to discourage rapid flip-flops between tracks when similarity is
    ambiguous.
    """

    def __init__(
        self,
        inertia: float = 0.15,
        stable_th: float = 0.25,
        break_th: float = 0.45,
        proto_len: int = 5,
    ) -> None:
        if not 0.0 <= stable_th <= break_th <= 2.0:
            raise ValueError("stable_th must be <= break_th and both within [0, 2]")

        self.inertia = float(inertia)
        self.stable_th = float(stable_th)
        self.break_th = float(break_th)
        self.proto_len = int(proto_len)

        self._tracks = {0: deque(maxlen=self.proto_len), 1: deque(maxlen=self.proto_len)}
        self._lock_state = {0: False, 1: False}
        self._last_debug: Dict[str, np.ndarray] = {}

    def assign(self, src_embs: List[np.ndarray]) -> Dict[int, int]:
        """
        Assign each source embedding to track A(0) or B(1).

        Args:
            src_embs: List of speaker embeddings (already separated).

        Returns:
            Mapping from source index to track id.
        """
        if not src_embs:
            return {}

        # Only two tracks are supported; ignore extra sources gracefully.
        capped = [self._normalize(e) for e in src_embs[:2]]
        num_sources = len(capped)
        tracks = [0, 1]

        cost = np.zeros((num_sources, len(tracks)), dtype=np.float32)
        sim_cache: Dict[tuple[int, int], float] = {}

        for i, emb in enumerate(capped):
            if emb is None:
                raise ValueError("Embedding cannot be None")
            for j, track_id in enumerate(tracks):
                proto = self._track_vector(track_id)
                if proto is None:
                    sim = 0.0
                    base_cost = 1.0
                else:
                    sim = float(np.clip(np.dot(emb, proto), -1.0, 1.0))
                    base_cost = 1.0 - sim
                penalty = self.inertia * base_cost if proto is not None else 0.0
                cost[i, j] = base_cost + penalty
                sim_cache[(i, track_id)] = sim

        row_idx, col_idx = linear_sum_assignment(cost)
        assignment: Dict[int, int] = {}
        for r, c in zip(row_idx, col_idx):
            if r >= num_sources:
                continue
            assignment[r] = tracks[c]

        self._last_debug = {
            "cost_matrix": cost.copy(),
            "sim_matrix": np.array(
                [[sim_cache.get((i, track_id), 0.0) for track_id in tracks] for i in range(num_sources)],
                dtype=np.float32,
            ),
        }

        # Apply hysteresis + update prototypes.
        for src_idx, track_id in assignment.items():
            sim = sim_cache[(src_idx, track_id)]
            dist = 1.0 - sim

            if dist >= self.break_th:
                # Speaker changed – reset prototype to the new embedding.
                self._tracks[track_id].clear()
                self._tracks[track_id].append(capped[src_idx])
                self._lock_state[track_id] = False
            elif dist <= self.stable_th:
                # Stable match – append new embedding to history.
                self._tracks[track_id].append(capped[src_idx])
                self._lock_state[track_id] = True
            else:
                # Ambiguous zone – blend slowly with existing prototype.
                proto = self._track_vector(track_id)
                blended = (
                    self._normalize(proto * (1.0 - self.inertia) + capped[src_idx] * self.inertia)
                    if proto is not None
                    else capped[src_idx]
                )
                self._tracks[track_id].append(blended)
                self._lock_state[track_id] = False

        return assignment

    def _track_vector(self, track_id: int) -> Optional[np.ndarray]:
        buf = self._tracks.get(track_id)
        if not buf:
            return None
        proto = np.mean(np.stack(buf, axis=0), axis=0)
        return self._normalize(proto)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return arr
        return arr / norm

    @property
    def last_debug(self) -> Dict[str, np.ndarray]:
        return self._last_debug
