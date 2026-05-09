"""Cross-camera person Re-Identification.

Two pieces:

* :class:`ReIDEngine` wraps torchreid's OSNet feature extractor. It turns a
  body crop (BGR ndarray) into an L2-normalised appearance embedding.
  Lazy-loaded; if torchreid is not installed the engine reports
  ``available=False`` and the rest of HomeShield keeps working.

* :class:`ReIDStore` is a process-wide gallery of *global identities*
  (one per real person seen across all cameras). It maps
  ``(camera_id, local_track_id) -> global_id`` and decides when a new
  embedding belongs to an existing global identity vs. a fresh one.

The store is intentionally in-memory: ReID identities are short-lived
("the person who walked through the house in the last few minutes"),
not permanent records like registered persons.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import cv2
import numpy as np


# ---- helpers --------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def crop_body(frame_bgr: np.ndarray, bbox,
              pad_ratio: float = 0.05,
              min_side: int = 32) -> Optional[np.ndarray]:
    """Crop a person bbox from a frame with a small margin.

    ``bbox`` is (x1, y1, x2, y2) in pixel coordinates. Returns ``None``
    if the crop would be empty or smaller than ``min_side`` on either side.
    """
    if frame_bgr is None or bbox is None:
        return None
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    x1i = max(0, int(round(x1 - px)))
    y1i = max(0, int(round(y1 - py)))
    x2i = min(W, int(round(x2 + px)))
    y2i = min(H, int(round(y2 + py)))
    if x2i - x1i < min_side or y2i - y1i < min_side:
        return None
    crop = frame_bgr[y1i:y2i, x1i:x2i]
    return crop if crop.size > 0 else None


# ---- ReIDEngine -----------------------------------------------------------

class ReIDEngine:
    """OSNet appearance-embedding extractor (lazy, GPU-aware)."""

    def __init__(self, model_name: str = "osnet_x1_0",
                 prefer_gpu: bool = True):
        self.model_name = model_name
        self.available: bool = False
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()
        self._extractor = None
        self.device: str = "cpu"
        self._init(prefer_gpu)

    def _init(self, prefer_gpu: bool) -> None:
        try:
            from torchreid.utils import FeatureExtractor  # type: ignore
        except Exception as e:
            self.last_error = (
                f"torchreid not installed ({type(e).__name__}: {e}). "
                f"Install with: pip install torchreid"
            )
            print(f"[reid] {self.last_error}")
            return

        device = "cpu"
        if prefer_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                pass

        try:
            self._extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path="",   # let torchreid pull pretrained weights
                device=device,
            )
            self.device = device
            self.available = True
            print(f"[reid] OSNet ready (model={self.model_name}, device={device})")
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"[reid] OSNet init failed: {e}")

    def embed(self, crops_bgr: list[np.ndarray]) -> Optional[np.ndarray]:
        """Embed a batch of body crops. Returns (N, D) float32 unit vectors,
        or ``None`` if the engine is unavailable / batch is empty."""
        if not self.available or not crops_bgr:
            return None
        # torchreid's FeatureExtractor accepts a list of BGR-or-RGB ndarrays
        # *or* file paths. Internally it converts BGR->RGB; we pass BGR
        # because OpenCV gives BGR by default.
        with self._lock:
            try:
                feats = self._extractor(crops_bgr)  # torch.Tensor (N, D)
            except Exception as e:
                self.last_error = f"embed: {e}"
                print(f"[reid] embed failed: {e}")
                return None
        try:
            arr = feats.detach().cpu().numpy().astype(np.float32, copy=False)
        except Exception:
            arr = np.asarray(feats, dtype=np.float32)
        # L2-normalise each row so cosine sim == dot product later.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        return arr / norms


# ---- ReIDStore ------------------------------------------------------------

class _Identity:
    """One global identity. Keeps the most recent K embeddings so the
    representation slowly tracks appearance changes (lighting, posture)."""

    __slots__ = ("global_id", "embeddings", "last_seen", "last_camera_id",
                 "last_camera_name", "first_seen", "name", "person_id")

    def __init__(self, global_id: int, embedding: np.ndarray, ts: float,
                 camera_id: Optional[int], camera_name: Optional[str]):
        self.global_id = global_id
        self.embeddings: list[np.ndarray] = [embedding]
        self.first_seen = ts
        self.last_seen = ts
        self.last_camera_id = camera_id
        self.last_camera_name = camera_name
        self.name: Optional[str] = None         # filled if face match resolves it
        self.person_id: Optional[int] = None    # registered person, if any

    def add_embedding(self, emb: np.ndarray, max_keep: int = 8) -> None:
        self.embeddings.append(emb)
        if len(self.embeddings) > max_keep:
            self.embeddings = self.embeddings[-max_keep:]

    def representative(self) -> np.ndarray:
        """Mean of the recent embeddings, re-normalised."""
        if len(self.embeddings) == 1:
            return self.embeddings[0]
        m = np.mean(np.stack(self.embeddings), axis=0)
        n = float(np.linalg.norm(m))
        return m / n if n > 1e-9 else m


class ReIDStore:
    """Process-wide gallery of global identities, with an
    ``(camera_id, local_track_id) -> global_id`` cache so the heavy
    embedding step doesn't need to run on every frame for the same track.
    """

    def __init__(self, *, match_threshold: float = 0.65,
                 ttl_seconds: float = 120.0,
                 max_embeddings_per_identity: int = 8):
        self.match_threshold = float(match_threshold)
        self.ttl_seconds = float(ttl_seconds)
        self.max_emb = int(max_embeddings_per_identity)
        # RLock so list_identities() can call identity_info() inside the lock.
        self._lock = threading.RLock()
        self._identities: dict[int, _Identity] = {}
        self._next_id: int = 1
        # (camera_id, local_track_id) -> (global_id, last_assigned_ts)
        self._track_map: dict[tuple[int, int], tuple[int, float]] = {}

    # ---- config tweakable at runtime -----------------------------------

    def update_config(self, *, match_threshold: Optional[float] = None,
                      ttl_seconds: Optional[float] = None) -> None:
        with self._lock:
            if match_threshold is not None:
                self.match_threshold = float(match_threshold)
            if ttl_seconds is not None:
                self.ttl_seconds = float(ttl_seconds)

    # ---- read paths -----------------------------------------------------

    def lookup_track(self, camera_id: int,
                     local_track_id: int) -> Optional[int]:
        """Return the cached global_id for a (camera, local track) pair, if any."""
        key = (int(camera_id), int(local_track_id))
        with self._lock:
            entry = self._track_map.get(key)
        return entry[0] if entry else None

    def identity_info(self, global_id: int) -> Optional[dict[str, Any]]:
        with self._lock:
            ident = self._identities.get(int(global_id))
            if ident is None:
                return None
            return {
                "global_id": ident.global_id,
                "first_seen": ident.first_seen,
                "last_seen": ident.last_seen,
                "last_camera_id": ident.last_camera_id,
                "last_camera_name": ident.last_camera_name,
                "name": ident.name,
                "person_id": ident.person_id,
                "embedding_count": len(ident.embeddings),
            }

    def list_identities(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.identity_info(g) for g in sorted(self._identities)]

    # ---- write paths ----------------------------------------------------

    def assign(self, *, camera_id: int, local_track_id: int,
               embedding: np.ndarray, ts: float,
               camera_name: Optional[str] = None) -> dict[str, Any]:
        """Match ``embedding`` against the live gallery and either link the
        track to an existing global identity or mint a new one.

        Returns a dict with the assignment outcome:
          {
            "global_id": int,
            "score": float,           # cosine similarity to chosen identity
            "is_new": bool,           # minted a new global id this call
            "is_handoff": bool,       # known global id appearing on a NEW camera
            "previous_camera_id": Optional[int],
            "previous_camera_name": Optional[str],
          }
        """
        emb = _unit(np.asarray(embedding, dtype=np.float32))
        if emb.size == 0:
            return {"global_id": -1, "score": 0.0, "is_new": False,
                    "is_handoff": False, "previous_camera_id": None,
                    "previous_camera_name": None}

        with self._lock:
            self._gc_locked(ts)

            # Build candidate matrix from live identities.
            candidates = list(self._identities.values())
            best_id, best_score = None, -1.0
            if candidates:
                mat = np.stack([c.representative() for c in candidates])  # (N, D)
                sims = mat @ emb  # (N,)
                idx = int(np.argmax(sims))
                best_score = float(sims[idx])
                if best_score >= self.match_threshold:
                    best_id = candidates[idx].global_id

            is_new = False
            is_handoff = False
            previous_camera_id: Optional[int] = None
            previous_camera_name: Optional[str] = None

            if best_id is None:
                gid = self._next_id
                self._next_id += 1
                self._identities[gid] = _Identity(
                    global_id=gid, embedding=emb, ts=ts,
                    camera_id=camera_id, camera_name=camera_name,
                )
                is_new = True
            else:
                gid = best_id
                ident = self._identities[gid]
                # Detect cross-camera handoff: same global identity, new camera.
                if (ident.last_camera_id is not None
                        and ident.last_camera_id != camera_id):
                    is_handoff = True
                    previous_camera_id = ident.last_camera_id
                    previous_camera_name = ident.last_camera_name
                ident.add_embedding(emb, self.max_emb)
                ident.last_seen = ts
                ident.last_camera_id = camera_id
                ident.last_camera_name = camera_name

            self._track_map[(int(camera_id), int(local_track_id))] = (gid, ts)

            return {
                "global_id": gid,
                "score": float(best_score) if best_score > 0 else 0.0,
                "is_new": is_new,
                "is_handoff": is_handoff,
                "previous_camera_id": previous_camera_id,
                "previous_camera_name": previous_camera_name,
            }

    def annotate_name(self, global_id: int, *, name: Optional[str],
                      person_id: Optional[int]) -> None:
        """Hook used by the face stage to attach a known name to a global id."""
        with self._lock:
            ident = self._identities.get(int(global_id))
            if ident is None:
                return
            if name:
                ident.name = name
            if person_id is not None:
                ident.person_id = int(person_id)

    # ---- maintenance ----------------------------------------------------

    def _gc_locked(self, now_ts: float) -> None:
        """Drop identities + track-mappings older than ttl_seconds.
        Caller must hold ``self._lock``."""
        cutoff = now_ts - self.ttl_seconds
        dead = [gid for gid, ident in self._identities.items()
                if ident.last_seen < cutoff]
        for gid in dead:
            self._identities.pop(gid, None)
        # Also drop stale track mappings.
        dead_keys = [k for k, (_, t) in self._track_map.items() if t < cutoff]
        for k in dead_keys:
            self._track_map.pop(k, None)

    def reset(self) -> None:
        with self._lock:
            self._identities.clear()
            self._track_map.clear()
            self._next_id = 1
