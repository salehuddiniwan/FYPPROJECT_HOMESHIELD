"""InsightFace wrapper. Prefers Face_Detection/face_recognizer.py if importable;
falls back to a direct FaceAnalysis init with explicit providers."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Make Face_Detection/ importable like fall_detection/.
_THIS = Path(__file__).resolve().parent
_FACE_DIR = _THIS.parent / "Face_Detection"
if _FACE_DIR.is_dir() and str(_FACE_DIR) not in sys.path:
    sys.path.insert(0, str(_FACE_DIR))


def _try_import_user_face_recognizer():
    try:
        from face_recognizer import FaceRecognizer  # type: ignore
        return FaceRecognizer
    except Exception as e:
        print(f"[face] Face_Detection/face_recognizer.py not loadable: {e}")
        return None


def _try_import_face_analysis():
    try:
        from insightface.app import FaceAnalysis  # type: ignore
        return FaceAnalysis
    except Exception as e:
        print(f"[face] insightface not loadable: {e}")
        return None


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class FaceEngine:
    """Thread-safe face detector + ArcFace embedder."""

    DEFAULT_THRESHOLD = 0.42  # matches Face_Detection/face_recognizer.py

    def __init__(self, model_name: str = "buffalo_l",
                 det_size: tuple[int, int] = (640, 640),
                 prefer_gpu: bool = True):
        self.available: bool = False
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()
        self._app = None              # raw FaceAnalysis (fallback path)
        self._user_recognizer = None  # user's FaceRecognizer (preferred path)

        use_gpu = prefer_gpu and _cuda_available()

        if self._try_user_recognizer(use_gpu):
            return
        self._try_face_analysis(model_name, det_size, use_gpu)

    # ---- init helpers ---------------------------------------------------

    def _try_user_recognizer(self, use_gpu: bool) -> bool:
        FaceRecognizer = _try_import_user_face_recognizer()
        if FaceRecognizer is None:
            return False
        for gpu in ([True, False] if use_gpu else [False]):
            try:
                rec = FaceRecognizer(use_gpu=gpu)
                if rec.is_enabled():
                    self._user_recognizer = rec
                    self.available = True
                    print(f"[face] using Face_Detection/face_recognizer.py (GPU={gpu})")
                    return True
            except Exception as e:
                self.last_error = f"FaceRecognizer init: {type(e).__name__}: {e}"
                print(f"[face] FaceRecognizer init failed (gpu={gpu}): {e}")
        if not self.last_error:
            self.last_error = "FaceRecognizer.is_enabled() returned False"
        return False

    def _try_face_analysis(self, model_name: str,
                           det_size: tuple[int, int], use_gpu: bool) -> bool:
        FaceAnalysis = _try_import_face_analysis()
        if FaceAnalysis is None:
            self.last_error = (
                "insightface not installed. "
                "Try: pip install insightface onnxruntime-gpu  "
                "(or use the wheel under Face_Detection/)"
            )
            return False
        for gpu in ([True, False] if use_gpu else [False]):
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if gpu else ["CPUExecutionProvider"])
            try:
                app = FaceAnalysis(name=model_name, providers=providers)
                app.prepare(ctx_id=0 if gpu else -1, det_size=det_size)
                self._app = app
                self.available = True
                self.last_error = None
                print(f"[face] direct FaceAnalysis ready ({providers}, {det_size})")
                return True
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                print(f"[face] FaceAnalysis init failed (gpu={gpu}): {e}")
        return False

    # ---- public API -----------------------------------------------------

    def detect(self, frame_bgr) -> list[dict[str, Any]]:
        if not self.available or frame_bgr is None or frame_bgr.size == 0:
            return []
        backend = (self._user_recognizer._app
                   if self._user_recognizer is not None else self._app)
        if backend is None:
            return []
        with self._lock:
            try:
                faces = backend.get(frame_bgr)
            except Exception as e:
                self.last_error = f"detect: {e}"
                return []
        return [_face_obj_to_dict(f) for f in faces]

    def best_face(self, frame_bgr) -> Optional[dict[str, Any]]:
        faces = self.detect(frame_bgr)
        return max(faces, key=lambda f: f["w"] * f["h"]) if faces else None


# ---- helpers --------------------------------------------------------------

def _face_obj_to_dict(face) -> dict[str, Any]:
    """Convert an InsightFace `Face` proxy to the homeshield dict shape."""
    bbox = np.asarray(face.bbox, dtype=float)
    emb = getattr(face, "normed_embedding", None)
    if emb is None and hasattr(face, "embedding"):
        e = np.asarray(face.embedding, dtype=np.float32)
        emb = e / (np.linalg.norm(e) or 1.0)
    if emb is not None:
        emb = np.asarray(emb, dtype=np.float32)
    return {
        "x": float(bbox[0]),
        "y": float(bbox[1]),
        "w": float(bbox[2] - bbox[0]),
        "h": float(bbox[3] - bbox[1]),
        "age": int(face.age) if getattr(face, "age", None) is not None else None,
        "gender": int(face.gender) if getattr(face, "gender", None) is not None else None,
        "det_score": float(getattr(face, "det_score", 0.0)),
        "embedding": emb,
    }


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a, b = _unit(a), _unit(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def best_match(query, gallery, threshold: float = FaceEngine.DEFAULT_THRESHOLD):
    """gallery: list of (id, embedding). Returns (best_id_or_None, score).

    Vectorised: stacks the gallery into one matrix and does a single
    matrix-vector dot, which is ~10x faster than per-loop cosine for a
    gallery of 10+ persons.
    """
    if not gallery or query is None:
        return None, 0.0
    valid = [(pid, _unit(emb)) for pid, emb in gallery if emb is not None]
    if not valid:
        return None, 0.0
    ids = [pid for pid, _ in valid]
    mat = np.stack([emb for _, emb in valid])  # (N, D), unit-normalised
    sims = mat @ _unit(query)                  # (N,)
    idx = int(np.argmax(sims))
    score = float(sims[idx])
    return (ids[idx], score) if score >= threshold else (None, score)
