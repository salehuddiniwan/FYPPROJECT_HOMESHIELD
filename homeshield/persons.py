"""PersonStore (registered faces) + IntruderStore (unrecognised faces).

Both keep a small in-memory cache of (id, embedding) tuples so the
detection loop can match without hitting SQLite per frame.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .db import read_conn, write_conn


# ---- helpers --------------------------------------------------------------

def emb_to_blob(emb: np.ndarray) -> bytes:
    return np.asarray(emb, dtype=np.float32).tobytes()


def blob_to_emb(blob: Optional[bytes]) -> Optional[np.ndarray]:
    return np.frombuffer(blob, dtype=np.float32) if blob else None


def save_face_thumbnail(frame_bgr, bbox, out_path: Path, side: int = 240) -> bool:
    """Crop face with 25% padding and resize so the longest side == ``side``."""
    x, y, w, h = bbox
    H, W = frame_bgr.shape[:2]
    px, py = w * 0.25, h * 0.25
    x1, y1 = max(0, int(x - px)), max(0, int(y - py))
    x2, y2 = min(W, int(x + w + px)), min(H, int(y + h + py))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    h_c, w_c = crop.shape[:2]
    scale = side / max(h_c, w_c)
    resized = cv2.resize(crop, (int(w_c * scale), int(h_c * scale)),
                         interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), resized))


def _maybe_save_thumb(frame_bgr, face_bbox, photo_path: Path) -> bool:
    if frame_bgr is None or face_bbox is None:
        return False
    return save_face_thumbnail(frame_bgr, face_bbox, photo_path)


def _unlink_quietly(p: Optional[str]) -> None:
    if p:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


# ---- PersonStore ----------------------------------------------------------

class PersonStore:
    """Registered known faces."""

    _CATEGORIES = ("adult", "child", "elderly")

    def __init__(self, db_path: str, photos_dir: Path):
        self.db_path = db_path
        self.photos_dir = Path(photos_dir)
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._gallery: list[tuple[int, np.ndarray]] = []
        self._meta: dict[int, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        with read_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT person_id, name, category, photo_path, "
                "embedding_blob, created_at "
                "FROM persons ORDER BY person_id ASC"
            ).fetchall()
        gallery, meta = [], {}
        for r in rows:
            pid = int(r["person_id"])
            emb = blob_to_emb(r["embedding_blob"])
            if emb is not None:
                gallery.append((pid, emb))
            meta[pid] = {
                "person_id": pid,
                "name": r["name"],
                "category": r["category"] or "adult",
                "photo_path": r["photo_path"],
                "created_at": r["created_at"],
            }
        with self._lock:
            self._gallery = gallery
            self._meta = meta

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            persons = list(self._meta.values())
        return [{
            "person_id": p["person_id"],
            "name": p["name"],
            "category": p["category"],
            "photo_url": (f"/person_photos/{Path(p['photo_path']).name}"
                          if p["photo_path"] else None),
            "created_at": p["created_at"],
        } for p in persons]

    def gallery(self) -> list[tuple[int, np.ndarray]]:
        with self._lock:
            return list(self._gallery)

    def name_of(self, person_id: int) -> Optional[str]:
        with self._lock:
            m = self._meta.get(int(person_id))
        return m["name"] if m else None

    def category_of(self, person_id: int) -> Optional[str]:
        with self._lock:
            m = self._meta.get(int(person_id))
        return m["category"] if m else None

    def add(self, *, name: str, category: str, embedding: np.ndarray,
            frame_bgr=None, face_bbox=None) -> dict[str, Any]:
        cat = category if category in self._CATEGORIES else "adult"
        photo_rel = f"person_{int(time.time() * 1000)}.jpg"
        photo_path = self.photos_dir / photo_rel
        ok = _maybe_save_thumb(frame_bgr, face_bbox, photo_path)
        with write_conn(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO persons (name, category, photo_path, embedding_blob)
                   VALUES (?, ?, ?, ?)""",
                (name.strip(), cat,
                 str(photo_path) if ok else None,
                 emb_to_blob(embedding)),
            )
            pid = cur.lastrowid
        self.reload()
        return {
            "person_id": pid,
            "name": name.strip(),
            "category": cat,
            "photo_url": f"/person_photos/{photo_rel}" if ok else None,
        }

    def delete(self, person_id: int) -> None:
        with write_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT photo_path FROM persons WHERE person_id = ?",
                (int(person_id),),
            ).fetchone()
            conn.execute(
                "DELETE FROM persons WHERE person_id = ?", (int(person_id),)
            )
        _unlink_quietly(row["photo_path"] if row else None)
        self.reload()


# ---- IntruderStore --------------------------------------------------------

class IntruderStore:
    """Faces detected on camera that didn't match anyone registered."""

    def __init__(self, db_path: str, photos_dir: Path):
        self.db_path = db_path
        self.photos_dir = Path(photos_dir)
        self.photos_dir.mkdir(parents=True, exist_ok=True)

    def add(self, *, embedding: np.ndarray, camera_id: Optional[int],
            camera_name: Optional[str], frame_bgr=None,
            face_bbox=None) -> Optional[int]:
        photo_rel = f"intruder_{int(time.time() * 1000)}.jpg"
        photo_path = self.photos_dir / photo_rel
        ok = _maybe_save_thumb(frame_bgr, face_bbox, photo_path)
        with write_conn(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO intruders
                   (photo_path, embedding_blob, camera_id, camera_name)
                   VALUES (?, ?, ?, ?)""",
                (str(photo_path) if ok else None,
                 emb_to_blob(embedding),
                 int(camera_id) if camera_id is not None else None,
                 camera_name),
            )
            return cur.lastrowid

    def list(self, include_dismissed: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM intruders"
        if not include_dismissed:
            sql += " WHERE dismissed = 0"
        sql += " ORDER BY detected_at DESC LIMIT 200"
        with read_conn(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
        return [{
            "intruder_id": r["intruder_id"],
            "photo_url": (f"/intruder_photos/{Path(r['photo_path']).name}"
                          if r["photo_path"] else None),
            "camera_id": r["camera_id"],
            "camera_name": r["camera_name"] or "Unknown",
            "detected_at": r["detected_at"],
            "dismissed": bool(r["dismissed"]),
        } for r in rows]

    def dismiss(self, intruder_id: int) -> None:
        with write_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE intruders SET dismissed = 1 WHERE intruder_id = ?",
                (int(intruder_id),),
            )

    def delete(self, intruder_id: int) -> None:
        with write_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT photo_path FROM intruders WHERE intruder_id = ?",
                (int(intruder_id),),
            ).fetchone()
            conn.execute(
                "DELETE FROM intruders WHERE intruder_id = ?", (int(intruder_id),)
            )
        _unlink_quietly(row["photo_path"] if row else None)

    def get(self, intruder_id: int) -> Optional[dict[str, Any]]:
        with read_conn(self.db_path) as conn:
            r = conn.execute(
                "SELECT * FROM intruders WHERE intruder_id = ?",
                (int(intruder_id),),
            ).fetchone()
        if not r:
            return None
        return {
            "intruder_id": r["intruder_id"],
            "photo_path": r["photo_path"],
            "embedding": blob_to_emb(r["embedding_blob"]),
            "camera_id": r["camera_id"],
            "camera_name": r["camera_name"],
            "detected_at": r["detected_at"],
            "dismissed": bool(r["dismissed"]),
        }
