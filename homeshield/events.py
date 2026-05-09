"""Event log + EventBus, backed by the `events` table.

Event types: fall_detected, lying_motionless, inactivity, zone_entry,
intruder_detected, fire_detected, normal, system.

`publish()` is non-blocking: it copies the frame and enqueues onto an
internal queue. A daemon thread does the snapshot + DB + SSE fan-out.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .db import read_conn, write_conn


EVENT_TYPES = {
    "fall_detected", "lying_motionless", "inactivity",
    "zone_entry", "intruder_detected", "fire_detected",
    "cross_camera_handoff",
    "normal", "system",
}


@dataclass
class Event:
    event_type: str
    ts: float = field(default_factory=time.time)
    camera_id: Optional[int] = None
    camera_name: Optional[str] = None
    person_category: str = "unknown"
    confidence: float = 0.0
    details: Optional[str] = None
    snapshot_path: Optional[str] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    meta: dict[str, Any] = field(default_factory=dict)
    event_id: Optional[int] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "ts": self.ts,
            "created_at": _fmt_ts(self.ts),
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "person_category": self.person_category,
            "confidence": float(self.confidence),
            "details": self.details,
            "snapshot_path": self.snapshot_path,
            "bbox": list(self.bbox) if self.bbox else None,
            "meta": self.meta,
        }


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ---- snapshot helpers -----------------------------------------------------

# BGR colour + label per event type.
_EV_VISUAL = {
    "fire_detected":        ((40, 60, 230),   "FIRE"),
    "fall_detected":        ((30, 210, 240),  "FALL"),
    "lying_motionless":     ((40, 60, 230),   "LYING"),
    "inactivity":           ((30, 210, 240),  "INACTIVITY"),
    "intruder_detected":    ((50, 140, 245),  "INTRUDER"),
    "zone_entry":           ((50, 140, 245),  "ZONE ENTRY"),
    "cross_camera_handoff": ((230, 170, 60),  "HANDOFF"),
    "system":               ((220, 160, 80),  "SYSTEM"),
    "normal":               ((100, 220, 130), "OK"),
}


def _draw_event_box(img: np.ndarray, bbox, color, label: str, conf: float) -> None:
    """Modern L-corner box + label background, drawn in-place."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    corner = max(8, min(28, (x2 - x1) // 6))
    th = 3
    for p1, p2 in (
        ((x1, y1), (x1 + corner, y1)), ((x1, y1), (x1, y1 + corner)),
        ((x2, y1), (x2 - corner, y1)), ((x2, y1), (x2, y1 + corner)),
        ((x1, y2), (x1 + corner, y2)), ((x1, y2), (x1, y2 - corner)),
        ((x2, y2), (x2 - corner, y2)), ((x2, y2), (x2, y2 - corner)),
    ):
        cv2.line(img, p1, p2, color, th)
    text = (f" {label}  {int(round(conf * 100))}% "
            if conf > 0 else f" {label} ")
    (tw, th_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    pad = 4
    if y1 - th_text - 2 * pad >= 0:
        by1, by2, ty = y1 - th_text - 2 * pad, y1, y1 - pad
    else:
        by1, by2, ty = y2, y2 + th_text + 2 * pad, y2 + th_text + pad
    cv2.rectangle(img, (x1, by1), (x1 + tw + 2 * pad, by2), color, -1)
    cv2.putText(img, text, (x1 + pad, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)


def save_snapshot(frame: np.ndarray, snapshot_dir: Path, ev: Event,
                  jpeg_quality: int = 85) -> Optional[str]:
    """JPEG with the event's bbox drawn on. Returns basename or None."""
    if frame is None:
        return None
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(ev.ts*1000)}_{ev.camera_id or 0}_{ev.event_type}.jpg"
        out = frame.copy()
        if ev.bbox is not None:
            color, label = _EV_VISUAL.get(
                ev.event_type, ((200, 200, 200), ev.event_type.upper())
            )
            try:
                _draw_event_box(out, ev.bbox, color, label,
                                float(ev.confidence or 0))
            except Exception as e:
                print(f"[events] snapshot annotate failed: {e}")
        cv2.imwrite(str(snapshot_dir / name), out,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        return name
    except Exception as e:
        print(f"[events] snapshot failed: {e}")
        return None


# ---- row mapping ----------------------------------------------------------

def _row_to_dict(r) -> dict[str, Any]:
    return {
        "event_id": r["event_id"],
        "ts": r["ts"],
        "created_at": r["created_at"],
        "event_type": r["event_type"],
        "camera_id": r["camera_id"],
        "camera_name": r["camera_name"],
        "person_category": r["person_category"],
        "confidence": r["confidence"],
        "details": r["details"],
        "snapshot_path": r["snapshot_path"],
        "bbox": json.loads(r["bbox_json"]) if r["bbox_json"] else None,
        "meta": json.loads(r["meta_json"]) if r["meta_json"] else {},
    }


# ---- EventBus -------------------------------------------------------------

class EventBus:
    """Persisted event log + non-blocking publish + SSE fan-out."""

    def __init__(self, db_path: str, snapshot_dir: Optional[Path] = None,
                 sub_queue_size: int = 64,
                 publish_queue_size: int = 256):
        self.db_path = db_path
        self.snapshot_dir = snapshot_dir
        self.sub_queue_size = sub_queue_size
        self._subs: list[queue.Queue] = []
        self._sub_lock = threading.Lock()
        self._pub_q: queue.Queue = queue.Queue(maxsize=publish_queue_size)
        self._pub_dropped = 0
        threading.Thread(target=self._publisher_loop,
                         name="hs-event-publisher", daemon=True).start()

    # ---- subscribers ----------------------------------------------------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self.sub_queue_size)
        with self._sub_lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    # ---- publish (non-blocking) ----------------------------------------

    def publish(self, ev: Event, frame: Optional[np.ndarray] = None) -> Event:
        """Enqueue for the background publisher. Copies frame for safety."""
        try:
            self._pub_q.put_nowait(
                (ev, frame.copy() if frame is not None else None)
            )
        except queue.Full:
            self._pub_dropped += 1
            if self._pub_dropped <= 5 or self._pub_dropped % 50 == 0:
                print(f"[events] publish queue full, dropping "
                      f"{ev.event_type} (dropped: {self._pub_dropped})")
        return ev

    def _publisher_loop(self) -> None:
        while True:
            try:
                ev, frame = self._pub_q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._do_publish(ev, frame)
            except Exception as e:
                print(f"[events] publisher error: {e}")

    def _do_publish(self, ev: Event, frame: Optional[np.ndarray]) -> None:
        if (frame is not None and self.snapshot_dir is not None
                and ev.snapshot_path is None):
            ev.snapshot_path = save_snapshot(frame, self.snapshot_dir, ev)

        with write_conn(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO events
                   (ts, created_at, event_type, camera_id, camera_name,
                    person_category, confidence, details, snapshot_path,
                    bbox_json, meta_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ev.ts, _fmt_ts(ev.ts), ev.event_type,
                    ev.camera_id, ev.camera_name, ev.person_category,
                    float(ev.confidence), ev.details, ev.snapshot_path,
                    json.dumps(list(ev.bbox)) if ev.bbox else None,
                    json.dumps(ev.meta) if ev.meta else None,
                ),
            )
            ev.event_id = cur.lastrowid

        # SSE fan-out (non-blocking; drop oldest if a subscriber lags)
        with self._sub_lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(ev)
                except Exception:
                    pass

    # ---- queries --------------------------------------------------------

    def list(self, *, limit: int = 50, event_type: Optional[str] = None,
             since_ts: Optional[float] = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events"
        clauses, params = [], []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(float(since_ts))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        with read_conn(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def count_today(self) -> int:
        midnight = time.mktime(datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0).timetuple())
        with read_conn(self.db_path) as conn:
            r = conn.execute(
                "SELECT COUNT(*) AS c FROM events WHERE ts >= ?", (midnight,)
            ).fetchone()
        return int(r["c"])

    def clear(self) -> int:
        with write_conn(self.db_path) as conn:
            cur = conn.execute("DELETE FROM events")
            return cur.rowcount or 0
