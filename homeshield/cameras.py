"""Multi-camera lifecycle: CameraStore, LatestFrame, CaptureWorker, CameraManager."""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .annotator import annotate, disconnected_placeholder
from .db import read_conn, write_conn
from .events import Event, EventBus
from .pipeline import CameraPipeline, Models


# ---- CameraStore ----------------------------------------------------------

class CameraStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def list(self) -> list[dict[str, Any]]:
        with read_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM cameras ORDER BY camera_id ASC"
            ).fetchall()
        return [{
            "camera_id": r["camera_id"],
            "name": r["name"],
            "url": r["url"],
            "location": r["location"] or "",
            "enabled": bool(r["enabled"]),
            "created_at": r["created_at"],
        } for r in rows]

    def add(self, *, name: str, url: str, location: str = "") -> int:
        with write_conn(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO cameras (name, url, location) VALUES (?, ?, ?)",
                (name.strip() or "Camera",
                 url.strip() or "0",
                 location.strip()),
            )
            return cur.lastrowid

    def delete(self, camera_id: int) -> None:
        with write_conn(self.db_path) as conn:
            conn.execute(
                "DELETE FROM cameras WHERE camera_id = ?", (int(camera_id),)
            )


# ---- LatestFrame ----------------------------------------------------------

class LatestFrame:
    """Thread-safe single-slot buffer (raw + JPEG-encoded copy)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._raw: Optional[np.ndarray] = None
        self._version = 0

    def set(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._raw = frame.copy()
            self._version += 1
            self._cond.notify_all()

    def get_blocking(self, last_version: int, timeout: float = 1.0):
        with self._cond:
            if self._version == last_version:
                self._cond.wait(timeout=timeout)
            return self._jpeg, self._version

    def jpeg(self) -> Optional[bytes]:
        with self._cond:
            return self._jpeg

    def raw(self) -> Optional[np.ndarray]:
        with self._cond:
            return None if self._raw is None else self._raw.copy()


# ---- CaptureWorker --------------------------------------------------------

@dataclass
class WorkerStatus:
    started_at: float = 0.0
    frames_total: int = 0
    last_error: Optional[str] = None
    camera_connected: bool = False
    fps: float = 0.0


def _parse_source(s):
    if isinstance(s, str):
        s = s.strip()
        if s.isdigit():
            return int(s)
    return s


class CaptureWorker(threading.Thread):
    """
    NOTE: do NOT name the stop flag ``self._stop`` -- it shadows
    ``threading.Thread._stop`` (called by Thread.join) and crashes.
    """

    MAX_READ_FAILS = 30   # ~1 s @ 30 FPS, survive transient hiccups
    MAX_PIPE_ERRORS = 60  # consecutive pipeline errors -> reconnect

    def __init__(self, camera: dict, pipeline: CameraPipeline,
                 bus: EventBus, latest: LatestFrame):
        super().__init__(daemon=True, name=f"hs-cam-{camera['camera_id']}")
        self.camera = camera
        self.pipeline = pipeline
        self.bus = bus
        self.latest = latest
        self.status = WorkerStatus()
        self._stop_flag = threading.Event()

    def request_stop(self):
        self._stop_flag.set()

    # ---- camera open ----------------------------------------------------

    def _open(self):
        src = _parse_source(self.camera["url"])
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            try:
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # Windows fallback
            except Exception:
                pass
            if not cap.isOpened():
                return None
        # Keep buffer tiny so inference lag doesn't pile up stale frames.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    # ---- main loop ------------------------------------------------------

    def run(self):
        self.status.started_at = time.time()
        backoff = 1.0
        cid = self.camera["camera_id"]
        cname = self.camera["name"]

        while not self._stop_flag.is_set():
            cap = self._open()
            if cap is None:
                if self.status.camera_connected:
                    self._sys_event(f"Camera disconnected ({self.camera['url']})", 0.0)
                self.status.camera_connected = False
                self.status.last_error = f"Could not open {self.camera['url']}"
                self.latest.set(disconnected_placeholder(text=f"{cname}: offline"))
                if self._stop_flag.wait(timeout=backoff):
                    break
                backoff = min(10.0, backoff * 1.7)
                continue

            if not self.status.camera_connected:
                self._sys_event(f"Camera connected ({self.camera['url']})", 1.0)
            self.status.camera_connected = True
            self.status.last_error = None
            backoff = 1.0

            try:
                self._loop(cap)
            except Exception as e:
                self.status.last_error = repr(e)
                self._sys_event(f"Pipeline error: {e}", 0.0)
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
                self.status.camera_connected = False

    def _sys_event(self, msg: str, conf: float) -> None:
        self.bus.publish(Event(
            event_type="system",
            camera_id=self.camera["camera_id"],
            camera_name=self.camera["name"],
            details=msg,
            confidence=conf,
        ))

    def _loop(self, cap: cv2.VideoCapture):
        """process_fps==0 -> unlimited; >0 -> soft cap."""
        read_fails = 0
        pipe_errors = 0
        cid = self.camera["camera_id"]

        while not self._stop_flag.is_set():
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok or frame is None:
                read_fails += 1
                self.status.last_error = (
                    f"Frame read failed ({read_fails}/{self.MAX_READ_FAILS})"
                )
                if read_fails >= self.MAX_READ_FAILS:
                    break
                if self._stop_flag.wait(timeout=0.05):
                    break
                continue
            read_fails = 0

            try:
                res = self.pipeline.process(frame, ts=loop_start)
                pipe_errors = 0
            except Exception as e:
                pipe_errors += 1
                self.status.last_error = repr(e)
                if pipe_errors <= 3 or pipe_errors % 30 == 0:
                    print(f"[pipeline cam={cid}] frame error #{pipe_errors}: {e}")
                    traceback.print_exc()
                if pipe_errors >= self.MAX_PIPE_ERRORS:
                    raise
                self.latest.set(frame)
                continue

            for ev in res.events:
                try:
                    self.bus.publish(ev, frame=frame)
                except Exception as e:
                    print(f"[pipeline cam={cid}] event publish error: {e}")

            try:
                annotated = annotate(frame.copy(), res,
                                     camera_name=self.camera["name"])
            except Exception as e:
                print(f"[pipeline cam={cid}] annotate error: {e}")
                traceback.print_exc()
                annotated = frame
            self.latest.set(annotated)
            self.status.frames_total += 1
            self.status.fps = res.fps

            max_fps = int(self.pipeline.settings.get("process_fps", 0) or 0)
            if max_fps > 0:
                slack = (1.0 / max_fps) - (time.time() - loop_start)
                if slack > 0 and self._stop_flag.wait(timeout=slack):
                    break


# ---- CameraManager --------------------------------------------------------

class CameraManager:
    def __init__(self, *, db_path: str, models: Models, settings,
                 bus: EventBus, person_store, intruder_store, zone_store,
                 reid_store=None):
        self.db_path = db_path
        self.models = models
        self.settings = settings
        self.bus = bus
        self.person_store = person_store
        self.intruder_store = intruder_store
        self.zone_store = zone_store
        self.reid_store = reid_store
        self.store = CameraStore(db_path)

        self._workers: dict[int, CaptureWorker] = {}
        self._frames: dict[int, LatestFrame] = {}
        self._pipelines: dict[int, CameraPipeline] = {}
        self._lock = threading.Lock()
        self._running = False

    # ---- public API -----------------------------------------------------

    def is_running(self) -> bool:
        return self._running

    def latest(self, camera_id: int) -> Optional[LatestFrame]:
        with self._lock:
            return self._frames.get(int(camera_id))

    def status(self) -> dict[str, Any]:
        cams = self.store.list()
        out = {
            "running": self._running,
            "cameras_total": len(cams),
            "cameras_online": 0,
            "people_count": 0,
            "alerts_today": self.bus.count_today(),
            "cameras": {},
        }
        for c in cams:
            cid = c["camera_id"]
            w = self._workers.get(cid)
            pipe = self._pipelines.get(cid)
            active = bool(w and w.status.camera_connected)
            cam_people = (len(pipe.fall_state.detectors) if pipe else 0)
            if active:
                out["cameras_online"] += 1
            out["people_count"] += cam_people
            out["cameras"][str(cid)] = {
                "name": c["name"],
                "location": c["location"],
                "url": c["url"],
                "active": active,
                "fps": round(w.status.fps, 1) if w else 0.0,
                "people": cam_people,
            }
        return out

    # ---- start / stop ---------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self.models.ensure_loaded()
            for c in self.store.list():
                if c["enabled"]:
                    self._spawn_locked(c)
            self._running = True

    def stop(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            pipes = list(self._pipelines.values())
            self._workers.clear()
            self._frames.clear()
            self._pipelines.clear()
            self._running = False
        # Stop face workers first (cheap), then capture workers.
        for p in pipes:
            try:
                p.shutdown()
            except Exception:
                pass
        # Join outside the lock so a worker stuck on cap.read() doesn't deadlock.
        for w in workers:
            w.request_stop()
        for w in workers:
            try:
                w.join(timeout=3.0)
            except Exception as e:
                print(f"[manager] worker join failed: {e}")

    # ---- live (re)config ------------------------------------------------

    def add_camera(self, *, name: str, url: str, location: str = "") -> int:
        cid = self.store.add(name=name, url=url, location=location)
        if self._running:
            cam = next((c for c in self.store.list() if c["camera_id"] == cid), None)
            if cam is not None:
                with self._lock:
                    self._spawn_locked(cam)
        return cid

    def delete_camera(self, camera_id: int) -> None:
        with self._lock:
            w = self._workers.pop(int(camera_id), None)
            self._frames.pop(int(camera_id), None)
            p = self._pipelines.pop(int(camera_id), None)
        if p is not None:
            try:
                p.shutdown()
            except Exception:
                pass
        if w is not None:
            w.request_stop()
            try:
                w.join(timeout=3.0)
            except Exception as e:
                print(f"[manager] camera {camera_id} join failed: {e}")
        self.store.delete(camera_id)

    def reload_settings(self) -> None:
        # Apply live-tunable settings on each per-camera pipeline.
        with self._lock:
            for pipe in self._pipelines.values():
                pipe.reload_settings()
        # Pick up fire_enabled / face_enabled toggles without restarting cameras.
        # ensure_loaded() loads if enabled-and-missing and unloads if disabled.
        if self._running:
            try:
                self.models.ensure_loaded()
            except Exception as e:
                print(f"[manager] ensure_loaded during reload failed: {e}")

    # ---- internals ------------------------------------------------------

    def _spawn_locked(self, cam):
        cid = cam["camera_id"]
        if cid in self._workers:
            return
        pipe = CameraPipeline(
            camera_id=cid, camera_name=cam["name"],
            models=self.models, settings=self.settings,
            person_store=self.person_store,
            zone_store=self.zone_store,
            intruder_store=self.intruder_store,
            bus=self.bus,
            reid_store=self.reid_store,
        )
        latest = LatestFrame()
        latest.set(disconnected_placeholder(text=f"{cam['name']}: starting..."))
        worker = CaptureWorker(camera=cam, pipeline=pipe,
                               bus=self.bus, latest=latest)
        self._pipelines[cid] = pipe
        self._frames[cid] = latest
        self._workers[cid] = worker
        worker.start()
