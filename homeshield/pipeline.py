"""Per-camera detection pipeline. Models are loaded once (in `Models`) and
shared across cameras; per-camera mutable state lives in `CameraPipeline`.

Stages per frame:
  1. Pose      (YOLO pose + research-tuned FSM, every frame)
  2. Fire      (YOLO detect, every fire_every_n frames)
  3. Face      (InsightFace, every face_every_n frames)
  4. Zone      (point-in-polygon vs configured zones)

Edge-triggered events with cooldowns:
  fall_detected, lying_motionless, inactivity, zone_entry,
  intruder_detected, fire_detected.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Make Fall_Detection importable.
_THIS = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS.parent
if str(_PROJECT_ROOT / "Fall_Detection") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "Fall_Detection"))


# ---- weights resolution ---------------------------------------------------

def resolve_weights(rel_or_abs: str, search_roots: list[Path]) -> str:
    p = Path(rel_or_abs)
    if p.is_absolute() and p.is_file():
        return str(p)
    candidates = (
        [Path.cwd() / rel_or_abs]
        + [r / rel_or_abs for r in search_roots]
        + [r / "weights" / rel_or_abs for r in search_roots]
        + [r / Path(rel_or_abs).name for r in search_roots]
    )
    for c in candidates:
        if c.is_file():
            return str(c)
    return str(p)


def _list_pt(*dirs: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for d in dirs:
        if d.is_dir():
            for p in sorted(d.glob("*.pt")):
                out.append({"value": p.name, "label": p.name, "path": str(p)})
    return out


def list_pose_models() -> list[dict[str, str]]:
    return _list_pt(_PROJECT_ROOT / "Fall_Detection" / "weights",
                    _PROJECT_ROOT / "weights")


def list_fire_models() -> list[dict[str, str]]:
    return _list_pt(_PROJECT_ROOT / "Fire_Detection",
                    _PROJECT_ROOT / "Fire_Detection" / "weights")


# ---- Shared models --------------------------------------------------------

class Models:
    """Lazy-loaded container for pose / fire YOLO + face engine."""

    def __init__(self, settings):
        self.settings = settings
        self._lock = threading.RLock()
        self.pose_model = None
        self.fire_model = None
        self.face_engine = None
        self.device = "cpu"
        self.pose_weights_path: str = ""
        self.fire_weights_path: str = ""

    def ensure_loaded(self) -> None:
        with self._lock:
            if self.pose_model is None:
                self._load_pose()
            self._sync_optional("fire_enabled", "fire_model", self._load_fire,
                                "fire detection")
            self._sync_optional("face_enabled", "face_engine", self._load_face,
                                "face recognition")

    def reload(self) -> None:
        with self._lock:
            self.pose_model = None
            self.fire_model = None
            self.face_engine = None
        self.ensure_loaded()

    # ---- internals ------------------------------------------------------

    def _sync_optional(self, enabled_key: str, attr: str,
                       loader, label: str) -> None:
        """Load if enabled and missing; drop if disabled."""
        if self.settings.get(enabled_key, True):
            if getattr(self, attr) is None:
                loader()
        elif getattr(self, attr) is not None:
            print(f"[models] {label} disabled - unloading")
            setattr(self, attr, None)

    def _resolve_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
        except Exception:
            pass
        return "cpu"

    def _load_pose(self) -> None:
        from ultralytics import YOLO
        weights = self.settings.get("yolo_model", "yolo11n-pose.pt")
        path = resolve_weights(weights, [
            _PROJECT_ROOT / "Fall_Detection", _PROJECT_ROOT, Path.cwd()
        ])
        if not Path(path).is_file():
            avail = list_pose_models()
            if avail:
                path = avail[0]["path"]
        print(f"[models] loading pose: {path}")
        self.pose_model = YOLO(path)
        self.device = self._resolve_device()
        try:
            self.pose_model.to(0 if self.device == "0" else self.device)
        except Exception:
            pass
        self.pose_weights_path = path

    def _load_fire(self) -> None:
        from ultralytics import YOLO
        weights = self.settings.get("fire_model", "best.pt")
        path = resolve_weights(weights, [
            _PROJECT_ROOT / "Fire_Detection", _PROJECT_ROOT, Path.cwd()
        ])
        if not Path(path).is_file():
            avail = list_fire_models()
            if avail:
                path = avail[0]["path"]
        if not Path(path).is_file():
            print(f"[models] fire weights not found at {weights}")
            return
        print(f"[models] loading fire: {path}")
        # Renaming so the attribute name matches "fire_model" exactly.
        self.fire_model = YOLO(path)
        try:
            self.fire_model.to(0 if self.device == "0" else self.device)
        except Exception:
            pass
        self.fire_weights_path = path

    # _load_face is called via self._sync_optional("face_engine", ...) but
    # the attribute name there is "face_engine" not "face_model" — so we
    # alias here for clarity.
    def _load_face(self) -> None:
        from .face import FaceEngine
        try:
            self.face_engine = FaceEngine()
            if self.face_engine.available:
                print("[models] face recognition: ready (InsightFace)")
            else:
                print(f"[models] face disabled: {self.face_engine.last_error}")
        except Exception as e:
            print(f"[models] face engine init failed: {e}")
            self.face_engine = None


# ---- Per-camera state -----------------------------------------------------

@dataclass
class FrameResult:
    ts: float
    persons: list[dict[str, Any]] = field(default_factory=list)
    fires: list[dict[str, Any]] = field(default_factory=list)
    faces: list[dict[str, Any]] = field(default_factory=list)
    events: list = field(default_factory=list)
    fps: float = 0.0
    fall_alert_ids: list[int] = field(default_factory=list)
    fire_alert: bool = False
    intruder_alert: bool = False
    danger_zones: list[dict[str, Any]] = field(default_factory=list)
    safe_zones: list[dict[str, Any]] = field(default_factory=list)


def _bbox4(b) -> Optional[tuple[float, float, float, float]]:
    arr = np.asarray(b).flatten().tolist()
    if len(arr) < 4:
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


# ---- FaceWorker -----------------------------------------------------------

class FaceWorker(threading.Thread):
    """
    Per-camera daemon thread that runs face detection + intruder logic
    OFF the main capture loop. The pipeline submits the latest frame via
    submit() (non-blocking, drops oldest if a frame is still pending) and
    reads back cached results via latest_faces() instantly.

    This keeps the main loop running at pose-only speed; face refreshes
    whenever this worker finishes a cycle.
    """

    def __init__(self, *, camera_id, camera_name, models, settings,
                 person_store, intruder_store, bus):
        super().__init__(daemon=True, name=f"hs-face-{camera_id}")
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.models = models
        self.settings = settings
        self.person_store = person_store
        self.intruder_store = intruder_store
        self.bus = bus
        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_faces: list[dict[str, Any]] = []
        self._intruder_emitted_at: float = 0.0
        self._stop_flag = threading.Event()

    def submit(self, frame_bgr: np.ndarray, ts: float) -> None:
        """Non-blocking. Drops the queued frame if one is still pending."""
        try:
            self._inbox.put_nowait((frame_bgr, ts))
        except queue.Full:
            try:
                self._inbox.get_nowait()
                self._inbox.put_nowait((frame_bgr, ts))
            except Exception:
                pass

    def latest_faces(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._latest_faces)

    def request_stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        from .events import Event
        from .face import best_match

        while not self._stop_flag.is_set():
            try:
                frame, ts = self._inbox.get(timeout=0.5)
            except queue.Empty:
                continue

            if not self.settings.get("face_enabled", True):
                with self._lock:
                    self._latest_faces = []
                continue

            engine = self.models.face_engine
            if engine is None or not engine.available or self.person_store is None:
                with self._lock:
                    self._latest_faces = []
                continue

            try:
                faces = engine.detect(frame)
            except Exception as e:
                print(f"[face-worker cam={self.camera_id}] detect: {e}")
                continue

            gallery = self.person_store.gallery()
            match_threshold = float(self.settings.get("face_match_threshold", 0.45))
            for f in faces:
                emb = f.get("embedding")
                if emb is None:
                    f["match_id"] = None
                    f["match_score"] = 0.0
                    f["match_name"] = None
                    f["match_category"] = None
                    continue
                pid, score = best_match(emb, gallery, threshold=match_threshold)
                f["match_id"] = pid
                f["match_score"] = score
                f["match_name"] = self.person_store.name_of(pid) if pid is not None else None
                f["match_category"] = self.person_store.category_of(pid) if pid is not None else None

            unknown = [f for f in faces
                       if f.get("match_id") is None
                       and f.get("embedding") is not None]
            if unknown:
                cooldown = float(self.settings.get("intruder_cooldown", 30))
                if ts - self._intruder_emitted_at >= cooldown:
                    self._intruder_emitted_at = ts
                    biggest = max(unknown, key=lambda x: x["w"] * x["h"])
                    if self.intruder_store is not None:
                        try:
                            self.intruder_store.add(
                                embedding=biggest["embedding"],
                                camera_id=self.camera_id,
                                camera_name=self.camera_name,
                                frame_bgr=frame,
                                face_bbox=(biggest["x"], biggest["y"],
                                           biggest["w"], biggest["h"]),
                            )
                        except Exception as e:
                            print(f"[face-worker intruder cam={self.camera_id}] {e}")
                    self.bus.publish(Event(
                        event_type="intruder_detected", ts=ts,
                        camera_id=self.camera_id, camera_name=self.camera_name,
                        person_category="unknown",
                        confidence=float(biggest.get("det_score", 0.5)),
                        details="Unknown face on camera",
                        bbox=(biggest["x"], biggest["y"],
                              biggest["x"] + biggest["w"],
                              biggest["y"] + biggest["h"]),
                        meta={"age": biggest.get("age")},
                    ), frame=frame)

            with self._lock:
                self._latest_faces = faces


class CameraPipeline:
    """Per-camera mutable state. Models are external (shared)."""

    def __init__(self, *, camera_id: int, camera_name: str,
                 models: Models, settings, person_store=None,
                 zone_store=None, intruder_store=None, bus=None):
        self.camera_id = int(camera_id)
        self.camera_name = camera_name
        self.models = models
        self.settings = settings
        self.person_store = person_store
        self.zone_store = zone_store
        self.intruder_store = intruder_store

        from fall_detection import Config, MultiPersonState, State
        self.cfg = Config(
            model_path=self.models.pose_weights_path or "yolo11n-pose.pt",
            device="auto",
            imgsz=int(self.settings.get("yolo_imgsz", 640)),
            conf=float(self.settings.get("yolo_confidence", 0.5)),
            tracker="bytetrack.yaml",
            show=False,
            save_path=None,
            inactivity_s=float(self.settings.get("inactivity_seconds", 300)),
        )
        self.fall_state = MultiPersonState(self.cfg)
        self.State = State

        # cooldown bookkeeping
        self._prev_states: dict[int, Any] = {}
        self._fall_emitted: dict[int, float] = {}
        self._motionless_emitted: dict[int, float] = {}
        self._inactivity_emitted: dict[int, float] = {}
        self._zone_emitted: dict[tuple[int, int], float] = {}
        self._fire_emitted: dict[str, float] = {}
        self._intruder_emitted_at: float = 0.0

        # frame-skip caches so the overlay still has data on skipped frames
        self._frame_idx: int = 0
        self._cached_fires: list[dict[str, Any]] = []
        self._cached_faces: list[dict[str, Any]] = []
        self._cached_fire_alert: bool = False

        self._smoothed_fps: float = 0.0
        self._last_t: Optional[float] = None

        # Async face worker (one per camera). Always running; it polls
        # face_enabled itself and sleeps when disabled.
        self.face_worker: Optional[FaceWorker] = None
        if bus is not None:
            try:
                self.face_worker = FaceWorker(
                    camera_id=self.camera_id, camera_name=self.camera_name,
                    models=self.models, settings=self.settings,
                    person_store=self.person_store,
                    intruder_store=self.intruder_store,
                    bus=bus,
                )
                self.face_worker.start()
            except Exception as e:
                print(f"[pipeline cam={self.camera_id}] face-worker start failed: {e}")
                self.face_worker = None

    def shutdown(self) -> None:
        if self.face_worker is not None:
            self.face_worker.request_stop()

    def reload_settings(self) -> None:
        self.cfg.imgsz = int(self.settings.get("yolo_imgsz", 640))
        self.cfg.conf = float(self.settings.get("yolo_confidence", 0.5))
        self.cfg.inactivity_s = float(self.settings.get("inactivity_seconds", 300))

    # ---- main entry point ----------------------------------------------

    def process(self, frame_bgr: np.ndarray,
                ts: Optional[float] = None) -> FrameResult:
        ts = ts if ts is not None else time.time()
        res = FrameResult(ts=ts)
        H, W = frame_bgr.shape[:2]
        device = 0 if self.models.device == "0" else self.models.device

        self._run_pose(frame_bgr, ts, H, W, device, res)
        self._collect_zones(W, H, res)
        self._emit_person_events(ts, W, H, res)
        res.fall_alert_ids = [
            pid for pid, det in self.fall_state.detectors.items() if det.fall_alert
        ]
        self._run_fire(frame_bgr, ts, device, res)
        self._run_face(frame_bgr, ts, res)

        # FPS smoothing (EMA, alpha=0.1)
        if self._last_t is not None:
            dt = max(1e-6, ts - self._last_t)
            inst = 1.0 / dt
            self._smoothed_fps = (inst if self._smoothed_fps == 0.0
                                  else 0.9 * self._smoothed_fps + 0.1 * inst)
        self._last_t = ts
        res.fps = self._smoothed_fps
        self._frame_idx += 1
        return res

    # ---- stages ---------------------------------------------------------

    def _run_pose(self, frame, ts, H, W, device, res: FrameResult) -> None:
        if self.models.pose_model is None:
            return
        try:
            tr = self.models.pose_model.track(
                frame,
                imgsz=self.cfg.imgsz, conf=self.cfg.conf,
                persist=True, tracker=self.cfg.tracker,
                verbose=False, device=device,
            )
            if tr:
                res.persons = self.fall_state.step(tr[0], H, W, ts)
        except Exception as e:
            print(f"[pipeline.pose cam={self.camera_id}] {e}")

    def _collect_zones(self, W: int, H: int, res: FrameResult) -> None:
        if self.zone_store is None:
            return
        from .zones import scale_polygon
        res.danger_zones = self.zone_store.danger_zones_for(self.camera_id, W, H)
        for sz in self.zone_store.for_camera(self.camera_id):
            if sz["zone_type"] == "safe":
                res.safe_zones.append({
                    **sz,
                    "polygon_scaled": scale_polygon(sz["polygon"], 640, 480, W, H),
                })

    def _emit_person_events(self, ts, W, H, res: FrameResult) -> None:
        from .events import Event
        from .zones import point_in_polygon
        cooldown = float(self.settings.get("alert_cooldown", 60))
        State = self.State
        fall_thresh = float(self.settings.get("fall_threshold", 0.8))

        for p in res.persons:
            pid = int(p["id"])
            cur = p["detector"].state
            prev = self._prev_states.get(pid)
            self._prev_states[pid] = cur

            bb = _bbox4(p["bbox"])
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            foot = ((x1 + x2) / 2.0, y2)

            in_safe = (self.zone_store is not None
                       and self.zone_store.is_in_safe_zone(
                           self.camera_id, W, H, foot[0], foot[1]))
            cat = "unknown"   # face-tag association is a future hook

            def _emit(emit_dict, event_type, conf, details, extra_meta=None):
                last = emit_dict.get(pid, 0.0)
                if (ts - last) < cooldown:
                    return
                emit_dict[pid] = ts
                meta = {"person_id": pid, "fsm_state": cur.value}
                if extra_meta:
                    meta.update(extra_meta)
                res.events.append(Event(
                    event_type=event_type, ts=ts,
                    camera_id=self.camera_id, camera_name=self.camera_name,
                    person_category=cat,
                    confidence=conf, details=details,
                    bbox=bb, meta=meta,
                ))

            # rising-edge events; suppressed inside safe zones
            if not in_safe and cur == State.FALL_DETECTED and prev != State.FALL_DETECTED:
                _emit(self._fall_emitted, "fall_detected",
                      fall_thresh, f"Person #{pid}")
            if not in_safe and cur == State.LYING_MOTIONLESS and prev != State.LYING_MOTIONLESS:
                _emit(self._motionless_emitted, "lying_motionless",
                      0.95, f"Person #{pid} unresponsive")
            if not in_safe and cur == State.INACTIVITY and prev != State.INACTIVITY:
                _emit(self._inactivity_emitted, "inactivity",
                      0.7, f"Person #{pid} idle")

            # Child-in-danger-zone (uses a separate cooldown dict keyed by pid+zone)
            if cat == "child":
                for zone in res.danger_zones:
                    if not point_in_polygon(foot[0], foot[1], zone["polygon_scaled"]):
                        continue
                    key = (pid, zone["zone_id"])
                    last = self._zone_emitted.get(key, 0.0)
                    if (ts - last) < cooldown:
                        continue
                    self._zone_emitted[key] = ts
                    res.events.append(Event(
                        event_type="zone_entry", ts=ts,
                        camera_id=self.camera_id, camera_name=self.camera_name,
                        person_category=cat, confidence=0.9,
                        details=f"Child entered {zone['zone_name']}",
                        bbox=bb,
                        meta={"person_id": pid, "zone_id": zone["zone_id"]},
                    ))

    def _run_fire(self, frame, ts, device, res: FrameResult) -> None:
        from .events import Event
        every = max(1, int(self.settings.get("fire_every_n", 2) or 1))
        run_now = (self._frame_idx % every == 0)
        enabled = self.settings.get("fire_enabled", True)
        if not enabled:
            return
        if not run_now or self.models.fire_model is None:
            res.fires = list(self._cached_fires)
            res.fire_alert = self._cached_fire_alert
            return
        try:
            fr = self.models.fire_model.predict(
                frame,
                conf=float(self.settings.get("fire_confidence", 0.35)),
                imgsz=self.cfg.imgsz, device=device, verbose=False,
            )
        except Exception as e:
            print(f"[pipeline.fire cam={self.camera_id}] {e}")
            return
        if not fr:
            self._cached_fires = []
            self._cached_fire_alert = False
            return
        r0 = fr[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            self._cached_fires = []
            self._cached_fire_alert = False
            return
        names = getattr(r0, "names", None) or {}
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        if xyxy.ndim != 2 or xyxy.shape[1] < 4:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)
            clss = np.zeros((0,), dtype=int)
        per_cls_best: dict[str, tuple[float, tuple]] = {}
        for row, c, k in zip(xyxy, confs, clss):
            if len(row) < 4:
                continue
            x1, y1, x2, y2 = (float(row[0]), float(row[1]),
                              float(row[2]), float(row[3]))
            cls_name = str(names.get(int(k), str(k))).lower()
            res.fires.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(c),
                "cls_name": cls_name,
            })
            best = per_cls_best.get(cls_name)
            if best is None or float(c) > best[0]:
                per_cls_best[cls_name] = (float(c), (x1, y1, x2, y2))
        alert_classes = {
            c.strip().lower()
            for c in str(self.settings.get("fire_classes", "fire,smoke")).split(",")
            if c.strip()
        }
        cooldown = float(self.settings.get("fire_cooldown", 5))
        for cls_name, (best_conf, bbox) in per_cls_best.items():
            if cls_name not in alert_classes:
                continue
            if ts - self._fire_emitted.get(cls_name, 0.0) < cooldown:
                continue
            self._fire_emitted[cls_name] = ts
            res.events.append(Event(
                event_type="fire_detected", ts=ts,
                camera_id=self.camera_id, camera_name=self.camera_name,
                person_category="unknown",
                confidence=best_conf,
                details=cls_name.upper(),
                bbox=bbox,
                meta={"class": cls_name},
            ))
            res.fire_alert = True
        self._cached_fires = list(res.fires)
        self._cached_fire_alert = res.fire_alert

    def _run_face(self, frame, ts, res: FrameResult) -> None:
        """Hand the frame to the async FaceWorker; read its latest result."""
        if self.face_worker is None:
            return
        if not self.settings.get("face_enabled", True):
            res.faces = []
            return
        every = max(1, int(self.settings.get("face_every_n", 5) or 1))
        if self._frame_idx % every == 0:
            self.face_worker.submit(frame, ts)
        res.faces = self.face_worker.latest_faces()
        if any(f.get("match_id") is None and f.get("embedding") is not None
               for f in res.faces):
            res.intruder_alert = True
