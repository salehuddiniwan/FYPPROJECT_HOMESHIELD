"""
Fall Detection with YOLO26 Pose Estimation  (research-tuned rewrite)
====================================================================

Design choices, with the papers they came from:

- Body-scale normalization. Distances/velocities are expressed in
  *body-units* (1 unit = ||shoulder_mid - hip_mid||), so metrics are
  independent of camera distance. (Nunez-Marcos et al., Pattern
  Recognition Letters 2022; PMC9185346 on 2D skeleton normalization.)

- Trunk-angle from shoulder->hip vector vs. vertical. Falls are
  characterised by trunk angle > ~45-60 deg. We fall back to the
  head->hip vector when shoulders are occluded. (MDPI Symmetry 2020,
  "Fall Detection Based on Key Points of Human-Skeleton Using
  OpenPose".)

- Two-stage decision. (A) impact = peak centroid descent velocity
  exceeded threshold. (B) lying = sustained horizontal + low. Both
  must fire within a short window, which rejects controlled
  sit-downs. (PIFR PLOS ONE 2024; arXiv 2401.01587.)

- Height-ratio against the last-3s maximum, not a decaying running
  max. current_h / recent_max_h drops below ~0.55 during a fall.
  (Sensors 2025 Enhanced HRNet+YOLO.)

- Confidence-weighted keypoints + EMA smoothing.

Run:
  python fall_detection.py
  python fall_detection.py --source clip.mp4
  python fall_detection.py --source clip.mp4 --save out.mp4 --no-show
"""
from __future__ import annotations
import argparse, os, time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# COCO-17 keypoint indices (Ultralytics pose models)
KP = {"nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3,
      "right_ear": 4, "left_shoulder": 5, "right_shoulder": 6,
      "left_elbow": 7, "right_elbow": 8, "left_wrist": 9,
      "right_wrist": 10, "left_hip": 11, "right_hip": 12,
      "left_knee": 13, "right_knee": 14, "left_ankle": 15,
      "right_ankle": 16}

SKELETON_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
                  (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
                  (14, 16), (0, 5), (0, 6)]


# ---- Config ---------------------------------------------------------------
@dataclass
class Config:
    model_path: str = "weights/yolo26x-pose.pt"
    fallback_model: str = "yolo11n-pose.pt"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.35
    kp_conf_min: float = 0.30

    kp_ema_alpha: float = 0.6
    fps_assumed: float = 30.0
    short_window_s: float = 0.5         # peak-velocity window
    impact_recent_s: float = 2.0        # how far back the peak still counts
    sustain_s: float = 0.5              # Stage-B must hold this long
    lying_motionless_s: float = 5.0
    inactivity_s: float = 5.0
    height_ref_window_s: float = 3.0

    # Spatial thresholds
    upright_angle_max: float = 30.0
    horizontal_angle_min: float = 60.0
    walking_motion_min: float = 0.05    # body-units / s
    sitting_aspect_min: float = 1.0
    standing_aspect_max: float = 0.7
    horizontal_aspect: float = 1.3
    height_collapse_ratio: float = 0.55

    # Temporal thresholds (BODY-UNITS / SECOND)
    impact_velocity_bu_s: float = 1.6
    motion_threshold_bu_s: float = 0.10

    # Multi-person tracking
    tracker: str = "bytetrack.yaml"        # "bytetrack.yaml" | "botsort.yaml"
    id_cleanup_after_s: float = 2.0        # remove IDs unseen this long
    max_persons_drawn: int = 16

    show: bool = True
    save_path: Optional[str] = None
    draw_skeleton: bool = True
    draw_hud: bool = True


class State(Enum):
    STANDING = "q0 Standing"
    WALKING = "q1 Walking"
    SITTING = "q2 Sitting"
    FALL_DETECTED = "q3 Fall_Detected"
    LYING_AFTER_FALL = "q4 Lying_After_Fall"
    LYING_MOTIONLESS = "q5 Lying_Motionless"
    INACTIVITY = "q6 Inactivity"


STATE_COLOR = {
    State.STANDING: (110, 220, 110),
    State.WALKING: (110, 220, 200),
    State.SITTING: (110, 200, 240),
    State.FALL_DETECTED: (60, 60, 255),
    State.LYING_AFTER_FALL: (50, 130, 255),
    State.LYING_MOTIONLESS: (40, 40, 220),
    State.INACTIVITY: (200, 120, 220),
}


@dataclass
class FrameFeatures:
    has_person: bool = False
    body_scale: float = 1.0
    trunk_angle: float = 0.0
    aspect_ratio: float = 0.0
    height_ratio: float = 1.0
    centroid_y_norm: float = 0.0
    centroid_velocity_bu_s: float = 0.0
    motion_energy_bu_s: float = 0.0
    is_horizontal: bool = False
    is_upright: bool = False
    is_low: bool = False
    is_still: bool = False


def _midpoint(kpts, a, b, conf_min):
    ka, kb = kpts[a], kpts[b]
    if ka[2] < conf_min or kb[2] < conf_min:
        return None
    return (ka[:2] + kb[:2]) * 0.5


def _kp_bbox(kpts, conf_min):
    valid = kpts[kpts[:, 2] >= conf_min]
    if len(valid) < 4:
        return None
    x1, y1 = float(valid[:, 0].min()), float(valid[:, 1].min())
    x2, y2 = float(valid[:, 0].max()), float(valid[:, 1].max())
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (x1, y1, x2, y2)


# ---- Feature extractor ----------------------------------------------------
class FeatureExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.smoothed_kpts = None
        self.prev_centroid = None
        self.prev_kpts_motion = None
        self.prev_t = None
        self.height_history = deque()  # (timestamp, h_px)

    def reset(self):
        self.smoothed_kpts = None
        self.prev_centroid = None
        self.prev_kpts_motion = None
        self.prev_t = None
        self.height_history.clear()

    def _smooth(self, kpts):
        a = self.cfg.kp_ema_alpha
        if self.smoothed_kpts is None or self.smoothed_kpts.shape != kpts.shape:
            self.smoothed_kpts = kpts.copy()
            return self.smoothed_kpts
        out = self.smoothed_kpts.copy()
        cur = kpts[:, 2] >= self.cfg.kp_conf_min
        prv = self.smoothed_kpts[:, 2] >= self.cfg.kp_conf_min
        both = cur & prv
        out[both, :2] = a * kpts[both, :2] + (1 - a) * self.smoothed_kpts[both, :2]
        out[both, 2] = kpts[both, 2]
        new_only = cur & ~prv
        out[new_only] = kpts[new_only]
        out[~cur, 2] = 0.0
        self.smoothed_kpts = out
        return out

    def _body_scale(self, kpts):
        sh = _midpoint(kpts, KP["left_shoulder"], KP["right_shoulder"],
                       self.cfg.kp_conf_min)
        hp = _midpoint(kpts, KP["left_hip"], KP["right_hip"],
                       self.cfg.kp_conf_min)
        if sh is not None and hp is not None:
            d = float(np.linalg.norm(sh - hp))
            if d > 1.0:
                return d
        nose = kpts[KP["nose"]]
        if hp is not None and nose[2] >= self.cfg.kp_conf_min:
            d = float(np.linalg.norm(hp - nose[:2])) * 0.6
            if d > 1.0:
                return d
        return None

    def _trunk_angle(self, kpts):
        sh = _midpoint(kpts, KP["left_shoulder"], KP["right_shoulder"],
                       self.cfg.kp_conf_min)
        hp = _midpoint(kpts, KP["left_hip"], KP["right_hip"],
                       self.cfg.kp_conf_min)
        if sh is None or hp is None:
            nose = kpts[KP["nose"]]
            if hp is None or nose[2] < self.cfg.kp_conf_min:
                return None
            sh = nose[:2]
        dx = sh[0] - hp[0]
        dy = hp[1] - sh[1]
        return float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-3))))

    def _centroid(self, kpts):
        ok = kpts[:, 2] >= self.cfg.kp_conf_min
        if ok.sum() < 4:
            return None
        return kpts[ok, :2].mean(axis=0)

    def _height_ref(self, now, h_px):
        self.height_history.append((now, h_px))
        cutoff = now - self.cfg.height_ref_window_s
        while self.height_history and self.height_history[0][0] < cutoff:
            self.height_history.popleft()
        return max(v for _, v in self.height_history)

    def extract(self, kpts_raw, frame_h, frame_w, now):
        f = FrameFeatures()
        if kpts_raw is None:
            self.reset()
            return f
        kpts = self._smooth(kpts_raw)
        scale = self._body_scale(kpts)
        if scale is None:
            return f
        f.body_scale = scale
        bbox = _kp_bbox(kpts, self.cfg.kp_conf_min)
        if bbox is None:
            return f
        x1, y1, x2, y2 = bbox
        w, h = (x2 - x1), (y2 - y1)
        f.has_person = True
        f.aspect_ratio = w / max(h, 1e-3)
        ref_h = self._height_ref(now, h)
        f.height_ratio = h / max(ref_h, 1e-3)
        ang = self._trunk_angle(kpts)
        f.trunk_angle = ang if ang is not None else (80.0 if f.aspect_ratio > 1.0 else 15.0)
        c = self._centroid(kpts)
        if c is not None:
            f.centroid_y_norm = float(c[1]) / float(frame_h)
            if self.prev_centroid is not None and self.prev_t is not None:
                dt = max(1e-3, now - self.prev_t)
                dy_px = float(c[1] - self.prev_centroid[1])
                f.centroid_velocity_bu_s = (dy_px / scale) / dt
            self.prev_centroid = c
        if (self.prev_kpts_motion is not None and self.prev_t is not None
                and self.prev_kpts_motion.shape == kpts.shape):
            both = ((kpts[:, 2] >= self.cfg.kp_conf_min) &
                    (self.prev_kpts_motion[:, 2] >= self.cfg.kp_conf_min))
            if both.any():
                d = kpts[both, :2] - self.prev_kpts_motion[both, :2]
                dt = max(1e-3, now - (self.prev_t - 1e-3))
                f.motion_energy_bu_s = float(
                    np.linalg.norm(d, axis=1).mean() / scale / dt)
        self.prev_kpts_motion = kpts.copy()
        self.prev_t = now
        f.is_horizontal = (f.trunk_angle > self.cfg.horizontal_angle_min
                           or f.aspect_ratio > self.cfg.horizontal_aspect)
        f.is_upright = f.trunk_angle < self.cfg.upright_angle_max
        f.is_low = f.height_ratio < self.cfg.height_collapse_ratio
        f.is_still = f.motion_energy_bu_s < self.cfg.motion_threshold_bu_s
        return f


# ---- Two-stage detector + FSM --------------------------------------------
class FallDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        max_buf_s = max(cfg.impact_recent_s, cfg.height_ref_window_s,
                        cfg.short_window_s, cfg.sustain_s) + 1.0
        self.history = deque(maxlen=int(cfg.fps_assumed * max_buf_s) + 5)
        self.state = State.STANDING
        self.entered_at = time.time()
        self.last_motion_at = time.time()
        self.last_impact_at = None
        self.fall_alert = False

    def _peak_velocity_in(self, now, window_s):
        cutoff = now - window_s
        peak = 0.0
        for t, f in self.history:
            if t < cutoff or not f.has_person:
                continue
            if f.centroid_velocity_bu_s > peak:
                peak = f.centroid_velocity_bu_s
        return peak

    def _stage_b_sustained(self, now):
        cutoff = now - self.cfg.sustain_s
        rel = [(t, f) for t, f in self.history if t >= cutoff and f.has_person]
        if len(rel) < 3:
            return False
        return all((f.is_horizontal and f.is_low) for _, f in rel)

    def _ratio_in(self, now, window_s, attr):
        cutoff = now - window_s
        rel = [f for t, f in self.history if t >= cutoff and f.has_person]
        if not rel:
            return 0.0
        return sum(1 for f in rel if getattr(f, attr)) / len(rel)

    def _enter(self, new):
        if new != self.state:
            print(f"[FSM] {self.state.value}  ->  {new.value}")
            self.state = new
            self.entered_at = time.time()
            if new == State.FALL_DETECTED:
                self.fall_alert = True
            elif new == State.STANDING:
                self.fall_alert = False

    def step(self, f, now):
        self.history.append((now, f))
        if not f.has_person:
            return
        if not f.is_still:
            self.last_motion_at = now
        cfg = self.cfg
        peak_v = self._peak_velocity_in(now, cfg.short_window_s)
        if peak_v > cfg.impact_velocity_bu_s:
            self.last_impact_at = now
        impact_recent = (self.last_impact_at is not None
                         and (now - self.last_impact_at) <= cfg.impact_recent_s)
        stage_b = self._stage_b_sustained(now)
        s = self.state

        if s in (State.STANDING, State.WALKING, State.SITTING):
            if impact_recent and stage_b:
                self._enter(State.FALL_DETECTED)
                return
            if s == State.STANDING:
                if (f.trunk_angle < cfg.upright_angle_max
                        and f.aspect_ratio < cfg.standing_aspect_max
                        and f.motion_energy_bu_s > cfg.walking_motion_min):
                    self._enter(State.WALKING)
                elif (f.trunk_angle < cfg.horizontal_angle_min
                      and f.aspect_ratio > cfg.sitting_aspect_min
                      and not stage_b):
                    self._enter(State.SITTING)
            elif s == State.WALKING:
                if (f.aspect_ratio > cfg.sitting_aspect_min
                        and f.motion_energy_bu_s < cfg.walking_motion_min
                        and not stage_b):
                    self._enter(State.SITTING)
                elif (f.aspect_ratio < cfg.standing_aspect_max
                      and f.motion_energy_bu_s < cfg.walking_motion_min):
                    self._enter(State.STANDING)
            elif s == State.SITTING:
                if (f.trunk_angle < cfg.upright_angle_max
                        and f.aspect_ratio < cfg.standing_aspect_max):
                    self._enter(State.STANDING)
                elif ((now - self.last_motion_at) > cfg.inactivity_s
                      and self._ratio_in(now, cfg.inactivity_s, "is_still") > 0.7):
                    self._enter(State.INACTIVITY)
            return

        if s == State.FALL_DETECTED:
            time_in = now - self.entered_at
            if stage_b and time_in <= 5.0:
                self._enter(State.LYING_AFTER_FALL)
            elif stage_b and time_in > 5.0:
                self._enter(State.LYING_MOTIONLESS)
            elif (self._ratio_in(now, 1.0, "is_upright") > 0.6
                  and not f.is_horizontal):
                self._enter(State.STANDING)
            return

        if s == State.LYING_AFTER_FALL:
            time_in = now - self.entered_at
            if (time_in > cfg.lying_motionless_s
                    and self._ratio_in(now, 2.0, "is_still") > 0.7):
                self._enter(State.LYING_MOTIONLESS)
            elif (self._ratio_in(now, 1.0, "is_upright") > 0.6
                  and not f.is_horizontal):
                self._enter(State.STANDING)
            return

        if s == State.LYING_MOTIONLESS:
            if (self._ratio_in(now, 1.0, "is_upright") > 0.6
                    and not f.is_horizontal):
                self._enter(State.STANDING)
            return

        if s == State.INACTIVITY:
            if impact_recent and stage_b:
                self._enter(State.FALL_DETECTED)
            elif (self._ratio_in(now, 1.0, "is_upright") > 0.6
                  and not f.is_horizontal):
                self._enter(State.STANDING)


# ---- Drawing -------------------------------------------------------------
def draw_skeleton(frame, kpts, conf_min):
    for x, y, c in kpts:
        if c >= conf_min:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
    for a, b in SKELETON_EDGES:
        if kpts[a, 2] >= conf_min and kpts[b, 2] >= conf_min:
            cv2.line(frame, (int(kpts[a, 0]), int(kpts[a, 1])),
                     (int(kpts[b, 0]), int(kpts[b, 1])), (0, 200, 255), 2)


def draw_person_label(frame, person):
    """Small ID + state badge under each person's bbox."""
    pid = person["id"]
    det = person["detector"]
    bbox = person["bbox"]
    color = STATE_COLOR.get(det.state, (200, 200, 200))
    x1, y1, x2, y2 = map(int, bbox)
    label = f"#{pid} {det.state.value}"
    if det.fall_alert:
        label = f"#{pid} !! FALL !!"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


def draw_hud(frame, state: "MultiPersonState", primary, fps):
    """Top banner = global alert; side panel = primary (largest) person metrics."""
    h, w = frame.shape[:2]
    n_tracked = len(state.detectors)
    fall_ids = state.fall_alert_ids()
    if fall_ids:
        banner_color = STATE_COLOR[State.FALL_DETECTED]
        ids_str = ", ".join(f"#{i}" for i in fall_ids)
        banner_text = f"!! FALL ALERT !!  Person(s) {ids_str}   ({n_tracked} tracked)"
    elif primary is not None:
        banner_color = STATE_COLOR.get(primary["detector"].state, (200, 200, 200))
        banner_text = (f"{primary['detector'].state.value}   "
                       f"(primary #{primary['id']}, {n_tracked} tracked)")
    else:
        banner_color = (90, 90, 90)
        banner_text = f"No person detected   ({n_tracked} tracked)"

    banner_h = 56
    cv2.rectangle(frame, (0, 0), (w, banner_h), banner_color, -1)
    cv2.putText(frame, banner_text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (20, 20, 20), 2, cv2.LINE_AA)

    if primary is None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, banner_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        return

    f = primary["feats"]
    det = primary["detector"]
    now = time.time()
    impact_age = "-"
    if det.last_impact_at is not None:
        impact_age = f"{now - det.last_impact_at:4.1f}s ago"
    lines = [
        f"FPS              : {fps:5.1f}",
        f"trunk angle      : {f.trunk_angle:5.1f} deg",
        f"aspect (w/h)     : {f.aspect_ratio:5.2f}",
        f"height ratio     : {f.height_ratio:5.2f}",
        f"body scale (px)  : {f.body_scale:5.0f}",
        f"centroid v (bu/s): {f.centroid_velocity_bu_s:+6.2f}",
        f"motion (bu/s)    : {f.motion_energy_bu_s:5.2f}",
        "",
        f"last impact      : {impact_age}",
        f"is_horizontal    : {f.is_horizontal}",
        f"is_upright       : {f.is_upright}",
        f"is_low           : {f.is_low}",
        f"is_still         : {f.is_still}",
    ]
    panel_w = 320
    x0, y0 = w - panel_w - 8, banner_h + 8
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + 22 * len(lines) + 12),
                  (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + 10, y0 + 22 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


# ---- Multi-person state manager ------------------------------------------
class MultiPersonState:
    """
    Holds one FeatureExtractor + one FallDetector per tracked person ID.
    Cleans up IDs that haven't been seen recently.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.extractors: dict = {}      # id -> FeatureExtractor
        self.detectors: dict = {}       # id -> FallDetector
        self.last_seen: dict = {}       # id -> timestamp
        self.last_kpts: dict = {}       # id -> kpts (for drawing)
        self.last_bbox: dict = {}       # id -> bbox (for drawing)
        self.next_anon_id = -1          # used when tracker doesn't return ids

    def _get_or_create(self, pid: int):
        if pid not in self.extractors:
            self.extractors[pid] = FeatureExtractor(self.cfg)
            self.detectors[pid] = FallDetector(self.cfg)
        return self.extractors[pid], self.detectors[pid]

    def _cleanup(self, now: float) -> None:
        cutoff = now - self.cfg.id_cleanup_after_s
        stale = [pid for pid, t in self.last_seen.items() if t < cutoff]
        for pid in stale:
            self.extractors.pop(pid, None)
            self.detectors.pop(pid, None)
            self.last_seen.pop(pid, None)
            self.last_kpts.pop(pid, None)
            self.last_bbox.pop(pid, None)

    def step(self, result, frame_h: int, frame_w: int, now: float) -> list:
        """
        Process a YOLO tracking result. Returns list of dicts:
        [{'id': int, 'kpts': ndarray, 'bbox': ndarray,
          'feats': FrameFeatures, 'detector': FallDetector}, ...]
        """
        out = []
        if (result.boxes is None or len(result.boxes) == 0
                or result.keypoints is None):
            self._cleanup(now)
            return out

        boxes = result.boxes.xyxy.cpu().numpy()
        kp_xy = result.keypoints.xy.cpu().numpy()
        kp_c = (result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(kp_xy.shape[:2], dtype=np.float32))

        # Track IDs (None when track() wasn't used or tracker has no IDs yet)
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().numpy().tolist()
        else:
            ids = [self._fresh_anon_id() for _ in range(len(boxes))]

        for i, pid in enumerate(ids):
            pid = int(pid)
            kpts = np.concatenate([kp_xy[i], kp_c[i][:, None]], axis=1)
            bbox = boxes[i]
            extractor, detector = self._get_or_create(pid)
            feats = extractor.extract(kpts, frame_h, frame_w, now)
            detector.step(feats, now)
            self.last_seen[pid] = now
            self.last_kpts[pid] = kpts
            self.last_bbox[pid] = bbox
            out.append({
                "id": pid, "kpts": kpts, "bbox": bbox,
                "feats": feats, "detector": detector,
            })

        self._cleanup(now)
        return out

    def _fresh_anon_id(self) -> int:
        self.next_anon_id -= 1
        return self.next_anon_id

    # --- queries used by drawing -----------------------------------------
    def any_fall_alert(self) -> bool:
        return any(d.fall_alert for d in self.detectors.values())

    def fall_alert_ids(self) -> list:
        return [pid for pid, d in self.detectors.items() if d.fall_alert]

    def primary_person(self, persons: list) -> Optional[dict]:
        """Largest-bbox person (used for the side metrics panel)."""
        if not persons:
            return None
        def area(p):
            x1, y1, x2, y2 = p["bbox"]
            return float((x2 - x1) * (y2 - y1))
        return max(persons, key=area)


# ---- Model loading -------------------------------------------------------
def load_model(cfg):
    path = cfg.model_path if os.path.exists(cfg.model_path) else cfg.fallback_model
    if path != cfg.model_path:
        print(f"[model] '{cfg.model_path}' not found, using '{path}'.")
    model = YOLO(path)
    if cfg.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device
    try:
        model.to(0 if device == "0" else device)
    except Exception as e:
        print(f"[model] .to({device}) failed ({e}); using default device.")
    return model, device




# ---- Main loop -----------------------------------------------------------
def run(cfg, source):
    model, device = load_model(cfg)
    print(f"[device] running on: {device} (cuda: {torch.cuda.is_available()})")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps_assumed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cfg.fps_assumed = float(src_fps if src_fps > 0 else cfg.fps_assumed)
    writer = None
    if cfg.save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg.save_path, fourcc, src_fps, (width, height))
        print(f"[save] writing -> {cfg.save_path}")

    state = MultiPersonState(cfg)
    smoothed_fps = float(src_fps)
    last_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # model.track() runs detection + persistent ID assignment via ByteTrack
        results = model.track(frame, imgsz=cfg.imgsz, conf=cfg.conf,
                              persist=True, tracker=cfg.tracker,
                              verbose=False,
                              device=0 if device == "0" else device)
        result = results[0]
        now = time.time()
        persons = state.step(result, frame.shape[0], frame.shape[1], now)

        # Per-person drawing (capped so a crowded scene doesn't spam the screen)
        for person in persons[:cfg.max_persons_drawn]:
            if cfg.draw_skeleton and person["kpts"] is not None:
                draw_skeleton(frame, person["kpts"], cfg.kp_conf_min)
            x1, y1, x2, y2 = map(int, person["bbox"])
            color = STATE_COLOR.get(person["detector"].state, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_person_label(frame, person)

        dt = max(1e-6, now - last_t)
        last_t = now
        smoothed_fps = 0.9 * smoothed_fps + 0.1 * (1.0 / dt)

        if cfg.draw_hud:
            primary = state.primary_person(persons)
            draw_hud(frame, state, primary, smoothed_fps)

        if writer is not None:
            writer.write(frame)
        if cfg.show:
            cv2.imshow("YOLO26 Fall Detection - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO26 multi-person fall detection (research-tuned)")
    p.add_argument("--source", default="0",
                   help="Webcam index (0/1/...) or video file path")
    p.add_argument("--model", default=Config.model_path)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu", "0"])
    p.add_argument("--imgsz", type=int, default=Config.imgsz)
    p.add_argument("--conf", type=float, default=Config.conf)
    p.add_argument("--tracker", default=Config.tracker,
                   choices=["bytetrack.yaml", "botsort.yaml"])
    p.add_argument("--impact-vel", type=float,
                   default=Config.impact_velocity_bu_s,
                   help="Stage-A peak descent threshold (body-units/sec)")
    p.add_argument("--horizontal-deg", type=float,
                   default=Config.horizontal_angle_min,
                   help="Trunk angle (deg) considered horizontal")
    p.add_argument("--collapse-ratio", type=float,
                   default=Config.height_collapse_ratio,
                   help="height_ratio below this = collapsed")
    p.add_argument("--save", default=None, help="Path to write annotated mp4")
    p.add_argument("--no-show", action="store_true",
                   help="Disable display window")
    args = p.parse_args()
    cfg = Config(model_path=args.model, device=args.device, imgsz=args.imgsz,
                 conf=args.conf, tracker=args.tracker,
                 impact_velocity_bu_s=args.impact_vel,
                 horizontal_angle_min=args.horizontal_deg,
                 height_collapse_ratio=args.collapse_ratio,
                 show=not args.no_show, save_path=args.save)
    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    return cfg, src


if __name__ == "__main__":
    cfg, src = parse_args()
    run(cfg, src)
