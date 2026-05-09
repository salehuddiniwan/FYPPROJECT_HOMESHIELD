"""Single drawing routine: zones, persons, fire boxes, face boxes, banner."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Pull the FSM colour palette + skeleton drawer from Fall_Detection.
_FALL_DIR = Path(__file__).resolve().parent.parent / "Fall_Detection"
if str(_FALL_DIR) not in sys.path:
    sys.path.insert(0, str(_FALL_DIR))


def _import_palette():
    try:
        from fall_detection import STATE_COLOR, draw_skeleton  # type: ignore
        return STATE_COLOR, draw_skeleton
    except Exception:
        return {}, None


_STATE_COLOR, _draw_skeleton = _import_palette()

# BGR colours
COLOR_FIRE = (40, 60, 230)
COLOR_SMOKE = (200, 200, 200)
COLOR_FACE_OK = (80, 220, 130)
COLOR_FACE_BAD = (60, 60, 240)
COLOR_BANNER_OK = (40, 130, 50)
COLOR_BANNER_CRIT = (50, 50, 230)
COLOR_DANGER_FILL = (50, 50, 230, 80)
COLOR_SAFE_FILL = (90, 220, 130, 70)
COLOR_TEXT_LIGHT = (235, 235, 235)
COLOR_TEXT_DARK = (20, 20, 20)


def _label(img, text, x, y, fg=COLOR_TEXT_LIGHT, bg=(20, 20, 20),
           size=0.5, pad=4, thick=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thick)
    x, y = int(x), int(y)
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y), bg, -1)
    cv2.putText(img, text, (x + pad, y - pad),
                cv2.FONT_HERSHEY_SIMPLEX, size, fg, thick, cv2.LINE_AA)


def _corner_box(img, x1, y1, x2, y2, color, thickness=2, corner=14):
    """Modern L-shape corner-style bounding box."""
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    for p1, p2 in (
        ((x1, y1), (x1 + corner, y1)), ((x1, y1), (x1, y1 + corner)),
        ((x2, y1), (x2 - corner, y1)), ((x2, y1), (x2, y1 + corner)),
        ((x1, y2), (x1 + corner, y2)), ((x1, y2), (x1, y2 - corner)),
        ((x2, y2), (x2 - corner, y2)), ((x2, y2), (x2, y2 - corner)),
    ):
        cv2.line(img, p1, p2, color, thickness)


def _draw_polygon(img, poly, edge, fill_rgba):
    if not poly or len(poly) < 3:
        return
    pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], fill_rgba[:3])
    alpha = fill_rgba[3] / 255.0
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [pts], True, edge, 2, cv2.LINE_AA)


def _bbox4(b):
    """Return (x1, y1, x2, y2) as floats, or None if not 4-element."""
    arr = np.asarray(b).flatten().tolist()
    return tuple(float(v) for v in arr[:4]) if len(arr) >= 4 else None


def _draw_zones(frame, zones, edge, fill, badge_bg, badge_fg, badge_prefix):
    for z in zones:
        poly = z["polygon_scaled"]
        _draw_polygon(frame, poly, edge, fill)
        if poly:
            x, y = poly[0]
            _label(frame, f"{badge_prefix}: {z['zone_name']}",
                   x, max(int(y) - 4, 14),
                   bg=badge_bg, fg=badge_fg, size=0.45, thick=1)


def _draw_fires(frame, fires):
    for d in fires:
        bb = _bbox4(d["bbox"])
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        is_fire = d["cls_name"].lower() == "fire"
        color = COLOR_FIRE if is_fire else COLOR_SMOKE
        _corner_box(frame, x1, y1, x2, y2, color, 2, 14)
        _label(frame, f"{d['cls_name'].upper()} {d['conf']:.2f}",
               int(x1), max(int(y1) - 2, 14),
               fg=COLOR_TEXT_LIGHT, bg=color, size=0.5, thick=1)


COLOR_GLOBAL_ID = (230, 170, 60)  # amber, matches the handoff event colour


def _draw_persons(frame, persons, kp_conf_min):
    for p in persons:
        bb = _bbox4(p["bbox"])
        if bb is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in bb)
        det = p["detector"]
        color = _STATE_COLOR.get(det.state, (200, 200, 200)) if det else (200, 200, 200)
        if p.get("kpts") is not None and _draw_skeleton is not None:
            try:
                _draw_skeleton(frame, p["kpts"], kp_conf_min)
            except Exception:
                pass
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        pid = p.get("id", "?")
        state_label = det.state.value if det is not None else "?"
        text = (f"#{pid} !! FALL !!" if det is not None and det.fall_alert
                else f"#{pid} {state_label}")
        _label(frame, text, x1, max(y1 - 2, 14),
               fg=COLOR_TEXT_DARK, bg=color, size=0.5, thick=1)

        # Cross-camera global identity badge (drawn underneath the box).
        gid = p.get("global_id")
        if gid is not None:
            gname = p.get("global_name")
            score = p.get("reid_score")
            badge = f"G{gid}"
            if gname:
                badge += f" ({gname})"
            if score is not None and score > 0:
                badge += f"  {score:.2f}"
            _label(frame, badge, x1, min(y2 + 18, frame.shape[0] - 2),
                   fg=COLOR_TEXT_DARK, bg=COLOR_GLOBAL_ID,
                   size=0.45, thick=1)


def _draw_faces(frame, faces):
    for f in faces:
        x, y, w, h = int(f["x"]), int(f["y"]), int(f["w"]), int(f["h"])
        is_known = f.get("match_id") is not None
        color = COLOR_FACE_OK if is_known else COLOR_FACE_BAD
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if is_known:
            name = f.get("match_name") or "Unknown"
            score = f.get("match_score", 0.0)
            text = f"{name} ({score:.2f})" if score > 0 else name
        else:
            text = "INTRUDER"
        _label(frame, text, x, max(y - 2, 14),
               fg=COLOR_TEXT_LIGHT, bg=color, size=0.45, thick=1)


def _draw_banner(frame, result, camera_name):
    H, W = frame.shape[:2]
    crit = []
    if result.fire_alert:
        crit.append("FIRE")
    if result.fall_alert_ids:
        crit.append("FALL #" + ",".join(str(i) for i in result.fall_alert_ids))
    if result.intruder_alert:
        crit.append("INTRUDER")
    if crit:
        color = COLOR_BANNER_CRIT
        text = " ALERT - " + "  &  ".join(crit) + " "
    else:
        color = COLOR_BANNER_OK
        text = f" {camera_name}  -  {len(result.persons)} person(s) "
    cv2.rectangle(frame, (0, 0), (W, 48), color, -1)
    cv2.putText(frame, text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2, cv2.LINE_AA)


def annotate(frame: np.ndarray, result, *,
             camera_name: str = "Camera",
             kp_conf_min: float = 0.30) -> np.ndarray:
    """Draw all overlays in-place (and return the frame for chaining)."""
    _draw_zones(frame, result.danger_zones,
                edge=(40, 50, 220), fill=COLOR_DANGER_FILL,
                badge_bg=(40, 50, 220), badge_fg=COLOR_TEXT_LIGHT,
                badge_prefix="DANGER")
    _draw_zones(frame, result.safe_zones,
                edge=(60, 200, 130), fill=COLOR_SAFE_FILL,
                badge_bg=(60, 180, 110), badge_fg=COLOR_TEXT_DARK,
                badge_prefix="SAFE")
    _draw_fires(frame, result.fires)
    _draw_persons(frame, result.persons, kp_conf_min)
    _draw_faces(frame, result.faces)
    _draw_banner(frame, result, camera_name)
    return frame


def disconnected_placeholder(width: int = 960, height: int = 540,
                             text: str = "Camera offline") -> np.ndarray:
    img = np.full((height, width, 3), 28, dtype=np.uint8)
    msg = f" ! {text} "
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(img, msg, ((width - tw) // 2, (height + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 80, 240), 2, cv2.LINE_AA)
    return img
