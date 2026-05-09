"""Tiny key-value settings store backed by the `settings` table."""

from __future__ import annotations

import json
import threading
from typing import Any

from .db import read_conn, write_conn


DEFAULTS: dict[str, Any] = {
    # Fall detector
    "fall_threshold": 0.80,
    "inactivity_seconds": 300,
    "alert_cooldown": 60,
    "alert_phones": "",

    # YOLO pose
    "yolo_model": "yolo11n-pose.pt",
    "yolo_confidence": 0.50,
    "yolo_imgsz": 640,
    "process_fps": 0,            # 0 = unlimited
    "use_fp16": True,

    # Fire detector
    "fire_enabled": True,
    "fire_model": "best.pt",
    "fire_confidence": 0.35,
    "fire_cooldown": 5,
    "fire_classes": "fire,smoke",
    "fire_every_n": 2,

    # Face recognition
    "face_enabled": True,
    "face_match_threshold": 0.45,
    "intruder_cooldown": 30,
    "face_every_n": 5,
}


class SettingsStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}
        self._load()
        self._ensure_defaults()
        self._migrate()

    def _load(self) -> None:
        with read_conn(self.db_path) as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        cache: dict[str, Any] = {}
        for r in rows:
            try:
                cache[r["key"]] = json.loads(r["value"])
            except Exception:
                cache[r["key"]] = r["value"]
        with self._lock:
            self._cache = cache

    def _ensure_defaults(self) -> None:
        missing = {k: v for k, v in DEFAULTS.items() if k not in self._cache}
        if missing:
            self._write_many(missing)
            self._cache.update(missing)

    def _migrate(self) -> None:
        """Patch settings whose default has changed in newer releases."""
        patches: dict[str, Any] = {}
        # process_fps used to default to 15 (artificial throttle); now 0.
        if self._cache.get("process_fps") == 15:
            patches["process_fps"] = 0
        if patches:
            print(f"[settings] migrating: {patches}")
            self.update(patches)

    def _write_many(self, items: dict[str, Any]) -> None:
        with write_conn(self.db_path) as conn:
            for k, v in items.items():
                conn.execute(
                    """INSERT INTO settings (key, value, updated_at)
                       VALUES (?, ?, datetime('now'))
                       ON CONFLICT(key) DO UPDATE SET
                           value      = excluded.value,
                           updated_at = excluded.updated_at""",
                    (k, json.dumps(v)),
                )

    # ---- public API -----------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        return DEFAULTS.get(key, default)

    def all(self) -> dict[str, Any]:
        with self._lock:
            return {**DEFAULTS, **self._cache}

    def update(self, partial: dict[str, Any]) -> dict[str, Any]:
        if not partial:
            return self.all()
        self._write_many(partial)
        with self._lock:
            self._cache.update(partial)
        return self.all()
