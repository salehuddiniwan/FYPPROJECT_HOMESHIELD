"""SQLite schema + connection helpers. One DB file backs the whole app."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Iterator, Union

PathArg = Union[str, PathLike]

# Serialise writes across threads; reads stay concurrent.
_WRITE_LOCK = threading.Lock()


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS cameras (
    camera_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    url         TEXT    NOT NULL,
    location    TEXT    NOT NULL DEFAULT '',
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    created_at      TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    camera_id       INTEGER,
    camera_name     TEXT,
    person_category TEXT    NOT NULL DEFAULT 'unknown',
    confidence      REAL    NOT NULL DEFAULT 0,
    details         TEXT,
    snapshot_path   TEXT,
    bbox_json       TEXT,
    meta_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events (ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_cam  ON events (camera_id);

CREATE TABLE IF NOT EXISTS zones (
    zone_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_name    TEXT    NOT NULL,
    camera_id    INTEGER NOT NULL,
    polygon_json TEXT    NOT NULL,
    zone_type    TEXT    NOT NULL DEFAULT 'danger',
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_zones_cam ON zones (camera_id);

CREATE TABLE IF NOT EXISTS persons (
    person_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL,
    category       TEXT    NOT NULL DEFAULT 'adult',
    photo_path     TEXT,
    embedding_blob BLOB,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_persons_name ON persons (name);

CREATE TABLE IF NOT EXISTS intruders (
    intruder_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_path     TEXT,
    embedding_blob BLOB,
    camera_id      INTEGER,
    camera_name    TEXT,
    detected_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    dismissed      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_intruders_at ON intruders (detected_at DESC);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db(db_path: PathArg) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, timeout=10.0) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def connect(db_path: PathArg) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def write_conn(db_path: PathArg) -> Iterator[sqlite3.Connection]:
    """Serialised writer; commits on success, always closes."""
    with _WRITE_LOCK:
        conn = connect(db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


@contextmanager
def read_conn(db_path: PathArg) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()
