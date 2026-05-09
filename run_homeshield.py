"""
HomeShield entry point.

Examples
--------
# Default: starts on http://localhost:5000/. Cameras you add via the UI
# are persisted in homeshield.db and reloaded next launch.
python run_homeshield.py

# Custom port and DB
python run_homeshield.py --port 8080 --db custom.db

# Bind only to localhost (default is 0.0.0.0 so phones can hit it)
python run_homeshield.py --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from homeshield.server import create_app


def main():
    p = argparse.ArgumentParser(description="HomeShield dashboard")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--db", default="homeshield.db")
    p.add_argument("--snapshots", default="snapshots")
    p.add_argument("--person-photos", default="person_photos")
    p.add_argument("--intruder-photos", default="intruder_photos")
    p.add_argument("--no-autostart", action="store_true",
                   help="Don't auto-start cameras on boot (useful for debugging)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    app = create_app(
        db_path=args.db,
        snapshot_dir=args.snapshots,
        person_photos_dir=args.person_photos,
        intruder_photos_dir=args.intruder_photos,
        auto_start=not args.no_autostart,
    )

    print(f"\n[HomeShield] Dashboard: http://{args.host}:{args.port}/")
    print(f"[HomeShield] Database : {Path(args.db).resolve()}")
    print(f"[HomeShield] Snapshots: {Path(args.snapshots).resolve()}\n")

    app.run(host=args.host, port=args.port, threaded=True,
            debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
