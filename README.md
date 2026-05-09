# 🛡️ HomeShield

**HomeShield** is a centralized dashboard for **real-time CCTV monitoring** with three GPU-accelerated anomaly detectors running side-by-side on every camera feed. One unified pipeline draws bounding-box overlays on the live MJPEG stream, persists every alert to SQLite with an annotated snapshot, and pushes updates to the dashboard via Server-Sent Events.

Built as a Final-Year Project at the **International Islamic University Malaysia (IIUM), Kulliyyah of Information and Communication Technology**.

> One pipeline. Three detectors. Per-camera worker threads sharing one set of YOLO / ONNX models on the GPU.

---

## ✨ Features

### Detection
- 🤸 **Fall detection** — any Ultralytics YOLO pose checkpoint (`yolov8`, `yolo11`, `yolo26`) feeds a research-tuned **7-state finite-state machine**: *Standing → Walking → Sitting → Fall_Detected → Lying_After_Fall → Lying_Motionless → Inactivity*. A two-stage decision (peak descent velocity **plus** sustained horizontal posture) rejects controlled sit-downs and false positives.
- 🔥 **Fire & smoke detection** — custom YOLO weights covering fire and smoke classes, with **per-class cooldown** to debounce repeated alerts.
- 👤 **Face / intruder detection** — InsightFace **ArcFace (buffalo_l)** generates 512-dimensional embeddings, cosine-matched against your registered Persons gallery. Unknown faces are auto-logged to the **Intruders** list with a snapshot.

### Identity & alerts
- **Registered persons** gallery — enrol family members from a still photo or a live capture.
- **Auto-intruder logging** — anyone not in the gallery is snapshot-logged with timestamp and camera.
- **Real-time alert stream** — Server-Sent Events push new events to the dashboard the instant they're published.
- **Annotated snapshots** — every event is saved as a JPEG with the bounding box / pose skeleton / face label burned in.

### Zones
- **Polygon zones per camera** drawn directly on the live preview.
- **Safe zones** suppress lying / inactivity alerts (e.g. on a bed or sofa).
- **Danger zones** trigger a child-entry alert when a person crosses into them (kitchen, balcony, pool, etc.).

### Platform
- **Multi-camera** — webcams, IP cameras (RTSP), DroidCam, IP Webcam, or a video file for testing.
- **Live MJPEG streams** with detector overlays.
- **SQLite event log** in WAL mode + filesystem snapshots.
- **Hot-reloadable settings** — toggle detectors, swap pose models, change FPS / imgsz / thresholds without restarting cameras.
- **Mobile-friendly UI** — responsive Flask dashboard you can hit from your phone on the same network.

---

## 📋 Requirements

### Recommended setup (what this is tuned for)
- **OS:** Windows 11
- **Python:** 3.11 (Anaconda virtual environment)
- **GPU:** NVIDIA GPU with **CUDA 12.x** drivers (tested on RTX-class hardware)
- **PyTorch:** 2.x with CUDA build
- **RAM:** 16 GB+
- **Disk:** ~5 GB for weights, snapshots, and the SQLite database

### Minimum (CPU-only, no intruder detection)
You can run HomeShield on a machine without a GPU, but with caveats:
- **CPU-only** PyTorch and ONNX Runtime work for **Fall** and **Fire** detection — expect 5–15 FPS on a modern laptop CPU at small image sizes.
- **Face / intruder detection should be disabled** — InsightFace inference on CPU is too slow to keep up with a live feed and will tank the pose framerate.
- 8 GB RAM minimum.
- Lower the **imgsz** and **target FPS** in *Settings → Performance* aggressively (e.g. imgsz 416, FPS cap 10).

### Python version note
**Python 3.11 is required.** The bundled InsightFace wheel (`insightface-0.7.3-cp311-cp311-win_amd64.whl`) is compiled specifically for **CPython 3.11 on Windows x64**. Other Python versions will fail to install it from source unless you have MSVC build tools configured. If you must use a different Python version, you'll need to compile InsightFace yourself or find a matching pre-built wheel.

---

## ⚡ Installation

### 1. Get Python 3.11
Install Python 3.11 via Anaconda (recommended) or the official installer:

```bash
# With Anaconda
conda create -n homeshield python=3.11
conda activate homeshield
```

Verify:

```bash
python --version
# Python 3.11.x
```

### 2. Clone and create a venv

```bash
git clone https://github.com/<your-user>/homeshield.git
cd homeshield

# If you skipped the conda step above, create a plain venv:
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install PyTorch (do this BEFORE requirements.txt)
**This step is critical.** If you let pip resolve PyTorch from `requirements.txt`, it will pull the **CPU-only** build and silently overwrite a working CUDA install. Install PyTorch with the matching CUDA index URL **first**:

```bash
# CUDA 12.x (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only fallback (no GPU available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify CUDA is detected:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available())"
# cuda True
```

### 4. Install InsightFace (Windows-specific)
On Windows without MSVC build tools, pip cannot compile InsightFace from source. Use the **bundled pre-built wheel** instead:

```bash
pip install Face_Detection/insightface-0.7.3-cp311-cp311-win_amd64.whl
```

On macOS / Linux (or Windows with MSVC installed), `pip install insightface==0.7.3` works directly and `requirements.txt` will handle it.

### 5. Install the rest

```bash
pip install -r requirements.txt
```

The pinned versions avoid two real ABI breaks:
- `numpy<2.0` — InsightFace wheels are compiled against NumPy 1.x (88-byte dtype struct).
- `opencv-python<4.11` — OpenCV 4.11+ dropped NumPy 1.x support.

Verify the full stack loads cleanly:

```bash
python -c "import torch, cv2, ultralytics, insightface, flask; \
           print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
           print('cv2', cv2.__version__); \
           print('insightface', insightface.__version__)"
```

Expected output:

```
torch 2.x.x cuda True
cv2 4.10.0.84
insightface 0.7.3
```

### 6. Configure
- Drop YOLO **pose weights** (`*-pose.pt`) into `Fall_Detection/weights/`. The dashboard's **Settings → Fall Detection** dropdown auto-populates from whatever's in that folder.
- Make sure **Fire_Detection/best.pt** exists (the custom fire/smoke weights).
- The first time you run the app, it creates `homeshield.db`, `snapshots/`, `person_photos/`, and `intruder_photos/` automatically.

### 7. Run

```bash
python run_homeshield.py
```

Then open the dashboard at **http://localhost:5000/**.

Common flags:

```bash
# Custom port and DB
python run_homeshield.py --port 8080 --db custom.db

# Bind only to localhost (default 0.0.0.0 lets phones on the LAN reach it)
python run_homeshield.py --host 127.0.0.1

# Don't auto-start cameras on boot (useful for debugging)
python run_homeshield.py --no-autostart
```

---

## 📸 Camera setup

HomeShield accepts any source OpenCV's `VideoCapture` understands — webcams, RTSP / HTTP streams, and video files. Add cameras from **Live feeds → Add camera** in the dashboard.

### Built-in webcam
Use the integer device index. `0` is your default webcam:

```
Source: 0
```

If you have multiple webcams, try `1`, `2`, etc.

### TP-Link Tapo C200 / C210
Enable RTSP in the **Tapo app → Camera Settings → Advanced Settings → Camera Account** and set a username + password. Then point HomeShield at the RTSP URL:

```
Source: rtsp://<user>:<password>@<camera-ip>:554/stream1   # high quality
Source: rtsp://<user>:<password>@<camera-ip>:554/stream2   # lower quality, lower bandwidth
```

`stream2` is recommended for 24/7 monitoring — it stays well within the GPU budget when you're running multiple cameras.

### Android phone as camera (DroidCam, IP Webcam)
Both apps expose a phone's camera as a network stream:

- **DroidCam** (Wi-Fi mode):
  ```
  Source: http://<phone-ip>:4747/video
  ```
- **IP Webcam**:
  ```
  Source: http://<phone-ip>:8080/video
  ```

Make sure both devices are on the same Wi-Fi network and the phone screen stays on.

### Video file for testing
Drop in any local video file to dry-run the detectors without a live camera:

```
Source: C:/path/to/sample_fall.mp4
Source: ./test_videos/kitchen_fire.mp4
```

The pipeline loops the file by default, so you can leave it running while tuning thresholds.

---

## 🎯 Using the system

### Register your family members (before you turn on intruder detection)
1. Go to **Persons → Add person**.
2. Enter the person's name.
3. Either upload a clear, well-lit photo **or** capture one from any active camera.
4. Repeat for every household member.

> ⚠️ **Important:** Do this *before* enabling face detection. Otherwise, every face — including yours — will be logged as an intruder until they're enrolled.

### Set up zones
1. Open **Zones**, pick a camera from the list.
2. Click on the live preview to lay polygon vertices, then close the polygon.
3. Tag the zone:
   - **Safe zone** → suppresses lying / inactivity alerts inside it (use this for beds, sofas, recliners).
   - **Danger zone** → triggers a child-entry alert when anyone enters (kitchen, balcony, pool, stairs).
4. Save. Zones apply live, no restart needed.

### Watch the live feed
**Live feeds** is the main dashboard:
- Each camera shows the annotated MJPEG stream with bounding boxes, pose skeletons, and face labels overlaid.
- The right-hand event log streams every detection in real time via Server-Sent Events.
- Click any event to open its annotated snapshot.

### Handle intruders
When an unknown face appears, an entry pops into **Persons → Intruders** with a snapshot and timestamp. From there you can:
- **Enrol** them as a known person if they were a friend / relative.
- **Delete** the entry if it was a false positive (poor lighting, motion blur, partial face).
- **Export** the snapshot if you need to share it.

---

## ⚙️ Tuning for your hardware

All knobs live in **Settings** and apply live without restarting cameras (only swapping the `.pt` model file forces a model reload).

| Setting | Effect | When to adjust |
|---|---|---|
| **Pose model** | Smaller (`yolo11n-pose`) = faster, less accurate. `yolo26x-pose` = slowest, best accuracy. | Drop to `n` on CPU or low-end GPU. |
| **imgsz** | Inference resolution (320 / 416 / 640). | Lower for more FPS, higher for far-away subjects. |
| **Target FPS cap** | Caps the capture loop. | Set to 10–15 on CPU. |
| **FP16 (half-precision)** | Halves GPU memory and boosts throughput on supported NVIDIA cards. | Leave **on** for RTX cards; **off** for very old GPUs / CPU. |
| **Frame skipping** | Pose runs every frame; fire every 2nd; face every 5th. | Bump face skip if face inference is the bottleneck. |
| **Fire / Face enabled** | Toggle the heavier detectors. | Disable face on CPU-only setups. |
| **Sensitivity (fall)** | Adjusts the FSM thresholds for descent velocity and lying duration. | Raise if false-falls are common; lower if real falls are missed. |
| **Cooldowns** | Min seconds between repeat alerts of the same class. | Raise for noisy environments. |

---

## 🏗️ Architecture

```
                   ┌─────────────────────────────────────┐
                   │         Flask + Jinja2 UI          │
                   │  (MJPEG iframe + SSE event stream) │
                   └────────────┬────────────────────────┘
                                │
                  ┌─────────────┴──────────────┐
                  │       homeshield.server    │
                  │  /api/...   /stream/<cam>  │
                  └──────┬───────────────┬─────┘
                         │               │
                ┌────────▼─────┐   ┌─────▼──────┐
                │ CameraManager│   │  EventBus  │
                │ (cameras.py) │   │ (events.py)│
                └────┬────┬────┘   └─────┬──────┘
                     │    │              │
        ┌────────────┘    └──────────┐   │  daemon thread
        ▼                            ▼   ▼
┌─────────────────┐         ┌─────────────────┐
│  Camera worker  │   ...   │  Camera worker  │ ── JPEG encode
│ (pipeline.py)   │         │ (pipeline.py)   │ ── SQLite write
│  capture loop   │         │  capture loop   │ ── SSE broadcast
└──┬──┬──┬────────┘         └─────────────────┘
   │  │  │
   │  │  └─► Face daemon thread (InsightFace, every 5th frame)
   │  └────► Fire YOLO         (every 2nd frame)
   └───────► Pose YOLO + Fall FSM (every frame)

   All workers share ONE set of models loaded on GPU (Models singleton).
```

Key design choices:
- **Per-camera worker threads** share one set of YOLO / ONNX models on the GPU. Adding a camera does not duplicate VRAM.
- **Frame skipping**: pose runs every frame to keep the FSM responsive; fire runs every 2nd frame; face every 5th.
- **Async face inference** — face detection runs on its own daemon thread per camera so the capture loop runs at pose-only speed regardless of how slow `app.get()` is.
- **Non-blocking event publishing** offloads JPEG encoding and SQLite writes to a daemon thread, so the capture loop never waits on disk I/O.
- **Hot reloading** — toggling `fire_enabled` / `face_enabled` or tweaking imgsz / FPS / thresholds applies live without restarting cameras.

---

## 📁 Project structure

```
FYP/
├── Fall_Detection/                 # Fall pipeline (imported as a library)
│   ├── fall_detection.py           #   YOLO pose + 7-state FSM
│   ├── weights/                    #   drop any *-pose.pt YOLO weights here
│   │   ├── yolo11n-pose.pt
│   │   ├── yolo11m-pose.pt
│   │   ├── yolo11x-pose.pt
│   │   ├── yolo26n-pose.pt
│   │   ├── yolo26m-pose.pt
│   │   └── yolo26x-pose.pt
│   ├── requirements.txt
│   └── README.md
│
├── Fire_Detection/                 # Custom YOLO fire/smoke detector
│   ├── detect_fire.py
│   └── best.pt                     #   custom-trained weights
│
├── Face_Detection/                 # InsightFace reference + bundled wheel
│   ├── face_recognizer.py
│   └── insightface-0.7.3-cp311-cp311-win_amd64.whl
│
├── Icon/                           # Dashboard SVG icons
│   ├── LOGO.svg
│   ├── Detection.svg
│   ├── Fall.svg
│   ├── Fire.svg
│   ├── Face.svg
│   ├── Camera.svg
│   └── Notifcation.svg
│
├── homeshield/                     # The unified Flask app
│   ├── __init__.py
│   ├── server.py                   #   Flask API + MJPEG endpoints + SSE
│   ├── pipeline.py                 #   per-camera pipeline + Models + FaceWorker
│   ├── cameras.py                  #   multi-camera lifecycle manager
│   ├── events.py                   #   async EventBus + SQLite log + snapshots
│   ├── persons.py                  #   registered persons + intruder log
│   ├── zones.py                    #   polygon zone storage + point-in-polygon
│   ├── settings.py                 #   hot-reloadable settings store
│   ├── face.py                     #   InsightFace embedding + matching helpers
│   ├── annotator.py                #   bounding boxes, pose skeleton, labels
│   ├── db.py                       #   SQLite (WAL mode) connection + schema
│   ├── static/                     #   compiled CSS + JS
│   │   ├── app.css
│   │   └── app.js
│   └── templates/
│       └── index.html              #   single-page dashboard UI
│
├── run_homeshield.py               # Entry point (argparse + create_app)
├── requirements.txt                # Pinned, ABI-consistent deps
├── homeshield.db                   # SQLite event/persons/zones store (created on first run)
├── events.db                       # Legacy event store (kept for reference)
├── .gitignore
├── .gitattributes
└── README.md                       # ← you are here
```

---

## 🔐 Privacy & security

- **All inference runs locally.** No frames, embeddings, or events leave your machine. There is no cloud component, no telemetry, and no third-party API calls during detection.
- **Face embeddings, not photos**, are used for matching at runtime. The 512-dim ArcFace vectors live in the local SQLite DB. Original enrolment photos are kept in `person_photos/` so you can re-enrol after a model swap.
- **Intruder snapshots** are stored in `intruder_photos/` on disk. Delete them whenever you want — the entry in the UI will disappear.
- **Network exposure:** by default the server binds to `0.0.0.0:5000`, meaning **anyone on your local network can reach the dashboard.** If that's not what you want, run with `--host 127.0.0.1`. There is no built-in authentication — for remote access, put it behind a reverse proxy with HTTPS + auth (Caddy, nginx + basic auth, Cloudflare Tunnel, Tailscale).
- **RTSP credentials** for IP cameras are stored in plaintext in the SQLite DB. Treat the DB file accordingly. Anyone with read access to it can pull camera passwords.

---

## 🧪 Performance notes

Indicative numbers on an **RTX-class GPU + Ryzen / Intel desktop CPU**, measured in the dashboard's FPS chip:

| Setup | Cameras | Pose model | imgsz | Detectors on | Avg FPS / cam |
|---|---|---|---|---|---|
| RTX 3060, FP16 | 1× 1080p | `yolo11n-pose` | 640 | Fall | ~55 |
| RTX 3060, FP16 | 1× 1080p | `yolo11n-pose` | 640 | Fall + Fire + Face | ~28 |
| RTX 3060, FP16 | 3× 1080p | `yolo11n-pose` | 640 | Fall + Fire + Face | ~14 each |
| RTX 3060, FP16 | 1× 1080p | `yolo26x-pose` | 640 | Fall + Fire + Face | ~9 |
| Laptop CPU only | 1× 720p | `yolo11n-pose` | 416 | Fall + Fire | ~8 |
| Laptop CPU only | 1× 720p | `yolo11n-pose` | 416 | Fall + Fire + Face | ~2 (unusable) |

Bottlenecks, in practice:
- **Face inference on CPU** is the single biggest performance cliff — keep it on GPU or off entirely.
- **RTSP decode** (especially the Tapo `stream1` 2K mode) eats a surprising amount of CPU. Prefer `stream2` for 24/7 use.
- **Disk I/O** never blocks the capture loop because event publishing is fully async.

---

## 🔧 Troubleshooting

**`numpy.dtype size changed, may indicate binary incompatibility`**
You ended up on NumPy 2.x. Pin back to NumPy 1.x: `pip install "numpy<2.0"`.

**`torch.cuda.is_available()` is `False`**
You either installed the CPU-only PyTorch build, or your NVIDIA driver / CUDA runtime is mismatched. Reinstall PyTorch with `--index-url https://download.pytorch.org/whl/cu121` **after** uninstalling the existing torch.

**InsightFace fails to install on Windows**
Use the bundled wheel: `pip install Face_Detection/insightface-0.7.3-cp311-cp311-win_amd64.whl`. The wheel is built for **Python 3.11 / Windows x64** specifically.

**Pose model dropdown is empty**
HomeShield only lists files in `Fall_Detection/weights/` whose names end in `-pose.pt`. Heavy `x` variants exceed GitHub's per-file limit and are excluded by `.gitignore`. Download them from <https://github.com/ultralytics/assets/releases> or train your own.

**RTSP camera connects but stutters**
You're probably on `stream1` (2K). Switch to `stream2` (lower bitrate). If still bad, check that your CPU isn't pinned at 100% from H.264 decode.

**Phone (DroidCam / IP Webcam) won't connect**
Both devices must be on the **same Wi-Fi network** (not Wi-Fi vs guest network, not cellular). Open the URL in a browser first to confirm the stream is reachable, then paste it into HomeShield.

**Every family member gets logged as an intruder**
You enabled face detection before enrolling them. Disable face detection, enrol everyone in **Persons**, then re-enable.

**Dashboard reachable from your laptop but not your phone**
Server is bound to `127.0.0.1`. Restart with `--host 0.0.0.0` (the default), and check Windows Firewall isn't blocking inbound TCP/5000.

**Fall alerts on someone just sitting down**
Lower the **fall sensitivity** in Settings, or raise the descent-velocity threshold. The FSM also requires sustained horizontal posture, so very brief lie-downs shouldn't trigger.

---

## 🗺️ Development roadmap

**Near-term**
- Push notifications (Telegram / Pushover / NTFY) for fall and intruder alerts.
- Multi-user authentication on the dashboard with per-user camera ACLs.
- Timeline / day-view of events with filtering.
- Export an event window as a clip (with annotated frames stitched into MP4).

**Mid-term**
- Audio anomaly detection (glass break, smoke alarm, scream classification) as a fourth detector lane.
- ONNX-only deployment path so the same dashboard can run on lighter hardware (Jetson, Raspberry Pi 5 + Coral).
- Replace the per-class cooldown with a smarter event deduplicator (e.g. event-window clustering across cameras).
- First-class Docker compose with GPU runtime and a single `up` command.

**Stretch**
- Federated person gallery so multiple HomeShield nodes on a LAN share the registered Persons / Intruder DB.
- Activity classifier (cooking, sleeping, watching TV) for elderly-care use cases.

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` (or the header in `run_homeshield.py`) for the full text.

Third-party model weights and libraries retain their own licenses:
- **Ultralytics YOLO** — AGPL-3.0 (commercial use requires an Ultralytics enterprise license).
- **InsightFace** — MIT.
- **PyTorch, OpenCV, Flask, NumPy, SciPy, Shapely** — their respective open-source licenses.

If you intend to deploy HomeShield commercially, audit Ultralytics' AGPL terms first.

---

## 🙏 Acknowledgements

- **Developer:** Mohammad Salehuddin bin Iwan
- **Supervisor:** Andi Fitriah binti Abdul Kadir
- **Institution:** International Islamic University Malaysia (IIUM), **Kulliyyah of Information and Communication Technology**
- **Project title:** Final Year Project — *HomeShield: Centralized Dashboard for Real-Time CCTV Monitoring and Anomaly Detection*

Built on the shoulders of:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — pose, fire, and smoke detection.
- [InsightFace](https://github.com/deepinsight/insightface) — ArcFace face recognition.
- [PyTorch](https://pytorch.org/), [ONNX Runtime](https://onnxruntime.ai/), [OpenCV](https://opencv.org/), [Flask](https://flask.palletsprojects.com/), [Shapely](https://shapely.readthedocs.io/).

Special thanks to the open-source CV community for making real-time anomaly detection on consumer hardware actually viable.
