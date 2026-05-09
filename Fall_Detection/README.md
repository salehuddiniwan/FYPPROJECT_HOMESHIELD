# Fall Detection with YOLO26 Pose

A real-time **multi-person** fall detection system. It uses YOLO26 pose estimation + ByteTrack to assign a stable ID to every person in the frame, then runs an independent finite-state machine (Standing / Walking / Sitting / Fall_Detected / Lying_After_Fall / Lying_Motionless / Inactivity) per person. A fall on **any** tracked person triggers the global alert.

The detector does **not** evaluate frames in isolation. For each tracked person it computes a vector of indicators (trunk angle, aspect ratio, height ratio, centroid descent velocity, motion energy) expressed in **body-scale units** so they're independent of camera distance. A two-stage temporal decision (impact peak → sustained lying) drives the state transitions, which makes the system tolerant to keypoint glitches and rejects controlled sit-downs.

---

## 1. Project layout

```
YoloV26/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── fall_detection.py         # Main script
└── weights/                  # (created by you) put your *.pt model here
```

---

## 2. Create the conda environment

Open **Anaconda Prompt** (or any terminal where `conda` is on PATH) and run:

```bash
# Move into the project folder
cd C:\Users\Admin\Documents\anaconda_projects\YoloV26

# Create a fresh environment with Python 3.11
conda create -n yolo26 python=3.11 -y

# Activate it
conda activate yolo26
```

> Python 3.11 is recommended. Ultralytics supports 3.8–3.12, but 3.11 has the best wheel coverage on Windows.

---

## 3. Install PyTorch with GPU (CUDA) support

**This step must come before `pip install -r requirements.txt`**, otherwise pip will pull the CPU-only `torch` from PyPI.

First check your CUDA driver version:

```bash
nvidia-smi
```

Look at the top-right corner — it shows the **CUDA Version** your driver supports (e.g. `12.4`). Pick the matching wheel:

| Driver supports        | Install command                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| CUDA 12.1+             | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`                       |
| CUDA 12.4+ (recommended)| `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`                       |
| CUDA 11.8              | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`                       |
| No NVIDIA GPU          | `pip install torch torchvision`                                                                          |

Verify the GPU is visible to PyTorch:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

You should see something like `2.4.0+cu124 True NVIDIA GeForce RTX ...`. If `is_available()` prints `False`, your driver / CUDA wheel mismatch — re-check `nvidia-smi`.

---

## 4. Install the rest of the requirements

```bash
pip install -r requirements.txt
```

> Do not install `torch` from this file — it was intentionally left out so the GPU build from step 3 is preserved.

---

## 5. Get the YOLO26 pose weights

Place your YOLO26 pose model into the `weights/` folder. If the folder doesn't exist, create it:

```bash
mkdir weights
```

Drop the `.pt` file in (e.g. `weights/yolo26n-pose.pt`). The script will look for it automatically. If you don't have a YOLO26 weight yet, the script will fall back to downloading `yolo11n-pose.pt` from Ultralytics' hub the first time it runs.

You can override the path on the command line:

```bash
python fall_detection.py --model weights/yolo26m-pose.pt
```

Sizes available (smallest → largest): `n`, `s`, `m`, `l`, `x`. Larger models are more accurate but slower. For real-time on a webcam, `n` or `s` is usually the right pick.

---

## 6. Run

**Webcam (default camera 0):**

```bash
python fall_detection.py
```

**Video file:**

```bash
python fall_detection.py --source path\to\video.mp4
```

**Save annotated output:**

```bash
python fall_detection.py --source path\to\video.mp4 --save out.mp4
```

**Run on CPU (force):**

```bash
python fall_detection.py --device cpu
```

**Headless (no window, just log events):**

```bash
python fall_detection.py --source video.mp4 --no-show
```

**Switch tracker** (default is ByteTrack — faster; BoT-SORT is more accurate but slower):

```bash
python fall_detection.py --tracker botsort.yaml
```

Press `q` in the display window to quit.

### Multi-person behaviour

- Each detected person gets a persistent **track ID** (shown as `#3`, `#5`, etc. on screen).
- Each ID has its own `FeatureExtractor` and `FallDetector`, so the FSMs are fully independent.
- The **top banner** shows the global state. When **any** tracked person falls, the banner turns red and lists the offending IDs (e.g. `!! FALL ALERT !! Person(s) #5`).
- The **side panel** (with the metric numbers) only shows the largest person on screen, to keep the screen readable when several people are visible.
- IDs that haven't been seen for 2 seconds are cleaned up automatically (`id_cleanup_after_s` in `Config`).

---

## 7. How the detection works (research-tuned)

The pipeline is a literature-grade port of three ideas that consistently appear in high-accuracy skeleton-based fall papers:

**a) Body-scale normalization.** Every distance and velocity is expressed in *body-units*, where 1 body-unit = `||shoulder_mid − hip_mid||` measured per frame. This single change makes the metrics independent of camera distance — a hip drop of "1.6 body-units/sec" means the same thing whether the person is 2 m or 6 m from the camera. (Núñez-Marcos et al. 2022; PMC9185346 on 2D skeleton normalization.)

**b) Two-stage decision (impact → lying).** A real fall always shows two things in sequence: an impact phase with a *peak* centroid descent velocity, followed by a sustained lying phase. A controlled sit-down satisfies the lying part but never produces the velocity peak. The detector requires both to fire within ~2 seconds, which is what eliminates the most common false alarm. (PIFR PLOS ONE 2024; arXiv 2401.01587.)

**c) Rolling-max height reference.** Instead of a decaying running maximum (which drifts), we track the **largest keypoint-bbox height seen in the last 3 seconds** as the standing reference. A `current_h / reference_h` below 0.55 means the person has visibly collapsed. (Sensors 2025 — Enhanced HRNet+YOLO.)

Per-frame features computed (all in body-scale units except angle):

| Indicator                | Meaning                                                                |
| ------------------------ | ---------------------------------------------------------------------- |
| **trunk angle**          | shoulder→hip vector vs vertical (0° upright, 90° horizontal)           |
| **aspect ratio**         | width / height of the *keypoint* bbox (more stable than YOLO bbox)     |
| **height ratio**         | current keypoint-bbox height / max in last 3 s                         |
| **centroid velocity**    | mean keypoint vertical speed in **body-units / second**                |
| **motion energy**        | mean per-joint speed in body-units / second                            |

Decision rule:

```
Stage A (impact)  : peak centroid_velocity in last 0.5 s  >  impact_velocity_bu_s   (default 1.6 bu/s)
Stage B (lying)   : (is_horizontal AND is_low) sustained for 0.5 s
FALL DETECTED     : Stage A occurred within last 2.0 s AND Stage B is currently true
```

`is_horizontal = trunk_angle > 60°  OR  aspect_ratio > 1.3`
`is_low        = height_ratio < 0.55`

The full state machine matches the diagram you provided:

```
q0 Standing  ──walking pose──►  q1 Walking
q0 Standing  ──seated pose──►   q2 Sitting
q1 Walking   ──seated pose──►   q2 Sitting
q0 / q1 / q2 ──b ∧ c (vote)──►  q3 Fall_Detected
q3           ──still horiz────► q4 Lying_After_Fall
q4           ──elapsed > 5 s──► q5 Lying_Motionless
q2           ──no motion──────► q6 Inactivity
q5 / q6      ──becomes upright──► q0 Standing
```

---

## 8. Tuning

All thresholds live at the top of `fall_detection.py` in the `Config` dataclass. Because of body-scale normalization, the velocity numbers are now portable across cameras — you almost never have to retune them when you change camera distance.

The ones you'll touch most often (also exposed as CLI flags):

- `impact_velocity_bu_s` (default 1.6, flag `--impact-vel`) — peak descent velocity in body-units/sec required to register an impact. Lower it (e.g. 1.2) if real falls are missed; raise it (e.g. 2.0) if soft sit-downs trigger.
- `horizontal_angle_min` (default 60°, flag `--horizontal-deg`) — trunk angle considered "horizontal". Lower if the camera is mounted high looking down (perspective compresses the angle).
- `height_collapse_ratio` (default 0.55, flag `--collapse-ratio`) — current/reference height below which the person is "low". Lower (e.g. 0.45) for a stricter test.
- `sustain_s` (default 0.5 s) — how long Stage-B lying must hold before counting.
- `kp_ema_alpha` (default 0.6) — keypoint smoothing. Higher = less smoothing, more responsive but jitterier.

Watch the HUD panel in the bottom-right: every metric the FSM uses is shown live, including which boolean flags are firing. That's the fastest way to figure out which threshold needs adjusting for your scene.

---

## 9. Troubleshooting

| Symptom                                          | Fix                                                                       |
| ------------------------------------------------ | ------------------------------------------------------------------------- |
| `torch.cuda.is_available()` returns False        | Reinstall torch with the correct `--index-url` matching your driver       |
| `ModuleNotFoundError: ultralytics`               | `pip install -r requirements.txt` inside the activated env                |
| Webcam window is black                           | Try `--source 1` or `--source 2` for a different camera index             |
| Falls aren't detected                            | `--impact-vel 1.2 --horizontal-deg 50 --collapse-ratio 0.6`               |
| Too many false alarms when sitting fast          | `--impact-vel 2.0 --collapse-ratio 0.45`                                  |
| Detector flickers between Standing/Sitting       | Increase `sustain_s` to 0.8 in Config and raise `kp_ema_alpha` to 0.4     |
| Person far from camera, no detection             | Switch to a larger model (`yolo26m-pose.pt` or `yolo26x-pose.pt`)         |
