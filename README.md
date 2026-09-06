<div align="center">

# ArUco Robot Navigation

**Real-time marker detection, 3D pose estimation, trajectory prediction, collision detection, and vision-based navigation.**

<p>
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white" alt="C++17">
  <img src="https://img.shields.io/badge/OpenCV-4.7%2B-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV 4.7+">
  <img src="https://img.shields.io/badge/CMake-Build-064F8C?logo=cmake&logoColor=white" alt="CMake">
  <img src="https://img.shields.io/github/actions/workflow/status/mitul-mishra5/arUco-robot-navigation/build.yml?label=Build&logo=github" alt="GitHub Actions Build">
  <img src="https://img.shields.io/github/license/mitul-mishra5/arUco-robot-navigation" alt="MIT License">
</p>

</div>

---

## Overview

A real-time computer-vision pipeline that detects ArUco markers, estimates their 3D pose, predicts motion with a Kalman filter, checks for collisions against defined zones, and drives a simple state-machine navigator toward a target marker — all visualised live with an on-screen HUD and a bird's-eye (top-down) view.

Built with **OpenCV 4.7+** using the `cv::aruco::ArucoDetector` API.

> **Scope:** This project produces navigation *commands* (`STOP` / `MOVE_FORWARD` / `MOVE_BACKWARD` / `TURN_LEFT` / `TURN_RIGHT` / `HOLD`) from a vision pipeline. It does **not** drive physical motors or hardware. To control a real robot, wire the `NavOutput::cmd` value from `RobotNavigator::tick()` to your own motor-control code, such as a serial link to a microcontroller or a ROS topic.

## Features

| Feature | Description |
|---|---|
| **Marker detection** | `cv::aruco::ArucoDetector` with `DICT_4X4_100` and sub-pixel corner refinement |
| **Pose estimation** | `cv::solvePnP` with `SOLVEPNP_IPPE_SQUARE` |
| **Trajectory prediction** | Per-marker 6-state Kalman filter for position/velocity smoothing and future-position prediction |
| **Collision detection** | Configurable 3D danger/warning zones with time-to-impact estimation |
| **Robot navigator** | Finite-state machine: `SEARCHING → APPROACHING → ALIGNING → HOLDING`, plus `EMERGENCY` |
| **Live visualisation** | On-screen HUD and bird's-eye top-down radar view |
| **Performance logging** | Per-frame timing data and average-FPS summary |

## Project Layout

```text
.
├── src/
│   └── main.cpp          # Full pipeline
├── .github/
│   └── workflows/
│       └── build.yml     # GitHub Actions CI
├── CMakeLists.txt        # CMake build configuration
├── Makefile              # g++ / pkg-config build
├── .gitignore
├── LICENSE
└── README.md
```

> The `markers/` folder (generated marker PNGs) and `evaluation/` folder (metrics CSV) are created at runtime and are git-ignored.

## Requirements

- A **C++17** compiler (GCC, Clang, or MSVC via MSYS2/MinGW)
- **OpenCV 4.7+** built with the `opencv_contrib` `aruco` module
- A webcam, or a video file to run against

### Installing OpenCV

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install libopencv-dev libopencv-contrib-dev cmake build-essential
```

> On many Ubuntu releases (e.g. 22.04/24.04), this installs OpenCV 4.5–4.6, which predates `cv::aruco::ArucoDetector` and won't compile this project. Check with `pkg-config --modversion opencv4`; if it is below 4.7, build OpenCV from source instead (see `.github/workflows/build.yml` for a working from-source recipe), or install a newer version via conda-forge or vcpkg.

**Windows (MSYS2 / MinGW)**

```bash
pacman -S mingw-w64-x86_64-opencv mingw-w64-x86_64-cmake mingw-w64-x86_64-toolchain
```

**macOS (Homebrew)**

```bash
brew install opencv cmake
```

## Build

### Option A — CMake (recommended)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/aruco_navigation --help
```

### Option B — Makefile / g++ directly

```bash
make

# or manually:
g++ -std=c++17 src/main.cpp -o aruco_navigation $(pkg-config --cflags --libs opencv4)
```

## Usage

### 1. Generate printable markers

```bash
./aruco_navigation --generate-markers
# writes markers/marker_0.png ... marker_9.png
```

Print `marker_0.png` (or any marker) on paper.

### 2. Run

```bash
./aruco_navigation                     # webcam
./aruco_navigation --video myvideo.mp4 # video file
./aruco_navigation --target 2          # navigate toward marker ID 2
./aruco_navigation --marker-len 0.08   # marker is 8 cm wide
./aruco_navigation --calib calib.yaml  # use a real camera calibration
./aruco_navigation --no-topdown        # disable the bird's-eye window
```

### Command-line Options

```text
--video <file>       video file (default: webcam)
--calib <yaml>       camera calibration file
--marker-len <m>     marker side length in metres, 0 < m <= 5.0 (default 0.05)
--target <id>        marker ID to navigate to, 0-99 (default 0)
--pred <n>           number of future steps to predict, 1-100 (default 10)
--no-topdown         disable bird's-eye window
--generate-markers   save marker PNGs and exit
--help               show usage
```

### Keys While Running

| Key | Action |
|---|---|
| `q` / Esc | Quit |
| `r` | Reset navigator + Kalman filters |
| `s` | Save the current frame as a PNG |

## Navigation State Machine

```text
SEARCHING
    │
    ▼
APPROACHING
    │
    ▼
ALIGNING
    │
    ▼
HOLDING

Any state ──────► EMERGENCY
```

The navigator generates high-level commands from the detected target marker and does not directly control motors.

## Camera Calibration

Without `--calib`, the app falls back to an approximate default intrinsic matrix for a generic 640×480 webcam. This is suitable for demos, but not for accurate metric distance measurements.

For real measurements, calibrate your camera with OpenCV's `calibrateCamera` workflow using a checkerboard pattern and save the result as a YAML file containing:

- `camera_matrix`
- `distortion_coefficients`
- `image_width`
- `image_height`

See `CameraParams::fromFile` / `CameraParams::save` in `src/main.cpp` for the expected format.

## Performance Metrics

Every run **appends** per-frame detection, pose, prediction, and total timings to `evaluation/metrics.csv`. The header is written only when the file does not already contain data.

Each row is tagged with a `run_id` based on the run's start time as a Unix timestamp, so data from different runs remains distinguishable even though the per-run `frame` counter restarts at 0.

An average-FPS summary is also printed on exit, making it easy to compare performance across machines or parameter settings.

## Continuous Integration

GitHub Actions automatically builds **OpenCV 4.9** from source and then builds this project using the workflow in `.github/workflows/build.yml`.

The repository currently uses the workflow as a build check for every push.

## License

Released under the [MIT License](LICENSE).
